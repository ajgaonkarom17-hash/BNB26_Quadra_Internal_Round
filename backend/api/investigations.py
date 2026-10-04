"""
Investigations router  (mounted at /api).

Endpoints (section 15 of the spec):
    POST /api/investigations
    POST /api/investigations/{id}/upload
    POST /api/investigations/{id}/analyze
    GET  /api/investigations
    GET  /api/investigations/{id}
    GET  /api/investigations/{id}/evidence
    GET  /api/investigations/{id}/graph

Plus a few read-only helpers the dashboard needs:
    GET  /api/investigations/{id}/results
    GET  /api/investigations/{id}/cross-modal
    GET  /api/investigations/{id}/explanation
    DELETE /api/investigations/{id}
    DELETE /api/evidence/{id}
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.database import db
from backend.models.schemas import CreateInvestigation
from backend.services.pipeline import run_analysis
from backend.services.explanation import build_explanation
from backend.services import cross_modal, uncertainty, fusion, evidence_graph
from backend.utils import upload as upload_utils

router = APIRouter()


def _representative_analyses(analyses: list[dict]) -> dict:
    """Choose the same usable, highest-risk evidence as the analysis pipeline."""
    grouped = {}
    for analysis in analyses:
        grouped.setdefault(analysis["modality"], []).append(analysis)

    selected = {}
    for modality, rows in grouped.items():
        usable = [row for row in rows
                  if (row.get("detail") or {}).get("quality") != "unusable"]
        candidates = usable or rows
        selected[modality] = max(
            candidates,
            key=lambda row: (float(row.get("score") or 0), row.get("created_at", "")),
        )
    return selected


# ---------------------------------------------------------------------------
# Create / list / detail
# ---------------------------------------------------------------------------
@router.post("/investigations")
def create_investigation(payload: CreateInvestigation):
    inv_id = str(uuid.uuid4())
    investigation = db.create_investigation(inv_id, payload.name.strip())
    return investigation


@router.get("/investigations")
def list_investigations():
    return {"investigations": db.list_investigations()}


@router.get("/investigations/{inv_id}")
def get_investigation(inv_id: str):
    inv = db.get_investigation(inv_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")
    evidence = db.list_evidence(inv_id)
    analyses = db.list_analyses(inv_id)
    cross = db.list_cross_modal(inv_id)
    inv["evidence_count"] = len(evidence)
    return {
        "investigation": inv,
        "evidence": evidence,
        "analysis_count": len(analyses),
        "cross_modal_count": len(cross),
    }


@router.delete("/investigations/{inv_id}")
def delete_investigation(inv_id: str):
    if not db.get_investigation(inv_id):
        raise HTTPException(status_code=404, detail="Investigation not found")
    # remove stored files
    from backend.config import UPLOAD_DIR
    import shutil
    folder = UPLOAD_DIR / inv_id
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    db.delete_investigation(inv_id)
    return {"ok": True, "message": "Investigation deleted"}


# ---------------------------------------------------------------------------
# Evidence / upload
# ---------------------------------------------------------------------------
@router.get("/investigations/{inv_id}/evidence")
def get_evidence(inv_id: str):
    if not db.get_investigation(inv_id):
        raise HTTPException(status_code=404, detail="Investigation not found")
    items = db.list_evidence(inv_id)
    for item in items:
        analysis = db.get_analysis(item["id"])
        item["analysis"] = analysis
    return {"evidence": items}


@router.post("/investigations/{inv_id}/upload")
async def upload_evidence(inv_id: str, file: UploadFile = File(...)):
    if not db.get_investigation(inv_id):
        raise HTTPException(status_code=404, detail="Investigation not found")

    filename = upload_utils.safe_filename(file.filename or "file")
    modality = upload_utils.detect_modality(filename, file.content_type)
    if modality is None:
        raise HTTPException(
            status_code=400,
            detail=(f"Unsupported file type '{filename}'. Supported: image, "
                    "video, audio, and text/pdf documents."),
        )

    data = await file.read()
    size_error = upload_utils.check_size(len(data))
    if size_error:
        raise HTTPException(status_code=400, detail=size_error)

    stored_path = upload_utils.new_stored_path(inv_id, filename)
    try:
        with open(stored_path, "wb") as fh:
            fh.write(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not store file: {exc}")

    evidence_id = str(uuid.uuid4())
    item = {
        "id": evidence_id,
        "investigation_id": inv_id,
        "filename": filename,
        "modality": modality,
        "content_type": file.content_type or "",
        "size_bytes": len(data),
        "stored_path": stored_path,
    }
    db.add_evidence(item)
    return {"ok": True, "evidence_id": evidence_id, "modality": modality,
            "filename": filename, "size_bytes": len(data)}


@router.delete("/evidence/{evidence_id}")
def delete_evidence(evidence_id: str):
    item = db.get_evidence(evidence_id)
    if not item:
        raise HTTPException(status_code=404, detail="Evidence not found")
    import os
    try:
        if item.get("stored_path") and os.path.exists(item["stored_path"]):
            os.remove(item["stored_path"])
    except Exception:
        pass
    db.delete_evidence(evidence_id)
    return {"ok": True, "message": "Evidence deleted"}


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------
@router.post("/investigations/{inv_id}/analyze")
def analyze(inv_id: str):
    if not db.get_investigation(inv_id):
        raise HTTPException(status_code=404, detail="Investigation not found")
    if not db.list_evidence(inv_id):
        raise HTTPException(status_code=400, detail="No evidence uploaded yet.")
    db.update_investigation(inv_id, status="analyzing")
    try:
        result = run_analysis(inv_id)
    except Exception as exc:
        db.update_investigation(inv_id, status="error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
    if result.get("error"):
        db.update_investigation(inv_id, status="error")
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# Results / graph / explanation (read back from DB)
# ---------------------------------------------------------------------------
@router.get("/investigations/{inv_id}/results")
def get_results(inv_id: str):
    inv = db.get_investigation(inv_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Investigation not found")
    evidence = db.list_evidence(inv_id)
    analyses = db.list_analyses(inv_id)
    cross = db.list_cross_modal(inv_id)

    by_modality = _representative_analyses(analyses)
    filenames = {item["id"]: item["filename"] for item in evidence}
    for analysis in by_modality.values():
        analysis["filename"] = filenames.get(analysis["evidence_id"], "")

    comparisons = [{
        "pair": c["pair"],
        "consistency": c["consistency"],
        "label": c["label"],
        "conflicts": c["conflicts"],
        "method": c["method"],
    } for c in cross]

    cross_summary = cross_modal.aggregate_inconsistency(comparisons)

    # rebuild lightweight individual results for the explanation module
    class _Res:
        def __init__(self, row, filename):
            self.modality = row["modality"]
            self.evidence_id = row["evidence_id"]
            self.filename = filename
            self.score = row.get("score") or 0.0
            self.label = row.get("label")
            self.confidence = 0.5
            self.features = row.get("features", {})
            self.detail = row.get("detail", {})
            self.findings = row.get("findings", [])
            # quality is persisted inside detail by the pipeline
            self.quality = (self.detail or {}).get("quality", "ok")
            self.error = None

    def _filename_for(evidence_id):
        return next((e["filename"] for e in evidence if e["id"] == evidence_id), "")

    individual_results = {}
    unusable_results = {}
    for m, a in by_modality.items():
        res = _Res(a, _filename_for(a["evidence_id"]))
        if res.quality == "unusable":
            unusable_results[m] = res
        else:
            individual_results[m] = res

    available = list(individual_results.keys())
    coverage = uncertainty.evidence_coverage(available)
    individual_risks = {m: r.score for m, r in individual_results.items()}
    mask = fusion.availability_mask(available)
    fusion_result = fusion.fuse(individual_risks, {}, mask, cross_summary, coverage)

    unc = uncertainty.analyze_uncertainty(available, comparisons,
                                          individual_results, fusion_result, coverage,
                                          unusable_results=unusable_results)
    expl = build_explanation(individual_results, comparisons, unc, fusion_result)

    # video timeline
    video_timeline = None
    if "video" in by_modality:
        vt = by_modality["video"].get("detail", {})
        video_timeline = {
            "duration": by_modality["video"].get("features", {}).get("duration_seconds", 0),
            "suspicious": vt.get("suspicious_timestamps", []),
            "frames": vt.get("frame_timeline", []),
        }

    return {
        "investigation": inv,
        "coverage": coverage,
        "fusion": fusion_result,
        "cross_modal": {"comparisons": comparisons, "summary": cross_summary},
        "uncertainty": unc,
        "explanation": expl,
        "individual": {m: by_modality[m] for m in by_modality},
        "video_timeline": video_timeline,
    }


@router.get("/investigations/{inv_id}/graph")
def get_graph(inv_id: str):
    if not db.get_investigation(inv_id):
        raise HTTPException(status_code=404, detail="Investigation not found")
    evidence = db.list_evidence(inv_id)
    relationships = db.list_relationships(inv_id)

    # Rebuild the full graph (including entity/claim nodes) from stored analyses.
    entries = {}
    analyses = db.list_analyses(inv_id)
    representatives = _representative_analyses(analyses)
    for modality, a in representatives.items():
        ev = next((e for e in evidence if e["id"] == a["evidence_id"]), None)
        if not ev:
            continue

        class _Res:
            def __init__(self, row, ev):
                self.modality = row["modality"]
                self.evidence_id = row["evidence_id"]
                self.filename = ev["filename"]
                self.score = row.get("score") or 0.0
                self.label = row.get("label")
                self.features = row.get("features", {})
                self.detail = row.get("detail", {})
                self.findings = row.get("findings", [])
                self.quality = "ok"

        entries[modality] = {
            "evidence_id": a["evidence_id"],
            "result": _Res(a, ev),
            "embedding": None,
            "stored_path": ev.get("stored_path", ""),
        }
    comparisons = [{
        "pair": c["pair"], "consistency": c["consistency"], "label": c["label"],
        "conflicts": c["conflicts"], "method": c["method"],
        "breakdown": {},
    } for c in db.list_cross_modal(inv_id)]

    if entries:
        graph = evidence_graph.build_graph(inv_id, entries, comparisons)
    else:
        graph = {"nodes": [], "edges": []}
    return {"graph": graph, "stored_relationships": relationships}


@router.get("/investigations/{inv_id}/cross-modal")
def get_cross_modal(inv_id: str):
    if not db.get_investigation(inv_id):
        raise HTTPException(status_code=404, detail="Investigation not found")
    return {"cross_modal": db.list_cross_modal(inv_id)}
