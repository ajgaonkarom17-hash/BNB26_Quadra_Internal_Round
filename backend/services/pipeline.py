"""
ANALYSIS ORCHESTRATOR.

Runs the full TrustLayer pipeline for one investigation:

    files -> individual analysis -> embeddings -> common representation
          -> cross-modal reasoning -> fusion -> evidence graph
          -> uncertainty -> explanation -> persisted result

This module is intentionally the ONLY place that knows the order of steps, so
individual services stay decoupled and replaceable.
"""

from __future__ import annotations

import traceback
from typing import Optional

import numpy as np

from backend.config import EMBEDDING_DIM, MODE
from backend.database import db
from backend.services import (
    image_analyzer, video_analyzer, audio_analyzer, document_analyzer,
    cross_modal, fusion, uncertainty, evidence_graph, explanation,
)
from backend.services.fusion import MODALITIES, availability_mask

ANALYZERS = {
    "image": image_analyzer.analyze,
    "video": video_analyzer.analyze,
    "audio": audio_analyzer.analyze,
    "document": document_analyzer.analyze,
}


def _uuid() -> str:
    import uuid
    return str(uuid.uuid4())


def run_analysis(investigation_id: str) -> dict:
    evidence_items = db.list_evidence(investigation_id)
    if not evidence_items:
        return {"error": "No evidence uploaded for this investigation."}

    db.clear_analysis_data(investigation_id)

    completed_steps = []

    # ---- 1. individual modality analysis --------------------------------
    results_by_evidence = {}
    grouped: dict = {m: [] for m in MODALITIES}
    errors = []

    for item in evidence_items:
        modality = item["modality"]
        analyzer = ANALYZERS.get(modality)
        meta = {
            "evidence_id": item["id"],
            "filename": item["filename"],
            "investigation_id": investigation_id,
            "stored_path": item.get("stored_path", ""),
            "mode": MODE,
        }
        if analyzer is None:
            errors.append(f"Unsupported modality for {item['filename']}")
            continue
        try:
            result = analyzer(item["stored_path"], meta)
        except Exception as exc:  # never crash the whole pipeline
            errors.append(f"{item['filename']}: {exc}")
            traceback.print_exc()
            continue

        results_by_evidence[item["id"]] = result
        grouped[modality].append({
            "evidence_id": item["id"],
            "result": result,
            "embedding": result.embedding,
            "stored_path": item.get("stored_path", ""),
            "quality": result.quality,
        })

        # persist individual analysis
        detail = dict(result.detail or {})
        detail["quality"] = result.quality
        db.add_analysis({
            "id": _uuid(),
            "evidence_id": item["id"],
            "investigation_id": investigation_id,
            "modality": modality,
            "score": result.score,
            "label": result.label,
            "features": result.features,
            "findings": [f.to_dict() for f in result.findings],
            "engine": result.engine,
            "detail": detail,
        })
        db.update_evidence(item["id"], individual_score=result.score,
                           individual_label=result.label)

    completed_steps += [
        "File intake",
        "Individual modality analysis",
        "Feature / embedding extraction",
        "Common representation",
    ]

    # ---- choose one representative per modality -------------------------
    entries = {}
    duplicates = {}
    unusable = {}     # modality -> result that was uploaded but could not be used
    for modality, group in grouped.items():
        if not group:
            continue
        usable = [g for g in group if g["result"].quality != "unusable"]
        if usable:
            pool_sorted = sorted(usable, key=lambda g: g["result"].score, reverse=True)
            entries[modality] = pool_sorted[0]
        else:
            # uploaded but nothing usable -> keep for missing/unusable reporting
            pool_sorted = sorted(group, key=lambda g: g["result"].score, reverse=True)
            entries_unusable = pool_sorted[0]
            unusable[modality] = entries_unusable["result"]
        if len(group) > 1:
            duplicates[modality] = [g["result"].filename for g in group]

    available = list(entries.keys())
    mask = availability_mask(available)
    coverage = uncertainty.evidence_coverage(available)

    # ---- 2. cross-modal reasoning ---------------------------------------
    combined_embeddings = {m: entries[m]["embedding"] for m in available}
    comparisons = cross_modal.compare_all(entries)

    # ---- 2b. structured claim extraction + comparison -------------------
    per_modality_claims: dict = {}
    for m in available:
        res = entries[m]["result"]
        detail = res.detail or {}
        claims = list(detail.get("claims") or [])
        if m == "document":
            from backend.services.claims import claims_from_entities
            claims += claims_from_entities(
                detail.get("entities", {}), "document",
                entries[m]["evidence_id"], 0.8, "rule-extraction")
        if m == "image":
            from backend.services.claim_extraction import claims_from_text
            ic = detail.get("image_claims") or {}
            from backend.services.claims import (normalize_datetime,
                                                 normalize_location,
                                                 make_claim)
            for dt in ic.get("datetimes", []) or []:
                n = normalize_datetime(dt)
                if n:
                    claims.append(make_claim("timestamp", n, dt, "image",
                                             entries[m]["evidence_id"], 0.7, "ocr"))
            for t in ic.get("times_of_day", []) or []:
                claims.append(make_claim("time_of_day", t, t, "image",
                                         entries[m]["evidence_id"], 0.7, "ocr"))
            for p in ic.get("places", []) or []:
                claims.append(make_claim("location", normalize_location(p), p, "image",
                                         entries[m]["evidence_id"], 0.7, "ocr"))
        per_modality_claims[m] = claims

    from backend.services.claim_compare import compare_claims
    claim_findings = compare_claims(per_modality_claims)

    # fold claim findings into the per-pair comparison rows so the UI, fusion
    # penalty and evidence graph all see them
    for cf in claim_findings:
        pair = "_".join(sorted((cf["modality_a"], cf["modality_b"])))
        target = next((c for c in comparisons if c["pair"] == pair), None)
        if cf["kind"] == "conflict":
            cmap = {"timestamp": "timestamp_conflict",
                    "time_of_day": "time_of_day_conflict",
                    "location": "location_conflict",
                    "person": "identity_uncertain"}
            conflict = {"type": cmap.get(cf["claim_type"], f"claim_{cf['claim_type']}_conflict"),
                        "description": cf["description"],
                        "severity": cf["severity"]}
            if target is not None:
                target["conflicts"].append(conflict)
                penalty = 0.18 if cf["severity"] == "high" else 0.10
                target["consistency"] = round(max(0.0, target["consistency"] - penalty), 4)
                target["breakdown"]["conflict_penalty"] = round(
                    target["breakdown"].get("conflict_penalty", 0) + penalty, 4)
                if target["label"] == "CONSISTENT":
                    target["label"] = "MIXED"
        elif cf["kind"] == "agree" and target is not None:
            target["breakdown"].setdefault("claim_agreements", []).append(cf["description"])
        elif cf["kind"] == "unavailable":
            note = {"type": "timestamp_missing" if cf["claim_type"] in ("timestamp", "time_of_day", "date")
                    else f"claim_{cf['claim_type']}_unavailable",
                    "description": cf["description"], "severity": "low"}
            if target is not None and note["description"] not in [c["description"] for c in target["conflicts"]]:
                target["conflicts"].append(note)

    cross_summary = cross_modal.aggregate_inconsistency(comparisons)
    completed_steps.append("Cross-modal consistency analysis")
    completed_steps.append("Claim extraction & comparison")

    # ---- 3. fusion ------------------------------------------------------
    individual_risks = {m: entries[m]["result"].score for m in available}
    fusion_result = fusion.fuse(individual_risks, combined_embeddings, mask,
                                cross_summary, coverage)
    completed_steps.append("Multimodal fusion")

    # ---- 4. uncertainty -------------------------------------------------
    individual_results = {m: entries[m]["result"] for m in available}
    uncertainty_result = uncertainty.analyze_uncertainty(
        available, comparisons, individual_results, fusion_result, coverage,
        unusable_results=unusable)
    completed_steps.append("Uncertainty analysis")

    # ---- 5. evidence graph ----------------------------------------------
    graph = evidence_graph.build_graph(investigation_id, entries, comparisons)
    completed_steps.append("Evidence graph construction")

    # ---- 6. explanation -------------------------------------------------
    explanation_result = explanation.build_explanation(
        individual_results, comparisons, uncertainty_result, fusion_result)
    completed_steps.append("Explainable assessment")
    completed_steps.append("Final assessment")

    # ---- persist cross-modal + relationships ----------------------------
    for comp in comparisons:
        db.add_cross_modal({
            "id": _uuid(),
            "investigation_id": investigation_id,
            "pair": comp["pair"],
            "consistency": comp["consistency"],
            "label": comp["label"],
            "conflicts": comp["conflicts"],
            "method": comp["method"],
        })
    for edge in graph["edges"]:
        if edge["relationship"] in ("mentions", "asserts_claim"):
            continue  # entity edges are regenerated, keep DB lean
        db.add_relationship({
            "id": edge["id"],
            "investigation_id": investigation_id,
            "source": edge["source"],
            "target": edge["target"],
            "relationship": edge["relationship"],
            "score": edge["score"],
            "detail": edge["detail"],
        })

    # ---- update investigation summary -----------------------------------
    db.update_investigation(
        investigation_id,
        status="analyzed",
        assessment=uncertainty_result["final_assessment"],
        confidence=fusion_result["confidence_label"],
        confidence_score=fusion_result["confidence"],
        coverage=coverage,
        risk_score=fusion_result["risk"],
        summary=_summary_text(uncertainty_result, fusion_result),
    )

    # ---- video timeline (for the frontend) ------------------------------
    video_result = individual_results.get("video")
    video_timeline = None
    if video_result is not None:
        video_timeline = {
            "duration": video_result.features.get("duration_seconds", 0),
            "suspicious": video_result.detail.get("suspicious_timestamps", []),
            "frames": video_result.detail.get("frame_timeline", []),
        }

    return {
        "investigation_id": investigation_id,
        "mode": MODE,
        "steps": completed_steps,
        "available_modalities": available,
        "modality_mask": [int(x) for x in mask.tolist()],
        "coverage": coverage,
        "fusion": fusion_result,
        "cross_modal": {
            "comparisons": comparisons,
            "summary": cross_summary,
        },
        "uncertainty": uncertainty_result,
        "explanation": explanation_result,
        "graph": graph,
        "video_timeline": video_timeline,
        "individual": {m: entries[m]["result"].to_dict() for m in available},
        "duplicates": duplicates,
        "errors": errors,
    }


def _summary_text(uncertainty_result: dict, fusion_result: dict) -> str:
    a = uncertainty_result["final_assessment"]
    c = fusion_result["confidence_label"]
    n = len(uncertainty_result["conflicts"])
    return (f"{a} assessment with {c} confidence. "
            f"{n} cross-modal conflict(s) detected. "
            "Heuristic prototype analysis - not proof.")
