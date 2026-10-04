"""
EXPLAINABILITY.

Turns the raw numbers into a decision-support narrative. We never output only
"FAKE 91%". Instead every investigation gets:

  * why_flagged      - the strongest reasons *against* authenticity
  * supporting       - what argues *for* authenticity
  * conflicting      - contradictions between pieces of evidence
  * missing          - what evidence was unavailable
  * limitations      - what the system cannot reliably determine
  * what_changed     - a per-aspect summary ("Face -> ...", "Timestamp -> ...")

All wording is deliberately hedged ("indicator", "possible", "suggests").
"""

from __future__ import annotations

from backend.config import DISCLAIMER
from backend.services.fusion import MODALITIES


def _severity_rank(sev: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(sev, 0)


def build_explanation(
    individual_results: dict,
    comparisons: list,
    uncertainty: dict,
    fusion_result: dict,
) -> dict:
    why_flagged = []
    supporting = []
    conflicting = []
    missing = []

    # ---- individual findings -------------------------------------------
    for modality, res in individual_results.items():
        if res is None:
            continue
        for finding in res.findings:
            f = finding if isinstance(finding, dict) else finding.to_dict()
            item = {
                "source": modality,
                "title": f["title"],
                "detail": f["detail"],
                "severity": f.get("severity", "info"),
            }
            if f.get("kind") == "contradicts":
                why_flagged.append(item)
            elif f.get("kind") == "supports":
                supporting.append(item)
            elif f.get("kind") == "missing":
                missing.append(item)

    # ---- cross-modal conflicts -----------------------------------------
    for comp in comparisons:
        for conflict in comp["conflicts"]:
            conflicting.append({
                "source": comp["pair"],
                "title": conflict["type"].replace("_", " ").title(),
                "detail": conflict["description"],
                "severity": conflict["severity"],
            })
        if comp["consistency"] <= 0.40:
            why_flagged.append({
                "source": comp["pair"],
                "title": f"Low cross-modal consistency ({comp['consistency']:.2f})",
                "detail": (f"{comp['pair'].replace('_', ' / ')} agree poorly in the "
                           "common representation and domain checks."),
                "severity": "medium",
            })

    # sort by severity, most important first
    why_flagged.sort(key=lambda x: _severity_rank(x.get("severity", "info")), reverse=True)
    conflicting.sort(key=lambda x: _severity_rank(x.get("severity", "info")), reverse=True)

    # ---- missing evidence ----------------------------------------------
    for m in uncertainty["missing"]:
        missing.append({
            "source": m["modality"],
            "title": f"Missing: {m['modality']}",
            "detail": f"{m['modality']} evidence is {m['reason']}.",
            "severity": "low",
        })

    # ---- limitations (always shown, honest) ----------------------------
    limitations = [
        "Individual detectors are lightweight heuristics (or small trained "
        "heads) and are not forensic-grade.",
        "This is NOT an AI-text detector.",
        "Suspicious video timestamps are INDICATORS, not confirmed manipulation "
        "localization.",
        "Consistency scores are heuristic, not calibrated probabilities.",
        DISCLAIMER,
    ]
    if "document" in individual_results and individual_results["document"] is not None:
        d = individual_results["document"]
        if d.detail.get("model_applied") is False and d.features.get("extractor") == "minimal-pdf-fallback":
            limitations.append(
                "PDF text was extracted with a minimal fallback parser; some "
                "content may be missing. Install PyMuPDF for better extraction.")
    if missing:
        limitations.append(
            "Cross-modal reasoning is limited when modalities are missing; "
            "conclusions rely on fewer relationships.")

    # ---- what changed ---------------------------------------------------
    what_changed = _what_changed(individual_results, comparisons)

    # ---- top-line reasons ----------------------------------------------
    top_reasons = [w["title"] for w in why_flagged[:4]]

    return {
        "why_flagged": why_flagged,
        "supporting": supporting,
        "conflicting": conflicting,
        "missing": missing,
        "limitations": limitations,
        "what_changed": what_changed,
        "top_reasons": top_reasons,
    }


def _what_changed(individual_results: dict, comparisons: list) -> list:
    """A compact per-aspect status card used on the results page."""
    out = []

    # image aspect
    img = individual_results.get("image")
    vid = individual_results.get("video")
    if img is not None:
        out.append({
            "aspect": "Image",
            "status": _aspect_status(img.score),
            "detail": ("Image-level heuristics: ELA / noise / metadata "
                       f"(risk {img.score:.2f})."),
        })
    if vid is not None:
        detail = "No significant frame anomaly detected."
        if vid.detail.get("suspicious_timestamps"):
            n = len(vid.detail["suspicious_timestamps"])
            detail = f"{n} suspicious frame indicator(s) flagged."
        out.append({
            "aspect": "Video",
            "status": _aspect_status(vid.score),
            "detail": detail,
        })
    aud = individual_results.get("audio")
    if aud is not None:
        out.append({
            "aspect": "Audio",
            "status": _aspect_status(aud.score),
            "detail": f"Audio-level heuristics (risk {aud.score:.2f}).",
        })
    doc = individual_results.get("document")
    if doc is not None:
        out.append({
            "aspect": "Text",
            "status": _aspect_status(doc.score),
            "detail": f"{doc.features.get('claim_count', 0)} claim(s), "
                      f"{doc.features.get('date_count', 0)} date(s) extracted.",
        })

    # cross aspects
    va = next((c for c in comparisons if c["pair"] == "video_audio"), None)
    if va is not None:
        out.append({
            "aspect": "Audio/Video sync",
            "status": ("Conflict" if va["label"] == "INCONSISTENT"
                       else "Possible inconsistency" if va["label"] == "MIXED"
                       else "Consistent"),
            "detail": f"Heuristic consistency {va['consistency']:.2f}.",
        })

    ts_conflict_types = ("timestamp", "time_of_day")
    ts_conflict = any(cf["type"].startswith(ts_conflict_types) for c in comparisons
                      for cf in c["conflicts"])
    tod_details = [cf for c in comparisons for cf in c["conflicts"]
                   if cf["type"] == "time_of_day_conflict"]
    out.append({
        "aspect": "Timestamp / Time of day",
        "status": "Conflict detected" if ts_conflict else "No conflict detected",
        "detail": (tod_details[0]["description"] if tod_details
                   else "Timestamps across modalities disagree." if ts_conflict
                   else "No time disagreement was found (or times were missing)."),
    })

    # speaker / identity
    speaker_conf = [cf for c in comparisons for cf in c["conflicts"]
                    if cf["type"] == "speaker_unverified"]
    if speaker_conf:
        out.append({
            "aspect": "Speaker / Identity",
            "status": "Could not verify",
            "detail": speaker_conf[0]["description"],
        })

    # location
    loc_conf = [cf for c in comparisons for cf in c["conflicts"]
                if cf["type"] == "location_conflict"]
    if loc_conf:
        out.append({
            "aspect": "Location",
            "status": "Conflict detected",
            "detail": loc_conf[0]["description"],
        })
    return out


def _aspect_status(score: float) -> str:
    if score >= 0.60:
        return "Possible alteration indicator"
    if score <= 0.35:
        return "No significant anomaly detected"
    return "Weak / inconclusive indicator"
