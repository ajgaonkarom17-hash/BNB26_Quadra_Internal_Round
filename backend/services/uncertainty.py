"""
UNCERTAINTY + MISSING/CONFLICTING EVIDENCE.

Never force a REAL/FAKE decision. This module works out:
  * evidence coverage (% of the 4 modalities present)
  * missing modalities
  * conflicting pairs
  * a list of unavailability reasons (so "missing" is explained)
and may DOWNGRADE a confident assessment to INCONCLUSIVE when evidence is weak.

Confidence (from fusion) and coverage are deliberately SEPARATE concepts:
confidence = how decisive/agreement-based the result is,
coverage   = how much of the evidence set we actually have.
"""

from __future__ import annotations

from backend.services.fusion import MODALITIES
from backend.utils.helpers import clamp


def evidence_coverage(available_modalities) -> float:
    present = len([m for m in MODALITIES if m in available_modalities])
    return round(present / len(MODALITIES), 4)


def analyze_uncertainty(
    available_modalities,
    comparisons: list,
    individual_results: dict,
    fusion_result: dict,
    coverage: float,
    unusable_results: dict = None,
) -> dict:
    unusable_results = unusable_results or {}
    missing = []
    for m in MODALITIES:
        if m in available_modalities:
            continue
        reason = "not uploaded"
        result = individual_results.get(m) or unusable_results.get(m)
        if result is not None and getattr(result, "error", None):
            reason = f"uploaded but unusable ({result.error})"
        elif m in unusable_results:
            reason = "uploaded but unusable (low quality or undecodable)"
        missing.append({"modality": m, "reason": reason})

    conflicts = []
    for c in comparisons:
        for conflict in c["conflicts"]:
            conflicts.append({
                "pair": c["pair"],
                "type": conflict["type"],
                "description": conflict["description"],
                "severity": conflict["severity"],
            })

    low_quality = []
    for m, res in individual_results.items():
        if res is not None and getattr(res, "quality", "ok") != "ok":
            low_quality.append({"modality": m, "quality": res.quality,
                                "note": res.error or "low quality media"})

    # ---- decide whether to downgrade to INCONCLUSIVE --------------------
    assessment = fusion_result["assessment"]
    confidence = fusion_result["confidence"]
    decisive_conflicts = [c for c in conflicts if c["severity"] == "high"]
    notes = []

    if coverage < 0.5:
        notes.append("Less than half of the expected modalities are available.")
    if len(available_modalities) <= 1:
        notes.append("Only one modality is available; cross-modal reasoning is impossible.")
    if decisive_conflicts:
        notes.append(f"{len(decisive_conflicts)} high-severity conflict(s) detected.")
    if confidence < 0.30:
        notes.append("The fused evidence is close to the decision boundary.")

    downgrade = (
        assessment == "AUTHENTIC"
        and (coverage < 0.5 or decisive_conflicts or confidence < 0.30)
    ) or (
        len(available_modalities) <= 1
    )

    final_assessment = "INCONCLUSIVE" if downgrade else assessment
    if downgrade:
        notes.append("Assessment downgraded to INCONCLUSIVE due to insufficient "
                     "or conflicting evidence.")

    # uncertainty level is the inverse of confidence, nudged by coverage
    uncertainty_score = clamp((1.0 - confidence) * 0.7 + (1.0 - coverage) * 0.3)
    if uncertainty_score >= 0.66:
        uncertainty = "HIGH"
    elif uncertainty_score >= 0.33:
        uncertainty = "MEDIUM"
    else:
        uncertainty = "LOW"

    return {
        "final_assessment": final_assessment,
        "pre_fusion_assessment": assessment,
        "downgraded": downgrade,
        "uncertainty": uncertainty,
        "uncertainty_score": round(uncertainty_score, 4),
        "coverage": coverage,
        "missing": missing,
        "conflicts": conflicts,
        "low_quality": low_quality,
        "notes": notes,
    }
