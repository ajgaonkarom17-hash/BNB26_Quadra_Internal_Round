"""
MULTIMODAL FUSION.

Inputs (exactly as specified):
    image_embedding   (128)
    video_embedding   (128)
    audio_embedding   (128)
    text_embedding    (128)   <- 'document' modality
    availability_mask (4)     e.g. [1,1,0,1]

Outputs:
    overall assessment  (AUTHENTIC / SUSPICIOUS / INCONCLUSIVE)
    confidence          (0..1)
    consistency score   (0..1)

DEMO mode uses a transparent, hand-set combination of the individual risks and
the cross-modal risk, plus an evidence-coverage term. MODEL mode loads weights
from models/fusion.npz if present and runs a small MLP.

The interface (`fuse(...)`) is deliberately stable so a Transformer /
cross-attention model can replace the internals later without touching callers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from backend.config import (
    MODE, EMBEDDING_DIM, FUSION_WEIGHTS,
    SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD, CONFLICT_OVERRIDE,
)
from backend.utils.helpers import clamp, risk_to_label, confidence_label

MODALITIES = ["image", "video", "audio", "document"]
FUSION_INPUT_DIM = EMBEDDING_DIM * 4 + 4  # 516 by default


def build_feature_vector(embeddings: dict, mask: np.ndarray) -> np.ndarray:
    """
    Build the fixed fusion input: [img|vid|aud|txt|mask].
    Missing modalities are zero vectors (and marked 0 in the mask).
    """
    parts = []
    for m in MODALITIES:
        vec = embeddings.get(m)
        if vec is None:
            parts.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
        else:
            vec = np.asarray(vec, dtype=np.float32).ravel()
            if vec.size != EMBEDDING_DIM:
                padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                n = min(vec.size, EMBEDDING_DIM)
                padded[:n] = vec[:n]
                vec = padded
            parts.append(vec)
    parts.append(np.asarray(mask, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def availability_mask(available_modalities) -> np.ndarray:
    return np.array([1.0 if m in available_modalities else 0.0 for m in MODALITIES],
                    dtype=np.float32)


def _demo_fuse(individual_risks: dict, cross: dict, coverage: float) -> dict:
    available = [m for m, r in individual_risks.items() if r is not None]
    if available:
        individual_risk = float(np.mean([individual_risks[m] for m in available]))
        # the most suspicious individual modality matters, but not exclusively
        individual_risk = 0.6 * individual_risk + 0.4 * max(
            individual_risks[m] for m in available)
    else:
        individual_risk = 0.5

    cross_risk = float(cross.get("cross_modal_risk", 0.5))
    w = FUSION_WEIGHTS
    base = (
        w["individual_risk"] * individual_risk
        + w["cross_modal_risk"] * cross_risk
        + w["coverage"] * (1.0 - coverage)
    )

    # ---- cross-modal conflict override ---------------------------------
    # The central research idea: evidence can look individually acceptable yet
    # be *collectively* inconsistent (coordinated manipulation). So strong
    # cross-modal disagreement adds suspicion on top of the individual scores.
    # override is 0 at cross_risk <= 0.5 and grows to CONFLICT_OVERRIDE at 1.0.
    conflict_override = clamp((cross_risk - 0.5) * 2.0) * CONFLICT_OVERRIDE
    fused_risk = clamp(base + conflict_override)

    return {
        "risk": fused_risk,
        "individual_risk": round(individual_risk, 4),
        "cross_modal_risk": round(cross_risk, 4),
        "base_fusion": round(base, 4),
        "conflict_override": round(conflict_override, 4),
        "coverage_term": round(1.0 - coverage, 4),
        "engine": "demo-transparent-fusion",
    }


def _model_fuse(feature_vector: np.ndarray) -> Optional[dict]:
    """Try to run a trained fusion model from models/fusion.npz."""
    try:
        from backend.models.registry import predict_fusion
        return predict_fusion(feature_vector)
    except Exception:
        return None


def fuse(
    individual_risks: dict,
    embeddings: dict,
    mask: np.ndarray,
    cross: dict,
    coverage: float,
) -> dict:
    """
    Returns a dict with: risk, assessment, confidence, confidence_label,
    consistency, engine, breakdown.
    """
    available_mods = [m for m in MODALITIES if mask[MODALITIES.index(m)] > 0]
    engine = "demo-transparent-fusion"
    model_out = None

    if MODE == "model":
        fv = build_feature_vector(embeddings, mask)
        model_out = _model_fuse(fv)

    if model_out is not None:
        risk = clamp(model_out.get("risk", 0.5))
        consistency = cluster_consistency(cross, coverage)
        breakdown = model_out
        engine = "model-mlp-fusion"
    else:
        breakdown = _demo_fuse(individual_risks, cross, coverage)
        risk = breakdown["risk"]
        # consistency = 1 - cross-modal risk, tempered by coverage
        consistency = cluster_consistency(cross, coverage)

    assessment = risk_to_label(risk, SUSPICIOUS_THRESHOLD, AUTHENTIC_THRESHOLD)

    # Confidence: how decisive + how much evidence we have. NOT a probability.
    decisiveness = abs(risk - 0.5) * 2.0            # 0 at the fence, 1 at extremes
    agreement = 1.0 - abs(breakdown.get("individual_risk", 0.5)
                          - breakdown.get("cross_modal_risk", 0.5))
    confidence = clamp(0.45 * decisiveness + 0.30 * agreement + 0.25 * coverage)
    if len(available_mods) <= 1:
        confidence = min(confidence, 0.35)          # single modality can't be confident

    return {
        "risk": round(risk, 4),
        "assessment": assessment,
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label(confidence),
        "consistency": round(consistency, 4),
        "engine": engine,
        "breakdown": breakdown,
        "available_modalities": available_mods,
    }


def cluster_consistency(cross: dict, coverage: float) -> float:
    mean_c = float(cross.get("mean_consistency", 0.5))
    # if there was nothing to compare, consistency is unknown -> 0.5
    if cross.get("pairs_compared", 0) == 0:
        return 0.5
    return clamp(0.85 * mean_c + 0.15 * coverage)
