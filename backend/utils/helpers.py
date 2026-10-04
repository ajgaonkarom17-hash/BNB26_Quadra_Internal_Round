"""Small shared helpers used across TrustLayer services."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------
def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of ``vector`` (safe for zero vectors)."""
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity mapped to 0..1 (0 = unrelated, 1 = identical)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    cos = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def stable_seed(text: str, base: int = 1337) -> int:
    """Deterministic integer seed derived from a string (for projections)."""
    digest = hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
    return (int(digest[:8], 16) + base) % (2**31 - 1)


def project_features(features: Sequence[float], dim: int, salt: str) -> np.ndarray:
    """
    Turn an arbitrary-length feature vector into a fixed ``dim`` embedding.

    This is a deterministic random projection (a real technique, Johnson-
    Lindenstrauss style). It is NOT a learned neural encoder, but it preserves
    relative similarity well enough for a prototype and can be swapped for a
    trained encoder later. See README "What is genuinely AI vs heuristic".
    """
    feats = np.asarray(list(features), dtype=np.float32).ravel()
    if feats.size == 0:
        return np.zeros(dim, dtype=np.float32)

    rng = np.random.default_rng(stable_seed(salt))
    projection = rng.standard_normal((feats.size, dim)).astype(np.float32)

    # nonlinearity keeps the projection from being purely linear (helps a
    # bit when comparing very different feature scales across modalities)
    embedded = np.tanh(projection.T @ feats)
    return l2_normalize(embedded)


# ---------------------------------------------------------------------------
# Numeric / misc helpers
# ---------------------------------------------------------------------------
def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def mean_or(values: Iterable[float], default: float = 0.0) -> float:
    values = [v for v in values if v is not None]
    if not values:
        return default
    return float(sum(values) / len(values))


def risk_to_label(risk: float, suspicious: float, authentic: float) -> str:
    """Map a 0..1 risk score to a coarse, honest label."""
    if risk >= suspicious:
        return "SUSPICIOUS"
    if risk <= authentic:
        return "AUTHENTIC"
    return "INCONCLUSIVE"


def confidence_label(confidence: float) -> str:
    if confidence >= 0.66:
        return "HIGH"
    if confidence >= 0.33:
        return "MEDIUM"
    return "LOW"
