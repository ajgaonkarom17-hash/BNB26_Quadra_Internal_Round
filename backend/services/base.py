"""
Shared contract for every modality analyzer.

An analyzer is a small, self-contained module that converts ONE evidence file
into a common result shape. Keeping the contract identical across modalities is
what lets fusion / cross-modal / explainability treat them uniformly, and it is
what makes it easy to drop in a stronger model later.

Every analyzer must expose::

    analyze(path: str, meta: dict) -> ModalityResult

The ``meta`` dict carries DB context: {"evidence_id", "filename", "investigation_id"}.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

from backend.config import MODE
from backend.utils.helpers import l2_normalize

# Engines are labelled honestly. Never call a heuristic a trained model.
ENGINE_HEURISTIC = "demo-heuristic"
ENGINE_MODEL = "model"


@dataclass
class Finding:
    """A single human-readable observation produced by an analyzer."""

    key: str            # machine key e.g. "compression_anomaly"
    title: str          # short human title
    detail: str         # one-sentence explanation
    severity: str = "info"   # info | low | medium | high
    kind: str = "neutral"    # neutral | supports | contradicts | missing

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModalityResult:
    modality: str                       # image | video | audio | document
    evidence_id: str
    filename: str
    score: float = 0.0                  # 0..1 risk / manipulation indicator
    label: str = "INCONCLUSIVE"
    confidence: float = 0.0             # how much we trust this individual result
    embedding: Optional[np.ndarray] = None
    features: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    engine: str = ENGINE_HEURISTIC
    detail: dict = field(default_factory=dict)
    quality: str = "ok"                 # ok | low | unusable
    warnings: list = field(default_factory=list)
    error: Optional[str] = None

    def embedding_list(self) -> list:
        if self.embedding is None:
            return []
        return [round(float(x), 6) for x in np.asarray(self.embedding).ravel()]

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "evidence_id": self.evidence_id,
            "filename": self.filename,
            "score": round(float(self.score), 4),
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "engine": self.engine,
            "quality": self.quality,
            "features": self.features,
            "findings": [f.to_dict() if isinstance(f, Finding) else f for f in self.findings],
            "detail": self.detail,
            "warnings": self.warnings,
            "error": self.error,
        }


def finalize(result: ModalityResult, dim: int, salt: str) -> ModalityResult:
    """
    Ensure the result has a fixed-size, L2-normalized embedding.

    If an analyzer produced raw features but no embedding, we build one with a
    deterministic random projection. If it produced a short embedding we pad /
    project it into ``dim``. This is the "common representation" step.
    """
    from backend.utils.helpers import project_features

    if result.embedding is not None:
        vec = l2_normalize(np.asarray(result.embedding, dtype=np.float32).ravel())
        if vec.size != dim:
            vec = project_features(vec.tolist(), dim, salt)
        result.embedding = vec
        return result

    if result.features:
        numeric = [v for v in result.features.values() if isinstance(v, (int, float))]
        if numeric:
            result.embedding = project_features(numeric, dim, salt)
        else:
            result.embedding = np.zeros(dim, dtype=np.float32)
    else:
        result.embedding = np.zeros(dim, dtype=np.float32)
    return result


def make_result(modality: str, meta: dict, **kwargs: Any) -> ModalityResult:
    return ModalityResult(
        modality=modality,
        evidence_id=meta.get("evidence_id", ""),
        filename=meta.get("filename", os.path.basename(meta.get("path", ""))),
        **kwargs,
    )
