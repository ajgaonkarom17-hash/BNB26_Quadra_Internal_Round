"""
MODEL MODE registry.

Looks inside the project ``models/`` directory for trained artifacts and loads
them if present. If anything is missing it returns ``None`` so the caller falls
back to the DEMO heuristic. Nothing here ever crashes the app.

Expected artifacts (all optional):
    models/image_head.npz      - logistic head over image indicators (w, b)
    models/video_head.npz
    models/audio_head.npz
    models/document_head.npz
    models/fusion.npz          - small MLP over the 516-d fusion vector

Each individual head file stores:  features (list[str]), w (k,), b (scalar).
The fusion file stores:            w1 (D,H), b1 (H,), w2 (H,3), b2 (3,)

PyTorch ``.pt`` checkpoints are also recognised when torch is installed, but the
numpy format is the primary, dependency-light path used by the training scripts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from backend.config import MODELS_DIR

HEAD_FILES = {
    "image": MODELS_DIR / "image_head.npz",
    "video": MODELS_DIR / "video_head.npz",
    "audio": MODELS_DIR / "audio_head.npz",
    "document": MODELS_DIR / "document_head.npz",
}
FUSION_FILE = MODELS_DIR / "fusion.npz"

_cache: dict = {}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def load_head(modality: str) -> Optional[dict]:
    key = f"head:{modality}"
    if key in _cache:
        return _cache[key]
    path = HEAD_FILES.get(modality)
    data = None
    if path is not None and path.exists():
        try:
            with np.load(path, allow_pickle=True) as npz:
                keys = set(npz.files)
                data = {
                    "features": list(npz["features"]) if "features" in keys else [],
                    "kind": "mlp" if "w1" in keys else "linear",
                }
                if data["kind"] == "mlp":
                    data.update({
                        "w1": npz["w1"].astype(np.float32),
                        "b1": npz["b1"].astype(np.float32),
                        "w2": npz["w2"].astype(np.float32),
                        "b2": npz["b2"].astype(np.float32),
                        "mu": npz["mu"].astype(np.float32) if "mu" in keys else None,
                        "sd": npz["sd"].astype(np.float32) if "sd" in keys else None,
                    })
                else:
                    data.update({
                        "w": npz["w"].astype(np.float32),
                        "b": float(npz["b"]) if "b" in keys else 0.0,
                        "mu": npz["mu"].astype(np.float32) if "mu" in keys else None,
                        "sd": npz["sd"].astype(np.float32) if "sd" in keys else None,
                    })
        except Exception:
            data = None
    _cache[key] = data
    return data


def predict_individual(modality: str, features: dict, path: str = "") -> Optional[float]:
    """
    Return a risk score in 0..1 from a trained head, or None if unavailable.
    Only the numeric features named in the head file are used. Supports both a
    linear logistic head and a small MLP head (w1/b1/w2/b2), with optional
    standardisation (mu/sd).
    """
    head = load_head(modality)
    if head is None:
        return None
    names = head["features"]
    if not names:
        return None
    vector = []
    for name in names:
        value = features.get(name, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 1.0 if value else 0.0
        vector.append(value)
    vector = np.asarray(vector, dtype=np.float32)

    if head.get("mu") is not None:
        vector = (vector - head["mu"]) / (head["sd"] + 1e-9)

    if head.get("kind") == "mlp":
        h = np.tanh(vector @ head["w1"] + head["b1"])
        logit = float(np.ravel(h @ head["w2"] + head["b2"])[0])
    else:
        logit = float(vector @ head["w"] + head["b"])
    return float(_sigmoid(logit))


def load_fusion() -> Optional[dict]:
    if "fusion" in _cache:
        return _cache["fusion"]
    data = None
    if FUSION_FILE.exists():
        try:
            with np.load(FUSION_FILE) as npz:
                data = {
                    "w1": npz["w1"].astype(np.float32),
                    "b1": npz["b1"].astype(np.float32),
                    "w2": npz["w2"].astype(np.float32),
                    "b2": npz["b2"].astype(np.float32),
                }
        except Exception:
            data = None
    _cache["fusion"] = data
    return data


def predict_fusion(feature_vector: np.ndarray) -> Optional[dict]:
    model = load_fusion()
    if model is None:
        return None
    x = np.asarray(feature_vector, dtype=np.float32)
    if x.shape[0] != model["w1"].shape[0]:
        return None
    h = np.tanh(x @ model["w1"] + model["b1"])
    logits = h @ model["w2"] + model["b2"]     # [authentic, suspicious, inconclusive]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    # risk = P(suspicious) + 0.5 * P(inconclusive)
    risk = float(probs[1] + 0.5 * probs[2])
    return {
        "risk": risk,
        "class_probs": {
            "authentic": round(float(probs[0]), 4),
            "suspicious": round(float(probs[1]), 4),
            "inconclusive": round(float(probs[2]), 4),
        },
        "engine": "model-mlp-fusion",
    }


def available_models() -> dict:
    """Describe which trained artifacts exist (used by the Evaluation page)."""
    info = {}
    for modality, path in HEAD_FILES.items():
        info[modality] = path.exists()
    info["fusion"] = FUSION_FILE.exists()
    return info
