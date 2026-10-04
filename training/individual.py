"""
Per-modality training scripts all share this pattern:

    Dataset -> Preprocessing -> Feature extraction -> Individual model -> save head

Because TrustLayer's analyzers already turn a file into a small numeric feature
dict, "training an individual detector" means fitting a tiny logistic head over
those features. That keeps the prototype honest (no fake deep learning) while
leaving a clean seam: replace ``extract_features`` with a CNN/transformer later.

This module contains the shared runner; train_image/video/audio call it.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List

import numpy as np

from backend.config import MODELS_DIR
from training.common import train_logistic, train_mlp, binary_metrics

# Which analyzer feature keys each modality head consumes. Keep in sync with the
# analyzers' ``indicators`` dicts.
FEATURE_KEYS = {
    "image": ["ela_anomaly", "smoothing_anomaly", "duplication_indicator",
              "metadata_anomaly"],
    "video": ["temporal_jump_indicator", "suspicious_frame_indicator",
              "lighting_variability_indicator", "overall_blur_indicator"],
    "audio": ["clipping_indicator", "silence_anomaly",
              "spectral_flatness_indicator", "dynamic_range_indicator",
              "zcr_anomaly"],
    "document": ["style_uniformity_indicator", "repetition_indicator",
                 "claim_density_indicator", "missing_dates_indicator"],
}


def _image_feature_names(feats: dict) -> List[str]:
    """
    Resolution-invariant detector features only.

    We deliberately EXCLUDE raw size / ELA / high-frequency-variance features
    because those change with image dimensions and would make the detector
    resolution-locked. All keys kept here are computed from the canonicalised
    image (see backend.services.image_analyzer.extract_detector_features).
    """
    keep_prefixes = ("spec_", "spec2_", "corr_", "mean_", "std_", "lap_",
                     "ela_feat", "noise_", "grad_", "edge_", "blockvar_")
    keep_exact = {"saturation", "brightness_mean", "contrast_std", "entropy"}
    names = []
    for k, v in feats.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if k in keep_exact or any(k.startswith(p) for p in keep_prefixes):
            names.append(k)
    names.sort()
    return names


def extract_features(modality: str, path: str) -> dict:
    """Run the real analyzer for a modality and return its indicator features."""
    from backend.services import (image_analyzer, video_analyzer,
                                   audio_analyzer, document_analyzer)
    analyzers = {
        "image": image_analyzer.analyze,
        "video": video_analyzer.analyze,
        "audio": audio_analyzer.analyze,
        "document": document_analyzer.analyze,
    }
    meta = {"evidence_id": "train", "filename": os.path.basename(path)}
    result = analyzers[modality](path, meta)
    return result.features


def load_manifest(manifest_path: str) -> List[dict]:
    """
    Manifest CSV columns (per modality training):
        path,label
    label = 1 (manipulated/suspicious) or 0 (authentic).
    """
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({"path": row["path"], "label": int(row["label"])})
    return rows


def run(modality: str, manifest_path: str, arch: str = "auto", limit: int = 0):
    rows = load_manifest(manifest_path)
    if not rows:
        print("Manifest is empty.")
        return None
    if limit:
        rows = rows[:limit]

    # choose feature names: image uses the full detector bank, others use the
    # small indicator set defined above.
    if modality == "image":
        probe = extract_features("image", rows[0]["path"])
        keys = _image_feature_names(probe)
    else:
        keys = FEATURE_KEYS[modality]

    X, y = [], []
    for i, row in enumerate(rows):
        if not os.path.exists(row["path"]):
            print(f"skip (missing): {row['path']}")
            continue
        try:
            feats = extract_features(modality, row["path"])
            X.append([float(feats.get(k, 0.0) or 0.0) for k in keys])
            y.append(row["label"])
            if (i + 1) % 250 == 0:
                print(f"  extracted {i + 1}/{len(rows)}")
        except Exception as exc:
            print(f"skip ({row['path']}): {exc}")
    if not X:
        print("No usable training samples.")
        return None

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    use_mlp = (arch == "mlp") or (arch == "auto" and len(keys) > 8)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sd

    if use_mlp:
        # Hold out 15% for validation-based early stopping (overfit protection).
        n = len(y)
        idx = np.arange(n)
        np.random.default_rng(7).shuffle(idx)
        cut = max(int(n * 0.85), 1)
        tr_i, va_i = idx[:cut], idx[cut:]
        if len(va_i) == 0:
            tr_i, va_i = idx, idx
        params = train_mlp(Xn[tr_i], y[tr_i], hidden=128, epochs=15000, lr=0.05,
                           l2=1e-3, val_X=Xn[va_i], val_y=y[va_i], patience=250)
        h = np.tanh(Xn @ params["w1"] + params["b1"])
        scores = 1 / (1 + np.exp(-(h @ params["w2"] + params["b2"]).ravel()))
        save_kwargs = {
            "features": np.array(keys),
            "w1": params["w1"], "b1": params["b1"],
            "w2": params["w2"], "b2": params["b2"],
            "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
        }
        headline = "MLP(128, early-stop)"
    else:
        w, b = train_logistic(Xn, y)
        scores = 1 / (1 + np.exp(-(Xn @ w + b)))
        save_kwargs = {
            "features": np.array(keys),
            "w": w.astype(np.float32), "b": float(b),
            "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
        }
        headline = "logistic"

    metrics = binary_metrics(y, (scores >= 0.5).astype(int), scores)
    print(f"[{modality}/{headline}] trained on {len(y)} samples ({len(keys)} features)"
          f" -> {metrics}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{modality}_head.npz"
    np.savez(out, **save_kwargs)
    print(f"saved {out}")
    return metrics


def main(modality: str):
    parser = argparse.ArgumentParser(description=f"Train the {modality} head")
    parser.add_argument("--manifest", required=True,
                        help="CSV with columns: path,label")
    parser.add_argument("--arch", default="auto", choices=["auto", "linear", "mlp"],
                        help="auto = MLP for rich image features, logistic otherwise")
    parser.add_argument("--limit", type=int, default=0,
                        help="only use the first N manifest rows (0 = all)")
    args = parser.parse_args()
    run(modality, args.manifest, arch=args.arch, limit=args.limit)
