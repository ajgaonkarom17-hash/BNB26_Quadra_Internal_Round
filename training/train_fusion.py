"""
MULTIMODAL DATASET + FUSION TRAINING.

This is the "Multimodal Dataset -> Fusion Model" stage from the spec.

The multimodal dataset is a table with one row per investigation:

    investigation_id
    image_path, video_path, audio_path, document_path
    image_label, video_label, audio_label, document_label
    cross_modal_label      (1 = consistent, 0 = inconsistent)   [optional]
    overall_label          (1 = suspicious, 0 = authentic)      [required]

We:
  1. run each modality analyzer to get its 128-d embedding (common representation),
  2. build the fusion feature vector [img|vid|aud|txt|mask] (516-d),
  3. apply MODALITY DROPOUT: randomly zero whole modalities and update the mask,
     so the model learns to work with missing evidence,
  4. train a small MLP on the fusion vector with a multi-task loss:
        - overall classification (3 classes: authentic / suspicious / inconclusive)
        - consistency classification (2 classes)
        - modality presence prediction (4 sigmoids)   [auxiliary]
  5. save models/fusion.npz compatible with backend/models/registry.py.

Run:
    python -m training.train_fusion --manifest data/manifests/multimodal.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np

from backend.config import EMBEDDING_DIM, MODELS_DIR
from backend.services.fusion import build_feature_vector, availability_mask, MODALITIES
from training.common import sigmoid

FUSION_DIM = EMBEDDING_DIM * 4 + 4


def _embed(modality: str, path: str):
    from backend.services import (image_analyzer, video_analyzer,
                                   audio_analyzer, document_analyzer)
    analyzers = {
        "image": image_analyzer.analyze, "video": video_analyzer.analyze,
        "audio": audio_analyzer.analyze, "document": document_analyzer.analyze,
    }
    if not path or not os.path.exists(path):
        return None
    meta = {"evidence_id": "train", "filename": os.path.basename(path)}
    try:
        return analyzers[modality](path, meta).embedding
    except Exception as exc:
        print(f"  embed failed [{modality}] {path}: {exc}")
        return None


def load_multimodal_manifest(path: str) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def build_fusion_dataset(rows: List[dict], modality_dropout: float = 0.3):
    """
    Returns X (n, FUSION_DIM), Y_overall (n,3), Y_consistency (n,2), Y_presence (n,4)
    """
    X, Yo, Yc, Yp = [], [], [], []
    rng = np.random.default_rng(0)
    for row in rows:
        embeddings = {}
        present = []
        for m in MODALITIES:
            path = row.get(f"{m}_path", "")
            emb = _embed(m, path)
            if emb is not None:
                embeddings[m] = emb
                present.append(m)
        if not present:
            continue

        # ---- modality dropout: randomly hide a present modality ----------
        dropped = set()
        for m in list(present):
            if rng.random() < modality_dropout and len(present) - len(dropped) > 1:
                dropped.add(m)
        kept = [m for m in present if m not in dropped]
        mask = availability_mask(kept)

        vec = build_feature_vector({m: e for m, e in embeddings.items() if m in kept}, mask)
        X.append(vec)

        overall = int(row.get("overall_label", 0))
        # 3-class one-hot: authentic / suspicious / inconclusive (2 == inconclusive)
        onehot = np.zeros(3, dtype=np.float32)
        onehot[min(max(overall, 0), 2)] = 1.0
        Yo.append(onehot)

        cross = int(float(row.get("cross_modal_label", 1)))
        Yc.append([1.0, 0.0] if cross == 0 else [0.0, 1.0])

        Yp.append(np.array([1.0 if m in kept else 0.0 for m in MODALITIES],
                           dtype=np.float32))

    return (np.asarray(X, dtype=np.float32),
            np.asarray(Yo, dtype=np.float32),
            np.asarray(Yc, dtype=np.float32),
            np.asarray(Yp, dtype=np.float32))


def train(X, Yo, Yc, Yp, hidden: int = 64, epochs: int = 800, lr: float = 0.04):
    n, d = X.shape
    rng = np.random.default_rng(42)
    w1 = (rng.standard_normal((d, hidden)) * 0.05).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = (rng.standard_normal((hidden, 3)) * 0.05).astype(np.float32)
    b2 = np.zeros(3, dtype=np.float32)
    # auxiliary heads
    wc = (rng.standard_normal((hidden, 2)) * 0.05).astype(np.float32)
    bc = np.zeros(2, dtype=np.float32)
    wp = (rng.standard_normal((hidden, 4)) * 0.05).astype(np.float32)
    bp = np.zeros(4, dtype=np.float32)

    for epoch in range(epochs):
        h = np.tanh(X @ w1 + b1)
        out = h @ w2 + b2
        e = np.exp(out - out.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        go = (p - Yo) / n
        # multi-task loss: overall (1.0) + consistency (0.5) + presence (0.3)
        hc = h
        outc = hc @ wc + bc
        ec = np.exp(outc - outc.max(axis=1, keepdims=True))
        pc = ec / ec.sum(axis=1, keepdims=True)
        gc = 0.5 * (pc - Yc) / n
        pp = sigmoid(h @ wp + bp)
        gp = 0.3 * (pp - Yp) / n

        gw2 = h.T @ go; gb2 = go.sum(0)
        gwc = h.T @ gc; gbc = gc.sum(0)
        gwp = h.T @ gp; gbp = gp.sum(0)
        gh = (go @ w2.T + gc @ wc.T + gp @ wp.T) * (1 - h ** 2)
        gw1 = X.T @ gh; gb1 = gh.sum(0)

        w1 -= lr * gw1; b1 -= lr * gb1
        w2 -= lr * gw2; b2 -= lr * gb2
        wc -= lr * gwc; bc -= lr * gbc
        wp -= lr * gwp; bp -= lr * gbp

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def main():
    parser = argparse.ArgumentParser(description="Train the multimodal fusion MLP")
    parser.add_argument("--manifest", required=True,
                        help="CSV: investigation_id,image_path,video_path,audio_path,"
                             "document_path,overall_label,cross_modal_label")
    args = parser.parse_args()

    rows = load_multimodal_manifest(args.manifest)
    if not rows:
        print("Empty manifest.")
        return
    X, Yo, Yc, Yp = build_fusion_dataset(rows)
    if X.shape[0] == 0:
        print("No usable multimodal samples (could not embed any modality).")
        return
    print(f"Fusion dataset: X={X.shape}, overall={Yo.shape}, consistency={Yc.shape}, presence={Yp.shape}")

    params = train(X, Yo, Yc, Yp)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "fusion.npz"
    np.savez(out, **params)
    print(f"saved {out}")

    # quick train accuracy
    h = np.tanh(X @ params["w1"] + params["b1"])
    pred = np.argmax(h @ params["w2"] + params["b2"], axis=1)
    acc = float((pred == np.argmax(Yo, axis=1)).mean())
    print(f"train overall accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
