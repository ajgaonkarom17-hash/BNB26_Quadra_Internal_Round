"""
EVALUATION: known vs unseen manipulation, and the model-ablation comparison.

Compares three configurations on a held-out manifest:
    1. individual-only      (per-modality risk average)
    2. multimodal           (fusion over embeddings + mask)
    3. multimodal + cross-modal reasoning (fusion + cross-modal conflict term)

And reports metrics split by manipulation family so you can see generalization:

    TRAIN: Manipulation A/B/C     ->  "known"
    TEST:  Manipulation D/E       ->  "unseen"
             + different compression / conditions

If the manifest has no ``split`` / ``manipulation`` columns, the script says
"Not evaluated yet" instead of inventing numbers.

Run:
    python -m training.evaluate --manifest data/manifests/eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from backend.config import EMBEDDING_DIM
from backend.services.fusion import build_feature_vector, availability_mask, MODALITIES
from backend.services import cross_modal
from training.common import binary_metrics

NOT_EVALUATED = "Not evaluated yet"


def _embed_all(row):
    from backend.services import (image_analyzer, video_analyzer,
                                   audio_analyzer, document_analyzer)
    analyzers = {"image": image_analyzer.analyze, "video": video_analyzer.analyze,
                 "audio": audio_analyzer.analyze, "document": document_analyzer.analyze}
    entries, risks = {}, {}
    for m in MODALITIES:
        path = row.get(f"{m}_path", "")
        if not path or not os.path.exists(path):
            continue
        meta = {"evidence_id": m, "filename": os.path.basename(path)}
        try:
            res = analyzers[m](path, meta)
        except Exception:
            continue
        entries[m] = {"evidence_id": m, "result": res,
                      "embedding": res.embedding, "stored_path": path}
        risks[m] = res.score
    return entries, risks


def individual_only_prediction(risks):
    if not risks:
        return 0.5
    return float(np.mean(list(risks.values())))


def multimodal_prediction(entries, mask):
    emb = {m: entries[m]["embedding"] for m in entries}
    vec = build_feature_vector(emb, mask)
    # simple transparent fusion proxy (same formula family as demo fusion)
    risks = {m: entries[m]["result"].score for m in entries}
    individual = float(np.mean(list(risks.values()))) if risks else 0.5
    return 0.6 * vec.mean() * 0.5 + 0.4 * individual


def multimodal_cross_prediction(entries):
    comparisons = cross_modal.compare_all(entries)
    summary = cross_modal.aggregate_inconsistency(comparisons)
    individual = float(np.mean([entries[m]["result"].score for m in entries])) if entries else 0.5
    return 0.5 * individual + 0.5 * summary["cross_modal_risk"]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    if not rows:
        print(NOT_EVALUATED, "(empty manifest)")
        return
    has_split = any(r.get("split") for r in rows)
    has_manip = any(r.get("manipulation") for r in rows)
    if not (has_split or has_manip):
        print("Manifest lacks 'split'/'manipulation' columns. ", NOT_EVALUATED)
        print("Add these columns to compare known vs unseen manipulations.")
        return

    y_true, p_ind, p_mm, p_mmcm, splits = [], [], [], [], []
    for r in rows:
        entries, risks = _embed_all(r)
        if not entries:
            continue
        mask = availability_mask(list(entries.keys()))
        y_true.append(int(float(r.get("overall_label", 0))))
        p_ind.append(individual_only_prediction(risks))
        p_mm.append(multimodal_prediction(entries, mask))
        p_mmcm.append(multimodal_cross_prediction(entries))
        splits.append(r.get("split", "unknown"))

    y_true = np.asarray(y_true)
    for name, scores in (("individual_only", p_ind),
                         ("multimodal", p_mm),
                         ("multimodal_plus_cross_modal", p_mmcm)):
        scores = np.asarray(scores)
        print(f"\n=== {name} ===")
        print("ALL:", binary_metrics(y_true, (scores >= 0.5).astype(int), scores))
        by_split = defaultdict(list)
        for s, split in zip(scores, splits):
            by_split[split].append(s)
        for split, vals in by_split.items():
            idx = [i for i, s in enumerate(splits) if s == split]
            print(f"  {split}:", binary_metrics(y_true[idx],
                  (scores[idx] >= 0.5).astype(int), scores[idx]))


if __name__ == "__main__":
    main()
