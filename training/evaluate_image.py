"""
HONEST EVALUATION for the IMAGE detector, on a real dataset (CIFAKE).

This script:
  1. trains the image head on the training split,
  2. evaluates it on the completely held-out test split (never seen in training),
  3. writes a JSON report to models/evaluation_report.json

The report is what the Evaluation page displays. If this script has not been
run, the page honestly says "Not evaluated yet" — numbers are never invented.

Manifest format (path,label,split):
    path,label,split
    data/raw/cifake/REAL/x.jpg,0,train
    data/raw/cifake_test/FAKE/y.jpg,1,test

Usage:
    python -m training.evaluate_image --manifest data/manifests/image_cifake_eval.csv
    # or use the built-in CIFAKE split folders:
    python -m training.evaluate_image --cifake
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from backend.config import MODELS_DIR
from backend.services.image_analyzer import analyze
from backend.models import registry
from training.individual import run as train_run
from training.common import binary_metrics

REPORT_PATH = MODELS_DIR / "evaluation_report.json"


def _build_cifake_eval_manifest(out_path: str):
    """Create a manifest from data/raw/cifake (train) + data/raw/cifake_test (test)."""
    rows = []
    for split, folder in (("train", "cifake"), ("test", "cifake_test")):
        base = os.path.join("data", "raw", folder)
        for label, val in (("REAL", 0), ("FAKE", 1)):
            d = os.path.join(base, label)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                rows.append({"path": f"{base}/{label}/{f}".replace("\\", "/"),
                             "label": val, "split": split})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "split"])
        w.writeheader()
        w.writerows(rows)
    return rows


def _load_manifest(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate(rows):
    """Train on 'train' rows, then score 'test' rows via the saved head."""
    train_rows = [r for r in rows if r.get("split") == "train"]
    test_rows = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        # fall back: split the rows 70/30
        n = int(len(rows) * 0.7)
        train_rows, test_rows = rows[:n], rows[n:]

    if not train_rows or not test_rows:
        return {"status": "insufficient_data"}

    # ---- train ---- (write a temporary manifest for the trainer)
    tmp = os.path.join("data", "manifests", "_tmp_image_train.csv")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label"])
        w.writeheader()
        for r in train_rows:
            w.writerow({"path": r["path"], "label": r["label"]})
    train_metrics = train_run("image", tmp)
    registry._cache.clear()  # ensure the freshly trained head is reloaded

    # ---- test ---- score held-out files through the real inference path
    y, scores = [], []
    for r in test_rows:
        if not os.path.exists(r["path"]):
            continue
        try:
            res = analyze(r["path"], {"evidence_id": "eval", "filename": os.path.basename(r["path"])})
            s = registry.predict_individual("image", res.features)
            if s is None:
                s = res.score
            y.append(int(r["label"]))
            scores.append(float(s))
        except Exception as exc:
            print(f"skip {r['path']}: {exc}")
    if not scores:
        return {"status": "no_test_samples"}

    y = np.asarray(y)
    scores = np.asarray(scores)
    metrics = binary_metrics(y, (scores >= 0.5).astype(int), scores)

    # ROC-AUC via the metric helper
    roc = metrics.get("roc_auc")
    return {
        "status": "ok",
        "dataset": "CIFAKE (train/ test/ split)",
        "n_train": len(train_rows),
        "n_test": len(y),
        "train_metrics": train_metrics,
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": roc,
        },
        "confusion": {"tp": metrics["tp"], "tn": metrics["tn"],
                      "fp": metrics["fp"], "fn": metrics["fn"]},
        "mean_score_real": round(float(scores[y == 0].mean()), 4) if (y == 0).any() else None,
        "mean_score_fake": round(float(scores[y == 1].mean()), 4) if (y == 1).any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the image detector")
    parser.add_argument("--manifest", help="path,label,split CSV")
    parser.add_argument("--cifake", action="store_true",
                        help="build the manifest from data/raw/cifake[_test]")
    args = parser.parse_args()

    if args.cifake:
        manifest = os.path.join("data", "manifests", "image_cifake_eval.csv")
        rows = _build_cifake_eval_manifest(manifest)
        print(f"built {len(rows)} rows from CIFAKE folders")
    elif args.manifest:
        rows = _load_manifest(args.manifest)
    else:
        print("Provide --manifest or --cifake")
        return

    report = evaluate(rows)
    report["modality"] = "image"
    report["mode_used"] = "model (trained MLP head)"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
