"""
One-command CIFAKE setup + training + evaluation.

    python -m training.setup_cifake --zip "C:\\path\\to\\archive.zip"

What it does:
  1. extracts N REAL + N FAKE images from the CIFAKE zip into data/raw/,
  2. builds manifests,
  3. trains the image detector head (MODEL mode),
  4. evaluates it on a held-out split and writes models/evaluation_report.json.

After this, run the app in MODEL mode to get real detection:

    set MODE=model            (Windows)
    export MODE=model         (mac/linux)
    python run.py
"""

from __future__ import annotations

import argparse
import os
import zipfile

from training.evaluate_image import _build_cifake_eval_manifest, evaluate
from training.individual import run as train_run
from backend.models import registry

DEFAULT_ZIP = r"C:\Users\reece\OneDrive\Documents\archive.zip"


def extract(zip_path: str, train_n: int, test_n: int):
    z = zipfile.ZipFile(zip_path)
    names = z.namelist()
    plan = [
        ("cifake", "train/REAL", "REAL", train_n),
        ("cifake", "train/FAKE", "FAKE", train_n),
        ("cifake_test", "test/REAL", "REAL", test_n),
        ("cifake_test", "test/FAKE", "FAKE", test_n),
    ]
    for folder, src_pref, label, n in plan:
        dest = os.path.join("data", "raw", folder, label)
        os.makedirs(dest, exist_ok=True)
        files = [x for x in names if x.startswith(src_pref + "/") and not x.endswith("/")][:n]
        for x in files:
            with z.open(x) as src, open(os.path.join(dest, os.path.basename(x)), "wb") as out:
                out.write(src.read())
        print(f"  {dest}: {len(files)} images")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=DEFAULT_ZIP, help="path to CIFAKE archive.zip")
    ap.add_argument("--train", type=int, default=1000, help="images per class for training")
    ap.add_argument("--test", type=int, default=500, help="images per class for testing")
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    if not args.skip_extract:
        if not os.path.exists(args.zip):
            raise SystemExit(f"zip not found: {args.zip}")
        print(f"Extracting from {args.zip} ...")
        extract(args.zip, args.train, args.test)

    print("Building manifest ...")
    manifest = os.path.join("data", "manifests", "image_cifake_eval.csv")
    rows = _build_cifake_eval_manifest(manifest)
    print(f"  {len(rows)} rows")

    print("Training + evaluating image detector ...")
    report = evaluate(rows)

    import json
    from backend.config import MODELS_DIR
    report["modality"] = "image"
    report["mode_used"] = "model (trained MLP head)"
    with open(MODELS_DIR / "evaluation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if report.get("status") == "ok":
        tm = report["test_metrics"]
        print("\nHELD-OUT RESULTS (real, non-inflated):")
        print(f"  accuracy={tm['accuracy']} precision={tm['precision']} "
              f"recall={tm['recall']} f1={tm['f1']} roc_auc={tm['roc_auc']}")
    print("\nDone. Now run the app with MODE=model.")


if __name__ == "__main__":
    main()
