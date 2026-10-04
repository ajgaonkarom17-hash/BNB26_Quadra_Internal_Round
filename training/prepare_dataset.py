"""
Prepare a training manifest from a folder structure.

Expected layout (one folder per class)::

    dataset/
      image/
        authentic/    *.jpg
        manipulated/  *.jpg
      audio/
        bonafide/     *.wav
        spoof/        *.wav
      video/
        authentic/
        manipulated/
      document/
        authentic/
        manipulated/

This does NOT download anything. Point it at data you already have (CIFAKE,
GenImage, FaceForensics++, Celeb-DF, ASVspoof, etc.).

Usage:
    python -m training.prepare_dataset --modality image --root data/raw/image --out data/manifests/image.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

POSITIVE_DIRS = {"manipulated", "spoof", "fake", "suspicious", "positive"}
NEGATIVE_DIRS = {"authentic", "bonafide", "real", "genuine", "negative"}


def infer_label(folder_name: str):
    name = folder_name.lower()
    if name in POSITIVE_DIRS:
        return 1
    if name in NEGATIVE_DIRS:
        return 0
    return None


def scan(root: str):
    rows = []
    for dirpath, _dirs, files in os.walk(root):
        folder = os.path.basename(dirpath)
        label = infer_label(folder)
        if label is None:
            continue
        for f in files:
            rows.append({"path": str(Path(dirpath) / f), "label": label})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--modality", default="image")
    args = parser.parse_args()

    rows = scan(args.root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
