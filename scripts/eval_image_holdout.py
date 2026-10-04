"""Honest held-out evaluation: train on train/ split, test on CIFAKE test/ split.

Run with MODE not set; this loads the saved image head and scores the held-out
test images directly through the same registry the API uses.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.getcwd())

from backend.services.image_analyzer import analyze
from backend.models import registry

TEST = os.path.join("data", "raw", "cifake_test")


def score_dir(label):
    d = os.path.join(TEST, label)
    fs = sorted(os.listdir(d))
    out = []
    for f in fs:
        r = analyze(os.path.join(d, f), {"evidence_id": "x", "filename": f})
        s = registry.predict_individual("image", r.features)
        out.append(s if s is not None else r.score)
    return np.array(out)


real = score_dir("REAL")
fake = score_dir("FAKE")
y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
p = np.r_[real, fake]
pred = (p >= 0.5).astype(int)

acc = (pred == y).mean()
tp = ((pred == 1) & (y == 1)).sum()
fp = ((pred == 1) & (y == 0)).sum()
fn = ((pred == 0) & (y == 1)).sum()
tn = ((pred == 0) & (y == 0)).sum()
prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(f"HELD-OUT TEST (CIFAKE test/ split, n={len(y)})")
print(f"  accuracy ={acc:.4f}")
print(f"  precision={prec:.4f}")
print(f"  recall   ={rec:.4f}")
print(f"  f1       ={f1:.4f}")
print(f"  tp={tp} tn={tn} fp={fp} fn={fn}")
print(f"  mean score REAL={real.mean():.3f} FAKE={fake.mean():.3f}")
