"""Lightweight, dependency-optional training helpers (numpy only)."""

from __future__ import annotations

from typing import Optional

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_logistic(X, y, epochs: int = 400, lr: float = 0.1, l2: float = 1e-3):
    """Plain gradient-descent logistic regression. Returns (w, b)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    n, d = X.shape
    w = np.zeros(d, dtype=np.float32)
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def binary_metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    auc = roc_auc(y_true, y_score) if y_score is not None else None
    return {"accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1": round(f1, 4), "roc_auc": auc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def roc_auc(y_true, y_score) -> Optional[float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    for idx, c in enumerate(counts):
        if c > 1:
            ranks[inv == idx] = ranks[inv == idx].mean()
    pos = ranks[y_true == 1].sum()
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    return round(float((pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 4)


def train_mlp(X, Y, hidden: int = 32, epochs: int = 600, lr: float = 0.05, l2: float = 1e-4,
              val_X=None, val_y=None, patience: int = 0):
    """
    Two-layer MLP for multi-class or multi-task labels.
    Y shape (n, k). Returns dict of weights for models/fusion.npz style saving.
    If val_X/val_y and patience are provided, keeps the weights from the epoch
    with the lowest validation loss (early stopping / overfit protection).
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, d = X.shape
    k = Y.shape[1]
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((d, hidden)).astype(np.float32) * 0.05
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.standard_normal((hidden, k)).astype(np.float32) * 0.05
    b2 = np.zeros(k, dtype=np.float32)
    best = None
    best_loss = float("inf")
    bad = 0
    for _ in range(epochs):
        h = np.tanh(X @ w1 + b1)
        out = h @ w2 + b2
        if k > 1:
            e = np.exp(out - out.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            grad_out = (p - Y) / n
        else:
            p = sigmoid(out)
            grad_out = (p - Y) / n
        gw2 = h.T @ grad_out + l2 * w2
        gb2 = grad_out.sum(axis=0)
        gh = (grad_out @ w2.T) * (1 - h ** 2)
        gw1 = X.T @ gh + l2 * w1
        gb1 = gh.sum(axis=0)
        w1 -= lr * gw1; b1 -= lr * gb1; w2 -= lr * gw2; b2 -= lr * gb2

        if patience and val_X is not None:
            vh = np.tanh(val_X @ w1 + b1)
            vout = vh @ w2 + b2
            if k > 1:
                e = np.exp(vout - vout.max(axis=1, keepdims=True))
                vp = e / e.sum(axis=1, keepdims=True)
            else:
                vp = sigmoid(vout)
            vloss = float(np.mean((vp - val_y.reshape(-1, 1) if val_y.ndim == 1 else val_y) ** 2))
            if vloss < best_loss - 1e-6:
                best_loss = vloss
                best = {"w1": w1.copy(), "b1": b1.copy(),
                        "w2": w2.copy(), "b2": b2.copy()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best is not None:
        return best
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
