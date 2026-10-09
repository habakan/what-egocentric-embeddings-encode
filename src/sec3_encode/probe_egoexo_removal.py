"""Ego-Exo4D: does removing the top principal components improve the within-participant affinity graph (3 encoders)?

For each participant, hold out one take at a time, propagate the task labels of the remaining takes over a video k-nearest-neighbour graph,
and predict the task of the held-out take's windows (leave-one-take-out). Sweep the number of removed components m.
Also measure the task purity of neighbours restricted to other takes.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(DATA_DIR + "/egoexo4d")
ENC = {"VideoMAEv2": "features", "CLIP": "features_clip", "DINOv2": "features_dinov2"}
MS = [0, 5, 10, 20, 30, 50]


def load(d):
    T = {t["take_name"]: t["task_name"] for t in json.load(open(ROOT / "takes.json"))}
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / d / "*.npz"))):
        z = np.load(f)
        ok = z["ok"]
        by[int(z["participant"])].append((Path(f).stem, T[Path(f).stem], z["F"].mean(1)[ok]))
    return by


def embed(F, m):
    X = F - F.mean(0)
    if m:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        X = X - (X @ Vt[:m].T) @ Vt[:m]
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def graph(X, k):
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argsort(-S, 1)[:, :k]
    A = np.zeros_like(S, dtype=bool)
    A[np.repeat(np.arange(len(X)), k), idx.ravel()] = True
    A = A & A.T                                                  # mutual k nearest neighbours
    W = np.where(A, np.maximum(S, 0), 0.0)
    return W / (W.sum(1, keepdims=True) + 1e-9)


def run(by, m, k=30, alpha=0.75, iters=20):
    y_all, p_all, pur = [], [], []
    for p, takes in by.items():
        F = np.concatenate([t[2] for t in takes])
        tid = np.concatenate([[i] * len(t[2]) for i, t in enumerate(takes)])
        names = sorted({t[1] for t in takes})
        y = np.concatenate([[names.index(t[1])] * len(t[2]) for t in takes])
        X = embed(F, m)
        W = graph(X, k)
        S = X @ X.T
        for i in range(len(takes)):
            te = tid == i
            if len(np.unique(y[~te])) < len(names):                 # cannot be predicted unless the other takes contain all tasks
                continue
            Q0 = np.zeros((len(X), len(names)))
            Q0[~te, y[~te]] = 1
            Q = Q0.copy()
            for _ in range(iters):
                Q = (1 - alpha) * Q0 + alpha * W @ Q
            y_all.append(y[te]); p_all.append(Q[te].argmax(1))
            # purity of neighbours restricted to other takes
            Sx = S[te][:, ~te]
            nn = np.argsort(-Sx, 1)[:, :10]
            pur.append((y[~te][nn] == y[te][:, None]).mean(1))
    y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
    return (y_all == p_all).mean(), np.concatenate(pur).mean()


if __name__ == "__main__":
    for name, d in ENC.items():
        by = load(d)
        print(f"== {name}")
        for m in MS:
            acc, pur = run(by, m)
            print(f"  m={m:3d}  LOTO propagation accuracy {acc:.3f}   other-take neighbour purity@10 {pur:.3f}")
