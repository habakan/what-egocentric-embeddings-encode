"""Measure the effect of transductive kNN smoothing on OOF.

test is "1-second non-overlapping tiles x one random sensor", so OOF is reduced to that form for evaluation:
  - use only windows with start % 50 == 0 (every other at stride 25 = 1-second non-overlapping)
  - pick one sensor at random per window
Then smooth the predicted probabilities with a within-subject kNN graph on video features and look at the change in macro-F1.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, SENSORS, WORK
from metric import macro_f1, per_class_f1

PREP = WORK / "prep"


def knn_smooth(P, F, k, alpha, iters):
    """P: (n, C) probabilities, F: (n, d) features -> smoothed probabilities"""
    F = F - F.mean(0)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = F @ F.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.take_along_axis(S, nn, 1)
    w = np.exp((w - w.max(1, keepdims=True)) * 10.0)
    w /= w.sum(1, keepdims=True)
    Q = P.copy()
    for _ in range(iters):
        Q = (1 - alpha) * P + alpha * (Q[nn] * w[:, :, None]).sum(1)
    return Q


def main(tag, k, alpha, iters, seed):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    n_win = len(meta)
    oof = np.load(WORK / tag / "oof.npy")                 # (n_win*4, C) sensor-major
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    valid = np.load(PREP / "valid_mask.npy")

    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]     # 1-second non-overlapping tiles
    sens = rng.randint(0, len(SENSORS), size=len(tile))
    ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    rows = tile + sens * n_win
    y = meta["label"].to_numpy()[tile]
    P = oof[rows]
    keep = P.sum(1) > 0
    tile, sens, rows, y, P = tile[keep], sens[keep], rows[keep], y[keep], P[keep]

    base = macro_f1(y, P.argmax(1))
    print(f"[{tag}] test-like (1-second tiles x one random sensor) n={len(y)}  macro-F1 = {base:.4f}")

    sbj = meta["sbj_id"].to_numpy()[tile]
    Q = P.copy()
    for s in np.unique(sbj):
        m = sbj == s
        F = np.asarray(video[tile[m]], np.float32).mean(1)      # (n_s, 768)
        Q[m] = knn_smooth(P[m], F, k, alpha, iters)
    sm = macro_f1(y, Q.argmax(1))
    print(f"  → kNN smoothing (k={k}, alpha={alpha}, iters={iters})  macro-F1 = {sm:.4f}  ({sm-base:+.4f})")
    print("  per-class before:", np.round(per_class_f1(y, P.argmax(1)), 2).tolist())
    print("  per-class after:", np.round(per_class_f1(y, Q.argmax(1)), 2).tolist())
    return base, sm


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="exp001_lgb_inertial")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    main(a.tag, a.k, a.alpha, a.iters, a.seed)
