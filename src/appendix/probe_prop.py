"""Sweep the breadth and strength of propagation, and 2-stage refinement (a graph version of the MS-TCN multi-stage idea).

Upper bound (true-segment mean) = 0.927 vs current 0.754. Locate where the gap comes from along 3 axes:
neighbourhood breadth / number of iterations / reweighting with probabilities in stage 2.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from metric import macro_f1
from postprocess import tune_tau

PREP = WORK / "prep"


def knn_graph(F, k, mutual=True, center=True):
    if center:
        F = F - F.mean(0)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = F @ F.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.take_along_axis(S, nn, 1)
    if mutual:
        rank = np.zeros(S.shape, bool)
        rank[np.arange(len(nn))[:, None], nn] = True
        keep = rank[np.arange(len(nn))[:, None], nn] & rank[nn, np.arange(len(nn))[:, None]]
        w = np.where(keep, w, -1e9)
    return nn, w


def propagate(P, nn, w, alpha, iters, sharp=10.0):
    ww = np.exp((w - w.max(1, keepdims=True)) * sharp)
    ww /= ww.sum(1, keepdims=True) + 1e-12
    Q = P.copy()
    for _ in range(iters):
        Q = (1 - alpha) * P + alpha * (Q[nn] * ww[:, :, None]).sum(1)
    return Q


def main():
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    oof = np.load(WORK / "exp001_lgb_inertial" / "oof.npy")
    valid = np.load(PREP / "valid_mask.npy")
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    P = oof[tile + sens * n]; keep = P.sum(1) > 0
    tile, P = tile[keep], P[keep]
    P = P / P.sum(1, keepdims=True)
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    subs = np.unique(sbj)
    F = {s: np.asarray(video[tile[sbj == s]], np.float32).mean(1) for s in subs}

    def run(k, alpha, iters, stage2_k=None, stage2_alpha=0.75, stage2_iters=5, lam=1.0):
        Q = P.copy()
        for s in subs:
            m = sbj == s
            nn, w = knn_graph(F[s], k)
            q = propagate(P[m], nn, w, alpha, iters)
            if stage2_k:
                # stage 2: add stage-1 probability similarity to the distance and rebuild the graph (multi-stage refinement)
                G = np.concatenate([F[s] / (np.linalg.norm(F[s], axis=1, keepdims=True) + 1e-8),
                                    lam * np.sqrt(q)], 1)
                nn2, w2 = knn_graph(G, stage2_k, center=False)
                q = propagate(q, nn2, w2, stage2_alpha, stage2_iters)
            Q[m] = q
        return tune_tau(Q, y)

    print(f"base = {tune_tau(P, y)[0]:.4f}   (upper bound: true-segment mean = 0.9270)")
    for k in [15, 30, 50, 80]:
        f1, tau = run(k, 0.75, 5)
        print(f"  1-stage mutual k={k:<3d} a=0.75 it=5            -> {f1:.4f} (tau={tau:.2f})")
    for it in [10, 20]:
        f1, tau = run(30, 0.85, it)
        print(f"  1-stage mutual k=30  a=0.85 it={it:<2d}           -> {f1:.4f} (tau={tau:.2f})")
    for lam in [0.5, 1.0, 2.0]:
        f1, tau = run(30, 0.75, 5, stage2_k=30, lam=lam)
        print(f"  2-stage refine k=30 lambda={lam}          -> {f1:.4f} (tau={tau:.2f})")


if __name__ == "__main__":
    main()
