"""VoI experiment comparing ways to build the transductive graph on OOF.

The effect of smoothing is decided by graph quality. Vary how features are built, normalized, and how neighbours are chosen,
and compare macro-F1 (with prior correction). One move = one variable.
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


def build_feats(V, kind):
    """V: (n, 15, 768) float32"""
    if kind == "mean":
        F = V.mean(1)
    elif kind == "mean_std":
        F = np.concatenate([V.mean(1), V.std(1)], 1)
    elif kind == "l2frame_mean":                       # L2-normalize per frame, then average
        Vn = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-8)
        F = Vn.mean(1)
    elif kind == "first_last":
        F = np.concatenate([V[:, 0], V[:, -1], V.mean(1)], 1)
    else:
        raise ValueError(kind)
    return F


def graph_smooth(P, F, k, alpha, iters, mutual=False, center=True, whiten=False):
    if center:
        F = F - F.mean(0)
    if whiten:
        F = F / (F.std(0) + 1e-6)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = F @ F.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.take_along_axis(S, nn, 1)
    if mutual:
        # drop edges that are not mutual kNN (hub suppression)
        rank = np.full(S.shape, k + 1, np.int16)
        for i in range(len(nn)):
            rank[i, nn[i]] = 1
        w = np.where(rank[np.arange(len(nn))[:, None], nn] + rank[nn, np.arange(len(nn))[:, None]] <= 2,
                     w, -1e9)
    w = np.exp((w - w.max(1, keepdims=True)) * 10.0)
    w /= w.sum(1, keepdims=True) + 1e-12
    Q = P.copy()
    for _ in range(iters):
        Q = (1 - alpha) * P + alpha * (Q[nn] * w[:, :, None]).sum(1)
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

    V = {s: np.asarray(video[tile[sbj == s]], np.float32) for s in np.unique(sbj)}

    def run(kind, k, alpha, **kw):
        Q = P.copy()
        for s in np.unique(sbj):
            m = sbj == s
            Q[m] = graph_smooth(P[m], build_feats(V[s], kind), k, alpha, 5, **kw)
        f1, tau = tune_tau(Q, y)
        return f1, tau

    print(f"base (no smoothing) = {tune_tau(P, y)[0]:.4f}")
    for kind in ["mean", "mean_std", "l2frame_mean", "first_last"]:
        f1, tau = run(kind, 8, 0.75)
        print(f"  features={kind:14s} k=8 a=0.75 -> {f1:.4f} (tau={tau:.2f})")
    for k in [8, 15, 25]:
        f1, tau = run("mean", k, 0.75, mutual=True)
        print(f"  mutual kNN k={k:<3d}          -> {f1:.4f} (tau={tau:.2f})")
    for k, a in [(15, 0.85), (25, 0.9), (40, 0.9)]:
        f1, tau = run("mean", k, a)
        print(f"  k={k:<3d} alpha={a}          -> {f1:.4f} (tau={tau:.2f})")


if __name__ == "__main__":
    main()
