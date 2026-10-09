"""Evaluate the whole pipeline (blend → multi-stage refinement → prior correction) on OOF.

Evaluation conditions are always test-equivalent: 1 s non-overlapping tiles × one random sensor per window.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from postprocess import tune_tau, tune_weights, prior_correct
from metric import macro_f1, per_class_f1
from refine import multistage

PREP = WORK / "prep"


def load_eval_set(tags, weights, seed=SEED, video_offset="head"):
    """Test-equivalent evaluation set. The randomness choosing one sensor per window is controlled by seed.
    Evaluating several times with different seeds and averaging reduces the variance from this randomness and stabilises comparisons.
    video_offset: the video rows used for the graph. The paper's results use "head" ([f0, f0+15)). The rows matching test are "sync" ([f0+8, f0+23))."""
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    rows = tile + sens * n

    P = None
    for t, w in zip(tags, weights):
        o = np.load(WORK / t / "oof.npy")[rows]
        s = o.sum(1, keepdims=True)
        o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(weights)
    keep = P.sum(1) > 1e-6
    tile, P = tile[keep], P[keep]
    P /= P.sum(1, keepdims=True)
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    video = np.load(PREP / f"video_raw_{video_offset}.npy", mmap_mode="r")
    F = {s: np.asarray(video[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    return P, y, sbj, F


def run(P, y, sbj, F, stages, k, alpha, iters, temp=1.0, conf_pow=0.0, conf_alpha=0.0):
    Q = P.copy()
    for s in np.unique(sbj):
        m = sbj == s
        Q[m] = multistage(P[m], F[s], stages, k=k, alpha=alpha, iters=iters, temp=temp,
                          conf_pow=conf_pow, conf_alpha=conf_alpha)
    return Q


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    P, y, sbj, F = load_eval_set(a.tags, w)
    print(f"tags={a.tags} weights={w}  n={len(y)}")
    print(f"  no refinement         : {tune_tau(P, y)[0]:.4f}")
    for stages in [[2.0], [4.0, 4.0]]:
        Q = run(P, y, sbj, F, stages, a.k, a.alpha, a.iters)
        f1, tau = tune_tau(Q, y)
        print(f"  stages={str(stages):12s}   : {f1:.4f} (tau={tau:.2f})")
        if stages == [4.0, 4.0]:
            fw, logw = tune_weights(Q, y)
            print(f"    + per-class weights  : {fw:.4f}")
            print("    per-class:", np.round(per_class_f1(y, prior_correct(Q, tau).argmax(1)), 2).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    main(p.parse_args())
