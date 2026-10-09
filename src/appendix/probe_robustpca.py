"""Does making the subspace estimate robust change the effect of removal?

Background (09-23):
  corr(projection share onto top 30, within-window AC energy) = -0.26 (0.00 for a random subspace).
  **Windows that move within the window fall outside the top 30 subspace.**
  Moreover, the top 30 have the lowest within-window share (0.107, overall 0.31), i.e. they are the most temporally stable directions.

Hypothesis:
  Moving windows are atypical points that deviate from "that subject's dominant configuration".
  Standard PCA maximizes variance, so such points may contaminate the principal-component estimate.
  A robust estimate might change the 30 removed directions, and with them the effect.

  Before building a robust PCA (low-rank + sparse decomposition), measure the same hypothesis cheaply:
    trim   : estimate the subspace only from windows in the bottom q quantile of AC, then remove it from all windows
    weight : estimate from a covariance weighting each window by 1/(1+AC/median)

  Both build "a subspace with reduced influence from moving windows".

kill criterion:
  If neither beats the current one (estimated on all windows) on all seeds, conclude that contamination is not a problem.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from postprocess import tune_tau
from probe_ens2 import graph_feat, load_oof, multistage

PREP = WORK / "prep"


def subspace(F, w=None, m=30):
    """Return the basis of the top m principal components from a weighted covariance. w=None gives ordinary PCA."""
    c = F - (F.mean(0) if w is None else (w[:, None] * F).sum(0) / w.sum())
    if w is not None:
        c = c * np.sqrt(w)[:, None]
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    return Vt[:m]


def remove(F, B):
    c = F - F.mean(0)
    return c - (c @ B.T) @ B


def main(a):
    wts = a.weights or [1.0] * len(a.tags)
    res = {}
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    import pandas as pd
    meta = pd.read_parquet(PREP / "win_meta.parquet")

    for sd in a.seeds:
        P, V, y, sbj, raw, Vd = load_oof(a.tags, wts, a.vbase, sd)
        # raw holds (n,15,768) per subject
        variants = {}
        for s in np.unique(sbj):
            w15 = raw[s]
            dc = w15.mean(1)
            ac = ((w15 - dc[:, None, :]) ** 2).mean((1, 2))
            B_all = subspace(dc, None, a.m)
            thr = np.quantile(ac, a.q)
            keep = ac <= thr
            B_trim = subspace(dc[keep], None, a.m) if keep.sum() > a.m + 5 else B_all
            wv = 1.0 / (1.0 + ac / (np.median(ac) + 1e-9))
            B_wt = subspace(dc, wv, a.m)
            variants.setdefault("current (estimated on all windows)", {})[s] = remove(dc, B_all)
            variants.setdefault(f"trim (estimated on bottom {a.q:.0%} of AC)", {})[s] = remove(dc, B_trim)
            variants.setdefault("weight (1/(1+AC))", {})[s] = remove(dc, B_wt)

        for nm, Fd in variants.items():
            Q = np.empty_like(P)
            for s in np.unique(sbj):
                msk = sbj == s
                F = Fd[s]
                F = F - F.mean(0)
                F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
                F = np.concatenate([F, a.gamma * np.sqrt(Vd[s])], 1)
                Q[msk] = multistage(P[msk], F, V[msk], a.stages, a.k, a.alpha,
                                    a.iters, a.temp, a.g)
            res.setdefault(nm, []).append(tune_tau(Q, y)[0])
        print(f"  seed {sd} done", flush=True)

    base = np.array(res["current (estimated on all windows)"])
    print(f"\n=== Robust subspace estimation (m={a.m}, {len(a.seeds)} seeds) ===")
    for nm, v in sorted(res.items(), key=lambda x: -np.mean(x[1])):
        line = f"  {np.mean(v):.4f}  {nm}"
        if nm != "current (estimated on all windows)":
            d = np.array(v) - base
            line += (f"   vs current {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d)
                     + f"]  all same sign {bool(np.all(d > 0) or np.all(d < 0))}")
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--vbase", required=True)
    p.add_argument("--m", type=int, default=30)
    p.add_argument("--q", type=float, default=0.7)
    p.add_argument("--g", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--seeds", nargs="*", type=int, default=[SEED, 7, 123])
    main(p.parse_args())
