"""Build a prior A from labelled anchors, and first check **redundancy**.

Results of probe_anchor:
  Cross-subject video neighbour purity is 0.4084 at m=0 (2.5x chance 0.1606). So it does work.
  But it degrades monotonically with PC removal (0.2714 at m=30), so **anchor edges are built on m=0 features**.
  This is a heterogeneous setup using a different feature space from the within-subject graph (m=30, purity 0.432).

Main concern:
  The video-only model V already injected (0.506 alone) **learned the mapping from video to activity
  on the same training subjects**. A is its non-parametric version with the same information source. If redundant, nothing is gained.

  Why it could differ: A is retrieval, not compression, so it can beat the learned model
  on **rare classes**. macro-F1 weights rare classes equally.

kill criteria:
  If the arg-max agreement between A and V exceeds 0.9, stop as redundant.
  Otherwise sweep h and inject into the production pipeline.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, WORK
from metric import macro_f1, per_class_f1
from probe_anchor import prep_subject

PREP = WORK / "prep"


def anchor_prior(Fq, Fb, yb, k, temp=10.0, chunk=512):
    """For each query window, build a weighted vote of the top-k nearest labelled anchor windows."""
    A = np.zeros((len(Fq), N_CLASSES), np.float32)
    for i in range(0, len(Fq), chunk):
        S = Fq[i:i + chunk] @ Fb.T
        kk = min(k, Fb.shape[0] - 1)
        nn = np.argpartition(-S, kk, axis=1)[:, :kk]
        sv = np.take_along_axis(S, nn, 1)
        w = np.exp((sv - sv.max(1, keepdims=True)) * temp)
        for c in range(N_CLASSES):
            A[i:i + chunk, c] = (w * (yb[nn] == c)).sum(1)
    A += 1e-6
    return A / A.sum(1, keepdims=True)


def main(a):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sbj_all = meta["sbj_id"].to_numpy()[tile]
    keep = []
    for s in np.unique(sbj_all):
        idx = np.where(sbj_all == s)[0]
        keep.append(rng.choice(idx, min(a.per_subject, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    tile, sbj = tile[keep], sbj_all[keep]
    y = meta["label"].to_numpy()[tile]
    raw = {s: np.asarray(vid[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    us = np.unique(sbj)
    Fn = {s: prep_subject(raw[s], a.m) for s in us}

    A = np.zeros((len(tile), N_CLASSES), np.float32)
    for s in us:
        others = [o for o in us if o != s]
        B = np.concatenate([Fn[o] for o in others])
        yb = np.concatenate([y[sbj == o] for o in others])
        A[sbj == s] = anchor_prior(Fn[s], B, yb, a.k, a.temp)

    # OOF of the video-only model V (take the rows of the same windows)
    n = len(meta)
    V = np.load(WORK / a.vmodel / "oof.npy")
    Vq = V[tile] if V.shape[0] == n else V[tile + 0 * n]
    Vq = Vq / (Vq.sum(1, keepdims=True) + 1e-9)

    pa, pv = A.argmax(1), Vq.argmax(1)
    print(f"=== anchor prior A (m={a.m}, k={a.k}) ===")
    print(f"  A  alone macro-F1 : {macro_f1(y, pa):.4f}")
    print(f"  V  alone macro-F1 : {macro_f1(y, pv):.4f}   ({a.vmodel})")
    print(f"  **agreement of A and V: {(pa == pv).mean():.4f}**   (above 0.9: stop as redundant)")
    print(f"  either correct: {((pa == y) | (pv == y)).mean():.4f}  "
          f"A only correct {((pa == y) & (pv != y)).mean():.4f}  "
          f"V only correct {((pv == y) & (pa != y)).mean():.4f}")

    fa, fv = per_class_f1(y, pa), per_class_f1(y, pv)
    from config import CLASS_NAMES
    cnt = np.bincount(y, minlength=N_CLASSES)
    print(f"\n  per-class F1 (classes where A beats V)")
    for c in np.argsort(fa - fv)[::-1][:6]:
        print(f"    {CLASS_NAMES[c]:28s} A {fa[c]:.3f}  V {fv[c]:.3f}  "
              f"({fa[c]-fv[c]:+.3f})  count {cnt[c]}")
    print(f"  mean over rare classes (bottom 8 by count): A {fa[np.argsort(cnt)[:8]].mean():.3f}  "
          f"V {fv[np.argsort(cnt)[:8]].mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=0)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--temp", type=float, default=10.0)
    p.add_argument("--per-subject", type=int, default=1200)
    p.add_argument("--vmodel", default="exp020_video_only")
    main(p.parse_args())
