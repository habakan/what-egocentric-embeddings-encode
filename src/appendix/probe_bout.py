"""Do the neighbours retrieve "the same segment" rather than "the same activity"? A quantity never measured.

Where the question comes from:
  The top 30 principal components were the best part of the representation on all 4 measures (largest variance, highest activity eta^2,
  strongest cross-subject transfer, most stable within a window). Physically they point to the window's **stable viewpoint configuration** ---
  which way the head is facing. What remains after removing them is the fine-grained appearance of the ground
  currently in front of the wearer.

  A participant performs one exercise roughly in one place. So the residual becomes a fingerprint of "which place /
  which point in the session the wearer is at".

Prediction:
  The graph's job is not classifying activity but **segment membership** (averaging over the true segment gives 0.927).
  Posture is shared by push-ups, burpees, and lunges, so linking by posture **mixes separate segments**.
  Therefore

      removing the top 30 **lowers label purity but raises segment purity**

  If this holds, it answers the paradox that went unexplained 16 times.
  If not, this interpretation goes the same way as the other 16.

Definitions:
  segment = a run of consecutive identical labels within one recording (label run). Different runs of the same activity are different segments.
  segment purity = fraction of the k neighbours that belong to the same segment as the query.
  control: what segment purity random neighbours with the same label purity would have
        (segment purity automatically drops when label purity drops, so that part must be subtracted).
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
from config import SEED, WORK
from refine import drop_top_pcs

PREP = WORK / "prep"


def bout_ids(labels, starts):
    """Number the runs of consecutive labels when ordered in time."""
    o = np.argsort(starts)
    lab = labels[o]
    b = np.zeros(len(lab), int)
    b[1:] = np.cumsum(lab[1:] != lab[:-1])
    out = np.empty(len(lab), int)
    out[o] = b
    return out


def stats(F, y, bout, k):
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(Fn) - 2)
    nn = np.argpartition(-S, kk, axis=1)[:, :kk]
    same_lab = y[nn] == y[:, None]
    same_bout = bout[nn] == bout[:, None]
    # Control: among neighbours with matching label, the fraction in the same segment
    #   segment purity automatically drops when label purity drops, so condition that part out
    cond = same_bout[same_lab].mean() if same_lab.sum() else np.nan
    return float(same_lab.mean()), float(same_bout.mean()), float(cond)


def main(a):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sbj = meta["sbj_id"].to_numpy()[tile]
    y = meta["label"].to_numpy()[tile]
    start = meta["start"].to_numpy()[tile]
    rec = meta["rec"].to_numpy()[tile]

    us = np.unique(sbj)[:a.n_subjects]
    print(f"  {'PCs rm':>7s}{'label purity':>12s}{'seg purity':>11s}"
          f"{'same seg in same label':>20s}{'segs/subject':>15s}")
    for m in a.drop_pcs:
        L, B, C, NB = [], [], [], []
        for s in us:
            msk = sbj == s
            F = np.asarray(vid[tile[msk]], np.float32).mean(1)
            if m:
                F = drop_top_pcs(F, m)
            # number segments per recording (a subject may have several recordings)
            bo = np.zeros(int(msk.sum()), int)
            off = 0
            for r in np.unique(rec[msk]):
                rm = rec[msk] == r
                bo[rm] = bout_ids(y[msk][rm], start[msk][rm]) + off
                off = bo[rm].max() + 1
            l_, b_, c_ = stats(F, y[msk], bo, a.k)
            L.append(l_); B.append(b_); C.append(c_); NB.append(len(np.unique(bo)))
        print(f"  {m:7d}{np.mean(L):12.4f}{np.mean(B):11.4f}"
              f"{np.nanmean(C):20.4f}{np.mean(NB):15.1f}")

    print("\n  Prediction: removal lowers label purity but raises the fraction of **same segment among same label**.")
    print("  (segment purity itself is dragged by label purity, so judge by the conditioned column)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--drop-pcs", nargs="*", type=int, default=[0, 10, 30, 50])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n-subjects", type=int, default=10)
    main(p.parse_args())
