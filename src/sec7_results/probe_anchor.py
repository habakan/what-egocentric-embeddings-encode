"""Do labelled anchors work? — measure the purity of video neighbours across subjects.

Motivation (from a critical review of the paper):
  §2 cites Zhou 2003 / Iscen 2019 / TPN, but those are methods where **the graph contains labelled nodes
  and labels flow from them**. Our system has not a single labelled node;
  it only smooths predictions among unlabelled test windows, which is not label propagation.
  The windows of the 22 training subjects have labels. Adding them as anchors **brings ground truth in from outside**.
  The current propagation can only average out errors; there is no ground truth anywhere, so it cannot be corrected from outside.

Obstacle, and the prediction that removes it:
  video appearance is dominated by subject and location (observed: cobblestones/grass/dappled sunlight/clothing colour differ completely).
  But the biggest win of this system is **removing the top 30 PCs**. If the top PCs are subject/scene appearance,
  **matching across subjects should work after removal**. The winning move connects to a new capability.

What is measured (leave-one-subject-out):
  for each window of subject s, take the top-k neighbours from windows of **subjects other than s** and check the label agreement rate.
  Centre and L2-normalise per subject, and remove the top m PCs before matching.

  Control: agreement rate with labels shuffled within subject = chance level.
        null makes up 44%, so it is not 1/19. Measured empirically.

Kill criterion:
  if no m clearly exceeds chance level, anchors do not work, so plan A is dropped entirely.
  If it exceeds chance and improves with m, build it into the production graph.
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


def prep_subject(F, m):
    """Centre within subject -> remove top m PCs -> L2-normalise. Puts features in a form comparable across subjects."""
    F = F - F.mean(0)
    if m:
        F = drop_top_pcs(F, m)
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


def topk_purity(Q, B, yb, yq, k, chunk=512):
    """For each row of Q, take the top-k neighbours from B and return the label agreement rate."""
    hit = tot = 0
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ B.T
        kk = min(k, B.shape[0] - 1)
        nn = np.argpartition(-S, kk, axis=1)[:, :kk]
        hit += (yb[nn] == yq[i:i + chunk, None]).sum()
        tot += nn.size
    return hit / tot


def main(a):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    # subsample a fixed number per subject (all of them would be 62k x 62k)
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
    print(f"  {len(us)} subjects x up to {a.per_subject} windows = {len(tile)} windows total")

    # chance level: from the label marginal distribution (null makes up 44%, so not 1/19)
    p = np.bincount(y, minlength=19) / len(y)
    print(f"  chance level (from label marginals) = {float((p**2).sum()):.4f}   "
          f"if uniform {1/19:.4f}\n")

    print(f"  {'PCs rm':>7s}{'cross-subj purity':>18s}{'within-subj (ref)':>20s}")
    for m in a.drop_pcs:
        Fn = {s: prep_subject(raw[s], m) for s in us}
        cross, within = [], []
        for s in us:
            q, yq = Fn[s], y[sbj == s]
            others = [o for o in us if o != s]
            B = np.concatenate([Fn[o] for o in others])
            yb = np.concatenate([y[sbj == o] for o in others])
            cross.append(topk_purity(q, B, yb, yq, a.k))
            within.append(topk_purity(q, q, yq, yq, a.k + 1))
        print(f"  {m:7d}{np.mean(cross):18.4f}{np.mean(within):20.4f}")

    print("\n  kill criterion: if no m clearly exceeds chance level, plan A is dropped.")
    print("  prediction: if the top PCs are subject/scene appearance, cross-subject purity rises with m.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--drop-pcs", nargs="*", type=int, default=[0, 10, 30, 50, 100])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--per-subject", type=int, default=1200)
    main(p.parse_args())
