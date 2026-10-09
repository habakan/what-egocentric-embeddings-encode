"""Can the "next window" be found from continuity at inertial boundaries? (idea from the 2nd-place public solution, own implementation)

The test is shuffled non-overlapping 1 s tiles, each window a random limb. If the next window is the same limb,
the last sample of the previous window and the first sample of the next are only 20 ms apart.
Measure with training data in test format (tiles, one limb per window by seed):
  P@1: for each window A, whether the B with the smallest boundary gap among same-limb candidates is the true successor
  gap: plain ||A[-1]-B[0]|| and linear extrapolation ||(2A[-1]-A[-2])-B[0]||
  video: cosine between A's last row and B's first row (candidates restricted to the same limb for a fair comparison)
  combined: sum of the two ranks
  accuracy in the top 20% of confidence (gap between 1st and 2nd)
Test side: does the distribution of minimum gaps of same-limb pairs resemble the training "true successor" or "others"?
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import SENSORS, WORK

dev = "cuda"
meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
rec = meta["rec"].to_numpy()[tile]; start = meta["start"].to_numpy()[tile]
xi = np.load(WORK / "prep" / "inertial.npy", mmap_mode="r")
vid = np.load(WORK / "prep" / "video_raw_head.npy", mmap_mode="r")
rng = np.random.RandomState(42)
limb = rng.randint(0, 4, len(tile))


def rank_of_true(D, true_idx):
    """D: (n, n) score, smaller is better (self is inf). true_idx: true successor (-1 = none)."""
    ok = true_idx >= 0
    r = (D[ok] < D[ok, true_idx[ok]][:, None]).sum(1)
    return r, ok


out = {"gap": [], "extrap": [], "video": [], "comb": []}
conf = {k: [] for k in out}
for g in np.unique(rec):
    m = np.where(rec == g)[0]
    X = torch.tensor(np.nan_to_num(np.asarray(xi[tile[m]], np.float32)), device=dev)[torch.arange(len(m)), torch.tensor(limb[m], device=dev)]  # (n,50,3)
    V = torch.tensor(np.asarray(vid[tile[m]], np.float32), device=dev)
    lm = torch.tensor(limb[m], device=dev)
    st = start[m]; pos = {s: i for i, s in enumerate(st)}
    true = np.array([pos.get(s + 50, -1) for s in st])
    true = np.where((true >= 0) & (limb[m][np.maximum(true, 0)] == limb[m]), true, -1)   # only successors of the same limb
    same = lm[:, None] == lm[None, :]
    inf = torch.tensor(float("inf"), device=dev)
    eye = torch.eye(len(m), dtype=torch.bool, device=dev)
    gap = torch.cdist(X[:, -1], X[:, 0])
    ext = torch.cdist(2 * X[:, -1] - X[:, -2], X[:, 0])
    a = torch.nn.functional.normalize(V[:, -1], dim=1); b = torch.nn.functional.normalize(V[:, 0], dim=1)
    vd = 1 - a @ b.T
    Ds = {}
    for nm, D in [("gap", gap), ("extrap", ext), ("video", vd)]:
        Ds[nm] = torch.where(same & ~eye, D, inf)
    rk = lambda D: torch.argsort(torch.argsort(D, 1), 1).float()
    Ds["comb"] = torch.where(same & ~eye, rk(Ds["extrap"]) + rk(Ds["video"]), inf)
    tt = torch.tensor(true, device=dev)
    for nm, D in Ds.items():
        r, ok = rank_of_true(D, tt)
        out[nm].append((r == 0).float().cpu().numpy())
        srt = torch.sort(D[ok], 1).values
        margin = (srt[:, 1] - srt[:, 0]) / (srt[:, 1].abs() + 1e-6)
        conf[nm].append(np.stack([margin.cpu().numpy(), (r == 0).float().cpu().numpy()], 1))
    step = (X[:, 1:] - X[:, :-1]).norm(dim=-1).median().item()
    if g == np.unique(rec)[0]:
        print(f"  reference ({g}): median 1-step within window {step:.4f} g / median boundary gap of true successor "
              f"{gap[torch.tensor(np.where(true>=0)[0],device=dev), tt[tt>=0]].median().item():.4f} / median over all same-limb pairs {gap[same & ~eye].median().item():.4f}")

print("\n=== Training (test format, only windows that have a same-limb successor) ===")
for nm in out:
    p1 = np.concatenate(out[nm]); c = np.concatenate(conf[nm])
    top = c[c[:, 0] >= np.percentile(c[:, 0], 80)]
    print(f"  {nm:7s} P@1 {p1.mean():.3f}  (n={len(p1)})  accuracy in top 20% confidence {top[:, 1].mean():.3f}")

# Test side: distribution of the minimum gap (extrapolated) of same-limb pairs
tm = pd.read_csv(WORK.parent / "input/test/test_meta_data.csv")
xt = np.load(WORK.parent / "input/test/test_inertial_data.npy").astype(np.float32)
sl = tm["sensor_location"].map({s: i for i, s in enumerate(SENSORS)}).to_numpy()
tr_best = []
for g in np.unique(rec)[:6]:
    m = np.where(rec == g)[0]
    X = torch.tensor(np.nan_to_num(np.asarray(xi[tile[m]], np.float32)), device=dev)[torch.arange(len(m)), torch.tensor(limb[m], device=dev)]
    lm = torch.tensor(limb[m], device=dev); same = (lm[:, None] == lm[None, :]) & ~torch.eye(len(m), dtype=torch.bool, device=dev)
    D = torch.where(same, torch.cdist(2 * X[:, -1] - X[:, -2], X[:, 0]), torch.tensor(float("inf"), device=dev))
    tr_best.append(D.min(1).values.cpu().numpy())
te_best = []
for s in tm.sbj_id.unique():
    m = np.where(tm.sbj_id.values == s)[0]
    X = torch.tensor(xt[m], device=dev); lm = torch.tensor(sl[m], device=dev)
    same = (lm[:, None] == lm[None, :]) & ~torch.eye(len(m), dtype=torch.bool, device=dev)
    D = torch.where(same, torch.cdist(2 * X[:, -1] - X[:, -2], X[:, 0]), torch.tensor(float("inf"), device=dev))
    te_best.append(D.min(1).values.cpu().numpy())
tr_best, te_best = np.concatenate(tr_best), np.concatenate(te_best)
print("\n=== Quantiles of the minimum gap (extrapolated): train vs test ===")
for q in [10, 25, 50, 75]:
    print(f"  {q}%: train {np.percentile(tr_best, q):.4f}  test {np.percentile(te_best, q):.4f}")
