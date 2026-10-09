"""Can the "next window" be identified regardless of limb (using the same video rows as the test, f0+8..f0+22 = video_raw_sync)?

For each window A, identify its successor B among all windows of the same participant (any limb). True successor = start+50 in the same recording.
Cues:
  v_last  : 1 - cos(last row of A, first row of B)
  v_ext   : distance between the row linearly extrapolated 16 frames ahead from the slope of A's last 4 rows and B's first row (after normalization)
  v_mean  : 1 - cos(mean of A, mean of B)
  i_ext   : inertial boundary extrapolation distance if same limb, missing if different limb (ranked last)
  comb    : sum of ranks (v_last + v_ext + i_ext)
Report P@1 for all windows / successor on same limb / successor on different limb (limb assignment of seed 42, all 20 participants).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from probe_chain_system import PREP, eval_rows

tile, sens, P, meta = eval_rows(42)
sbj = meta["sbj_id"].to_numpy()[tile]; rec = meta["rec"].to_numpy()[tile]; start = meta["start"].to_numpy()[tile]
xi = np.load(PREP / "inertial.npy", mmap_mode="r")
vs = np.load(PREP / "video_raw_sync.npy", mmap_mode="r")
X = np.nan_to_num(np.asarray(xi[tile], np.float32))[np.arange(len(tile)), sens]
inf = float("inf")
hits = {k: [] for k in ["v_last", "v_ext", "v_mean", "i_ext", "comb"]}
same_flag = []
for s in np.unique(sbj):
    m = np.where(sbj == s)[0]
    Vr = torch.tensor(np.asarray(vs[tile[m]], np.float32), device="cuda")            # (n, 15, 768)
    Vn = torch.nn.functional.normalize(Vr, dim=2)
    eye = torch.eye(len(m), dtype=torch.bool, device="cuda")
    D = {}
    D["v_last"] = 1 - Vn[:, -1] @ Vn[:, 0].T
    slope = (Vr[:, -1] - Vr[:, -4]) / 3.0
    ext = torch.nn.functional.normalize(Vr[:, -1] + 16 * slope, dim=1)
    D["v_ext"] = torch.cdist(ext, Vn[:, 0])
    mu = torch.nn.functional.normalize(Vr.mean(1), dim=1)
    D["v_mean"] = 1 - mu @ mu.T
    Xt = torch.tensor(X[m], device="cuda"); lm = torch.tensor(sens[m], device="cuda")
    ie = torch.cdist(2 * Xt[:, -1] - Xt[:, -2], Xt[:, 0])
    D["i_ext"] = torch.where(lm[:, None] == lm[None, :], ie, torch.tensor(inf, device="cuda"))
    for k in D:
        D[k] = D[k].masked_fill(eye, inf)
    rk = lambda A: torch.argsort(torch.argsort(A, 1), 1).float()
    D["comb"] = (rk(D["v_last"]) + rk(D["v_ext"]) + rk(D["i_ext"])).masked_fill(eye, inf)
    pos = {(r, t): i for i, (r, t) in enumerate(zip(rec[m], start[m]))}
    true = np.array([pos.get((r, t + 50), -1) for r, t in zip(rec[m], start[m])])
    ok = true >= 0
    same_flag.append((sens[m][np.maximum(true, 0)] == sens[m])[ok])
    for k in D:
        pred = torch.argmin(D[k], 1).cpu().numpy()
        hits[k].append((pred == true)[ok])
same_flag = np.concatenate(same_flag)
print(f"windows with a successor {len(same_flag)}, of which successor on same limb {same_flag.mean():.3f}")
for k, h in hits.items():
    h = np.concatenate(h)
    print(f"  {k:7s} P@1 all {h.mean():.3f}  successor same limb {h[same_flag].mean():.3f}  different limb {h[~same_flag].mean():.3f}")
