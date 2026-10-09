"""Do the top components of the egocentric embedding encode "where in the world one is looking from, and in which direction" (Ego-Exo4D)?

Tilt (pitch, roll) alone explained almost none of the variance of WEAR's top components (probe_camtilt.py).
From the 6-DoF head trajectory, compute per window
  tilt     : gravity direction in device coordinates (3)
  height   : z in world coordinates
  rotation : device rotation matrix in world coordinates (9; includes heading = rotation about gravity)
  position : x, y in world coordinates (and quadratic terms)
and compare how well bands of principal components are predicted as these are added step by step.
World coordinates may be reset per take, so comparisons are made within takes: center per take, and average the R^2 of
cross-validation over 5 time blocks within each take, weighted by window count. Principal components are computed per participant (pooling takes).
Also report the fraction of top-component variance due to differences between takes. Both egocentric and exocentric (features_exo_*).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from egoexo_viewpoint import ROOT, quat_to_R

DIRS = {"ego VideoMAEv2": "features", "ego CLIP": "features_clip", "ego DINOv2": "features_dinov2",
        "exo VideoMAEv2": "features_exo_videomae", "exo CLIP": "features_exo_clip", "exo DINOv2": "features_exo_dinov2",
        "ego VC-1": "features_vc1", "exo VC-1": "features_exo_vc1"}
CACHE = ROOT / "viewpose.npz"


def pose6(take):
    d = ROOT / take["root_dir"]
    vid = sorted((d / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4"))[0]
    aria = vid.name.split("_")[0]
    cap = [c for c in (ROOT / "captures").iterdir() if take["take_name"].startswith(c.name)]
    col = f"{aria}_214-1_capture_timestamp_ns"
    ts = pd.read_csv(cap[0] / "timesync.csv", usecols=[col])[col].to_numpy()
    tr = pd.read_csv(d / "trajectory/closed_loop_trajectory.csv")
    return ts, tr


def compute():
    if CACHE.exists():
        return dict(np.load(CACHE, allow_pickle=True))
    takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
    out = {}
    for fb in sorted((ROOT / "features").glob("*.npz")):
        z = np.load(fb, allow_pickle=True); nwin = len(z["pose"])
        t = takes[fb.stem]
        ts, tr = pose6(t)
        t_us = tr["tracking_timestamp_us"].to_numpy()
        fidx = np.array([30 * w + o + 8 for w in range(nwin) for o in (0, 7, 14)])
        t_frame = ts[t["timesync_start_idx"] + fidx] / 1000.0
        j = np.clip(np.searchsorted(t_us, np.nan_to_num(t_frame)), 0, len(t_us) - 1)
        R = quat_to_R(tr[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()[j])
        g = tr[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].to_numpy()[j]
        g_dev = np.einsum("nji,nj->ni", R, g); g_dev /= np.linalg.norm(g_dev, axis=1, keepdims=True) + 1e-9
        pos = tr[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()[j]
        feat = np.column_stack([g_dev, pos[:, 2:3], R.reshape(len(R), 9), pos[:, :2]]).reshape(nwin, 3, -1).mean(1)
        out[fb.stem] = feat.astype(np.float32)
    np.savez(CACHE, **out)
    return out


def sets(A):
    tilt, h, rot, xy = A[:, :3], A[:, 3:4], A[:, 4:13], A[:, 13:15]
    xyq = np.column_stack([xy, xy ** 2, xy[:, :1] * xy[:, 1:2]])
    return {"tilt": tilt, "+height": np.column_stack([tilt, h]), "+rotation (heading)": np.column_stack([tilt, h, rot]),
            "+position": np.column_stack([tilt, h, rot, xyq])}


def within_take_r2(X, Z):
    n = len(X); blk = np.arange(n) * 5 // n
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-9); Zc = Z - Z.mean(0)
    pr = np.zeros_like(Zc)
    for b in range(5):
        tr, te = blk != b, blk == b
        pr[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xc[tr], Zc[tr]).predict(Xc[te])
    return ((Zc - pr) ** 2).sum(0), (Zc ** 2).sum(0)


def analyse(poses):
    bands = [(1, 2), (3, 10), (11, 30), (31, 100)]
    for name, d in DIRS.items():
        by = collections.defaultdict(list)
        for f in sorted((ROOT / d).glob("*.npz")):
            if f.stem not in poses:
                continue
            z = np.load(f, allow_pickle=True); ok = z["ok"]
            A = poses[f.stem]
            n = min(len(ok), len(A))
            ok = ok[:n] & np.isfinite(A[:n]).all(1)
            by[int(z["participant"])].append((z["F"][:n].mean(1)[ok], A[:n][ok]))
        err = collections.defaultdict(lambda: np.zeros(100)); tot = np.zeros(100); between = []
        for p, v in by.items():
            F = np.concatenate([a for a, _ in v]); F = F - F.mean(0)
            _, _, Vt = np.linalg.svd(F, full_matrices=False)
            i0 = 0; bt = np.zeros(100); tt = np.zeros(100)
            for a, A in v:
                Z = F[i0:i0 + len(a)] @ Vt[:100].T; i0 += len(a)
                bt += len(Z) * Z.mean(0) ** 2; tt += (Z ** 2).sum(0)
                if len(Z) < 40:
                    continue
                for k, X in sets(A).items():
                    e, t_ = within_take_r2(X, Z)
                    err[k] += e
                tot += ((Z - Z.mean(0)) ** 2).sum(0)
            between.append(bt / tt)
        print(f"\n== {name}")
        bw = np.mean(between, 0)
        print("  fraction of top-component variance due to between-take differences: " + "  ".join(f"PC{b[0]}-{b[1]} {bw[b[0] - 1:b[1]].mean():.2f}" for b in bands))
        print("  within-take R^2 (time-block CV):")
        for k in ["tilt", "+height", "+rotation (heading)", "+position"]:
            r2 = 1 - err[k] / tot
            print(f"    {k:12s} " + "  ".join(f"PC{b[0]}-{b[1]} {r2[b[0] - 1:b[1]].mean():+.3f}" for b in bands))


if __name__ == "__main__":
    analyse(compute())
