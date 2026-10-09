"""Realign the ground truth of egoexo_motion_valid.py to "change between frames".

The instantaneous angular velocity of the IMU can be dominated by high-frequency vibration that does not show up in the video. Video displacement measures rotation and translation
between two consecutive frames (10fps, 0.1 s), so from the trajectory poses we compute the relative rotation between frames (rotation vector in device coordinates, degrees) and
the translation distance (m), and use per window the mean absolute value per axis, mean rotation angle, and mean translation distance as ground truth. Estimates use the cache.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from egoexo_motion_valid import CACHE, ROOT

res = dict(np.load(CACHE, allow_pickle=True))
takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
by = {}
for k, A in res.items():
    if k.endswith("__p"):
        continue
    t = takes[k]; d = ROOT / t["root_dir"]
    vid = sorted((d / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4"))[0]
    aria = vid.name.split("_")[0]
    cap = [c for c in (ROOT / "captures").iterdir() if t["take_name"].startswith(c.name)]
    col = f"{aria}_214-1_capture_timestamp_ns"
    ts = pd.read_csv(cap[0] / "timesync.csv", usecols=[col])[col].to_numpy()
    tr = pd.read_csv(d / "trajectory/closed_loop_trajectory.csv")
    t_us = tr["tracking_timestamp_us"].to_numpy()
    n = len(A)
    fidx = np.array([30 * w + 3 * j for w in range(n) for j in range(10)])
    tf = ts[t["timesync_start_idx"] + fidx] / 1000.0
    jj = np.clip(np.searchsorted(t_us, np.nan_to_num(tf)), 0, len(t_us) - 1)
    q = tr[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()[jj].reshape(n, 10, 4)
    p = tr[["tx_world_device", "ty_world_device", "tz_world_device"]].to_numpy()[jj].reshape(n, 10, 3)
    Rw = Rotation.from_quat(q.reshape(-1, 4))
    Ra, Rb = Rotation.from_quat(q[:, :-1].reshape(-1, 4)), Rotation.from_quat(q[:, 1:].reshape(-1, 4))
    rv = np.degrees((Ra.inv() * Rb).as_rotvec()).reshape(n, 9, 3)             # relative rotation in device coordinates
    dist = np.linalg.norm(np.diff(p, axis=1), axis=2)                          # (n, 9)
    gt = np.column_stack([np.abs(rv).mean(1), np.linalg.norm(rv, axis=2).mean(1), dist.mean(1)])
    by.setdefault(int(res[k + "__p"]), []).append(np.column_stack([A[:, :14], gt, A[:, -1:]]))

tn = ["rotation |x|", "rotation |y|", "rotation |z|", "rotation angle", "translation"]
print("=== Shake measured from video -> measured inter-frame rotation/translation (R^2 with CV by take, mean over participants) ===")
for nm, sl in {"phase correlation": slice(0, 6), "flow similarity": slice(6, 14), "both": slice(0, 14)}.items():
    r2s = []
    for p, vs in by.items():
        A = np.concatenate(vs); grp = np.concatenate([[i] * len(v) for i, v in enumerate(vs)])
        ok = (A[:, -1] > 0) & np.isfinite(A).all(1); A, grp = A[ok], grp[ok]
        X = np.log1p(np.abs(A[:, sl])); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Y = np.log1p(A[:, 14:19])
        pr = np.zeros_like(Y)
        for tr_, te in GroupKFold(min(5, len(np.unique(grp)))).split(X, Y, grp):
            pr[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr_], Y[tr_]).predict(X[te])
        r2s.append(1 - ((Y - pr) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0))
    print(f"  {nm:8s} " + "  ".join(f"{t} {x:+.3f}" for t, x in zip(tn, np.mean(r2s, 0))))
