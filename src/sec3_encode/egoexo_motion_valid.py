"""Validate camera shake measured from video against head motion measured by the Aria in Ego-Exo4D.

Estimates (Aria RGB 448 read at 10 fps, 160x160 grayscale, per consecutive pair):
  phase correlation : whole-frame translation (dx, dy) and frame difference (same as wear_motion.py)
  flow similarity   : goodFeaturesToTrack + calcOpticalFlowPyrLK -> estimateAffinePartial2D (RANSAC) giving
               translation (tx, ty), rotation (rot), scaling (log scale)
  Summarize per window (1 s, 9 pairs) as mean absolute value and standard deviation.
Ground truth (closed_loop_trajectory, interpolated at the window's 10 time points): mean absolute angular velocity (3 device axes), mean translational speed magnitude,
  std of angular speed magnitude.
Comparison: per participant, R^2 of ridge regression with cross-validation by take (per ground-truth quantity).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from egoexo_viewpoint import ROOT

S = 160
CACHE = ROOT / "motion_valid.npz"


def phase_shift(a, b, win):
    Fa = np.fft.rfft2(a * win); Fb = np.fft.rfft2(b * win)
    R = Fa * np.conj(Fb); R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    y, x = np.unravel_index(np.argmax(r), r.shape)
    if y > a.shape[0] // 2: y -= a.shape[0]
    if x > a.shape[1] // 2: x -= a.shape[1]
    return x, y


def flow_sim(a, b, mask):
    p0 = cv2.goodFeaturesToTrack(a, 150, 0.01, 5, mask=mask)
    if p0 is None or len(p0) < 8:
        return np.array([0, 0, 0, 0], np.float32), False
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(15, 15), maxLevel=2)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return np.array([0, 0, 0, 0], np.float32), False
    M, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return np.array([0, 0, 0, 0], np.float32), False
    sc = np.hypot(M[0, 0], M[1, 0]); rot = np.arctan2(M[1, 0], M[0, 0])
    return np.array([M[0, 2], M[1, 2], rot, np.log(sc + 1e-9)], np.float32), True


def per_window(fr):
    """fr: (10k, S, S) uint8 -> (k, 6 + 8)"""
    win = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)
    mask = np.zeros((S, S), np.uint8); cv2.circle(mask, (S // 2, S // 2), int(S * 0.45), 255, -1)   # exclude the fisheye rim
    k = len(fr) // 10
    out = np.zeros((k, 14), np.float32)
    ff = fr.astype(np.float32)
    for t in range(k):
        f = fr[10 * t:10 * t + 10]; g = ff[10 * t:10 * t + 10]
        sh = np.array([phase_shift(g[i], g[i + 1], win) for i in range(9)], np.float32)
        d = np.abs(np.diff(g, axis=0)).mean((1, 2))
        fs = np.array([flow_sim(f[i], f[i + 1], mask)[0] for i in range(9)])
        out[t] = np.concatenate([[np.abs(sh[:, 0]).mean(), np.abs(sh[:, 1]).mean(), sh[:, 0].std(), sh[:, 1].std(),
                                  np.hypot(sh[:, 0], sh[:, 1]).max(), d.mean()],
                                 np.abs(fs).mean(0), fs.std(0)])
    return out


def compute():
    if CACHE.exists():
        return dict(np.load(CACHE, allow_pickle=True))
    takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
    res = {}
    for i, fb in enumerate(sorted((ROOT / "features").glob("*.npz"))):
        z = np.load(fb, allow_pickle=True); nwin = len(z["pose"])
        t = takes[fb.stem]; d = ROOT / t["root_dir"]
        vid = sorted((d / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4"))[0]
        aria = vid.name.split("_")[0]
        buf = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(vid), "-vf",
                              f"select='not(mod(n\\,3))',scale={S}:{S},format=gray", "-vsync", "0", "-f", "rawvideo", "-"],
                             stdout=subprocess.PIPE, check=True).stdout
        fr = np.frombuffer(buf, np.uint8).reshape(-1, S, S)[:10 * nwin]
        est = per_window(fr)
        # ground truth: interpolate the trajectory at the times of 30 fps frames 30w + 3j (j=0..9) of window w
        cap = [c for c in (ROOT / "captures").iterdir() if t["take_name"].startswith(c.name)]
        col = f"{aria}_214-1_capture_timestamp_ns"
        ts = pd.read_csv(cap[0] / "timesync.csv", usecols=[col])[col].to_numpy()
        tr = pd.read_csv(d / "trajectory/closed_loop_trajectory.csv")
        t_us = tr["tracking_timestamp_us"].to_numpy()
        k = len(est)
        fidx = np.array([30 * w + 3 * j for w in range(k) for j in range(10)])
        tf = ts[t["timesync_start_idx"] + fidx] / 1000.0
        ok = np.isfinite(tf).reshape(k, 10).all(1)
        jj = np.clip(np.searchsorted(t_us, np.nan_to_num(tf)), 0, len(t_us) - 1)
        av = tr[["angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"]].to_numpy()[jj].reshape(k, 10, 3)
        lv = tr[["device_linear_velocity_x_device", "device_linear_velocity_y_device", "device_linear_velocity_z_device"]].to_numpy()[jj].reshape(k, 10, 3)
        gt = np.column_stack([np.abs(av).mean(1), np.linalg.norm(lv, axis=2).mean(1), np.linalg.norm(av, axis=2).std(1)])
        res[fb.stem] = np.column_stack([est, gt, ok[:, None]]).astype(np.float32)
        res[fb.stem + "__p"] = np.array(int(z["participant"]))
        if i % 10 == 9:
            print(f"  {i + 1} takes", flush=True)
    np.savez(CACHE, **res)
    return res


def main():
    res = compute()
    by = {}
    for k, v in res.items():
        if k.endswith("__p"):
            continue
        by.setdefault(int(res[k + "__p"]), []).append(v)
    tnames = ["ang. vel. |x|", "ang. vel. |y|", "ang. vel. |z|", "transl. speed", "ang. vel. fluctuation"]
    feats = {"phase corr.": slice(0, 6), "flow similarity": slice(6, 14), "both": slice(0, 14)}
    print("=== shake measured from video -> Aria measurements (R^2 of by-take CV, mean over participants) ===")
    for nm, sl in feats.items():
        r2s = []
        for p, vs in by.items():
            A = np.concatenate(vs); grp = np.concatenate([[i] * len(v) for i, v in enumerate(vs)])
            ok = A[:, -1] > 0; A, grp = A[ok], grp[ok]
            X = A[:, sl]; X = np.log1p(np.abs(X)); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            Y = A[:, 14:19]
            pr = np.zeros_like(Y)
            for tr, te in GroupKFold(min(5, len(np.unique(grp)))).split(X, Y, grp):
                pr[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], Y[tr]).predict(X[te])
            r2s.append(1 - ((Y - pr) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0))
        r = np.mean(r2s, 0)
        print(f"  {nm:8s} " + "  ".join(f"{t} {x:+.3f}" for t, x in zip(tnames, r)))


if __name__ == "__main__":
    main()
