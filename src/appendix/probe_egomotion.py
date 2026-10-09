"""Are the top principal components "the camera's own motion" = egomotion?

Why this candidate differs from the others:
  Of the 14 candidates eliminated so far, this is the first to **survive** the exclusion of the nuisance family (removing top PCs
  monotonically lowers cross-subject purity, 0.408 -> 0.198).
  Lighting, location, clothing, and session are all subject-specific, so they fell to that measurement.
  But egomotion gives the same optical flow on grass or cobblestones, so it **transfers across subjects**.
  This is consistent with §5.3, "the top directions carry an activity signal that transfers across subjects".

  Further: in head-mounted video the largest variance is the camera's own motion,
  which also correlates strongly with activity (running / standing still). It explains both the variance ratio of the top PCs and eta^2.

What is measured (same form as probe_person, but the label is egomotion rather than visibility):
  1. Grab frames at 10 fps and find the global translation by **phase correlation** between consecutive frames
     (phaseCorrelate is robust to brightness changes and does not depend on the amount of texture)
  2. Average the displacement per second and map it to the tiled window of that second
  3. Linear regression from VideoMAE features (recording hold-out) -> R^2
  4. Share of the obtained direction lying in the subspace of the top 30 principal components

Decision:
  R^2 low                         -> the features do not encode egomotion. Candidate is dead
  R^2 high, share ~= 0.039        -> encoded, but distinct from the removed directions
  R^2 high, share >> 0.039        -> **the first surviving candidate for the mechanism that failed 14 times**
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK

PREP = WORK / "prep"
VURL = "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/camera/{}.mp4"


def decode_gray(rec, start, dur, fps, width=320):
    h = int(round(width * 9 / 16 / 2)) * 2
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start),
           "-i", VURL.format(rec), "-t", str(dur),
           "-vf", f"fps={fps},scale={width}:{h},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    n = len(buf) // (width * h)
    return np.frombuffer(buf, np.uint8, n * width * h).reshape(n, h, width)


def egomotion(frames, fps):
    """Find the global translation by phase correlation between consecutive frames and average per second."""
    import cv2
    win = cv2.createHanningWindow((frames.shape[2], frames.shape[1]), cv2.CV_32F)
    mag = []
    prev = frames[0].astype(np.float32)
    for f in frames[1:]:
        cur = f.astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(prev, cur, win)
        mag.append(float(np.hypot(dx, dy)))
        prev = cur
    mag = np.array(mag, np.float32)
    nsec = len(mag) // fps
    return mag[:nsec * fps].reshape(nsec, fps).mean(1)


def main(a):
    from sklearn.linear_model import Lasso, Ridge
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")

    X, yv, sb = [], [], []
    for rec in a.recs:
        g = meta[meta["rec"] == rec]
        tmax = g["start"].max() / 50.0
        st = max((tmax - a.dur) / 2, 0)
        fr = decode_gray(rec, st, a.dur + 1, a.fps)
        em = egomotion(fr, a.fps)
        tiles = g[g["start"] % 50 == 0]
        sec2row = {int(s_ // 50): r for s_, r in
                   zip(tiles["start"].to_numpy(), tiles.index.to_numpy())}
        secs = st + np.arange(len(em))
        rows = np.array([sec2row.get(int(s_), -1) for s_ in secs])
        ok = rows >= 0
        X.append(np.asarray(vid[rows[ok]], np.float32).mean(1))
        yv.append(em[ok]); sb.append(np.full(int(ok.sum()), rec))
        print(f"  {rec}: {int(ok.sum())} s  egomotion mean {em[ok].mean():.2f}px "
              f"(std {em[ok].std():.2f}, range {em[ok].min():.1f}-{em[ok].max():.1f})",
              flush=True)

    X = np.concatenate(X); yv = np.concatenate(yv); sb = np.concatenate(sb)
    # ⚠ Standardize the target per recording.
    # The scale of egomotion differs a lot between recordings (mean 6.6-10.4, max 97px).
    # Taking R^2 on raw values gives negative R^2 from offset/scale mismatch alone,
    # **even if the within-recording ordering is perfectly right**. What we want is not transfer of absolute values but
    # "whether the direction exists", so look only at within-recording variation.
    for r in np.unique(sb):
        m = sb == r
        X[m] -= X[m].mean(0)
        yv[m] = (yv[m] - yv[m].mean()) / (yv[m].std() + 1e-8)

    print(f"\n=== Linear prediction of egomotion (recording hold-out) ===")
    for nm, mk in [("Ridge", lambda: Ridge(alpha=10.0)),
                   ("Lasso", lambda: Lasso(alpha=a.lasso_alpha, max_iter=5000))]:
        r2s, cors, ws = [], [], []
        for r in np.unique(sb):
            te = sb == r
            mdl = mk().fit(X[~te], yv[~te])
            p = mdl.predict(X[te])
            ss = ((yv[te] - p) ** 2).sum()
            r2s.append(1 - ss / max(((yv[te] - yv[te].mean()) ** 2).sum(), 1e-9))
            cors.append(float(np.corrcoef(p, yv[te])[0, 1]))
            ws.append(mdl.coef_)
        w = np.mean(ws, 0); w /= np.linalg.norm(w) + 1e-12
        nz = int((np.abs(np.mean(ws, 0)) > 1e-8).sum())
        frac = []
        for r in np.unique(sb):
            m = sb == r
            _, _, Vt = np.linalg.svd(X[m] - X[m].mean(0), full_matrices=False)
            frac.append(float((Vt[:a.m] @ w) @ (Vt[:a.m] @ w)))
        print(f"  {nm:6s} R^2 {np.mean(r2s):+.4f}  within-recording corr {np.mean(cors):+.4f}  "
              f"nonzero {nz:4d}/768  share in top {a.m} PCs {np.mean(frac):.4f}  "
              f"(random would be {a.m/768:.4f})")
    print("\n  Judge by **within-recording correlation**. R^2 is after standardizing the target per recording.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recs", nargs="*", default=["sbj_0", "sbj_3", "sbj_10", "sbj_18"])
    p.add_argument("--dur", type=float, default=400.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--m", type=int, default=30)
    p.add_argument("--lasso-alpha", type=float, default=0.02)
    main(p.parse_args())
