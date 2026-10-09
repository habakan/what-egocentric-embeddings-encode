"""Does the "is a person visible" direction lie among the removed top principal components?

Motivation:
  §5.3 showed that the head camera never captures the performer, and that what encodes activity on the video side is
  head orientation and ground flow rather than body motion.
  But hands and feet are sometimes visible. So we measure "body-part visibility" with an **external label source**
  and check whether the direction explaining it lies in the subspace of the top 30 principal components whose removal helps.

Why this is new:
  All 13 candidate explanations so far were **internal statistics** (neighbour purity, hubness, redundancy, anisotropy,
  temporal drift, cross-subject transfer). This is the first time an external label source is brought in.
  The possibility that the failures were "because we looked from the inside" has not yet been ruled out.

Procedure:
  1. Take frames at 1fps from the training recordings and get the number of visible keypoints with YOLO11x-pose
  2. Map each frame to a window index and fetch that window's VideoMAE features
  3. Linearly regress visibility on the within-subject centered features (Ridge / Lasso, subject hold-out)
  4. Measure the fraction of the obtained direction lying in the subspace spanned by the top 30 principal components

Verdict:
  low R^2                         -> features do not encode person visibility. Idea is dead
  high R^2, direction outside     -> not reproduced by removal (still possible as a gate)
  high R^2, direction inside      -> first interpretable clue to the mechanism that failed 13 times

  The expected fraction of a random unit direction in the top 30 dimensions is 30/768 = 0.039. This is the reference.
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
YOLO_W = _os.environ.get("YOLO_POSE_WEIGHTS", "yolo11x-pose.pt")


def decode_1fps(rec, start, dur, width=640):
    """Grab a contiguous span at 1fps. HTTP seeks are expensive, so grab it in one go."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start),
           "-i", VURL.format(rec), "-t", str(dur),
           "-vf", f"fps=1,scale={width}:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    h = int(round(width * 9 / 16 / 2)) * 2
    n = len(buf) // (width * h * 3)
    return np.frombuffer(buf, np.uint8, n * width * h * 3).reshape(n, h, width, 3)


def visibility(model, frames, batch=32, conf=0.5):
    """Number of visible keypoints (thresholded by person-detection confidence). With a head camera, mostly one's own hands and feet are visible."""
    out = []
    for i in range(0, len(frames), batch):
        res = model.predict(list(frames[i:i + batch]), verbose=False, device=0)
        for r in res:
            if r.keypoints is None or r.keypoints.conf is None or len(r.keypoints) == 0:
                out.append(0.0)
            else:
                out.append(float((r.keypoints.conf > conf).sum()))
    return np.array(out, np.float32)


def main(a):
    from sklearn.linear_model import Lasso, Ridge
    from ultralytics import YOLO
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    model = YOLO(YOLO_W)

    X, yv, sb = [], [], []
    for rec in a.recs:
        g = meta[meta["rec"] == rec]
        tmax = g["start"].max() / 50.0
        st = max((tmax - a.dur) / 2, 0)
        fr = decode_1fps(rec, st, a.dur)
        v = visibility(model, fr)
        # frame at t seconds -> tile window for that second.
        # windows overlap, so there are several per second. Restricting to tiles with start%50==0 makes it unique.
        tiles = g[g["start"] % 50 == 0]
        sec2row = {int(s_ // 50): r for s_, r in
                   zip(tiles["start"].to_numpy(), tiles.index.to_numpy())}
        secs = st + np.arange(len(v))
        rows = np.array([sec2row.get(int(s_), -1) for s_ in secs])
        ok = rows >= 0
        rows = rows[ok]
        X.append(np.asarray(vid[rows], np.float32).mean(1))
        yv.append(v[ok]); sb.append(np.full(int(ok.sum()), rec))
        print(f"  {rec}: {ok.sum()} frames  visible KP mean {v[ok].mean():.2f} "
              f"(fraction of 0 {(v[ok]==0).mean():.2f})", flush=True)

    X = np.concatenate(X); yv = np.concatenate(yv); sb = np.concatenate(sb)
    # center within subject
    for r in np.unique(sb):
        m = sb == r
        X[m] -= X[m].mean(0)

    print(f"\n=== linear prediction of visibility (recording hold-out) ===")
    for nm, mk in [("Ridge", lambda: Ridge(alpha=10.0)),
                   ("Lasso", lambda: Lasso(alpha=a.lasso_alpha, max_iter=5000))]:
        r2s, ws = [], []
        for r in np.unique(sb):
            te = sb == r
            mdl = mk().fit(X[~te], yv[~te])
            p = mdl.predict(X[te])
            ss = ((yv[te] - p) ** 2).sum()
            r2s.append(1 - ss / max(((yv[te] - yv[~te].mean()) ** 2).sum(), 1e-9))
            ws.append(mdl.coef_)
        w = np.mean(ws, 0); w /= np.linalg.norm(w) + 1e-12
        nz = int((np.abs(np.mean(ws, 0)) > 1e-8).sum())
        # fraction lying in the top 30 PC subspace (measured per recording and averaged)
        frac = []
        for r in np.unique(sb):
            m = sb == r
            _, _, Vt = np.linalg.svd(X[m] - X[m].mean(0), full_matrices=False)
            frac.append(float((Vt[:a.m] @ w) @ (Vt[:a.m] @ w)))
        print(f"  {nm:6s} R^2 {np.mean(r2s):+.4f}  nonzero coefs {nz:4d}/768  "
              f"fraction in top {a.m} PCs {np.mean(frac):.4f}  (random: {a.m/768:.4f})")

    print("\n  Verdict: if R^2 is low, the idea is dead. If high with a fraction near 0.039, removal does not reproduce it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recs", nargs="*", default=["sbj_0", "sbj_3", "sbj_10", "sbj_18"])
    p.add_argument("--dur", type=float, default=400.0)
    p.add_argument("--m", type=int, default=30)
    p.add_argument("--lasso-alpha", type=float, default=0.01)
    main(p.parse_args())
