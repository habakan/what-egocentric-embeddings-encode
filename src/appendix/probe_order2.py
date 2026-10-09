"""Another attempt at order recovery: directional matching.

Window i is 15 consecutive frames at 30fps (= 0.5 s); window i+1 starts 30 frames later.
Matching on window means throws away the direction of time, so
match "the last frame of window i" with "the first frame of window j" (15 frames apart if j is the true successor),
and also try the distance to a point extrapolated with the within-window feature change (velocity).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import FPS, SR, TRAIN_VIDEOMAE

WIN_V = 15


def load(rec, n_max=1500):
    vid = np.load(TRAIN_VIDEOMAE / f"{rec}.npy", mmap_mode="r")
    n = min(vid.shape[0] // FPS, n_max)
    return np.stack([np.asarray(vid[t * FPS: t * FPS + WIN_V], np.float32) for t in range(n)])


def norm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def topk_acc(S, offset, k):
    """The larger S[i, j], the more likely j is the successor of i. The true successor is j = i + 1"""
    n = len(S)
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    hit1 = np.mean([order[i, 0] == i + offset for i in range(n - 1)])
    hitk = np.mean([(i + offset) in order[i, :k] for i in range(n - 1)])
    return hit1, hitk


def main(rec):
    V = load(rec)
    n = len(V)
    mean = norm(V.mean(1))
    first, last = norm(V[:, 0]), norm(V[:, -1])
    vel = V[:, -1] - V[:, 0]                     # change within the window (over 15 frames)
    extrap = norm(V[:, -1] + vel)                # linear extrapolation 15 more frames ahead = near the start of the next window

    print(f"{rec}: n={n}  (true successor = i+1)")
    for name, S in [("window mean vs window mean", mean @ mean.T),
                    ("last -> first", last @ first.T),
                    ("extrapolated -> first", extrap @ first.T),
                    ("extrapolated -> window mean", extrap @ mean.T)]:
        h1, hk = topk_acc(S, 1, 5)
        print(f"  {name:14s}: top1={h1:.3f}  top5={hk:.3f}")


if __name__ == "__main__":
    for r in sys.argv[1:] or ["sbj_3", "sbj_20"]:
        main(r)
