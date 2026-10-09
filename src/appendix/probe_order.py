"""VoI probe: can the temporal order of test windows be recovered from video features alone?

On train subjects, try "cut into 1-second non-overlapping windows -> shuffle -> recover" and compare with the true order.
The goal is to verify the premise of a 2-stage pipeline (order recovery -> sequence labelling) without using the LB.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FPS, LABEL_TO_ID, SR, TRAIN_INERTIAL, TRAIN_VIDEOMAE

WIN_V = 15


def load_rec(rec: str):
    import polars as pl
    d = pl.read_csv(TRAIN_INERTIAL / f"{rec}.csv", columns=["label"],
                    schema_overrides={"label": pl.String})
    y = np.array([LABEL_TO_ID.get(v, 0) for v in d["label"].fill_null("null").to_list()], np.int8)
    vid = np.load(TRAIN_VIDEOMAE / f"{rec}.npy", mmap_mode="r")
    n_sec = min(len(y) // SR, vid.shape[0] // FPS)
    # same as test: 1-second non-overlapping windows, video is the first 15 frames (0.5 s) of each second
    V = np.stack([np.asarray(vid[t * FPS: t * FPS + WIN_V], np.float32) for t in range(n_sec)])
    lab = np.array([np.bincount(y[t * SR:(t + 1) * SR], minlength=19).argmax() for t in range(n_sec)])
    return V, lab


def evaluate(rec: str, k: int = 10):
    V, lab = load_rec(rec)
    n = len(V)
    F = V.mean(1)                                  # (n, 768) mean feature per window
    F = F - F.mean(0)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    S = F @ F.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]

    t = np.arange(n)
    succ_top1 = np.mean([(abs(nn[i, 0] - i) == 1) for i in range(n)])
    succ_topk = np.mean([np.any(np.abs(nn[i] - i) == 1) for i in range(n)])
    same_lab = np.mean(lab[nn] == lab[:, None])
    # fraction judged similar despite being far apart in time (>30 s) = false-link rate
    far = np.mean(np.abs(nn - t[:, None]) > 30)
    print(f"{rec}: n_sec={n}")
    print(f"  adjacent (t±1) is nearest top1 : {succ_top1:.3f}")
    print(f"  adjacent (t±1) within top{k}   : {succ_topk:.3f}")
    print(f"  top{k} neighbour label match : {same_lab:.3f}   (upper-bound-like smoothing effect for single windows)")
    print(f"  top{k} more than 30 s apart : {far:.3f}")
    print(f"  median label segment length    : {np.median([len(list(g)) for _, g in __import__('itertools').groupby(lab)]):.0f} s")
    return same_lab


if __name__ == "__main__":
    for rec in sys.argv[1:] or ["sbj_3", "sbj_12", "sbj_20"]:
        evaluate(rec)
