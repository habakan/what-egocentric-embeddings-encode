"""Build aggregated VideoMAE features per window.

test only has **15 consecutive frames (= 0.5 s)** at 30fps, so the train side is also matched to 15 frames.
Where the 0.5 s of video sits relative to the 1 s inertial window is unknown, so the offset is selectable:
  head   : first 0.5 s of the window  [vs, vs+15)
  center : middle 0.5 s of the window  [vs+7, vs+22)
  tail   : last 0.5 s of the window  [vs+15, vs+30)
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FPS, SR, TEST_DIR, TRAIN_VIDEOMAE, WORK

WIN_V = 15
# From the official data description (Kaggle Data tab):
#   "for each frame i, the model takes 16 frames as input (frame i, 8 past, 7 future)"
#   "test windows contain 15 feature vectors instead of 30. This is to avoid frames that leak
#     context from outside the 1-second window"
# When a 1-second window covers video frames [f0, f0+30), row i does not see outside the window iff
#   i-8 >= f0 and i+7 <= f0+29  ->  i in [f0+8, f0+22]  = exactly 15 rows.
# So the 15 test rows start at **frame 8 from the window start**.
# Unless the training side matches this, train/test are offset by 0.267 s.
OFFSETS = {"head": 0, "center": 7, "tail": 15, "sync": 8}


def agg(frames: np.ndarray) -> np.ndarray:
    """frames: (w, 15, 768) float32 -> (w, 1536)"""
    return np.concatenate([frames.mean(1), frames.std(1)], axis=1)


def build_train(offset_name: str) -> None:
    off = OFFSETS[offset_name]
    prep = WORK / "prep"
    meta = pd.read_parquet(prep / "win_meta.parquet")
    out = np.empty((len(meta), 1536), np.float16)
    for rec, g in meta.groupby("rec", sort=False):
        vid = np.load(TRAIN_VIDEOMAE / f"{rec}.npy", mmap_mode="r")
        vs = (g["start"].to_numpy() * FPS // SR) + off
        vs = np.clip(vs, 0, vid.shape[0] - WIN_V)
        rows = g.index.to_numpy()
        for i in range(0, len(vs), 2048):
            sl = slice(i, i + 2048)
            f = np.stack([np.asarray(vid[s:s + WIN_V], np.float32) for s in vs[sl]])
            out[rows[sl]] = agg(f).astype(np.float16)
        print(rec, len(vs), flush=True)
    np.save(prep / f"video_{offset_name}.npy", out)
    print("saved", offset_name, out.shape)


def build_test() -> None:
    a = np.load(TEST_DIR / "test_videomae_data_f16.npy", mmap_mode="r")  # (n, 768, 15)
    n = a.shape[0]
    out = np.empty((n, 1536), np.float16)
    for i in range(0, n, 2048):
        f = np.asarray(a[i:i + 2048], np.float32).transpose(0, 2, 1)     # (w, 15, 768)
        out[i:i + 2048] = agg(f).astype(np.float16)
    np.save(WORK / "prep" / "video_test.npy", out)
    print("saved test", out.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--offset", default="head", choices=list(OFFSETS))
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    if a.test:
        build_test()
    else:
        build_train(a.offset)
