"""Materialize raw per-window VideoMAE frames (15, 768). For the NN.

Output: experiments/prep/video_raw_<offset>.npy  (n_win, 15, 768) float16 (~3.2GB)
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
from prepare_video import OFFSETS, WIN_V

PREP = WORK / "prep"


def build_train(offset_name: str) -> None:
    off = OFFSETS[offset_name]
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    out = np.lib.format.open_memmap(PREP / f"video_raw_{offset_name}.npy", mode="w+",
                                    dtype=np.float16, shape=(len(meta), WIN_V, 768))
    for rec, g in meta.groupby("rec", sort=False):
        vid = np.load(TRAIN_VIDEOMAE / f"{rec}.npy", mmap_mode="r")
        vs = np.clip((g["start"].to_numpy() * FPS // SR) + off, 0, vid.shape[0] - WIN_V)
        rows = g.index.to_numpy()
        for i, (r, s) in enumerate(zip(rows, vs)):
            out[r] = vid[s:s + WIN_V]
        print(rec, len(vs), flush=True)
    out.flush()
    print("saved", offset_name, out.shape)


def build_test() -> None:
    a = np.load(TEST_DIR / "test_videomae_data_f16.npy", mmap_mode="r")   # (n, 768, 15)
    out = np.lib.format.open_memmap(PREP / "video_raw_test.npy", mode="w+",
                                    dtype=np.float16, shape=(a.shape[0], WIN_V, 768))
    for i in range(0, a.shape[0], 4096):
        out[i:i + 4096] = np.asarray(a[i:i + 4096]).transpose(0, 2, 1)
    out.flush(); print("saved test", out.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--offset", default="head", choices=list(OFFSETS))
    p.add_argument("--test", action="store_true")
    a = p.parse_args()
    build_test() if a.test else build_train(a.offset)
