"""Cut the continuous train streams into the same "1 s windows" as test.

test is (12234, 50, 3) 1 s windows + sensor_location, so train is also
expanded with "window x sensor placement" as 1 sample.
Video features are 30 fps with window = 30 frames and do not depend on sensor placement, so they are aggregated once per window.

Output (experiments/prep/):
  win_meta.parquet   : rec, sbj_id, start(inertial index), label, purity
  inertial.npy       : (n_win, 4, 50, 3) float32   4 = order of SENSORS
  video_mean_std.npy : (n_win, 1536) float16       [mean(768), std(768)]
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from config import (AXES, FPS, LABEL_TO_ID, SENSORS, SR, TRAIN_INERTIAL,
                    TRAIN_VIDEOMAE, WIN_INERTIAL, WIN_VIDEO, WORK)


def recording_key(path: Path) -> tuple[str, int]:
    stem = path.stem                      # sbj_14_2 / sbj_3
    sbj = int(stem.split("_")[1])
    return stem, sbj


def build(stride: int, do_video: bool) -> None:
    assert stride % 5 == 0, "inertial index must be a multiple of 5 for the video index to be an integer"
    out = WORK / "prep"
    out.mkdir(parents=True, exist_ok=True)

    metas, inert_chunks, video_chunks = [], [], []
    cols = [f"{s}_acc_{a}" for s in SENSORS for a in AXES]

    for csv in sorted(TRAIN_INERTIAL.glob("sbj_*.csv")):
        rec, sbj = recording_key(csv)
        df = pl.read_csv(csv, columns=cols + ["label"],
                         schema_overrides={c: pl.Float32 for c in cols} | {"label": pl.String})
        n = df.height
        # The null class in the CSV is the string "null". Unknown labels and missing values are also mapped to 0 (null).
        lab_str = df["label"].fill_null("null").to_list()
        y = np.array([LABEL_TO_ID.get(v, 0) for v in lab_str], np.int8)
        x = df.select(cols).to_numpy().astype(np.float32).reshape(n, len(SENSORS), len(AXES))
        del df, lab_str

        starts = np.arange(0, n - WIN_INERTIAL + 1, stride, dtype=np.int64)

        vid = None
        if do_video:
            vid = np.load(TRAIN_VIDEOMAE / f"{rec}.npy", mmap_mode="r")
            vstart = starts * FPS // SR
            starts = starts[vstart + WIN_VIDEO <= vid.shape[0]]

        # Label: mode within the window and purity
        idx = starts[:, None] + np.arange(WIN_INERTIAL)[None, :]
        wl = y[idx]                                            # (w, 50)
        counts = np.zeros((len(starts), len(LABEL_TO_ID)), np.int16)
        for c in range(len(LABEL_TO_ID)):
            counts[:, c] = (wl == c).sum(1)
        lab = counts.argmax(1).astype(np.int8)
        purity = counts.max(1) / WIN_INERTIAL

        metas.append(pd.DataFrame({"rec": rec, "sbj_id": sbj, "start": starts,
                                   "label": lab, "purity": purity.astype(np.float32)}))
        inert_chunks.append(x[idx].transpose(0, 2, 1, 3).copy())   # (w, 4, 50, 3)
        del x, idx, wl, counts

        if do_video:
            vs = starts * FPS // SR
            agg = np.empty((len(vs), 1536), np.float16)
            for i, s in enumerate(vs):
                f = np.asarray(vid[s:s + WIN_VIDEO], np.float32)
                agg[i, :768] = f.mean(0)
                agg[i, 768:] = f.std(0)
            video_chunks.append(agg)
            del vid

        print(f"{rec}: n={n} win={len(starts)}", flush=True)

    meta = pd.concat(metas, ignore_index=True)
    meta.to_parquet(out / "win_meta.parquet")
    np.save(out / "inertial.npy", np.concatenate(inert_chunks))
    if do_video:
        np.save(out / "video_mean_std.npy", np.concatenate(video_chunks))
    print(meta.shape, meta.label.value_counts().sort_index().to_dict())
    print("fraction with purity==1:", (meta.purity == 1).mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    build(a.stride, a.video)
