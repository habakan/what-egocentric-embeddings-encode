"""VoI probe: measure the ceiling of the transductive approach.

Approximate "what macro-F1 would be with a perfect neighbour graph" with an oracle that groups by
the true activity segments. If the ceiling is high, investing in graph improvements is worthwhile;
if low, the only option is to strengthen the base model itself.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from metric import macro_f1
from postprocess import tune_tau

PREP = WORK / "prep"


def main():
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    oof = np.load(WORK / "exp001_lgb_inertial" / "oof.npy")
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    P = oof[tile + sens * n]; keep = P.sum(1) > 0
    tile, P = tile[keep], P[keep]
    P = P / P.sum(1, keepdims=True)
    m = meta.iloc[tile].reset_index(drop=True)
    y = m["label"].to_numpy()

    # True activity segment ID (runs of consecutive labels within a recording)
    seg = (m["label"].ne(m["label"].shift()) | m["rec"].ne(m["rec"].shift())).cumsum().to_numpy()

    print(f"base                          : {tune_tau(P, y)[0]:.4f}")
    for name, g in [("oracle: mean over true segment", seg),
                    ("oracle: subject x true label", pd.factorize(
                        m["sbj_id"].astype(str) + "_" + m["label"].astype(str))[0])]:
        Q = np.zeros_like(P)
        df = pd.DataFrame(P); df["g"] = g
        mean = df.groupby("g").transform("mean").to_numpy()
        Q[:] = mean
        print(f"{name:30s}: {tune_tau(Q, y)[0]:.4f}")
    # Distribution of segment lengths
    _, cnt = np.unique(seg, return_counts=True)
    print(f"segments={len(cnt)} segment length median={np.median(cnt):.0f} s mean={cnt.mean():.1f}")


if __name__ == "__main__":
    main()
