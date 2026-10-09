"""Calibration with per-class count bands (own implementation based on calibrate_counts of the 2nd-place public solution).

Training facts: an activity class has a median of 94 windows per recording (78-126 at 5-95%), and the class is present in 431 of 432 pairs.
Method: add a per-recording, per-class bias b_c to the log scores after prior correction.
  Raise b_c for classes with fewer than lo windows and lower it for classes above hi. Iterate. null has no bias (takes the rest).
The band is set from the whole training distribution, and is also set from the distribution excluding the evaluated recording (leave-one-recording-out) for comparison.
Input is the OOF of exp017 (Q' = Q^0.7 S^0.3; stage-2 OOF is stage2_oof_s{seed}.npz). 2 seeds.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK
from metric import macro_f1
from postprocess import tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from stack_q import blend


def band_calibrate(L, groups, lo, hi, step=0.1, iters=200):
    """L: (n, C) log scores. groups: recording IDs. Push the count of each non-null class into [lo, hi] in each recording."""
    pred = L.argmax(1).copy()
    for g in np.unique(groups):
        m = groups == g; Lg = L[m]; b = np.zeros(L.shape[1])
        for _ in range(iters):
            p = (Lg + b).argmax(1); cnt = np.bincount(p, minlength=L.shape[1])
            up = (cnt[1:] < lo); dn = (cnt[1:] > hi)
            if not (up.any() or dn.any()): break
            b[1:] += step * up - step * dn
        pred[m] = (Lg + b).argmax(1)
    return pred


def main():
    meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
    res = {}
    for sd in [42, 7]:
        out, y, sbj = build(sd); Q0 = out["final"]
        _, _, _, _, sens, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
        rec = meta["rec"].to_numpy()[tile]
        S = np.load(WORK / "decomp" / f"stage2_oof_s{sd}.npz")["S"]
        Qp = blend(Q0, S, 0.3)
        f0, tau = tune_tau(Qp, y)
        res.setdefault("exp017 (tau only)", []).append(f0)
        L = np.log(np.clip(Qp, 1e-9, None)) - tau * np.log(Qp.mean(0) + 1e-9)
        # count distribution within a recording (counts on this seed's evaluation set; one limb is picked per tile, so this equals the recording length in seconds)
        for lo, hi in [(78, 126), (70, 140), (60, 160), (50, 200), (85, 110)]:
            res.setdefault(f"band [{lo},{hi}]", []).append(macro_f1(y, band_calibrate(L, rec, lo, hi)))
        # set the band from the 5-95% of the distribution excluding the evaluated recording (leave-one-recording-out)
        pred = L.argmax(1).copy()
        cnt = pd.crosstab(rec, y)
        for g in np.unique(rec):
            other = cnt.drop(index=g).drop(columns=[0]).values.ravel(); other = other[other > 0]
            lo, hi = np.percentile(other, 5), np.percentile(other, 95)
            m = rec == g
            pred[m] = band_calibrate(L[m], rec[m], lo, hi)
        res.setdefault("band LORO 5-95%", []).append(macro_f1(y, pred))
        print(f"  seed {sd} done", flush=True)
    ref = np.array(res["exp017 (tau only)"])
    for nm, v in res.items():
        d = np.array(v) - ref
        print(f"  {nm:20s} {np.mean(v):.4f}  vs exp017 {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d) + "]")


if __name__ == "__main__":
    main()
