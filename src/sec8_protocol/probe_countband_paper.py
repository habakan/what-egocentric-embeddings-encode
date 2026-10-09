"""OOF of the paper's system (exp015, no stage 2) with count-band calibration applied. Also reports the change in per-class F1.

Band: 5-95% of the training recordings excluding the evaluated recording (leave-one-recording-out), and fixed [78,126]. 3 seeds.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, WORK
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_countband import band_calibrate
from probe_invariance import load_all
from probe_prf import TAGS, W, build

meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
res, pcf = {}, {}
for sd in [42, 7, 1337]:
    out, y, sbj = build(sd); Q = out["final"]
    _, _, _, _, _, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    rec = meta["rec"].to_numpy()[tile]
    f0, tau = tune_tau(Q, y)
    p0 = prior_correct(Q, tau).argmax(1)
    L = np.log(np.clip(Q, 1e-9, None)) - tau * np.log(Q.mean(0) + 1e-9)
    p1 = band_calibrate(L, rec, 78, 126)
    cnt = pd.crosstab(rec, y)
    p2 = p0.copy()
    for g in np.unique(rec):
        other = cnt.drop(index=g).drop(columns=[0]).values.ravel(); other = other[other > 0]
        m = rec == g
        p2[m] = band_calibrate(L[m], rec[m], np.percentile(other, 5), np.percentile(other, 95))
    for nm, p in [("exp015 (tau)", p0), ("+ band [78,126]", p1), ("+ band LORO 5-95%", p2)]:
        res.setdefault(nm, []).append(macro_f1(y, p))
        pcf.setdefault(nm, []).append(f1_score(y, p, average=None, labels=range(19)))
    # fraction of (recording x class) that were outside the band
    c0 = pd.crosstab(rec, p0).reindex(columns=range(19), fill_value=0).drop(columns=[0]).values
    res.setdefault("out-of-band pair fraction (before calibration)", []).append(((c0 < 78) | (c0 > 126)).mean())
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["exp015 (tau)"])
for nm, v in res.items():
    if nm.startswith("out-of-band"):
        print(f"  {nm}: {np.mean(v):.3f}"); continue
    d = np.array(v) - ref
    print(f"  {nm:18s} {np.mean(v):.4f} ± {np.std(v):.4f}  vs exp015 {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d) + "]")
a, b = np.mean(pcf["exp015 (tau)"], 0), np.mean(pcf["+ band LORO 5-95%"], 0)
print("\n  change in per-class F1 (LORO band - exp015), largest first:")
for c in np.argsort(-np.abs(b - a))[:10]:
    print(f"    {CLASS_NAMES[c]:28s} {a[c]:.3f} -> {b[c]:.3f}  ({b[c]-a[c]:+.3f})")
