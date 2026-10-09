"""Which participants does the count-band calibration help? (final Q of the paper's system exp015, 2 seeds)

Band: 5-95% over the training recordings other than the evaluated one (leave-one-recording-out), same as probe_countband_paper.
Per participant: macro-F1 before/after calibration, predicted null fraction (compared with the true value), number of classes outside the band.
Hypothesis: it helps most for participants who break down at the null boundary (sbj_10: mistakes null for activity / sbj_4, 9: mistake activity for null).
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
from postprocess import prior_correct, tune_tau
from probe_countband import band_calibrate
from probe_invariance import load_all
from probe_prf import TAGS, W, build

meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
rows = {}
for sd in [42, 7]:
    out, y, sbj = build(sd); Q = out["final"]
    _, _, _, _, _, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    rec = meta["rec"].to_numpy()[tile]
    tau = tune_tau(Q, y)[1]
    p0 = prior_correct(Q, tau).argmax(1)
    L = np.log(np.clip(Q, 1e-9, None)) - tau * np.log(Q.mean(0) + 1e-9)
    cnt = pd.crosstab(rec, y)
    p1 = p0.copy()
    for g in np.unique(rec):
        other = cnt.drop(index=g).drop(columns=[0]).values.ravel(); other = other[other > 0]
        m = rec == g
        p1[m] = band_calibrate(L[m], rec[m], np.percentile(other, 5), np.percentile(other, 95))
    for s in np.unique(sbj):
        m = sbj == s
        c0 = np.bincount(p0[m], minlength=19)[1:]
        outb = int(((c0 < 78) | (c0 > 126)).sum()) if len(np.unique(rec[m])) == 1 else np.nan
        r = rows.setdefault(s, {"f0": [], "f1": [], "null_true": [], "null0": [], "null1": [], "out": []})
        r["f0"].append(macro_f1(y[m], p0[m])); r["f1"].append(macro_f1(y[m], p1[m]))
        r["null_true"].append((y[m] == 0).mean()); r["null0"].append((p0[m] == 0).mean()); r["null1"].append((p1[m] == 0).mean())
        r["out"].append(outb)
    print(f"  seed {sd} done", flush=True)
tab = pd.DataFrame([{"sbj": f"sbj_{s}", "F1_before": np.mean(r["f0"]), "F1_after": np.mean(r["f1"]),
                     "gain": np.mean(r["f1"]) - np.mean(r["f0"]), "null_true": np.mean(r["null_true"]),
                     "null_pred_before": np.mean(r["null0"]), "null_pred_after": np.mean(r["null1"]),
                     "classes_out_of_band": np.nanmean(r["out"])} for s, r in rows.items()]).sort_values("F1_before")
pd.set_option("display.width", 220)
print(tab.round(3).to_string(index=False))
print("\nCorrelation: gain vs F1 before calibration %.2f / gain vs |pred null - true null| (before calibration) %.2f" % (
    np.corrcoef(tab.gain, tab.F1_before)[0, 1],
    np.corrcoef(tab.gain, (tab.null_pred_before - tab.null_true).abs())[0, 1]))
print("mean null error |pred - true|: before calibration %.3f -> after calibration %.3f" % (
    (tab.null_pred_before - tab.null_true).abs().mean(), (tab.null_pred_after - tab.null_true).abs().mean()))
hard = tab.sbj.isin(["sbj_10", "sbj_4", "sbj_5", "sbj_9"])
print("mean gain: hard 4 participants %+.3f / others %+.3f" % (tab.gain[hard].mean(), tab.gain[~hard].mean()))
