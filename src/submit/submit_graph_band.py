"""Fallback for slot 2: a submission applying the count band [78,126] to the paper's system without the second-stage classifier (exp015 = graph with top 30 removed).

The test Q is the one saved when exp018 was made (Q in experiments/decomp/exp018_band_test_probs.npz = output of the exp015 system).
The prior-correction tau is set with tune_tau on the OOF (seed 42) Q, as in exp015. Check that without the band it matches the exp015 submission.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from postprocess import prior_correct, tune_tau
from probe_countband import band_calibrate
from probe_prf import build
from submit_stack import ROOT, test_Q

out, y, sbj = build(42)
f0, tau0 = tune_tau(out["final"], y)
d = np.load(ROOT / "experiments" / "decomp" / "exp018_band_test_probs.npz")
Qt = d["Q"]
_, _, te, _ = test_Q()
sbj_t = te["sbj_id"].to_numpy()
plain = prior_correct(Qt, tau0).argmax(1)
ref = pd.read_csv(ROOT / "submissions" / "exp015_graphpc30.csv")["target_feature"].to_numpy()
print(f"OOF exp015 {f0:.4f} (tau {tau0:.2f}). agreement of no-band predictions with the exp015 submission {(plain == ref).mean():.4f}")
L = np.log(np.clip(Qt, 1e-9, None)) - tau0 * np.log(Qt.mean(0) + 1e-9)
pred = band_calibrate(L, sbj_t, 78, 126)
pd.DataFrame({"id": te["id"], "target_feature": pred}).to_csv(ROOT / "submissions" / "exp024_graph_band.csv", index=False)
b18 = pd.read_csv(ROOT / "submissions" / "exp018_band.csv")["target_feature"].to_numpy()
print(f"wrote exp024_graph_band.csv: labels changed (vs exp015) {(pred != ref).mean():.4f}, agreement with exp018 {(pred == b18).mean():.4f}, null rate {(pred == 0).mean():.4f}")
for s in np.unique(sbj_t):
    m = sbj_t == s
    cnt = np.bincount(pred[m], minlength=19)[1:]
    print(f"  sbj {s}: windows per activity class min {cnt.min()} max {cnt.max()}")
