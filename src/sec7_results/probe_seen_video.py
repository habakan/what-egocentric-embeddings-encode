"""How much does a video-only classifier rely on participants' appearance and location? (seen participants vs unseen participants)

Input: per-window video features (mean of 15 rows, 768 dims). LightGBM. Splits are the same as probe_seen_vs_unseen:
  (a) 5-fold by participant / (b) 5-fold over each recording's sets (runs of the same label)
Evaluation: macro-F1 over all tiled windows (prior-correction tau tuned on OOF). Also per-participant F1.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from metric import macro_f1
from postprocess import prior_correct, tune_tau

meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
vid = np.load(WORK / "prep" / "video_raw_head.npy", mmap_mode="r")
X = np.concatenate([np.asarray(vid[tile[i:i + 4096]], np.float32).mean(1) for i in range(0, len(tile), 4096)])
# 768 dims are too slow for LightGBM, so reduce to 64 principal components (no labels used)
Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc[::7], full_matrices=False); X = (Xc @ Vt[:64].T).astype(np.float32)
y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
us = np.random.RandomState(0).permutation(np.unique(sbj))
fold_a = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
mt = meta.iloc[tile].sort_values(["rec", "start"])
seg = (mt["label"].ne(mt["label"].shift()) | mt["rec"].ne(mt["rec"].shift())).cumsum()
seg_of = dict(zip(mt.index.to_numpy(), seg.to_numpy()))
segid = np.array([seg_of[t] for t in tile])
fold_b = np.random.RandomState(1).permutation(segid.max() + 1)[segid] % 5
params = dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.1, num_leaves=63,
              min_data_in_leaf=50, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, num_threads=8, verbosity=-1, seed=0)
res = {}
for nm, fold in [("(a) unseen participant", fold_a), ("(b) seen participant", fold_b)]:
    p = np.zeros((len(y), N_CLASSES))
    for k in range(5):
        tr = fold != k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=200)
        p[~tr] = m.predict(X[~tr])
        print(f"  {nm} fold {k} done", flush=True)
    f, t = tune_tau(p, y)
    pr = prior_correct(p, t).argmax(1)
    res[nm] = (f, {s: macro_f1(y[sbj == s], pr[sbj == s]) for s in np.unique(sbj)})
print("\n=== macro-F1 of video-only LightGBM ===")
for nm, (f, _) in res.items():
    print(f"  {nm:14s} {f:.4f}")
print(f"  seen participant - unseen participant: {res['(b) seen participant'][0] - res['(a) unseen participant'][0]:+.4f}")
print("  reference (inertial LightGBM, probe_seen_vs_unseen): unseen 0.5991 / seen 0.6005 / diff +0.0014")
pa, pb = res["(a) unseen participant"][1], res["(b) seen participant"][1]
d = np.array([pb[s] - pa[s] for s in pa])
print(f"  per-participant difference (seen - unseen): mean {d.mean():+.3f}, min {d.min():+.3f}, max {d.max():+.3f}")
