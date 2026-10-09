"""Unseen participants vs seen participants: how much does the model rely on participant-specific information?

Train the inertial LightGBM (78 features + sensor placement) on the same data (tiled windows x four limbs) with 2 kinds of split.
  (a) unseen participants: 5-fold by participant (the current evaluation)
  (b) seen participants: 5-fold over each recording's sets (segments with the same label continuing). Train on other sets of the same participant, predict the rest
      (split by segment, so there is no leak from window overlap or within-segment continuity)
Evaluation on the same evaluation set as current (tiled windows, one limb per window by seed). Base macro-F1 and macro-F1 after
graph propagation (final configuration, GPU). Also report the spread of F1 per unseen participant. 2 seeds (limb assignment).
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
from config import N_CLASSES, SENSORS, WORK
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W
from refine_gpu import final_with

meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet"); n = len(meta)
tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
fi = np.load(WORK / "prep" / "feat_inertial.npy", mmap_mode="r")
X = np.concatenate([np.asarray(fi[tile + l * n], np.float32) for l in range(4)])
X = np.column_stack([X, np.repeat(np.eye(4, dtype=np.float32), len(tile), 0)])
T = np.tile(tile, 4); L = np.repeat(np.arange(4), len(tile))
y = meta["label"].to_numpy()[T]; sbj = meta["sbj_id"].to_numpy()[T]; rec = meta["rec"].to_numpy()[T]
start = meta["start"].to_numpy()[T]

# (a) by participant
us = np.random.RandomState(0).permutation(np.unique(sbj))
fold_a = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
# (b) by set: give an ID to each segment with the same label continuing per recording, and 5-fold the segments
mt = meta.iloc[tile].sort_values(["rec", "start"])
seg = (mt["label"].ne(mt["label"].shift()) | mt["rec"].ne(mt["rec"].shift())).cumsum()
seg_of = dict(zip(mt.index.to_numpy(), seg.to_numpy()))
segid = np.array([seg_of[t] for t in T])
rs = np.random.RandomState(1); perm = rs.permutation(segid.max() + 1)
fold_b = perm[segid] % 5
print(f"number of segments {segid.max() + 1}, median windows per segment {np.median(np.bincount(segid)) / 4:.0f}", flush=True)

params = dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.1, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, num_threads=8, verbosity=-1, seed=0)
oof = {}
for nm, fold in [("(a) unseen participants", fold_a), ("(b) seen participants", fold_b)]:
    p = np.zeros((len(y), N_CLASSES), np.float32)
    for k in range(5):
        tr = fold != k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=300)
        p[~tr] = m.predict(X[~tr])
        print(f"  {nm} fold {k} done", flush=True)
    oof[nm] = p

key = {(t, l): i for i, (t, l) in enumerate(zip(T, L))}
res, per = {}, {}
for sd in [42, 7]:
    _, _, ye, sbe, sens, _, _, _, te = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    idx = np.array([key[(t, l)] for t, l in zip(te, sens)])
    _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(ye), N_CLASSES), np.float32)
        for s in np.unique(sbe): M[sbe == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    for nm, p in oof.items():
        P = p[idx].astype(np.float64); P /= P.sum(1, keepdims=True)
        fb, tb = tune_tau(P, ye)
        Q = final_with(P, F0, Vg, V, sbe, 30)
        ff, tf = tune_tau(Q, ye)
        res.setdefault((nm, "base"), []).append(fb); res.setdefault((nm, "after propagation"), []).append(ff)
        pb, pf = prior_correct(P, tb).argmax(1), prior_correct(Q, tf).argmax(1)
        for s in np.unique(sbe):
            mm = sbe == s
            per.setdefault((nm, s), []).append((macro_f1(ye[mm], pb[mm]), macro_f1(ye[mm], pf[mm])))
    print(f"  seed {sd} done", flush=True)

print("\n=== macro-F1 with only the inertial LightGBM as base (2 seeds) ===")
for nm in oof:
    b, f = np.mean(res[(nm, "base")]), np.mean(res[(nm, "after propagation")])
    print(f"  {nm:14s} base {b:.4f}   after graph propagation {f:.4f}")
ga = np.mean(res[("(b) seen participants", "base")]) - np.mean(res[("(a) unseen participants", "base")])
gf = np.mean(res[("(b) seen participants", "after propagation")]) - np.mean(res[("(a) unseen participants", "after propagation")])
print(f"  seen participants - unseen participants: base {ga:+.4f}, after propagation {gf:+.4f}")
print("\n=== macro-F1 per unseen participant (base -> after propagation, mean of 2 seeds) ===")
rows = []
for (nm, s), v in per.items():
    if nm.startswith("(a)"):
        rows.append((s, np.mean([a for a, _ in v]), np.mean([b for _, b in v])))
rows.sort(key=lambda r: r[2])
for s, b, f in rows:
    print(f"  sbj_{s:<3d} {b:.3f} -> {f:.3f}")
fs = np.array([r[2] for r in rows])
print(f"  after propagation: min {fs.min():.3f}, median {np.median(fs):.3f}, max {fs.max():.3f}, std {fs.std():.3f}")
