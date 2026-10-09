"""Measure the effect of subject adaptation (transductive self-training) on OOF.

Take high-confidence windows from the refined predictions Q as pseudo-labels,
train a small classifier on **that subject's own inertial data only**, and re-predict all windows.
The aim is to adapt to the quirks of an unseen subject (sensor placement shifts, individual differences in movement).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, WORK
from eval_pipeline import load_eval_set, run
from metric import macro_f1
from postprocess import tune_tau
from refine import multistage

PREP = WORK / "prep"


def self_train(Xs, Q, conf_q, seed=SEED):
    """Xs: (n_s, d) that subject's inertial features, Q: (n_s, C) refined probabilities"""
    conf = Q.max(1)
    thr = np.quantile(conf, conf_q)
    m = conf >= thr
    yq = Q.argmax(1)
    if len(np.unique(yq[m])) < 5 or m.sum() < 200:
        return None
    params = dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.08,
                  num_leaves=15, min_data_in_leaf=20, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  num_threads=8, verbosity=-1, seed=seed)
    mdl = lgb.train(params, lgb.Dataset(Xs[m], yq[m]), num_boost_round=120)
    return mdl.predict(Xs).astype(np.float32)


def main(a):
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    P, y, sbj, F = load_eval_set(a.tags, a.weights or [1.0] * len(a.tags))
    Q = run(P, y, sbj, F, [4.0, 4.0], 30, 0.75, 5)
    print(f"base (3-stage refine + tau) : {tune_tau(Q, y)[0]:.4f}")

    # look up the inertial features corresponding to the evaluation rows
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile2, sens2 = tile[ok], sens[ok]
    Xi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    oof0 = np.load(WORK / a.tags[0] / "oof.npy")[tile2 + sens2 * n]
    keep = oof0.sum(1) > 1e-6
    rows = (tile2 + sens2 * n)[keep]
    X = np.asarray(Xi[rows], np.float32)
    X = np.concatenate([X, np.eye(4, dtype=np.float32)[sens2[keep]]], 1)
    assert len(X) == len(y)

    for conf_q in a.conf:
        for beta in a.beta:
            R = Q.copy()
            for s in np.unique(sbj):
                m = sbj == s
                ps = self_train(X[m], Q[m], conf_q)
                if ps is None:
                    continue
                blended = (1 - beta) * Q[m] + beta * ps
                R[m] = multistage(blended, F[s], [4.0, 4.0], k=30, alpha=0.75, iters=5)
            f1, tau = tune_tau(R, y)
            print(f"  self-training top {(1-conf_q)*100:.0f}% confident beta={beta} -> {f1:.4f} (tau={tau:.2f})",
                  flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["exp001_lgb_inertial"])
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--conf", nargs="*", type=float, default=[0.5, 0.7])
    p.add_argument("--beta", nargs="*", type=float, default=[0.3, 0.5])
    main(p.parse_args())
