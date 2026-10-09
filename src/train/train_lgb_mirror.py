"""LightGBM on inertial data canonicalised by left-right mirroring (following the public solution's mirror-canonicalised IMU features; no leak used).

Apply sign flips s (chosen from 8 options) to the 3 axes of the left limb to map it into the right limb's coordinate frame, so a single model can learn
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
from left and right data jointly. s is the combination that minimises the left-right difference of the window-mean acceleration (gravity direction) on the same training windows
(arms: left arm -> right arm, legs: left leg -> right leg). Instead of a 4-way one-hot of the mounting position, [is leg, is left] is attached.
Features, splits and settings are the same as exp001 (train_lgb.py). Output: experiments/<tag>/ (same layout as exp001).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, SENSORS, TEST_DIR, WORK, seed_everything
from cv import subject_folds
from features import window_features
from metric import macro_f1

PAIRS = {"arm": (SENSORS.index("left_arm"), SENSORS.index("right_arm")),
         "leg": (SENSORS.index("left_leg"), SENSORS.index("right_leg"))}


def choose_signs(arr):
    M = np.nanmean(np.asarray(arr[::7], np.float32), axis=2)          # (n', 4, 3) window mean
    out = {}
    for k, (l, r) in PAIRS.items():
        ok = np.isfinite(M[:, l]).all(1) & np.isfinite(M[:, r]).all(1)
        best = None
        for s in itertools.product([1, -1], repeat=3):
            d = np.linalg.norm(M[ok, l] * np.array(s) - M[ok, r], axis=1).mean()
            print(f"  {k} s={s}: left-right difference {d:.3f}")
            if best is None or d < best[0]:
                best = (d, np.array(s, np.float32))
        out[k] = best[1]
        print(f"  -> {k}: s = {best[1].tolist()} (diff {best[0]:.3f})", flush=True)
    return out


def side_feats(sensor_idx):
    left = np.isin(sensor_idx, [PAIRS["arm"][0], PAIRS["leg"][0]]).astype(np.float32)
    leg = np.isin(sensor_idx, [PAIRS["leg"][0], PAIRS["leg"][1]]).astype(np.float32)
    return np.column_stack([leg, left])


def canon(x, sensor_idx, signs):
    x = np.array(x, np.float32, copy=True)
    x[sensor_idx == PAIRS["arm"][0]] *= signs["arm"]
    x[sensor_idx == PAIRS["leg"][0]] *= signs["leg"]
    return x


def main(a):
    seed_everything()
    prep = WORK / "prep"
    meta = pd.read_parquet(prep / "win_meta.parquet"); n = len(meta)
    arr = np.load(prep / "inertial.npy", mmap_mode="r")
    signs = choose_signs(arr)
    parts = []
    for s in range(len(SENSORS)):
        x = canon(np.asarray(arr[:, s]), np.full(n, s), signs)
        f, _ = window_features(x); parts.append(f)
    sensor_idx = np.repeat(np.arange(len(SENSORS)), n)
    X = np.concatenate([np.concatenate(parts), side_feats(sensor_idx)], axis=1)
    meta4 = pd.concat([meta] * len(SENSORS), ignore_index=True); meta4["sensor_idx"] = sensor_idx
    y = meta4["label"].to_numpy(np.int32)
    xt = np.load(TEST_DIR / "test_inertial_data.npy").astype(np.float32)
    tm = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    si = tm["sensor_location"].map({s: i for i, s in enumerate(SENSORS)}).to_numpy()
    ft, _ = window_features(canon(xt, si, signs))
    Xte = np.concatenate([ft, side_feats(si)], axis=1)
    folds, _ = subject_folds(meta4["sbj_id"].to_numpy(), n_splits=5)
    params = dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=8,
                  verbosity=-1, seed=SEED)
    oof = np.zeros((len(y), N_CLASSES), np.float32); test_p = np.zeros((len(Xte), N_CLASSES), np.float32)
    used = np.zeros(len(y), bool)
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=a.rounds, valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration); used[va] = True
        test_p += m.predict(Xte, num_iteration=m.best_iteration) / len(folds)
        print(f"fold{k} iter={m.best_iteration} F1={macro_f1(y[va], oof[va].argmax(1)):.4f} ({time.time() - t0:.0f}s)", flush=True)
    out = WORK / a.tag; out.mkdir(parents=True, exist_ok=True)
    print("== OOF macro-F1 =", round(macro_f1(y[used], oof[used].argmax(1)), 4))
    np.save(out / "oof.npy", oof); np.save(out / "test_pred.npy", test_p)
    meta4.loc[used, ["rec", "sbj_id", "start", "label", "sensor_idx"]].to_parquet(out / "oof_meta.parquet")
    np.save(out / "signs.npy", np.stack([signs["arm"], signs["leg"]]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=400)
    p.add_argument("--tag", default="exp025_lgb_mirror")
    main(p.parse_args())
