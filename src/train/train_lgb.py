"""Inertial-only LightGBM baseline. Expands window x sensor placement as 1 sample."""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, SENSORS, TEST_DIR, WORK, seed_everything
from cv import subject_folds
from features import window_features
from metric import macro_f1, per_class_f1


def load_train(cache=True):
    prep = WORK / "prep"
    cache_f = prep / "feat_inertial.npy"
    meta = pd.read_parquet(prep / "win_meta.parquet")
    n = len(meta)
    if cache and cache_f.exists():
        X = np.load(cache_f)
    else:
        arr = np.load(prep / "inertial.npy", mmap_mode="r")     # (n, 4, 50, 3)
        parts = []
        for s in range(len(SENSORS)):
            f, names = window_features(np.asarray(arr[:, s]))
            parts.append(f)
        X = np.concatenate(parts, axis=0)                       # sensor-major
        np.save(cache_f, X)
    # repeat meta 4 times to match sensor-major order
    sensor_idx = np.repeat(np.arange(len(SENSORS)), n)
    meta4 = pd.concat([meta] * len(SENSORS), ignore_index=True)
    meta4["sensor_idx"] = sensor_idx
    X = np.concatenate([X, np.eye(len(SENSORS), dtype=np.float32)[sensor_idx]], axis=1)
    return X, meta4


def load_test():
    x = np.load(TEST_DIR / "test_inertial_data.npy").astype(np.float32)
    meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    f, _ = window_features(x)
    si = meta["sensor_location"].map({s: i for i, s in enumerate(SENSORS)}).to_numpy()
    f = np.concatenate([f, np.eye(len(SENSORS), dtype=np.float32)[si]], axis=1)
    return f, meta


def main(n_folds, only_fold, rounds, tag):
    seed_everything()
    X, meta = load_train()
    y = meta["label"].to_numpy(np.int32)
    print("train", X.shape, "classes", np.bincount(y, minlength=N_CLASSES))

    folds, _ = subject_folds(meta["sbj_id"].to_numpy(), n_splits=n_folds)
    Xte, te_meta = load_test()

    oof = np.zeros((len(y), N_CLASSES), np.float32)
    test_p = np.zeros((len(Xte), N_CLASSES), np.float32)
    used = np.zeros(len(y), bool)
    params = dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.1,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=8, verbosity=-1, seed=SEED)

    for k, (tr, va) in enumerate(folds):
        if only_fold is not None and k != only_fold:
            continue
        t0 = time.time()
        ds_tr = lgb.Dataset(X[tr], y[tr])
        ds_va = lgb.Dataset(X[va], y[va])
        m = lgb.train(params, ds_tr, num_boost_round=rounds, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(50)])
        p = m.predict(X[va], num_iteration=m.best_iteration)
        oof[va] = p; used[va] = True
        test_p += m.predict(Xte, num_iteration=m.best_iteration) / len(folds)
        f1 = macro_f1(y[va], p.argmax(1))
        print(f"fold{k} subj={sorted(meta['sbj_id'].to_numpy()[va].tolist()[:0] or set(meta['sbj_id'].to_numpy()[va]))} "
              f"iter={m.best_iteration} F1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)

    out = WORK / tag; out.mkdir(parents=True, exist_ok=True)
    yp = oof[used].argmax(1)
    print("== OOF macro-F1 (incl. null) =", round(macro_f1(y[used], yp), 4))
    print("== OOF macro-F1 (excl. null) =", round(macro_f1(y[used], yp, include_null=False), 4))
    print("per-class:", np.round(per_class_f1(y[used], yp), 3).tolist())
    np.save(out / "oof.npy", oof); np.save(out / "test_pred.npy", test_p)
    meta.loc[used, ["rec", "sbj_id", "start", "label", "sensor_idx"]].to_parquet(out / "oof_meta.parquet")
    if only_fold is None:
        sub = pd.DataFrame({"id": te_meta["id"], "target_feature": test_p.argmax(1)})
        sub.to_csv(Path(__file__).parents[2] / "submissions" / f"{tag}.csv", index=False)
        print("wrote submission", tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=None)
    p.add_argument("--rounds", type=int, default=400)
    p.add_argument("--tag", default="exp001_lgb_inertial")
    a = p.parse_args()
    main(a.folds, a.only_fold, a.rounds, a.tag)
