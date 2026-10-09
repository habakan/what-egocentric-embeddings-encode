"""Build a submission in which a second stage (Q + inertial features of the window and its neighbours) corrects Q's confusions.

OOF evidence (probe_nbfeat, stack_q):
  input     inertial features only -0.0001 / Q only +0.0006 / **Q + inertial features +0.0082** (both seeds)
  It only helps in combination. The second stage uses fine inertial cues conditioned on Q.

Concerns about transfer to test, and verified facts:
  The test Q is reproduced with the same procedure as OOF (agreement with exp015 1.0000).
  The shift in Q's distribution is small (median max probability 0.958 -> 0.973). The largest difference is the class composition
  (predicted null 0.376 -> 0.427), which is consistent with the true test null rate 0.445.

Procedure:
  features = [own inertial features (within-subject z), sensor location one-hot, neighbours' inertial features averaged per sensor location, Q]
  neighbours = k=30 on video features (top 30 PCs removed), including self. No labels used.
  second stage = multiclass LightGBM. Trained on all training windows and applied to test.
  Q' ∝ Q^(1-w) S^w. tau is set from the 5-fold OOF S.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, TEST_DIR, WORK
from final_submit import vsource_test
from postprocess import prior_correct, tune_tau
from probe_invariance import load_all, zscore
from probe_nbfeat import pooled_features
from probe_prf import TAGS, W, build
from stack_q import blend, test_Q
from train_lgb import load_test

PREP = WORK / "prep"
ROOT = Path(__file__).resolve().parents[2]


def lgbm(sd, jobs):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.4, min_child_samples=20, verbose=-1,
                              random_state=sd, n_jobs=jobs)


def feats(Q, own_by_s, Fv_by_s, sens, sbj, k):
    n = len(Q)
    D = next(iter(own_by_s.values())).shape[1]
    own = np.zeros((n, D), np.float32); pool = np.full((n, 4 * D + 4), np.nan, np.float32)
    for s in np.unique(sbj):
        m = np.where(sbj == s)[0]
        Fi = zscore(own_by_s[s]); own[m] = Fi
        pool[m] = pooled_features(Fv_by_s[s], Fi, sens[m], k)
    return np.column_stack([own, np.eye(4)[sens], pool, Q]).astype(np.float32)


def main(a):
    # ---------- training side (OOF)
    sd = a.seed
    out, y, sbj = build(sd)
    Q0 = out["final"]
    _, _, y2, sbj2, sens0, _, Fv0, Fi0, _ = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    assert (y2 == y).all()
    X0 = feats(Q0, Fi0, Fv0, sens0, sbj, a.k)

    us = np.random.RandomState(sd).permutation(np.unique(sbj))
    fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
    S0 = np.zeros((len(y), N_CLASSES))
    for kf in range(5):
        tr = fold != kf
        mdl = lgbm(sd, a.jobs).fit(X0[tr], y[tr])
        pr = np.zeros(((~tr).sum(), N_CLASSES)); pr[:, mdl.classes_] = mdl.predict_proba(X0[~tr])
        S0[~tr] = pr
    f0, tau0 = tune_tau(Q0, y)
    f1, tau1 = tune_tau(blend(Q0, S0, a.w), y)
    print(f"OOF: current {f0:.4f} (tau {tau0:.2f})  ->  second stage {f1:.4f} (tau {tau1:.2f})  diff {f1-f0:+.4f}")

    # ---------- test side
    Qt, Pt, te, sens_t = test_Q()
    sbj_t = te["sbj_id"].to_numpy()
    ft, _ = load_test()
    D = next(iter(Fi0.values())).shape[1]
    ft = np.nan_to_num(ft[:, :D].astype(np.float32))
    video = np.load(PREP / "video_raw_test.npy", mmap_mode="r")
    Fi_t, Fv_t = {}, {}
    for s in np.unique(sbj_t):
        m = np.where(sbj_t == s)[0]
        Fi_t[s] = ft[m]
        Fv_t[s] = np.asarray(video[m], np.float32).mean(1)
    Xt = feats(Qt, Fi_t, Fv_t, sens_t, sbj_t, a.k)
    full = lgbm(sd, a.jobs).fit(X0, y)
    St = np.zeros((len(Qt), N_CLASSES)); St[:, full.classes_] = full.predict_proba(Xt)
    Bt = blend(Qt, St, a.w)
    np.savez(ROOT / "experiments" / "decomp" / f"{a.out}_test_probs.npz", Q=Qt, S=St, B=Bt, tau=tau1)
    pred = prior_correct(Bt, tau1).argmax(1)
    if a.band:
        # calibration by per-class count band (probe_countband.band_calibrate). Per subject; null takes the remainder
        from probe_countband import band_calibrate
        L = np.log(np.clip(Bt, 1e-9, None)) - tau1 * np.log(Bt.mean(0) + 1e-9)
        pred = band_calibrate(L, sbj_t, a.band[0], a.band[1])
    outp = ROOT / "submissions" / f"{a.out}.csv"
    pd.DataFrame({"id": te["id"], "target_feature": pred}).to_csv(outp, index=False)

    # ---------- screen
    ref = pd.read_csv(ROOT / "submissions" / a.ref)["target_feature"].to_numpy()
    vo = vsource_test(["exp020_video_only"]).argmax(1)
    print(f"\nwrote {outp.name}")
    for nm, p in [(a.ref, ref), (a.out, pred)]:
        nr = [(p[sbj_t == s] == 0).mean() for s in np.unique(sbj_t)]
        print(f"  {nm:18s} null rate {(p==0).mean():.4f}  between-subject null range {max(nr)-min(nr):.3f}  "
              f"agreement with video model {(p == vo).mean():.4f}  classes {len(np.unique(p))}")
    print(f"  agreement with current {(pred == ref).mean():.4f}  (LB probe null rate 0.445)")
    from config import CLASS_NAMES
    chg = pd.crosstab(pd.Series(ref, name="current"), pd.Series(pred, name="new"))
    moves = [(chg.index[i], chg.columns[j], chg.iat[i, j]) for i in range(chg.shape[0])
             for j in range(chg.shape[1]) if chg.index[i] != chg.columns[j] and chg.iat[i, j] > 0]
    moves.sort(key=lambda z: -z[2])
    print("  main shifts among windows whose prediction changed (current -> new):")
    for a_, b_, c_ in moves[:8]:
        print(f"    {CLASS_NAMES[a_]:26s} -> {CLASS_NAMES[b_]:26s} {c_}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--w", type=float, default=0.3)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--out", default="exp017_stack")
    p.add_argument("--band", nargs=2, type=int, default=None, help="count band lo hi (e.g. 78 126)")
    p.add_argument("--ref", default="exp015_graphpc30.csv", help="submission to compare against in the screen")
    main(p.parse_args())
