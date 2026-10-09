"""Pre-submission check of the idea of stacking a second stage (correcting Q's confusions) on top of the production probabilities Q.

Background (probe_nbfeat):
  Feeding Q into the second stage gives OOF +0.0082 (both seeds). Without Q the gain disappears.
  So the second stage corrects Q's habitual confusions (pulling complex too far toward plain, etc.).
  Concern: on the training side Q comes from OOF, where each subject is predicted by one fold model; on the test side Q comes
  from the average of 5 fold predictions. If the probability calibration differs, the second-stage correction does not transfer
  (the same pattern that got per-class weights OOF +0.025 / LB -0.006).

This script:
  1. Reproduces the production test-side Q with the same steps as final_submit and checks that it gives the same labels as exp015
  2. Measures the distribution shift between OOF Q and test Q (max probability, entropy, predicted class distribution)
  3. Checks on OOF whether the gain remains when the second-stage input is restricted to [Q] / [Q, sensor position]
  4. With --write, applies a second stage trained on all training windows to test, writes the submission, and prints numbers for screening
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
from config import CLASS_NAMES, N_CLASSES, SENSORS, TEST_DIR, WORK
from final_submit import vsource_test
from postprocess import prior_correct, tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from refine import drop_top_pcs, multistage_at

PREP = WORK / "prep"
ROOT = Path(__file__).resolve().parents[2]


def test_Q():
    te = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    Pt = None
    for t, ww in zip(TAGS, W):
        p = np.load(WORK / t / "test_pred.npy"); p = p / p.sum(1, keepdims=True)
        Pt = ww * p if Pt is None else Pt + ww * p
    Pt /= sum(W)
    video = np.load(PREP / "video_raw_test.npy", mmap_mode="r")
    Vt = vsource_test(["exp020_video_only", "exp022_vonly_center_5f"])
    vg = np.load(WORK / "exp013_nn_video_5fold" / "test_pred.npy"); vg = vg / vg.sum(1, keepdims=True)
    Qt = Pt.copy()
    for s in te["sbj_id"].unique():
        m = (te["sbj_id"] == s).to_numpy()
        Ft = np.asarray(video[np.where(m)[0]], np.float32).mean(1)
        Ft = drop_top_pcs(Ft, 30)
        Ft = Ft - Ft.mean(0); Ft /= np.linalg.norm(Ft, axis=1, keepdims=True) + 1e-8
        Ft = np.concatenate([Ft, 0.5 * np.sqrt(vg[m])], 1)
        Qt[m] = multistage_at(Pt[m], Ft, Vt[m], [4.0, 4.0], k=30, alpha=0.75, iters=5,
                              temp=1.4, g=0.2, where="every")
    sens = te["sensor_location"].map({s: i for i, s in enumerate(SENSORS)}).to_numpy()
    return Qt, Pt, te, sens


def ent(Q):
    q = np.clip(Q, 1e-9, 1); return -(q * np.log(q)).sum(1)


def describe(nm, Q, P):
    print(f"  {nm:10s} Q: max prob mean {Q.max(1).mean():.3f} (median {np.median(Q.max(1)):.3f}), "
          f"entropy mean {ent(Q).mean():.3f} | base P: max prob {P.max(1).mean():.3f}, "
          f"entropy {ent(P).mean():.3f}")


def fit_stage2(X, y, sd, jobs):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05,
                              num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, min_child_samples=20, verbose=-1,
                              random_state=sd, n_jobs=jobs).fit(X, y)


def blend(Q, S, w):
    Z = np.exp((1 - w) * np.log(Q + 1e-9) + w * np.log(S + 1e-9))
    return Z / Z.sum(1, keepdims=True)


def main(a):
    # ---------- 1. reproduce test-side Q
    Qt, Pt, te, sens_t = test_Q()
    ref = pd.read_csv(ROOT / "submissions" / "exp015_graphpc30.csv")["target_feature"].to_numpy()

    # ---------- OOF Q
    oof = {}
    for sd in a.seeds:
        out, y, sbj = build(sd)
        _, _, y2, sbj2, sens, *_ = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
        assert (y2 == y).all()
        oof[sd] = (out["final"], out["base"], y, sbj, sens)
    Q0, P0, y0, sbj0, sens0 = oof[a.seeds[0]]
    _, tau = tune_tau(Q0, y0)
    lab_t = prior_correct(Qt, tau).argmax(1)
    print(f"=== 1. Reproducing test-side Q === tau={tau:.2f}  agreement with exp015 {(lab_t == ref).mean():.4f}")

    # ---------- 2. distribution shift
    print("\n=== 2. Distribution of Q on OOF and test ===")
    describe("OOF", Q0, P0)
    describe("test", Qt, Pt)
    po = np.bincount(prior_correct(Q0, tau).argmax(1), minlength=N_CLASSES) / len(Q0)
    pt = np.bincount(lab_t, minlength=N_CLASSES) / len(Qt)
    print("  Classes with the largest difference in predicted class distribution (test - OOF):")
    for c in np.argsort(-np.abs(pt - po))[:5]:
        print(f"    {CLASS_NAMES[c]:28s} OOF {po[c]:.3f}  test {pt[c]:.3f}")

    # ---------- 3. second stage with restricted input
    print("\n=== 3. Does the gain remain with restricted second-stage input? (OOF, 5-fold by subject) ===")
    res = {}
    for sd, (Q, P, y, sbj, sens) in oof.items():
        us = np.random.RandomState(sd).permutation(np.unique(sbj))
        fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
        f0 = tune_tau(Q, y)[0]
        for nm, X in [("[Q]", Q), ("[Q, sensor position]", np.column_stack([Q, np.eye(4)[sens]]))]:
            S = np.zeros((len(y), N_CLASSES))
            for k in range(5):
                tr = fold != k
                mdl = fit_stage2(X[tr], y[tr], sd, a.jobs)
                S[~tr][:, :] = 0
                pr = np.zeros(((~tr).sum(), N_CLASSES)); pr[:, mdl.classes_] = mdl.predict_proba(X[~tr])
                S[~tr] = pr
            for w in a.ws:
                res.setdefault((nm, w), []).append(tune_tau(blend(Q, S, w), y)[0] - f0)
        print(f"  seed {sd} done", flush=True)
    for (nm, w), v in res.items():
        print(f"  {nm:14s} w={w}: vs current {np.mean(v):+.4f}  [" + " ".join(f"{x:+.4f}" for x in v) + "]")

    if not a.write:
        return
    # ---------- 4. submission
    nm = a.feat
    X0 = Q0 if nm == "Q" else np.column_stack([Q0, np.eye(4)[sens0]])
    Xt = Qt if nm == "Q" else np.column_stack([Qt, np.eye(4)[sens_t]])
    mdl = fit_stage2(X0, y0, a.seeds[0], a.jobs)
    St = np.zeros((len(Qt), N_CLASSES)); St[:, mdl.classes_] = mdl.predict_proba(Xt)
    # re-tune tau on the OOF distribution with the second stage (using S built from the 5 folds)
    us = np.random.RandomState(a.seeds[0]).permutation(np.unique(sbj0))
    fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj0])
    S0 = np.zeros((len(y0), N_CLASSES))
    for k in range(5):
        tr = fold != k
        m_ = fit_stage2(X0[tr], y0[tr], a.seeds[0], a.jobs)
        pr = np.zeros(((~tr).sum(), N_CLASSES)); pr[:, m_.classes_] = m_.predict_proba(X0[~tr])
        S0[~tr] = pr
    f2, tau2 = tune_tau(blend(Q0, S0, a.w), y0)
    pred = prior_correct(blend(Qt, St, a.w), tau2).argmax(1)
    outp = ROOT / "submissions" / f"{a.out}.csv"
    pd.DataFrame({"id": te["id"], "target_feature": pred}).to_csv(outp, index=False)
    print(f"\n=== 4. Submission {outp.name} (OOF {f2:.4f}, tau {tau2:.2f}) ===")
    sb = te["sbj_id"].to_numpy()
    vo = vsource_test(["exp020_video_only"]).argmax(1)
    for nm2, p in [("exp015 (current)", ref), (a.out, pred)]:
        nr = [(p[sb == s] == 0).mean() for s in np.unique(sb)]
        print(f"  {nm2:22s} null rate {(p==0).mean():.4f}  between-subject null range {max(nr)-min(nr):.3f}  "
              f"agreement with video model {(p == vo).mean():.4f}  n classes {len(np.unique(p))}")
    print(f"  agreement with current {(pred == ref).mean():.4f}  (LB probe null rate 0.445)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7])
    p.add_argument("--ws", nargs="*", type=float, default=[0.3, 0.5])
    p.add_argument("--w", type=float, default=0.3)
    p.add_argument("--feat", default="Q+sens", choices=["Q", "Q+sens"])
    p.add_argument("--write", action="store_true")
    p.add_argument("--out", default="exp017_stackq")
    p.add_argument("--jobs", type=int, default=6)
    main(p.parse_args())
