"""Train a second stage on "features" pooled over neighbours. A flat version and a hierarchical version per variant family.

Background (09-23):
  Half of the final system's errors are between variants of the same exercise (push-ups vs complex confusion 17%,
  sit-ups 8.7%, lunges 8.8%, jogging and rotating arms). The family is mostly right.
  Variant differences are like "complex has about 2x the limb sway": weak in one window / one limb, but
  stable when the whole segment is viewed.

  The current system pools neighbours' "predictions" but not their "features". The classifier always sees
  a single 1 s, single-limb window.

Proposal:
  For each window, **average the inertial features of its 30 video neighbours (after removing the top 30 PCs, same space as production)
  per sensor placement**, building features that view the segment with four limbs. Add own features, placement, and the production Q.
  A flat : 19-class LightGBM. Q' ∝ Q^(1-w) S^w
  B hier.: a model per family {push-ups 2, sit-ups 2, lunges 2, jogging 5} that only discriminates within the family.
           The family's total probability stays as in Q; only the within-family split is replaced (null boundary untouched).

Train/eval: 5-fold by subject. Neighbour pooling is within-subject and uses no labels (transductive).
kill: stop if neither beats the current system on all seeds.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES
from postprocess import tune_tau
from probe_invariance import drop_top_pcs, load_all, zscore
from probe_prf import TAGS, W, build

FAMS = {"push-ups": [11, 12], "sit-ups": [13, 14], "lunges": [16, 17], "jogging": [1, 2, 3, 4, 5]}


def pooled_features(Fv, Fi, sens, k):
    f = drop_top_pcs(Fv, 30)
    f = f - f.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    S = f @ f.T
    kk = min(k, len(f) - 1)
    nn = np.argpartition(-S, kk, axis=1)[:, :kk + 1]        # includes self
    D = Fi.shape[1]
    out = np.full((len(f), 4 * D + 4), np.nan, np.float32)
    for li in range(4):
        msk = (sens[nn] == li)                               # (n, k+1)
        cnt = msk.sum(1)
        s = (Fi[nn] * msk[:, :, None]).sum(1)
        mean = s / np.maximum(cnt, 1)[:, None]
        mean[cnt == 0] = np.nan
        out[:, li * D:(li + 1) * D] = mean
        out[:, 4 * D + li] = cnt
    return out


def fold_ids(sbj, sd, n=5):
    us = np.random.RandomState(sd).permutation(np.unique(sbj))
    return {s: i % n for i, s in enumerate(us)}


def pair_conf(y, p, a, b):
    t = (y == a) | (y == b)
    return float(((y == a) & (p == b)).sum() + ((y == b) & (p == a)).sum()) / max(t.sum(), 1)


def main(a):
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    from postprocess import prior_correct
    res, pconf, pcf = {}, {}, {}
    for sd in a.seeds:
        out, y, sbj = build(sd)
        Q = out["final"]
        P, V, y2, sbj2, sens, start, Fv, Fi_raw, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
        assert (y2 == y).all() and (sbj2 == sbj).all(), "window order does not match"
        n = len(y)
        own = np.zeros((n, next(iter(Fi_raw.values())).shape[1]), np.float32)
        pool = None
        for s in np.unique(sbj):
            m = np.where(sbj == s)[0]
            Fi = zscore(Fi_raw[s])
            own[m] = Fi
            pf = pooled_features(Fv[s], Fi, sens[m], a.k)
            if pool is None:
                pool = np.full((n, pf.shape[1]), np.nan, np.float32)
            pool[m] = pf
        # ⚠ If Q is an input, probability calibration differs between the train side (OOF, 1-fold model) and the test side (5-fold average),
        #   and the second stage may learn that quirk, which would not transfer
        #   (the same issue as when per-class weights gave OOF +0.025 / LB -0.006).
        #   --no-q restricts the input to the inertial features and the sensor placement.
        cols = [own, np.eye(4)[sens], pool] + ([] if a.no_q else [Q])
        X = np.column_stack(cols).astype(np.float32)
        fid = fold_ids(sbj, sd)
        fold = np.array([fid[s] for s in sbj])

        def lgbm(nc):
            kw = dict(n_estimators=a.trees, learning_rate=0.05, num_leaves=31,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.4,
                      min_child_samples=20, verbose=-1, random_state=sd, n_jobs=a.jobs)
            return lgb.LGBMClassifier(objective="multiclass" if nc > 2 else "binary", **kw)

        # --- A flat
        S = np.zeros((n, N_CLASSES))
        for f_ in range(5):
            te = fold == f_
            mdl = lgbm(N_CLASSES).fit(X[~te], y[~te])
            pr = np.zeros((te.sum(), N_CLASSES)); pr[:, mdl.classes_] = mdl.predict_proba(X[te])
            S[te] = pr
        # --- B hierarchical (within-family split only)
        FP = {}
        for nm, mem in FAMS.items():
            Pm = np.zeros((n, len(mem)))
            for f_ in range(5):
                te = fold == f_
                tr = (~te) & np.isin(y, mem)
                yy = np.searchsorted(mem, y[tr])
                mdl = lgbm(len(mem)).fit(X[tr], yy)
                pr = np.zeros((te.sum(), len(mem))); pr[:, mdl.classes_] = mdl.predict_proba(X[te])
                Pm[te] = pr
            FP[nm] = Pm

        def score(Qx, key):
            f, tau = tune_tau(Qx, y)
            res.setdefault(key, []).append(f)
            pr = prior_correct(Qx, tau).argmax(1)
            pconf.setdefault(key, []).append(pair_conf(y, pr, 11, 12))
            pcf.setdefault(key, []).append(
                f1_score(y, pr, labels=list(range(N_CLASSES)), average=None, zero_division=0))

        score(Q, "current")
        for w in a.ws:
            Qa = np.exp((1 - w) * np.log(Q + 1e-9) + w * np.log(S + 1e-9)); Qa /= Qa.sum(1, keepdims=True)
            score(Qa, f"A flat w={w}")
            Qb = Q.copy()
            for nm, mem in FAMS.items():
                mass = Qb[:, mem].sum(1, keepdims=True)
                within = Qb[:, mem] / np.maximum(mass, 1e-12)
                nw = np.exp((1 - w) * np.log(within + 1e-9) + w * np.log(FP[nm] + 1e-9))
                nw /= nw.sum(1, keepdims=True)
                Qb[:, mem] = mass * nw
            score(Qb, f"B hier. w={w}")
        # Within-family accuracy of the family models (on windows of the true family)
        for nm, mem in FAMS.items():
            t = np.isin(y, mem)
            acc = (np.array(mem)[FP[nm][t].argmax(1)] == y[t]).mean()
            accq = (np.array(mem)[Q[t][:, mem].argmax(1)] == y[t]).mean()
            print(f"  seed {sd} {nm}: within-family accuracy  current Q {accq:.3f} -> family model {acc:.3f}", flush=True)
        print(f"  seed {sd} done", flush=True)

    base = np.array(res["current"])
    print(f"\n=== Second stage on neighbour-pooled features ({len(a.seeds)} seeds) ===")
    print(f"  {'config':18s}{'macro-F1':>10s}{'vs current':>9s}{'push-up pair confusion':>20s}  per seed")
    for nm, v in res.items():
        d = np.array(v) - base
        print(f"  {nm:18s}{np.mean(v):10.4f}{d.mean():+9.4f}{np.mean(pconf[nm]):20.3f}  ["
              + " ".join(f"{x:+.4f}" for x in d) + "]")
    best = max((k for k in res if k != "current"), key=lambda k: np.mean(res[k]))
    dd = np.mean(pcf[best], 0) - np.mean(pcf["current"], 0)
    print(f"\n  change in per-class F1 ({best} - current), largest first:")
    for c in np.argsort(-np.abs(dd))[:8]:
        print(f"    {CLASS_NAMES[c]:28s}{np.mean(pcf['current'],0)[c]:6.3f} -> "
              f"{np.mean(pcf[best],0)[c]:6.3f}  ({dd[c]:+.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--ws", nargs="*", type=float, default=[0.3, 0.5, 0.8])
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7])
    p.add_argument("--no-q", action="store_true")
    main(p.parse_args())
