"""Label-free check on the test side before submitting.

Motivation: exp006 (CV +0.025 / LB -0.006) and exp009 (CV +0.017 / LB -0.038)
both "only revealed their failure after submission". The only signal is OOF, so
the symptom is: the more post-processing parameters we add, the higher CV goes, but it does not carry over to LB.

The synchronization module of the master's thesis (VSFF, §4.4 Algorithm 2) "matches the predictions of
two independent domains and sets parameters without labels". Here we take that principle,
not limited to synchronization parameters, and reuse it as a **test-side sanity check for post-processing in general**.

All the metrics below need no labels, so they can be computed directly on the 4 test subjects:

  agree_f1   : macro-F1 using the video model's argmax as pseudo ground truth.
               Independence of base=inertial / graph=video has been confirmed empirically
               (bidirectional check in probe_inertial_edge.py), so it gives an independent second view.
  agree_nmi  : same as above, with normalized mutual information. Insensitive to shifts in class priors.
  null_rate  : null fraction of the predictions. **The LB probe value 0.445 is the external anchor**
               (back-computed from the all-null submission 0.03242). The only ground truth we can check against without submitting.
  null_spread: min-max of per-subject null fraction. exp009's Sinkhorn flattened this
               to a uniform q_null and hit LB -0.038. Measured on the 22 train subjects: 0.14-0.57.
  nn_purity  : fraction of each window's video k-nearest neighbours that share its predicted label.
               ⚠ **Do not use for decisions**. In the probe_calib.py calibration, Spearman with LB is -0.086
               (-0.900 when restricted to the 5 model-only changes). Over-smoothing scores higher, so better configurations
               score lower. Kept only for diagnostics.
  n_used     : number of classes with predicted share >= 1% / min_share. macro-F1 is fatal when a
               single class collapses, so this detects collapse.

Usage:
  # 1) Calibration: put true macro-F1 and the proxies side by side on OOF, and check that the proxies rank the true values correctly
  uv run python src/probe_agree.py --tags exp001_lgb_inertial exp014_nn_inertial_rot \
      exp015_nn_inertial_s1337 exp016_nn_inertial_rot30 exp017_nn_aux \
      exp018_nn_aux_s99 exp019_nn_aux_s555 \
      --weights 1.5 1 1 1 1 1 1 --vgraph exp013_nn_video_5fold --temp 1.4 \
      --judge exp020_video_only --label base
  # 2) Run the same proxies on known failing configurations (--sinkhorn is the configuration that got LB -0.038)
  ...same command + --sinkhorn --label sinkhorn
  ...same command + --class-weights --label classw
  # If base's proxy > the failing configuration's proxy, this checker can be trusted.

⚠ If --judge and --vgraph get the same tag, graph propagation is pulled toward the judge's
  predictions and agree_* rises mechanically (contamination). A warning is printed when they are identical.

## ★ Scope (found in the probe_calib.py calibration; ignoring it leads to wrong decisions)

Calibrating on 6 earlier submissions (with measured LB), **which metric can be trusted switches by regime**:

| Kind of change | Metric to trust | Evidence |
|---|---|---|
| Model / blend / number of refinement stages | **true OOF F1** | Spearman +0.900 with LB on these 5 points |
| **Post-processing** (prior correction, per-class weights, Sinkhorn) | **agree_f1 / null_rate / spread_ratio** | OOF points in exactly the wrong direction for exp009 |

Over all n=6, OOF +0.371 / agree_f1 +0.829 / null_rate closeness +0.829, but
this difference is decided **solely by whether exp009 (a post-processing failure) is included**.
So do not generalize to "the proxies are better than OOF".
**OOF breaks only when post-processing is changed**, and that is the territory of this checker.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent))
from config import TEST_DIR, WORK
from eval_pipeline import load_eval_set, run
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import prior_correct, tune_tau, tune_weights
from refine import multistage
from sinkhorn import apply_per_subject

PREP = WORK / "prep"

# LB probe measurement: all-null submission 0.03242 -> F1_null = 0.616 -> null fraction 0.445.
# The only "already-submitted ground truth" for the public test, so it is used as an external anchor.
LB_NULL_RATE = 0.445
# Range of per-subject null fraction measured on the 22 train subjects (the evidence for exp009's failure)
TRAIN_NULL_RANGE = (0.14, 0.57)


# ---------------------------------------------------------------- label-free metrics

def nn_purity(F: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Fraction of each window's video k-nearest neighbours that have the same predicted label.

    The OOF "segment purity" needs ground-truth segments and cannot be computed on test. We substitute
    consistency on the neighbour graph. It is high when the grouping works.
    """
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(Fn) - 2)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return float((pred[nn] == pred[:, None]).mean())


def diagnose(pred, sbj, judge_pred, Fraw, k):
    """Diagnostics that use no labels at all. Returns the same thing on OOF and on test."""
    share = np.bincount(pred, minlength=19) / len(pred)
    us = np.unique(sbj)
    null_by_sbj = {int(s): float((pred[sbj == s] == 0).mean()) for s in us}
    # The judge also estimates the null fraction independently per subject. If post-processing flattens subject differences,
    # only our own spread shrinks and diverges from the judge's spread (the cause of exp009's failure).
    jnull_by_sbj = {int(s): float((judge_pred[sbj == s] == 0).mean()) for s in us}
    a = np.array(list(null_by_sbj.values()))
    b = np.array(list(jnull_by_sbj.values()))
    return {
        "agree_f1": macro_f1(judge_pred, pred),
        "agree_nmi": normalized_mutual_info_score(judge_pred, pred),
        "null_rate": float(share[0]),
        "null_by_sbj": null_by_sbj,
        "judge_null_by_sbj": jnull_by_sbj,
        "null_spread": (a.min(), a.max()),
        "judge_null_spread": (b.min(), b.max()),
        # How well subject differences are preserved. Closer to 1 means as much subject difference is kept as in the judge.
        "spread_ratio": float((a.max() - a.min()) / (b.max() - b.min() + 1e-9)),
        "null_corr": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"),
        "nn_purity": float(np.mean([nn_purity(Fraw[s], pred[sbj == s], k) for s in us])),
        "n_used": int((share >= 0.01).sum()),
        "min_share": float(share[1:].min()),
        "judge_null_rate": float((judge_pred == 0).mean()),
    }


def fmt(d, true_f1=None):
    lo, hi = d["null_spread"]
    jlo, jhi = d["judge_null_spread"]
    lines = []
    if true_f1 is not None:
        lines.append(f"    true macro-F1  : {true_f1:.4f}   <- uses labels; not visible on test")
    lines += [
        f"    agree_f1       : {d['agree_f1']:.4f}   (judge null fraction {d['judge_null_rate']:.3f})",
        f"    agree_nmi      : {d['agree_nmi']:.4f}",
        f"    null_rate      : {d['null_rate']:.3f}    <- LB probe measured {LB_NULL_RATE:.3f}",
        f"    null_spread    : {lo:.3f} - {hi:.3f}  (judge {jlo:.3f} - {jhi:.3f} / "
        f"train 22 subjects {TRAIN_NULL_RANGE[0]:.2f} - {TRAIN_NULL_RANGE[1]:.2f})",
        f"    spread_ratio   : {d['spread_ratio']:.3f}   null_corr(judge)={d['null_corr']:.3f}",
        f"    nn_purity      : {d['nn_purity']:.4f}",
        f"    n_used(>=1%)   : {d['n_used']}/19   min_share={d['min_share']:.4f}",
        f"    null/subject   : " + ", ".join(
            f"{s}:{v:.3f}(j{d['judge_null_by_sbj'][s]:.3f})"
            for s, v in d["null_by_sbj"].items()),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- pipeline reproduction

def build_oof(a, seed):
    """Produce OOF post-refinement probabilities with the same steps as final_submit.py."""
    w = a.weights or [1.0] * len(a.tags)
    P, y, sbj, F = load_eval_set(a.tags, w, seed=seed)
    Fraw = {s: F[s].copy() for s in F}          # for nn_purity; measured on raw features before the γ concatenation
    if a.vgraph:
        Vs = load_vgraph_oof(a.vgraph, a.tags[0], seed=seed)
        for s in F:
            f = F[s] - F[s].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            F[s] = np.concatenate([f, a.gamma * np.sqrt(Vs[s])], 1)
    Q = run(P, y, sbj, F, a.stages, a.k, a.alpha, a.iters, temp=a.temp)

    Vj = load_vgraph_oof(a.judge, a.tags[0], seed=seed)
    judge = np.empty((len(y), 19), np.float32)
    for s in np.unique(sbj):
        judge[sbj == s] = Vj[s]
    return Q, y, sbj, Fraw, judge


def build_test(a, tau, q_null, logw, qmap=None):
    """Same steps as the test side of final_submit.py. Returns only the predictions, no CSV is written."""
    w = a.weights or [1.0] * len(a.tags)
    te = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    Pt = None
    for t, ww in zip(a.tags, w):
        p = np.load(WORK / t / "test_pred.npy")
        p = p / p.sum(1, keepdims=True)
        Pt = ww * p if Pt is None else Pt + ww * p
    Pt /= sum(w)

    video = np.load(PREP / "video_raw_test.npy", mmap_mode="r")
    sbj = te["sbj_id"].to_numpy()
    Qt, Fraw = Pt.copy(), {}
    for s in np.unique(sbj):
        m = sbj == s
        Ft = np.asarray(video[np.where(m)[0]], np.float32).mean(1)
        Fraw[s] = Ft.copy()
        if a.vgraph:
            vt = np.load(WORK / a.vgraph / "test_pred.npy")[m]
            vt = vt / vt.sum(1, keepdims=True)
            Ft = Ft - Ft.mean(0)
            Ft /= np.linalg.norm(Ft, axis=1, keepdims=True) + 1e-8
            Ft = np.concatenate([Ft, a.gamma * np.sqrt(vt)], 1)
        Qt[m] = multistage(Pt[m], Ft, a.stages, k=a.k, alpha=a.alpha,
                           iters=a.iters, temp=a.temp)
    if q_null is not None:
        Qt = apply_per_subject(Qt, sbj, q_null)
    if qmap is not None:
        from final_submit import apply_qnull
        # qmap is a dict, or a callable returning a dict from Qt and sbj (for the anchor search)
        Qt = apply_qnull(Qt, sbj, qmap(Qt, sbj) if callable(qmap) else qmap)

    if logw is not None:
        pred = (np.log(Qt + 1e-9) + logw).argmax(1)
    elif q_null is not None or qmap is not None:
        pred = Qt.argmax(1)
    else:
        pred = prior_correct(Qt / Qt.sum(1, keepdims=True), tau).argmax(1)

    jt = np.load(WORK / a.judge / "test_pred.npy")
    judge = jt / jt.sum(1, keepdims=True)
    return pred, sbj, Fraw, judge


# ---------------------------------------------------------------- main

def main(a):
    if a.vgraph and a.judge == a.vgraph:
        print(f"⚠ --judge and --vgraph are identical ({a.judge}). Graph propagation is pulled toward the judge and\n"
              f"  agree_* rises mechanically. Pass a different video model to --judge.\n")

    # --- OOF: put the true value and the proxies side by side. tau/q_null/logw are also set here (same as final_submit)
    print(f"[{a.label}] tags={a.tags} vgraph={a.vgraph} judge={a.judge} "
          f"sinkhorn={a.sinkhorn} class_weights={a.class_weights}")
    tau = q_null = logw = None
    oof_rows = []
    for i, seed in enumerate(a.seeds):
        Q, y, sbj, Fraw, judge = build_oof(a, seed)
        f_tau, t = tune_tau(Q, y)

        if a.sinkhorn:
            f_sk, qn = max((macro_f1(y, apply_per_subject(Q, sbj, q).argmax(1)), q)
                           for q in np.arange(0.14, 0.50, 0.03))
            Qs = apply_per_subject(Q, sbj, qn)
            pred, true_f1 = Qs.argmax(1), f_sk
        elif a.class_weights:
            f_w, lw = tune_weights(Q, y)
            pred, true_f1 = (np.log(Q + 1e-9) + lw).argmax(1), f_w
        else:
            pred, true_f1 = prior_correct(Q, t).argmax(1), f_tau

        if i == 0:      # hyperparameters passed to the test side are those of seed[0] (same SEED as final_submit)
            tau = t
            q_null = qn if a.sinkhorn else None
            logw = lw if a.class_weights else None

        d = diagnose(pred, sbj, judge.argmax(1), Fraw, a.k)
        d["true_f1"] = true_f1
        oof_rows.append(d)
        print(f"\n  === OOF (seed={seed}, n={len(y)}) ===")
        print(fmt(d, true_f1))

    if len(oof_rows) > 1:
        print("\n  === OOF mean ===")
        for key in ["true_f1", "agree_f1", "agree_nmi", "null_rate", "nn_purity"]:
            v = np.array([r[key] for r in oof_rows])
            print(f"    {key:14s} : {v.mean():.4f} +- {v.std():.4f}")

    # --- test: run with the same hyperparameters and look only at the label-free metrics
    pred, sbj, Fraw, judge = build_test(a, tau, q_null, logw)
    dt = diagnose(pred, sbj, judge.argmax(1), Fraw, a.k)
    print(f"\n  === TEST (n={len(pred)}, tau={tau:.2f}"
          + (f", q_null={q_null:.2f}" if q_null is not None else "") + ") ===")
    print(fmt(dt))

    # --- decision guide
    m = np.mean([r["agree_f1"] for r in oof_rows])
    print("\n  === Decision ===")
    print(f"    agree_f1  OOF {m:.4f} -> TEST {dt['agree_f1']:.4f} "
          f"({dt['agree_f1'] - m:+.4f})")
    gap = abs(dt["null_rate"] - LB_NULL_RATE)
    flag = "OK" if gap < 0.05 else "⚠ divergent"
    print(f"    null_rate TEST {dt['null_rate']:.3f} vs LB probe {LB_NULL_RATE:.3f} "
          f"-> {flag} ({gap:+.3f})")
    sr = dt["spread_ratio"]
    print(f"    spread_ratio TEST {sr:.3f} "
          f"(our between-subject null difference / judge's between-subject null difference)")
    if sr < 0.7:
        lo, hi = dt["null_spread"]
        jlo, jhi = dt["judge_null_spread"]
        print(f"    ⚠ Subject differences have shrunk to {sr:.0%} of the judge's "
              f"(ours {lo:.3f}-{hi:.3f} / judge {jlo:.3f}-{jhi:.3f}). "
              f"Post-processing may be flattening subject differences (the cause of exp009's failure)")
    if dt["n_used"] < 19:
        print(f"    ⚠ {19 - dt['n_used']} classes have a predicted share below 1%. "
              f"macro-F1 is vulnerable to class collapse")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--vgraph", default=None)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--sinkhorn", action="store_true")
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--judge", default="exp020_video_only",
                   help="video model used as the independent view. Pass a different one from --vgraph")
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument("--label", default="cfg")
    main(p.parse_args())
