"""Calibration of probe_agree.py. Apply it to past submissions (with measured LB) and check whether the proxy ranks them in the same order as LB.

The weakness of probe_agree is the small number of calibration points. They can be added without using submission slots, so do this first.
Only **submissions whose config can be reliably reproduced from the plan** are listed here.
A calibration point with a mistaken config would break the calibration itself, so such points are excluded.

The judge is fixed to exp020_video_only for all configs (if the judge changes, the absolute values of agree_* move
and configs can no longer be compared). Only exp011 uses vgraph=exp013, which differs from the judge.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from final_submit import LB_NULL_RATE as ANCHOR, apply_qnull, fit_qnull_map
from sinkhorn import apply_per_subject
from config import WORK
from probe_agree import LB_NULL_RATE, build_oof, build_test, diagnose

B2 = ["exp001_lgb_inertial", "exp014_nn_inertial_rot"]
B4 = B2 + ["exp015_nn_inertial_s1337", "exp016_nn_inertial_rot30"]
B7 = B4 + ["exp017_nn_aux", "exp018_nn_aux_s99", "exp019_nn_aux_s555"]

# (label, measured LB, reproduction confidence, settings)
CONFIGS = [
    ("exp005_2stage",  0.80457, "mid", dict(tags=["exp001_lgb_inertial"], weights=[1.0],
                                           stages=[2.0], temp=1.0, vgraph=None)),
    ("exp008_blend4",  0.81439, "high", dict(tags=B4, weights=[1.5, 1, 1, 1],
                                           stages=[4.0, 4.0], temp=1.0, vgraph=None)),
    ("exp007_blend2",  0.81728, "high", dict(tags=B2, weights=[1.0, 1.0],
                                           stages=[4.0, 4.0], temp=1.0, vgraph=None)),
    # exp009 and exp010 are a control pair differing **only in whether Sinkhorn is used**.
    # The only pair with measured LB on the validator's home ground (post-processing changes).
    ("exp009_sinkhorn", 0.78573, "high", dict(tags=B4, weights=[1.5, 1, 1, 1],
                                            stages=[4.0, 4.0], temp=1.4, vgraph=None,
                                            sinkhorn=True)),
    ("exp010_sharpen", 0.82366, "high", dict(tags=B4, weights=[1.5, 1, 1, 1],
                                           stages=[4.0, 4.0], temp=1.4, vgraph=None)),
    ("exp011_vgraph",  0.84074, "high", dict(tags=B7, weights=[1.5, 1, 1, 1, 1, 1, 1],
                                           stages=[4.0, 4.0], temp=1.4,
                                           vgraph="exp013_nn_video_5fold")),
    # Second point of the post-processing regime. A control pair with exp011 differing **only in per-subject q_null**.
    ("exp012_qnull",   0.83891, "high", dict(tags=B7, weights=[1.5, 1, 1, 1, 1, 1, 1],
                                           stages=[4.0, 4.0], temp=1.4,
                                           vgraph="exp013_nn_video_5fold",
                                           qnull_judge="exp020_video_only")),
]

DEFAULTS = dict(k=30, alpha=0.75, iters=5, gamma=0.5, sinkhorn=False,
                class_weights=False, judge="exp020_video_only", seeds=[42], label="",
                qnull_judge=None)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def run_one(cfg):
    a = SimpleNamespace(**{**DEFAULTS, **cfg})
    Q, y, sbj, Fraw, judge = build_oof(a, a.seeds[0])
    f_tau, tau = tune_tau(Q, y)
    q_null, true_oof, oof_pred = None, f_tau, prior_correct(Q, tau).argmax(1)
    if a.sinkhorn:
        # determine q_null from OOF with the same procedure as final_submit.py
        true_oof, q_null = max((macro_f1(y, apply_per_subject(Q, sbj, q).argmax(1)), q)
                               for q in np.arange(0.14, 0.50, 0.03))
        oof_pred = apply_per_subject(Q, sbj, q_null).argmax(1)
    qmap = None
    if a.qnull_judge:
        # same procedure as final_submit. The anchor search needs Qt, so pass it as a callable.
        a_, b_, _, _ = fit_qnull_map(a.qnull_judge, a.tags[0], y, sbj)
        jt = np.load(WORK / a.qnull_judge / "test_pred.npy")
        jpt = (jt / jt.sum(1, keepdims=True)).argmax(1)

        def qmap(Qt, st, _a=a_, _b=b_, _jp=jpt):
            ust = np.unique(st)
            qraw = {s: _a + _b * (_jp[st == s] == 0).mean() for s in ust}
            shift = min(np.arange(-0.25, 0.25, 0.005),
                        key=lambda dz: abs(
                            (apply_qnull(Qt, st, {s: qraw[s] + dz for s in ust}).argmax(1) == 0).mean()
                            - ANCHOR))
            return {s: qraw[s] + shift for s in ust}

    d_oof = diagnose(oof_pred, sbj, judge.argmax(1), Fraw, a.k)
    pred, sbj_t, Fraw_t, judge_t = build_test(a, tau, q_null, None, qmap=qmap)
    d = diagnose(pred, sbj_t, judge_t.argmax(1), Fraw_t, a.k)
    d["true_oof"] = true_oof
    d["oof_agree_f1"] = d_oof["agree_f1"]
    return d


def main():
    rows = []
    for name, lb, conf, cfg in CONFIGS:
        d = run_one(cfg)
        d.update(name=name, lb=lb, conf=conf)
        rows.append(d)
        print(f"  {name:16s} done", flush=True)

    print("\n" + "=" * 96)
    hdr = (f"{'config':17s}{'conf':4s}{'LB':>9s}{'trueOOF':>8s}"
           f"{'agree_f1':>10s}{'agree_nmi':>10s}{'null_rate':>10s}{'nn_purity':>10s}{'spread':>8s}")
    print(hdr); print("-" * 96)
    for r in sorted(rows, key=lambda x: x["lb"]):
        print(f"{r['name']:17s}{r['conf']:4s}{r['lb']:9.5f}{r['true_oof']:8.4f}"
              f"{r['agree_f1']:10.4f}{r['agree_nmi']:10.4f}{r['null_rate']:10.3f}"
              f"{r['nn_purity']:10.4f}{r['spread_ratio']:8.3f}")

    lb = np.array([r["lb"] for r in rows])
    print("\n=== Rank correlation with LB (Spearman, n=%d) ===" % len(rows))
    print(f"  {'true OOF F1':16s}: {spearman(lb, [r['true_oof'] for r in rows]):+.3f}"
          f"   <- the criterion used so far")
    for key, nm in [("agree_f1", "agree_f1"), ("agree_nmi", "agree_nmi"),
                    ("nn_purity", "nn_purity")]:
        print(f"  {nm:16s}: {spearman(lb, [r[key] for r in rows]):+.3f}")
    gap = [-abs(r["null_rate"] - LB_NULL_RATE) for r in rows]
    print(f"  {'null_rate close':16s}: {spearman(lb, gap):+.3f}   "
          f"(smallness of |null_rate - {LB_NULL_RATE}|)")

    # control pair differing only in post-processing = the validator's home ground
    d = {r["name"]: r for r in rows}
    if "exp009_sinkhorn" in d and "exp010_sharpen" in d:
        s, o = d["exp009_sinkhorn"], d["exp010_sharpen"]
        print("\n=== Control pair: exp010 vs exp009 (differ only in whether Sinkhorn is used) ===")
        print(f"  {'':14s}{'exp010':>10s}{'exp009':>10s}   verdict")
        for key, nm, hi in [("lb", "measured LB", True), ("true_oof", "true OOF F1", True),
                            ("agree_f1", "agree_f1", True), ("agree_nmi", "agree_nmi", True),
                            ("nn_purity", "nn_purity", True), ("spread_ratio", "spread_ratio", True)]:
            win = "exp010" if (o[key] > s[key]) == hi else "exp009"
            mark = "OK" if win == "exp010" else "x  opposite to LB"
            print(f"  {nm:14s}{o[key]:10.4f}{s[key]:10.4f}   {mark}")
        print(f"  {'null_rate diff':14s}{abs(o['null_rate']-LB_NULL_RATE):10.4f}"
              f"{abs(s['null_rate']-LB_NULL_RATE):10.4f}   "
              f"{'OK' if abs(o['null_rate']-LB_NULL_RATE) < abs(s['null_rate']-LB_NULL_RATE) else 'x  opposite to LB'}")


if __name__ == "__main__":
    main()
