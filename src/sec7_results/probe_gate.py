"""Removing the qualifier from the independence principle: let video affect decisions "through a confidence gate".

The plan's "independence principle" was set up from two measurements:
  (a) mixing video into the base classifier **blend** -> 0.7859 -> 0.7640, worse
  (b) adding inertial edges to the graph              -> 0.8013 -> 0.7014, worse
Both were **uniform mixing**, and (a) was an operation **before refinement**.

probe_residual.py found that of the 106 segments the student gets entirely wrong, **15.1% can be rescued only by video**
(invisible to the inertial teacher). The video-only model is weak, with per-window F1 0.5064, yet
its residual rescue rate is 0.453, above the 4-sensor inertial teacher's 0.425. Especially for class 0 (null):
video 0.80 / inertial 0.57. So the information exists; it is a matter of allocation.

when-stuck.md Step 6: "a signal worthless on its own can become valuable through a gate (applied only where uncertain)".
Here, **after refinement**,
video is applied **only to windows where the inertial side is uncertain**:

    Q' ∝ Q^(1-g_i) · V^(g_i)          (log-linear pooling)
    g_i = g0 · (1 - max_c Q_ic)       gated version: the lower the confidence, the more video is applied
    g_i = g0                          control (uniform version): isolates whether the gate is essential

One parameter, g0. Always run the control alongside (if the control gains just as much,
it is not "the gate" but simply "adding video").

⚠ Note for decisions: probe_agree's judge is a video model, so mixing video into decisions
  raises agree_f1 mechanically (contamination). **This experiment does not use agree_f1 for accept/reject.**
  Look at OOF F1 and the judge-independent null_rate (measured anchor 0.445).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK
from eval_pipeline import load_eval_set, run
from final_submit import load_vgraph_oof
from postprocess import gate_blend, prior_correct, tune_tau

PREP = WORK / "prep"


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    P, y, sbj, F = load_eval_set(a.tags, w)
    if a.vgraph:
        Vs = load_vgraph_oof(a.vgraph, a.tags[0])
        for s in F:
            f = F[s] - F[s].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            F[s] = np.concatenate([f, a.gamma * np.sqrt(Vs[s])], 1)
    Q = run(P, y, sbj, F, a.stages, a.k, a.alpha, a.iters, temp=a.temp)
    base_f1, base_tau = tune_tau(Q, y)
    print(f"base: macro-F1 = {base_f1:.4f} (tau={base_tau:.2f})   n={len(y)}")
    print(f"gate source = {a.vsource}\n")

    Vd = load_vgraph_oof(a.vsource, a.tags[0])
    V = np.empty_like(Q)
    for s in np.unique(sbj):
        V[sbj == s] = Vd[s]

    print(f"{'g0':>6s} | {'gated':>10s} {'tau':>5s} | {'control(uniform)':>10s} {'tau':>5s}")
    print("-" * 48)
    for g0 in a.g0:
        fg, tg = tune_tau(gate_blend(Q, V, g0, True), y)
        fu, tu = tune_tau(gate_blend(Q, V, g0, False), y)
        mark = "  <- beats base" if fg > base_f1 else ""
        print(f"{g0:6.2f} | {fg:10.4f} {tg:5.2f} | {fu:10.4f} {tu:5.2f}{mark}")


def verify(a, g0, seeds=(42, 7, 123)):
    """For accept/reject: measure OOF over multiple seeds, and also report judge-independent metrics on the test side.

    ⚠ agree_f1 is contaminated (mixing video into decisions = pulling toward the judge), so it is not used.
    """
    import pandas as pd
    from config import TEST_DIR
    from probe_agree import LB_NULL_RATE, diagnose
    from refine import multistage

    w = a.weights or [1.0] * len(a.tags)
    out = []
    for sd in seeds:
        P, y, sbj, F = load_eval_set(a.tags, w, seed=sd)
        Vs = load_vgraph_oof(a.vgraph, a.tags[0], seed=sd)
        for s in F:
            f = F[s] - F[s].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            F[s] = np.concatenate([f, a.gamma * np.sqrt(Vs[s])], 1)
        Q = run(P, y, sbj, F, a.stages, a.k, a.alpha, a.iters, temp=a.temp)
        Vd = load_vgraph_oof(a.vsource, a.tags[0], seed=sd)
        V = np.empty_like(Q)
        for s in np.unique(sbj):
            V[sbj == s] = Vd[s]
        out.append((tune_tau(Q, y)[0], tune_tau(gate_blend(Q, V, g0, False), y)[0]))
    b = np.array([o[0] for o in out]); g = np.array([o[1] for o in out])
    print(f"\nOOF multi-seed{list(seeds)}:")
    print(f"  base      : {b.mean():.4f} +- {b.std():.4f}")
    print(f"  g0={g0:.2f} uniform: {g.mean():.4f} +- {g.std():.4f}   diff {g.mean()-b.mean():+.4f}")

    # --- test side
    te = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    sbjt = te["sbj_id"].to_numpy()
    Pt = None
    for t, ww in zip(a.tags, w):
        p = np.load(WORK / t / "test_pred.npy"); p = p / p.sum(1, keepdims=True)
        Pt = ww * p if Pt is None else Pt + ww * p
    Pt /= sum(w)
    video = np.load(PREP / "video_raw_test.npy", mmap_mode="r")
    Qt, Frawt = Pt.copy(), {}
    for s in np.unique(sbjt):
        m = sbjt == s
        Ft = np.asarray(video[np.where(m)[0]], np.float32).mean(1)
        Frawt[s] = Ft.copy()
        vt = np.load(WORK / a.vgraph / "test_pred.npy")[m]; vt = vt / vt.sum(1, keepdims=True)
        Ft = Ft - Ft.mean(0); Ft /= np.linalg.norm(Ft, axis=1, keepdims=True) + 1e-8
        Ft = np.concatenate([Ft, a.gamma * np.sqrt(vt)], 1)
        Qt[m] = multistage(Pt[m], Ft, a.stages, k=a.k, alpha=a.alpha, iters=a.iters, temp=a.temp)
    Vt = np.load(WORK / a.vsource / "test_pred.npy"); Vt = Vt / Vt.sum(1, keepdims=True)
    jt = np.load(WORK / "exp020_video_only" / "test_pred.npy")
    jp = (jt / jt.sum(1, keepdims=True)).argmax(1)

    _, tau_b = tune_tau(Q, y)
    print(f"\nTEST (agree_* is contaminated, reference only. Look at null_rate):")
    hdr = f"  {'config':14s}{'null_rate':>10s}{'|-0.445|':>10s}{'spread':>9s}{'n_used':>8s}{'(agree_f1)':>12s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for nm, QQ in [("base", Qt), (f"g0={g0:.2f} uniform", gate_blend(Qt, Vt, g0, False))]:
        _, tt = tune_tau(gate_blend(Q, V, 0.0 if nm == "base" else g0, False), y)
        pr = prior_correct(QQ / QQ.sum(1, keepdims=True), tt).argmax(1)
        d = diagnose(pr, sbjt, jp, Frawt, a.k)
        print(f"  {nm:14s}{d['null_rate']:10.3f}{abs(d['null_rate']-LB_NULL_RATE):10.4f}"
              f"{d['spread_ratio']:9.3f}{d['n_used']:8d}{d['agree_f1']:12.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--vsource", default="exp020_video_only",
                   help="video model applied through the gate")
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--vgraph", default="exp013_nn_video_5fold")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--g0", nargs="*", type=float,
                   default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    p.add_argument("--verify", type=float, default=None, help="run through to the accept/reject decision with this g0")
    args = p.parse_args()
    main(args)
    if args.verify is not None:
        verify(args, args.verify)
