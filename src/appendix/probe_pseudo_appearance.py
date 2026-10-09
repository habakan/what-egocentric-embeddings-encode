"""Idea 2: within a test participant, estimate per-activity means from the system's predictions (pseudo-labels), subtract them from the video features, and run the system again.

Idea 1 (an appearance subspace learned from other participants) did not work: egocentric appearance differs per participant (location, orientation).
What worked in the diagnosis was the within-participant per-activity mean, so we approximate it with pseudo-labels.
  pass 1: main system (top 30 removed) -> Q1
  estimate: soft  mu_c = sum_i Q1[i,c] F_i / sum_i Q1[i,c], C_i = sum_c Q1[i,c] mu_c
          hard  mean over assignment by argmax after prior correction
  pass 2: run the main system on F - lam * C combined with removal of the top m
Upper-bound reference: the same with ground-truth labels. Evaluation: OOF macro-F1 (tune_tau), seeds 42/7/1.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import prior_correct, tune_tau
from probe_prf import TAGS, W
from refine_gpu import final_with


def class_means(F, A):
    """F: (n,d) centered, A: (n,C) assignment (soft or one-hot) -> C_i for each window"""
    w = A.sum(0) + 1e-9
    mu = (A.T @ F) / w[:, None]
    mu -= (A.sum(0)[:, None] * mu).sum(0) / A.sum()          # make the mean over all assignments zero
    return A @ mu


def main():
    configs = [("top30 (current)", None, 0, 30)]
    for kind in ["soft", "hard", "ground truth (upper bound)"]:
        for lam in [0.5, 1.0]:
            for m in [0, 10, 20, 30]:
                configs.append((f"{kind} λ{lam} +top{m}", kind, lam, m))
    res = {c[0]: [] for c in configs}
    for sd in [42, 7, 1]:
        P, y, sbj, F0 = load_eval_set(TAGS, W, seed=sd)
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
            M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj): M[sbj == s] = dd[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        subs = np.unique(sbj)
        Fc = {s: F0[s] - F0[s].mean(0) for s in subs}
        Q1 = final_with(P, Fc, Vg, V, sbj, 30)
        f1, tau = tune_tau(Q1, y)
        res["top30 (current)"].append(f1)
        hard = prior_correct(Q1, tau).argmax(1)
        A = {"soft": Q1 / Q1.sum(1, keepdims=True), "hard": np.eye(N_CLASSES)[hard], "ground truth (upper bound)": np.eye(N_CLASSES)[y]}
        for name, kind, lam, m in configs[1:]:
            Fd = {s: Fc[s] - lam * class_means(Fc[s], A[kind][sbj == s]) for s in subs}
            res[name].append(tune_tau(final_with(P, Fd, Vg, V, sbj, m), y)[0])
        print(f"seed {sd} done: current {f1:.4f}", flush=True)
    ref = np.array(res["top30 (current)"])
    print("\n=== OOF macro-F1 (mean of 3 seeds, difference from current) ===")
    for name, *_ in configs:
        v = np.array(res[name])
        print(f"  {name:22s} {v.mean():.4f}  vs current {(v - ref).mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]")


if __name__ == "__main__":
    main()
