"""Implement the "adaptive graph fusion" prescribed by the multi-view literature and compare it with the fixed choice.

Why this is needed (literature survey 09-23):
  Modern multi-view semi-supervised methods handle which view to build the graph from
  via **adaptive weight learning** (the adaptive graph fusion family). We fixed the video graph,
  tried only adding inertial edges with fixed weights, and saw monotonic degradation. The learned version is untested, and
  reviewers will surely ask "why not learn the weights". Close that gap.

Objective from the literature (needs no labels, so usable as-is in the transductive setting):
  min_Q,alpha  sum_v alpha_v * tr(Q^T L_v Q) + reg(alpha),   sum alpha_v = 1
  Solve alternately: fix Q and update alpha, fix alpha and propagate.
  Under L2-type regularization, alpha_v is proportional to the inverse of the smoothness:
      alpha_v ∝ (1 / s_v)^(1/(r-1)),  s_v = tr(Q^T L_v Q)

Prediction (if this fails, the literature's prescription was right):
  Smoothness s_v measures nearly the same thing as neighbour purity. In this system purity predicts **in the wrong direction**
  (§diss3: operations that lower purity raise the score). So choosing weights by smoothness should miss.
  If correct, this is the third instance of "obvious graph-quality metrics mislead".

Numbers reported:
  - converged alpha (video vs inertial)
  - the resulting macro-F1 vs a sweep of fixed alpha
  - **the smoothness s_v of each view itself** — whether the criterion orders the views correctly
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED
from postprocess import tune_tau
from probe_invariance import drop_top_pcs, load_all, zscore
from refine import knn_graph, sharpen


def norm_rows(nn, w, sharp=10.0):
    ww = np.exp((w - w.max(1, keepdims=True)) * sharp)
    return ww / (ww.sum(1, keepdims=True) + 1e-12)


def smoothness(Q, nn, ww):
    """tr(Q^T L Q) = sum_ij W_ij ||Q_i - Q_j||^2 / 2. Uses the row-normalized W."""
    d = ((Q[:, None, :] - Q[nn]) ** 2).sum(-1)          # (n, k)
    return float((ww * d).sum() / 2.0)


def fuse_propagate(P, graphs, alpha_v, alpha, iters):
    """Propagate with a convex combination of the row-stochastic matrices of multiple views."""
    Q = P.copy()
    for _ in range(iters):
        agg = np.zeros_like(Q)
        for (nn, ww), av in zip(graphs, alpha_v):
            if av > 0:
                agg += av * (Q[nn] * ww[:, :, None]).sum(1)
        Q = (1 - alpha) * P + alpha * agg
    return Q


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    res, alphas, smooths = {}, [], []
    for sd in a.seeds:
        P, V, y, sbj, sens, start, Fv, Fi_raw, _ = load_all(a.tags, w, a.vbase, seed=sd)
        us = np.unique(sbj)
        Qfix = {av: np.empty_like(P) for av in a.fixed}
        Qada = np.empty_like(P)
        for s in us:
            m = sbj == s
            Fvs = drop_top_pcs(Fv[s], a.m) if a.m else Fv[s]
            Fvs = Fvs - Fvs.mean(0); Fvs /= np.linalg.norm(Fvs, axis=1, keepdims=True) + 1e-8
            Fis = zscore(Fi_raw[s])
            Fis = Fis / (np.linalg.norm(Fis, axis=1, keepdims=True) + 1e-8)
            gv = knn_graph(Fvs, a.k, center=False)
            gi = knn_graph(Fis, a.k, center=False)
            G = [(gv[0], norm_rows(*gv)), (gi[0], norm_rows(*gi))]

            for av in a.fixed:                                   # sweep of fixed weights
                Qfix[av][m] = sharpen(fuse_propagate(P[m], G, [av, 1 - av],
                                                     a.alpha, a.iters), a.temp)
            # adaptive: solve alternately, updating alpha by the inverse smoothness
            al = np.array([0.5, 0.5])
            for _ in range(a.outer):
                Q = fuse_propagate(P[m], G, al, a.alpha, a.iters)
                sv = np.array([smoothness(Q, nn, ww) for nn, ww in G])
                inv = (1.0 / np.maximum(sv, 1e-12)) ** (1.0 / (a.r - 1))
                al = inv / inv.sum()
            Qada[m] = sharpen(Q, a.temp)
            alphas.append(al); smooths.append(sv)

        for av in a.fixed:
            res.setdefault(f"fixed video={av:.2f}", []).append(tune_tau(Qfix[av], y)[0])
        res.setdefault("adaptive (learned by smoothness)", []).append(tune_tau(Qada, y)[0])
        print(f"  seed {sd} done", flush=True)

    al = np.mean(alphas, 0); sv = np.mean(smooths, 0)
    print(f"\n=== adaptive graph fusion ({len(a.seeds)} seeds) ===")
    print(f"  learned weights: video {al[0]:.3f} / inertial {al[1]:.3f}")
    print(f"  smoothness tr(Q^T L Q): video {sv[0]:.1f} / inertial {sv[1]:.1f}"
          f"   (smaller is judged a 'better view')")
    print(f"\n  {'config':24s}{'macro-F1':>10s}")
    for nm, v in sorted(res.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {nm:24s}{np.mean(v):10.4f}")
    print("\n  Verdict: do the learned weights match the actually best fixed weights?")
    print("  If not, the smoothness criterion is wrong for this system (for the same reason as purity).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--vbase", required=True)
    p.add_argument("--fixed", nargs="*", type=float, default=[1.0, 0.9, 0.75, 0.5, 0.25, 0.0])
    p.add_argument("--m", type=int, default=30)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--r", type=float, default=2.0, help="exponent of the weight update (r>1 in the literature)")
    p.add_argument("--seeds", nargs="*", type=int, default=[SEED, 7])
    main(p.parse_args())
