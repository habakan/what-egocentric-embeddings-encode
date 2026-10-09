"""Inject video "between stages". A different position from exp013 (injected at the end, LB +0.0142).

General rule: **the same signal can change sign when its position in the pipeline changes.**
Only two positions have been tried so far:
  - **before** refinement (base blend)            -> harmful (0.7859 -> 0.7640)
  - **after** refinement (pool into final probs)  -> **+0.0142** (exp013, adopted)

Two positions remain untested:
  - inject **between** stages every time (every)
  - inject **only right after stage 1** (once)

Injecting between stages means the injected video information **also enters the next stage's graph construction**
(the graph features are [Fn, lam*sqrt(Q)], so if video enters Q, the edges change too).
Injecting at the end moves only the probabilities and leaves the edges unchanged. **Which is better is not obvious.**

Compare with the total video weight matched (to avoid repeating the confound hit in probe_blend_post):
every uses g/n at each stage; once and last use g a single time.
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
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from refine import multistage_at

PREP = WORK / "prep"


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    out = {k: [] for k in ("none", "last", "every", "once")}
    for sd in a.seeds:
        P, y, sbj, F = load_eval_set(a.tags, w, seed=sd)
        Vs = load_vgraph_oof(a.vgraph, a.tags[0], seed=sd)
        for s in F:
            f = F[s] - F[s].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            F[s] = np.concatenate([f, a.gamma * np.sqrt(Vs[s])], 1)
        # video used in later stages: geometric mean of head and center (measured to be better than either alone)
        src = []
        for tag in a.vsources:
            d = load_vgraph_oof(tag, a.tags[0], seed=sd)
            M = np.empty((len(y), 19), np.float32)
            for s in np.unique(sbj):
                M[sbj == s] = d[s]
            src.append(M)
        V = src[0] if len(src) == 1 else np.prod(src, axis=0) ** (1 / len(src))
        V = V / V.sum(1, keepdims=True)

        for where in out:
            Q = np.empty_like(P)
            for s in np.unique(sbj):
                m = sbj == s
                Q[m] = multistage_at(P[m], F[s], V[m], a.stages, a.k, a.alpha,
                                     a.iters, a.temp, a.g, where)
            out[where].append(tune_tau(Q, y)[0])

    print(f"video sources = {a.vsources}  total weight g={a.g}  seeds={a.seeds}\n")
    print(f"{'injection point':>22s}{'macro-F1':>12s}{'std':>9s}{'diff vs last':>13s}")
    print("-" * 58)
    ref = np.mean(out["last"])
    for where, nm in [("none", "no injection"), ("last", "last only (exp013)"),
                      ("once", "only after stage 1"), ("every", "every stage")]:
        v = np.array(out[where])
        print(f"{nm:>22s}{v.mean():12.4f}{v.std():9.4f}{v.mean() - ref:+13.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--vsources", nargs="+",
                   default=["exp020_video_only", "exp022_vonly_center_5f"])
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--vgraph", default="exp013_nn_video_5fold")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--g", type=float, default=0.2)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123])
    main(p.parse_args())
