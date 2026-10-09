"""(1) Graph ensembles and (2) recalibration of the structural parameters.

Why now:
  - Almost all of the system's value is in the refinement, yet **that is the one part not diversified**.
    The geometric mean of video "sources" (head x center) gives +0.0017, but the graph **geometry** is still a single one.
  - k (15-80), number of stages (2-4) and alpha were all **tuned only on the 09-03 configuration (OOF 0.78)**.
    Since then top-30 PC removal (+0.0087) and per-stage injection (+0.0093) were added, and the graph geometry changed
    (neighbour activity purity 0.73 -> 0.63). Follow the principle: when a component changes, recalibrate downstream.
    This is not "unverified" but "**stale**".

Where we measure:
  multistage_at(P, F, V, stages, k, alpha, iters, temp, g, where="every") —
  The production path including video injection. Not run_cell (refinement only).

kill criteria:
  (2) If no point beats the current point (k=30, alpha=0.75, iters=5, stages=[4,4]), stop the recalibration.
  (1) If the mean of 3 configurations does not beat the best single one on OOF, stop the ensemble.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED
from postprocess import tune_tau
from probe_invariance import drop_top_pcs, load_all
from refine import multistage_at


def run(P, V, Fd, sbj, m, k, stages, alpha, iters, temp, g):
    """Run the production path once with m principal components removed + the given parameters, and return Q concatenated over subjects."""
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        msk = sbj == s
        F = drop_top_pcs(Fd[s], m) if m else Fd[s]
        Q[msk] = multistage_at(P[msk], F, V[msk], stages, k=k, alpha=alpha,
                               iters=iters, temp=temp, g=g, where="every")
    return Q


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    base, ens = {}, {}
    for sd in a.seeds:
        P, V, y, sbj, sens, start, Fv, _, _ = load_all(a.tags, w, a.vbase, seed=sd)

        # ---------- (2) recalibration of the structural parameters (on the current configuration m=30, g=0.2)
        cur = dict(k=a.k, stages=a.stages, alpha=a.alpha, iters=a.iters)
        grid = [("k", kk, dict(cur, k=kk)) for kk in a.ks] \
             + [("alpha", a2, dict(cur, alpha=a2)) for a2 in a.alphas] \
             + [("iters", it, dict(cur, iters=it)) for it in a.iters_list] \
             + [("stages", len(st), dict(cur, stages=st)) for st in
                [[4.0], [4.0, 4.0], [4.0] * 3, [4.0] * 4]]
        for nm, val, kw in grid:
            Q = run(P, V, Fv, sbj, a.m, kw["k"], kw["stages"], kw["alpha"],
                    kw["iters"], a.temp, a.g)
            base.setdefault((nm, val), []).append(tune_tau(Q, y)[0])

        # ---------- (1) graph ensembles
        members = {}
        for m, k in a.members:
            members[(m, k)] = run(P, V, Fv, sbj, m, k, a.stages, a.alpha,
                                  a.iters, a.temp, a.g)
            ens.setdefault(f"single m={m} k={k}", []).append(
                tune_tau(members[(m, k)], y)[0])
        keys = list(members)
        for r in [2, 3, len(keys)]:
            if r > len(keys):
                continue
            for comb in itertools.combinations(keys, r):
                Qa = np.mean([members[c] for c in comb], 0)
                lg = np.mean([np.log(members[c] + 1e-9) for c in comb], 0)
                Qg = np.exp(lg); Qg /= Qg.sum(1, keepdims=True)
                tag = "+".join(f"({m},{k})" for m, k in comb)
                ens.setdefault(f"arith mean {tag}", []).append(tune_tau(Qa, y)[0])
                ens.setdefault(f"geo mean {tag}", []).append(tune_tau(Qg, y)[0])
        print(f"  seed {sd} done", flush=True)

    print(f"\n=== (2) Recalibration of the structural parameters (m={a.m}, g={a.g}, {len(a.seeds)} seeds) ===")
    print(f"  current point: k={a.k} alpha={a.alpha} iters={a.iters} stages={len(a.stages)}")
    last = None
    for (nm, val), v in base.items():
        if nm != last:
            print(f"  --- {nm} ---"); last = nm
        cur_mark = " <- current" if (nm, val) in [("k", a.k), ("alpha", a.alpha),
                                              ("iters", a.iters), ("stages", len(a.stages))] else ""
        print(f"    {val:<8} {np.mean(v):.4f}{cur_mark}")

    print(f"\n=== (1) Graph ensembles ===")
    for nm, v in sorted(ens.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {nm:34s} {np.mean(v):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--vbase", required=True)
    p.add_argument("--m", type=int, default=30)
    p.add_argument("--g", type=float, default=0.2)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--ks", nargs="*", type=int, default=[15, 20, 30, 45, 60])
    p.add_argument("--alphas", nargs="*", type=float, default=[0.65, 0.75, 0.85])
    p.add_argument("--iters-list", nargs="*", type=int, default=[3, 5, 8])
    p.add_argument("--members", nargs="*",
                   type=lambda s: tuple(int(x) for x in s.split(",")),
                   default=[(30, 30), (0, 30), (50, 30), (30, 15), (30, 60)])
    p.add_argument("--seeds", nargs="*", type=int, default=[SEED, 7])
    main(p.parse_args())
