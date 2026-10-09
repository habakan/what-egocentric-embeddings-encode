"""Inject the anchor prior A into the production system.

Results of probe_anchor2:
  A alone 0.4494 / V alone 0.5255 / **agreement 0.5976** / only A correct 0.1058.
  Non-redundancy is clear. However, the rationale "A wins on rare classes" was wrong
  (rare-class mean A 0.449 vs V 0.531). The source of the complementarity has not been identified.
  Still, what we use is **the fact of complementarity**, not an explanation, so we proceed.

Three variants measured:
  1. current (no A)                      <- reference
  2. inject A **in addition to** V       Q' ∝ Q^(1-h) A^h after each stage
  3. inject A **replacing** V            tells whether the complementarity is "something different" or "part of a superset"

Kill criterion:
  If no h beats current on all seeds, idea A is dropped.
  The only remaining structural axis is then the heterogeneous graph.
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
from config import N_CLASSES, SEED, WORK
from postprocess import gate_blend, tune_tau
from probe_anchor import prep_subject
from probe_anchor2 import anchor_prior
from probe_ens2 import graph_feat, load_oof, multistage

PREP = WORK / "prep"


def ms_anchor(P, F, V, A, stages, k, alpha, iters, temp, g, h):
    """Same as multistage. After each stage, log-linearly pool V with weight g/n and A with h/n."""
    n_inj = len(stages) + 1
    gi, hi = g / n_inj, h / n_inj
    from refine import knn_graph, propagate, sharpen
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    nn, w = knn_graph(Fn, k, center=False)
    Q = sharpen(propagate(P, nn, w, alpha, iters), temp)
    if gi:
        Q = gate_blend(Q, V, gi)
    if hi:
        Q = gate_blend(Q, A, hi)
    for lam in stages:
        G = np.concatenate([Fn, lam * np.sqrt(Q)], 1)
        nn, w = knn_graph(G, k, center=False)
        Q = sharpen(propagate(Q, nn, w, alpha, iters), temp)
        if gi:
            Q = gate_blend(Q, V, gi)
        if hi:
            Q = gate_blend(Q, A, hi)
    return Q


def build_anchor(meta, tile, sbj, y, k, temp, m):
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    us = np.unique(sbj)
    Fn = {s: prep_subject(np.asarray(vid[tile[sbj == s]], np.float32).mean(1), m)
          for s in us}
    A = np.zeros((len(tile), N_CLASSES), np.float32)
    for s in us:
        others = [o for o in us if o != s]
        B = np.concatenate([Fn[o] for o in others])
        yb = np.concatenate([y[sbj == o] for o in others])
        A[sbj == s] = anchor_prior(Fn[s], B, yb, k, temp)
    return A


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    res = {}
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    for sd in a.seeds:
        P, V, y, sbj, raw, Vd = load_oof(a.tags, w, a.vbase, sd)
        # rebuild tile to get the same windows as load_oof
        n = len(meta)
        valid = np.load(PREP / "valid_mask.npy")
        rng = np.random.RandomState(sd)
        t0 = np.where(meta["start"].to_numpy() % 50 == 0)[0]
        se = rng.randint(0, 4, len(t0)); ok = valid[t0, se]
        t0, se = t0[ok], se[ok]
        Pf = None
        for t, ww in zip(a.tags, w):
            o = np.load(WORK / t / "oof.npy")[t0 + se * n]
            s_ = o.sum(1, keepdims=True)
            o = np.divide(o, s_, out=np.zeros_like(o), where=s_ > 0)
            Pf = ww * o if Pf is None else Pf + ww * o
        keepm = (Pf / sum(w)).sum(1) > 1e-6
        tile = t0[keepm]
        A = build_anchor(meta, tile, sbj, y, a.anchor_k, a.anchor_temp, a.anchor_m)

        for nm, g, hs in [("current (no A)", a.g, [0.0]),
                          ("A added to V", a.g, a.hs),
                          ("A replaces V", 0.0, a.hs)]:
            for h in hs:
                if nm != "current (no A)" and h == 0.0:
                    continue
                Q = np.empty_like(P)
                for s in np.unique(sbj):
                    msk = sbj == s
                    F = graph_feat(raw[s], 0, 15, a.m, Vd[s], a.gamma)
                    Q[msk] = ms_anchor(P[msk], F, V[msk], A[msk], a.stages, a.k,
                                       a.alpha, a.iters, a.temp, g, h)
                key = nm if h == 0.0 else f"{nm} h={h}"
                res.setdefault(key, []).append(tune_tau(Q, y)[0])
        print(f"  seed {sd} done", flush=True)

    base = np.array(res["current (no A)"])
    print(f"\n=== anchor injection ({len(a.seeds)} seeds) ===")
    for nm, v in sorted(res.items(), key=lambda x: -np.mean(x[1])):
        line = f"  {np.mean(v):.4f}  {nm}"
        if nm != "current (no A)":
            d = np.array(v) - base
            line += (f"   vs current {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d)
                     + f"]  all same sign {bool(np.all(d > 0) or np.all(d < 0))}")
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--vbase", required=True)
    p.add_argument("--hs", nargs="*", type=float, default=[0.1, 0.2, 0.35])
    p.add_argument("--g", type=float, default=0.2)
    p.add_argument("--m", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--anchor-k", type=int, default=30)
    p.add_argument("--anchor-temp", type=float, default=10.0)
    p.add_argument("--anchor-m", type=int, default=0)
    p.add_argument("--seeds", nargs="*", type=int, default=[SEED, 7, 123])
    main(p.parse_args())
