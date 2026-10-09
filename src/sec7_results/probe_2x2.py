"""Controlled mechanism experiment: is "use different modalities for the base and the graph" a symmetric principle?

The paper draft claims the following mechanism:
  mixing video into the base hurts because the base's errors become correlated with the graph, which loses its corrective power.

But there is a competing hypothesis:
  test has one sensor per window, so windows from the same segment come from different limbs. Inertial features from
  different limbs do not look alike, so maybe **inertial features simply cannot build a good affinity graph** in the first place.

The existing measurement (base=inertial x graph=inertial is bad) is consistent with both, so it does not separate them.
Separate them with a 2x2:

                graph=inertial   graph=video
  base=inertial      A             B (current)
  base=video         C  <- key     D

  - both A and C bad  -> the inertial graph itself is bad. **The correlation hypothesis is wrong**
  - A bad but C good -> the correlation hypothesis holds and the principle is **symmetric**

We also measure the graph's own quality directly with labels (analysis only; not used in the pipeline):
  neighbour label purity = fraction of each window's k nearest neighbours whose true label matches.
  If this is video >> inertial, the evidence favours the competing hypothesis.
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
from config import SEED, WORK
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import tune_tau
from refine import multistage

PREP = WORK / "prep"


def load_cells(tags, weights, vbase_tag, seed=SEED):
    """Build the 2x2 inputs on the same row selection."""
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile))
    ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    rows = tile + sens * n

    # base 1: inertial ensemble
    P = None
    for t, w in zip(tags, weights):
        o = np.load(WORK / t / "oof.npy")[rows]
        s = o.sum(1, keepdims=True)
        o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(weights)
    keep = P.sum(1) > 1e-6
    tile, sens, rows, P = tile[keep], sens[keep], rows[keep], P[keep]
    P /= P.sum(1, keepdims=True)

    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]

    # base 2: video-only model
    Vd = load_vgraph_oof(vbase_tag, tags[0], seed=seed)
    V = np.empty_like(P)
    for s in np.unique(sbj):
        V[sbj == s] = Vd[s]

    # graph source 1: video features (15-frame mean, 768-dim)
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    Fv = {s: np.asarray(vid[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}

    # graph source 2: inertial features (78-dim hand-crafted). Scales differ by orders of magnitude, so z-score within subject
    fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    Fi = {}
    for s in np.unique(sbj):
        a = np.asarray(fi[rows[sbj == s]], np.float32)
        a = np.nan_to_num(a)
        Fi[s] = (a - a.mean(0)) / (a.std(0) + 1e-6)
    return P, V, y, sbj, Fv, Fi, sens


def neighbor_stats(F, y, sens, k):
    """Directly measure what the graph groups windows by.

    label : fraction of k nearest neighbours whose true label matches (= does it group by activity)
    sensor: fraction of k nearest neighbours with the same sensor position (= does it group by a nuisance)
            Under test-like conditions the sensor is uniformly random, so chance is 0.25.
            A graph far above 0.25 is looking at "wearing position, not activity".
    """
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(Fn) - 2)
    nn = np.argpartition(-S, kk, axis=1)[:, :kk]
    return (float((y[nn] == y[:, None]).mean()),
            float((sens[nn] == sens[:, None]).mean()))


def run_cell(B, F, sbj, stages, k, alpha, iters, temp):
    Q = np.empty_like(B)
    for s in np.unique(sbj):
        m = sbj == s
        Q[m] = multistage(B[m], F[s], stages, k=k, alpha=alpha, iters=iters, temp=temp)
    return Q


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    res = {}
    pur = {"video": [], "inertial": []}
    sen = {"video": [], "inertial": []}
    for sd in a.seeds:
        P, V, y, sbj, Fv, Fi, sens = load_cells(a.tags, w, a.vbase, seed=sd)
        for gname, F in [("video", Fv), ("inertial", Fi)]:
            st = [neighbor_stats(F[s], y[sbj == s], sens[sbj == s], a.k)
                  for s in np.unique(sbj)]
            pur[gname].append(np.mean([x[0] for x in st]))
            sen[gname].append(np.mean([x[1] for x in st]))
        for bname, B in [("inertial", P), ("video", V)]:
            res.setdefault((bname, "none"), []).append(tune_tau(B, y)[0])
            for gname, F in [("inertial", Fi), ("video", Fv)]:
                Q = run_cell(B, F, sbj, a.stages, a.k, a.alpha, a.iters, a.temp)
                res.setdefault((bname, gname), []).append(tune_tau(Q, y)[0])
        print(f"  seed {sd} done", flush=True)

    print(f"\nn seeds={len(a.seeds)}  k={a.k} alpha={a.alpha} iters={a.iters} "
          f"stages={a.stages} temp={a.temp}")
    print("\n=== What the graph groups by (analysis only) ===")
    print(f"  {'graph src':10s}{'label match':>12s}{'sensor match':>12s}  (sensor chance = 0.25)")
    for gname in ("video", "inertial"):
        v, s2 = np.array(pur[gname]), np.array(sen[gname])
        print(f"  {gname:10s}{v.mean():12.4f}{s2.mean():12.4f}")

    print("\n=== 2x2: macro-F1 after refinement ===")
    hdr = f"  {'':14s}{'no graph':>12s}{'graph=inertial':>14s}{'graph=video':>14s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for bname in ("inertial", "video"):
        cells = []
        for gname in ("none", "inertial", "video"):
            v = np.array(res[(bname, gname)])
            cells.append(f"{v.mean():.4f}")
        print(f"  base={bname:4s}  {cells[0]:>12s}{cells[1]:>14s}{cells[2]:>14s}")

    print("\n=== Effect (difference from no graph) ===")
    for bname in ("inertial", "video"):
        b0 = np.mean(res[(bname, "none")])
        for gname in ("inertial", "video"):
            d = np.mean(res[(bname, gname)]) - b0
            same = " (same modality)" if gname == bname else " (different modality)"
            print(f"  base={bname} + graph={gname}: {d:+.4f}{same}")

    import json
    out = Path(__file__).parents[2] / "paper" / "figdata" / "twoxtwo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "k": a.k, "seeds": a.seeds,
        "purity": {g: float(np.mean(pur[g])) for g in pur},
        "sensor": {g: float(np.mean(sen[g])) for g in sen},
        "cells": {f"{b}|{g}": {"mean": float(np.mean(res[(b, g)])),
                               "std": float(np.std(res[(b, g)]))}
                  for (b, g) in res},
    }, ensure_ascii=False, indent=1))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--vbase", default="exp020_video_only")
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123])
    main(p.parse_args())
