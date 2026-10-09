"""Bias graph edges toward "pairs with different sensor locations".

The dose-response in `probe_base_ceiling.py` showed that averaging the base predictions of the same window across sensor locations
improves monotonically (k=1 0.8322 / k=2 0.8401 / k=3 0.8454 / k=4 0.8464).
This is privileged information (test has one sensor per window), so it cannot be used directly, but
it identifies that **what the pipeline needs is "averaging across sensor locations"**.

Graph propagation should in principle do this (a segment mixes windows from 4 sensors), but
edges are built from video similarity alone and ignore sensor location entirely.
About 1/4 of neighbours share the same sensor location, diluting the averaging effect accordingly.

So we **apply a penalty delta to same-sensor-location edges** to bias neighbour selection
toward cross-sensor pairs. sensor_location is also in test_meta, so this is usable on test.

Consistent with a related known result: "adding same-sensor inertial edges to the graph" degraded monotonically
(0.8013 -> 0.7014). Both point to same-sensor links being harmful.
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
from refine import propagate, sharpen

PREP = WORK / "prep"


def load_with_sens(tags, weights, seed=SEED):
    """Same as load_eval_set, but also returns the per-window assigned sensor location sens."""
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile))
    ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    rows = tile + sens * n

    P = None
    for t, w in zip(tags, weights):
        o = np.load(WORK / t / "oof.npy")[rows]
        s = o.sum(1, keepdims=True)
        o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(weights)
    keep = P.sum(1) > 1e-6
    tile, P, sens = tile[keep], P[keep], sens[keep]
    P /= P.sum(1, keepdims=True)
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    F = {s: np.asarray(video[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    return P, y, sbj, F, sens


def knn_graph_xs(F, k, sens, delta):
    """Subtract delta from the similarity of same-sensor-location pairs, then take the top-k."""
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = F @ F.T
    if delta != 0.0:
        S = S - delta * (sens[:, None] == sens[None, :])
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(F) - 2)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.take_along_axis(S, nn, 1)
    r = np.zeros(S.shape, bool)
    r[np.arange(len(nn))[:, None], nn] = True
    keep = r[np.arange(len(nn))[:, None], nn] & r[nn, np.arange(len(nn))[:, None]]
    return nn, np.where(keep, w, -1e9)


def multistage_xs(P, F, sens, stages, k, alpha, iters, temp, delta):
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    nn, w = knn_graph_xs(Fn, k, sens, delta)
    Q = sharpen(propagate(P, nn, w, alpha, iters), temp)
    for lam in stages:
        G = np.concatenate([Fn, lam * np.sqrt(Q)], 1)
        nn, w = knn_graph_xs(G, k, sens, delta)
        Q = sharpen(propagate(Q, nn, w, alpha, iters), temp)
    return Q


def cross_rate(F, k, sens, delta):
    """Fraction of selected neighbours whose sensor location differs from the window's own. Baseline at delta=0 is about 0.75."""
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    nn, _ = knn_graph_xs(Fn, k, sens, delta)
    return float((sens[nn] != sens[:, None]).mean())


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    P, y, sbj, F, sens = load_with_sens(a.tags, w)
    if a.vgraph:
        Vs = load_vgraph_oof(a.vgraph, a.tags[0])
        for s in F:
            f = F[s] - F[s].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            F[s] = np.concatenate([f, a.gamma * np.sqrt(Vs[s])], 1)

    print(f"n={len(y)}  k={a.k} alpha={a.alpha} iters={a.iters} temp={a.temp}\n")
    print(f"{'delta':>7s} {'cross-sensor nbr rate':>12s} {'macro-F1':>10s} {'tau':>6s}")
    print("-" * 42)
    for d in a.deltas:
        Q = np.empty_like(P)
        xr = []
        for s in np.unique(sbj):
            m = sbj == s
            Q[m] = multistage_xs(P[m], F[s], sens[m], a.stages, a.k, a.alpha,
                                 a.iters, a.temp, d)
            xr.append(cross_rate(F[s], a.k, sens[m], d))
        f, tau = tune_tau(Q, y)
        print(f"{d:7.3f} {np.mean(xr):12.4f} {f:10.4f} {tau:6.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--vgraph", default=None)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--deltas", nargs="*", type=float,
                   default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.20])
    main(p.parse_args())
