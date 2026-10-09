"""Actively test the §7 criterion: does removing the nuisance from the affinity source flip the sign?

The current evidence is only negative: "the inertial graph groups by placement, so it hurts".
If the criterion is causal, **an inertial graph with the placement component removed should no longer hurt**.
Sensor placement is known at test time too (sensor_location in test_meta), so this can actually be run.

Experiment A: z-score the inertial features per (subject, placement) before building the graph.
  This removes the per-placement mean and variance, so placement-driven clusters disappear.
  -> If placement purity drops from 0.60 to around 0.25 and the pipeline contribution improves, causality is confirmed.

Experiment B: aim the same criterion at the video graph. Video is invariant to placement, but not
  necessarily to lighting/appearance drift within a recording. Test order is unknown, so temporal smoothing cannot be used,
  but removing the top components of a within-subject PCA is order-free and can be applied.
  The plan's exp012 (per-recording standardization) failed as an operation on the **classifier input**,
  but this is an operation on the **affinity**, which is exactly the "placement" difference this paper is about.
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
from postprocess import tune_tau
from probe_2x2 import neighbor_stats, run_cell

PREP = WORK / "prep"


def load_all(tags, weights, vbase_tag, seed=SEED):
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
    tile, sens, rows, P = tile[keep], sens[keep], rows[keep], P[keep]
    P /= P.sum(1, keepdims=True)
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    start = meta["start"].to_numpy()[tile]          # time within the recording (for analysis)

    Vd = load_vgraph_oof(vbase_tag, tags[0], seed=seed)
    V = np.empty_like(P)
    for s in np.unique(sbj):
        V[sbj == s] = Vd[s]

    # Default is head; switch with the WEAR_VIDEO environment variable.
    # The array matching the official test setup is "sync" (starts at the 8th frame from the window start).
    import os
    vname = os.environ.get("WEAR_VIDEO", "head")
    vid = np.load(PREP / f"video_raw_{vname}.npy", mmap_mode="r")
    Fv = {s: np.asarray(vid[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    Fi_raw = {s: np.nan_to_num(np.asarray(fi[rows[sbj == s]], np.float32))
              for s in np.unique(sbj)}
    return P, V, y, sbj, sens, start, Fv, Fi_raw, tile


def zscore(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


def zscore_per_group(a, g):
    """z-score per group. Group = the nuisance value (here, sensor placement)."""
    out = np.empty_like(a)
    for v in np.unique(g):
        m = g == v
        out[m] = zscore(a[m]) if m.sum() > 2 else 0.0
    return out


def drop_top_pcs(a, m):
    """Remove the top m components of a within-subject PCA. Uses no ordering, so it can run on test too."""
    if m <= 0:
        return a
    c = a - a.mean(0)
    # SVD rather than an eigendecomposition of the covariance (fast enough for 768 dims, a few thousand samples)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    B = Vt[:m]
    return c - (c @ B.T) @ B


def drop_group_subspace(a, g):
    """Project out the subspace spanned by the group (placement) centroids.

    Per-dimension z-scoring only removes the 1st and 2nd moments. If placement shows up as "which axis dominates",
    removing the directions that linearly predict placement is more direct.
    With 4 placements the centroids span a subspace of at most 3 dimensions.
    """
    c = a - a.mean(0)
    cents = np.stack([c[g == v].mean(0) for v in np.unique(g) if (g == v).sum() > 2])
    cents = cents - cents.mean(0)
    q, _ = np.linalg.qr(cents.T)            # (d, r) orthonormal basis
    return c - (c @ q) @ q.T


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    accA, accB, statA, statB = {}, {}, {}, {}
    for sd in a.seeds:
        P, V, y, sbj, sens, start, Fv, Fi_raw, _ = load_all(a.tags, w, a.vbase, seed=sd)
        us = np.unique(sbj)

        # ---------- Experiment A: remove sensor placement from the inertial graph
        variants = {
            "inertial (within-subject z)": {s: zscore(Fi_raw[s]) for s in us},
            "inertial (per-placement z)": {s: zscore_per_group(Fi_raw[s], sens[sbj == s]) for s in us},
            "inertial (placement subspace removed)": {s: drop_group_subspace(zscore(Fi_raw[s]), sens[sbj == s])
                                for s in us},
            "video (current)": Fv,
        }
        for nm, F in variants.items():
            st = [neighbor_stats(F[s], y[sbj == s], sens[sbj == s], a.k) for s in us]
            statA.setdefault(nm, []).append((np.mean([x[0] for x in st]),
                                             np.mean([x[1] for x in st])))
        for bnm, B in [("inertial base", P), ("video base", V)]:
            accA.setdefault((bnm, "none"), []).append(tune_tau(B, y)[0])
            for nm, F in variants.items():
                Q = run_cell(B, F, sbj, a.stages, a.k, a.alpha, a.iters, a.temp)
                accA.setdefault((bnm, nm), []).append(tune_tau(Q, y)[0])

        # ---------- Experiment B: remove top PCs from the video graph
        for m in a.drop_pcs:
            F = {s: drop_top_pcs(Fv[s], m) for s in us}
            st = [neighbor_stats(F[s], y[sbj == s], sens[sbj == s], a.k) for s in us]
            # Temporal drift indicator: median absolute difference of within-recording time to neighbours (smaller = clustered by time)
            td = []
            for s in us:
                Fs = F[s] - F[s].mean(0)
                Fs = Fs / (np.linalg.norm(Fs, axis=1, keepdims=True) + 1e-8)
                S = Fs @ Fs.T
                np.fill_diagonal(S, -np.inf)
                kk = min(a.k, len(Fs) - 2)
                nn = np.argpartition(-S, kk, axis=1)[:, :kk]
                t0 = start[sbj == s] / 50.0                       # seconds
                td.append(np.median(np.abs(t0[nn] - t0[:, None])))
            statB.setdefault(m, []).append((np.mean([x[0] for x in st]),
                                            np.mean([x[1] for x in st]), np.mean(td)))
            Q = run_cell(P, F, sbj, a.stages, a.k, a.alpha, a.iters, a.temp)
            accB.setdefault(m, []).append(tune_tau(Q, y)[0])
        print(f"  seed {sd} done", flush=True)

    print(f"\n=== Experiment A: does removing sensor placement from the affinity source flip the sign? ===")
    print(f"  {'graph source':22s}{'act. purity':>10s}{'plc. purity':>10s}   (placement chance 0.25)")
    for nm in statA:
        v = np.array(statA[nm])
        print(f"  {nm:22s}{v[:,0].mean():10.4f}{v[:,1].mean():10.4f}")
    print(f"\n  {'':22s}{'inertial base':>14s}{'video base':>14s}   (diff vs no graph)")
    for nm in list(statA):
        row = []
        for bnm in ("inertial base", "video base"):
            d = np.mean(accA[(bnm, nm)]) - np.mean(accA[(bnm, "none")])
            row.append(f"{d:+.4f}")
        print(f"  {nm:22s}{row[0]:>14s}{row[1]:>14s}")

    print(f"\n=== Experiment B: remove top PCs from the video graph (base = inertial) ===")
    base = np.mean(accB[a.drop_pcs[0]]) if 0 in a.drop_pcs else None
    print(f"  {'PCs removed':>10s}{'act. purity':>10s}{'plc. purity':>10s}{'nbr time diff (s)':>18s}"
          f"{'macro-F1':>11s}{'diff vs m=0':>12s}")
    for m in a.drop_pcs:
        v = np.array(statB[m]); f = np.array(accB[m])
        d = f.mean() - base if base is not None else float("nan")
        print(f"  {m:10d}{v[:,0].mean():10.4f}{v[:,1].mean():10.4f}{v[:,2].mean():18.1f}"
              f"{f.mean():11.4f}{d:+12.4f}")


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
    p.add_argument("--drop-pcs", nargs="*", type=int, default=[0, 1, 2, 3, 5, 10])
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123])
    main(p.parse_args())


