"""Are the top principal components "common factors"? Checking the interpretation as factor neutralization.

Observation: removing the top 30 within-subject PCA components from the video graph features gives LB 0.85491 -> 0.87059.
But the activity purity of neighbours **drops** 0.73 -> 0.56, so purity could not explain it.

Hypothesis (in the framework of quant factor models):
  top PCs = the recording's **common factors** (lighting, exposure, scene, slow appearance drift).
  Large variance, but they do not distinguish windows cross-sectionally. Cosine similarity is dominated by high-variance directions, so
  neighbours in the raw graph become "temporally adjacent windows". Neutralizing leaves the **specific component**,
  which can link windows of the same segment that are far apart in time.

  This also explains why performance rises while purity drops:
  raw purity is inflated by the trivial agreement "adjacent seconds are of course the same activity",
  and after neutralization it measures **genuine activity similarity across segments**. Propagation needs the latter
  (activity segments are scattered, median 32 s).

What is measured:
  (1) per component: "variance / correlation with time / activity discriminability" -> are the top ones common factors
  (2) neighbour purity conditioned on time distance                                  -> is there inflation from trivial agreement
  (3) temporal spread of neighbours                                                  -> do they cross segments
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
from refine import drop_top_pcs

PREP = WORK / "prep"


def load(seed=SEED):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile))
    ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    tsec = meta["start"].to_numpy()[tile] / 50.0
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    F = {s: np.asarray(vid[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    return F, y, sbj, tsec


def eta2(proj, y):
    """Discriminability for the activity label: between-group variance / total variance. 0 means it does not distinguish activities at all."""
    tot = proj.var()
    if tot < 1e-12:
        return 0.0
    gm = np.array([proj[y == c].mean() for c in np.unique(y)])
    gn = np.array([(y == c).sum() for c in np.unique(y)], float)
    between = (gn * (gm - proj.mean()) ** 2).sum() / len(proj)
    return float(between / tot)


def comp_stats(F, y, sbj, tsec, ncomp):
    """Per component, variance ratio / R^2 with time / activity eta^2 (mean over subjects)."""
    V, T, E = [], [], []
    for s in np.unique(sbj):
        m = sbj == s
        c = F[s] - F[s].mean(0)
        _, sv, Vt = np.linalg.svd(c, full_matrices=False)
        k = min(ncomp, len(sv))
        var = sv[:k] ** 2 / (sv ** 2).sum()
        t = tsec[m] - tsec[m].mean()
        pr = c @ Vt[:k].T                                  # (n, k)
        r2 = np.array([np.corrcoef(pr[:, i], t)[0, 1] ** 2 for i in range(k)])
        e2 = np.array([eta2(pr[:, i], y[m]) for i in range(k)])
        V.append(var); T.append(np.nan_to_num(r2)); E.append(e2)
    n = min(len(v) for v in V)
    return (np.mean([v[:n] for v in V], 0),
            np.mean([t[:n] for t in T], 0),
            np.mean([e[:n] for e in E], 0))


def purity_by_time(F, y, tsec, k, gaps):
    """Neighbour purity conditioned on time distance, and the median time difference of neighbours."""
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(Fn) - 2)
    nn = np.argpartition(-S, kk, axis=1)[:, :kk]
    same = y[nn] == y[:, None]
    dt = np.abs(tsec[nn] - tsec[:, None])
    out = {"all": float(same.mean()), "dt_med": float(np.median(dt))}
    for g in gaps:
        m = dt > g
        out[f">{g}s"] = float(same[m].mean()) if m.sum() > 0 else float("nan")
        out[f"n>{g}s"] = float(m.mean())
    return out


def main(a):
    F, y, sbj, tsec = load()
    us = np.unique(sbj)

    var, tr2, e2 = comp_stats(F, y, sbj, tsec, a.ncomp)
    print("=== (1) Properties per component (mean over 22 subjects) ===")
    print(f"  {'component':>8s}{'var ratio':>10s}{'R^2 w/ time':>13s}{'act eta^2':>12s}")
    for i in [0, 1, 2, 4, 9, 19, 29, 49, 99]:
        if i < len(var):
            print(f"  {i+1:8d}{var[i]:10.4f}{tr2[i]:13.4f}{e2[i]:12.4f}")
    for lo, hi, nm in [(0, 30, "top 1-30 (removed)"), (30, 100, "31-100"),
                       (100, len(var), f"101-{len(var)}")]:
        if lo < len(var):
            sl = slice(lo, min(hi, len(var)))
            print(f"  {nm:22s} var {var[sl].sum():.3f}  "
                  f"time R^2 {tr2[sl].mean():.4f}  act eta^2 {e2[sl].mean():.4f}")

    print(f"\n=== (2)(3) Neighbour purity conditioned on time distance (k={a.k}) ===")
    print(f"  {'PCs removed':>8s}{'all':>9s}{'median dt':>13s}" +
          "".join(f"{'>'+str(g)+'s purity':>14s}" for g in a.gaps) +
          "".join(f"{'>'+str(g)+'s share':>14s}" for g in a.gaps))
    for m in a.drop_pcs:
        rows = [purity_by_time(drop_top_pcs(F[s], m), y[sbj == s], tsec[sbj == s],
                               a.k, a.gaps) for s in us]
        agg = {kk: np.nanmean([r[kk] for r in rows]) for kk in rows[0]}
        print(f"  {m:8d}{agg['all']:9.4f}{agg['dt_med']:13.1f}" +
              "".join(f"{agg[f'>{g}s']:14.4f}" for g in a.gaps) +
              "".join(f"{agg[f'n>{g}s']:14.4f}" for g in a.gaps))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--ncomp", type=int, default=200)
    p.add_argument("--drop-pcs", nargs="*", type=int, default=[0, 10, 30, 50])
    p.add_argument("--gaps", nargs="*", type=int, default=[10, 60])
    main(p.parse_args())
