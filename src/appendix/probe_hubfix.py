"""Hypothesis 10: if reducing hubness is the mechanism, explicitly reducing hubs with m=0 should give the same gain.

Measured in probe_hub.py: PC removal reduces the max in-degree from 3.3k -> 2.4k (m=30), a 27% drop.
This is the only one of the four measured quantities that moved substantially, and it is consistent with the plan's
"mutual kNN (hub suppression) gives +0.014". So the same test as for α/iters/γ applies:
**with m=0, does adding a different hub reduction reach 0.8516?**

reaches      -> hubness is the mechanism (PC removal is merely a proxy for it)
does not     -> hub reduction is a side effect, not the cause

For hub reduction we use local scaling (Zelnik-Manor & Perona).
Distances are normalised by each point's local scale sigma_i (distance to its r-th neighbour),
which suppresses points in dense regions becoming everyone's neighbour. A different mechanism from mutual kNN.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, WORK
from final_submit import load_vgraph_oof
from postprocess import gate_blend, tune_tau
from refine import drop_top_pcs, propagate, sharpen

PREP = WORK / "prep"


def knn_graph_ls(F, k, r, mutual=True):
    """Mutual kNN with local scaling. r=0 gives plain cosine (identical to current)."""
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = F @ F.T
    if r > 0:
        D = 1.0 - S
        np.fill_diagonal(D, np.inf)
        rr = min(r, len(F) - 2)
        sig = np.partition(D, rr, axis=1)[:, rr]          # distance to the r-th neighbour
        sig = np.maximum(sig, 1e-6)
        np.fill_diagonal(D, 0.0)                          # restore the diagonal inf (forgetting this makes everything NaN)
        S = -D / np.sqrt(sig[:, None] * sig[None, :])     # normalise by local scale
        lo, hi = S.min(), S.max()
        S = (S - lo) / (hi - lo + 1e-9)                   # map back to [0,1] to stay consistent with sharp
        assert np.isfinite(S).all(), "local scaling produced NaN/inf" 
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(F) - 2)
    nn = np.argpartition(-S, kk, axis=1)[:, :kk]
    w = np.take_along_axis(S, nn, 1)
    if mutual:
        rmat = np.zeros(S.shape, bool)
        rmat[np.arange(len(nn))[:, None], nn] = True
        keep = rmat[np.arange(len(nn))[:, None], nn] & rmat[nn, np.arange(len(nn))[:, None]]
        w = np.where(keep, w, -1e9)
    return nn, w


def multistage_ls(P, F, V, stages, k, alpha, iters, temp, g, r):
    Fn = F - F.mean(0); Fn /= np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8
    gi = g / (len(stages) + 1)
    nn, w = knn_graph_ls(Fn, k, r)
    Q = gate_blend(sharpen(propagate(P, nn, w, alpha, iters), temp), V, gi)
    for lam in stages:
        G = np.concatenate([Fn, lam * np.sqrt(Q)], 1)
        nn, w = knn_graph_ls(G, k, r)
        Q = gate_blend(sharpen(propagate(Q, nn, w, alpha, iters), temp), V, gi)
    return Q


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    res = {}
    for sd in a.seeds:
        valid = np.load(PREP / "valid_mask.npy")
        rng = np.random.RandomState(sd)
        tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
        sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
        tile = tile[ok]; rows = tile + sens[ok] * n
        P = None
        for t, ww in zip(a.tags, w):
            o = np.load(WORK / t / "oof.npy")[rows]; s = o.sum(1, keepdims=True)
            o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
            P = ww * o if P is None else P + ww * o
        P /= sum(w); keep = P.sum(1) > 1e-6
        tile, P = tile[keep], P[keep]; P /= P.sum(1, keepdims=True)
        y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
        vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
        F0 = {s: np.asarray(vid[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
        Vg = load_vgraph_oof("exp013_nn_video_5fold", a.tags[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            d = load_vgraph_oof(tg, a.tags[0], seed=sd); M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj): M[sbj == s] = d[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)

        for mdrop, r in a.combos:
            Q = np.empty_like(P)
            for s in np.unique(sbj):
                msk = sbj == s
                f = drop_top_pcs(F0[s], mdrop)
                fn = f - f.mean(0); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
                G = np.concatenate([fn, 0.5 * np.sqrt(Vg[s])], 1)
                Q[msk] = multistage_ls(P[msk], G, V[msk], [4.0, 4.0], a.k, a.alpha,
                                       a.iters, a.temp, 0.2, r)
            res.setdefault((mdrop, r), []).append(tune_tau(Q, y)[0])
        print(f"  seed {sd} done", flush=True)

    print(f"\n=== Can hub reduction (local scaling) replace PC removal? ===")
    print(f"  {'PCs rm':>7s}{'local scaling r':>17s}{'macro-F1':>11s}{'std':>9s}")
    for (mdrop, r) in a.combos:
        v = np.array(res[(mdrop, r)])
        tag = "current" if (mdrop, r) == (0, 0) else ("target" if (mdrop, r) == (30, 0) else "")
        print(f"  {mdrop:7d}{(r if r else '—'):>17}{v.mean():11.4f}{v.std():9.4f}  {tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123])
    a = p.parse_args()
    a.combos = [(0, 0), (30, 0), (0, 5), (0, 10), (0, 30), (30, 10)]
    main(a)
