"""Why does subtracting the activity-appearance component improve propagation? (WEAR, 3 encoders, seed 42)

Conditions: raw / top 30 removed / subtract the four-limb prediction H / subtract the per-activity mean C (diagnostic, uses true labels)
System: feature-only graph, propagation of P with 3 stages (2 re-links; g=0) and 1 stage (no re-link). Decision is argmax after tune_tau prior correction.
1. macro-F1 for 1 stage and 3 stages (does the gain depend on re-linking?)
2. Connectivity: in the stage-1 and stage-3 graphs, fraction of (isolated) windows with 0 mutual edges, mean degree,
   and fraction isolated among windows where P is wrong
3. Number of windows that changed "wrong->right" and "right->wrong" vs raw, and for those windows the degree, neighbour purity, and P correctness in the raw stage-1 graph.
   Top classes of the fixed windows
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_limbpred import oof
from probe_wear_encoders import eval_tiles
from refine_gpu import _t, drop_top_pcs, knn_graph, propagate, sharpen
from sklearn.linear_model import Ridge


def pipeline(P, f, stages):
    nn, w = knn_graph(f, 30); g1 = (nn, w)
    Q = sharpen(propagate(P, nn, w, 0.75, 5), 1.4)
    for _ in range(stages):
        nn, w = knn_graph(torch.cat([f, 4.0 * torch.sqrt(Q)], 1), 30)
        Q = sharpen(propagate(Q, nn, w, 0.75, 5), 1.4)
    return Q, g1, (nn, w)


def deg(g):
    return (g[1] > -1e8).sum(1).cpu().numpy()


def main():
    d = oof(None); pos_of = {t: i for i, t in enumerate(d["tile"])}
    tile0, P, meta = eval_tiles(42)
    use = np.array([t in pos_of for t in tile0]); tile, P = tile0[use], P[use]
    pos = np.array([pos_of[t] for t in tile])
    y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
    pa = P.argmax(1); Y1 = np.eye(N_CLASSES)[y]
    for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
        Yc, H = d[f"Yc_{e}"][pos], d[f"H_{e}"][pos]
        C = np.zeros_like(Yc)
        for s in np.unique(sbj):
            m = sbj == s
            C[m] = Ridge(alpha=1e-3).fit(Y1[m], Yc[m]).predict(Y1[m]) - Yc[m].mean(0)
        res = {}
        for nm in ["raw", "top30 removed", "subtract H", "subtract per-activity mean"]:
            out = {}
            for stages in [0, 2]:
                Q = np.empty_like(P); d1 = np.zeros(len(y)); d3 = np.zeros(len(y)); pur1 = np.zeros(len(y))
                for s in np.unique(sbj):
                    m = np.where(sbj == s)[0]
                    X = Yc[m] - (H[m] if nm == "subtract H" else C[m] if nm == "subtract per-activity mean" else 0)
                    f = _t(X); f = drop_top_pcs(f, 30) if nm == "top30 removed" else f
                    f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
                    Qs, g1, g3 = pipeline(_t(P[m]), f, stages)
                    Q[m] = Qs.cpu().numpy(); d1[m] = deg(g1); d3[m] = deg(g3)
                    nn = g1[0].cpu().numpy(); keep = (g1[1] > -1e8).cpu().numpy()
                    pur1[m] = ((y[m][nn] == y[m][:, None]) & keep).sum(1) / keep.sum(1).clip(1)
                f1, tau = tune_tau(Q, y)
                out[stages] = dict(f1=f1, pred=prior_correct(Q, tau).argmax(1), d1=d1, d3=d3, pur1=pur1)
            res[nm] = out
        print(f"\n===== {e} =====")
        print(f"  {'':16s} {'1-stg F1':>7s} {'3-stg F1':>7s} | {'iso 1stg':>7s} {'iso 3stg':>7s} {'iso 1stg P wrong':>12s} {'mean deg 1stg':>9s}")
        for nm, o in res.items():
            wrong = pa != y
            print(f"  {nm:16s} {o[0]['f1']:7.4f} {o[2]['f1']:7.4f} | {(o[2]['d1'] == 0).mean():7.3f} {(o[2]['d3'] == 0).mean():7.3f} "
                  f"{(o[2]['d1'][wrong] == 0).mean():12.3f} {o[2]['d1'].mean():9.1f}")
        base = res["raw"][2]
        for nm in ["top30 removed", "subtract H", "subtract per-activity mean"]:
            o = res[nm][2]
            fixed = (base["pred"] != y) & (o["pred"] == y); broke = (base["pred"] == y) & (o["pred"] != y)
            print(f"  [{nm}] wrong->right {fixed.sum()}  right->wrong {broke.sum()}  (diff {fixed.sum() - broke.sum():+d})")
            for lab, msk in [("fixed", fixed), ("broken", broke), ("all", np.ones(len(y), bool))]:
                print(f"     {lab:6s}: raw stage-1 degree {base['d1'][msk].mean():5.1f} isolated {(base['d1'][msk] == 0).mean():.3f} "
                      f"nbr purity {base['pur1'][msk].mean():.3f}  P right {(pa[msk] == y[msk]).mean():.3f}")
            cc = collections.Counter(y[fixed]); cb = collections.Counter(y[broke])
            top = sorted(set(cc) | set(cb), key=lambda c: -(cc[c] - cb[c]))[:5]
            print("     classes with most fixed-minus-broken: " + ", ".join(f"{CLASS_NAMES[c]} {cc[c] - cb[c]:+d}" for c in top))


if __name__ == "__main__":
    main()
