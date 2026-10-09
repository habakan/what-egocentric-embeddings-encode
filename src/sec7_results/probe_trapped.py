"""Were the windows fixed by removal "trapped in a cluster of a different, similar-looking activity"? (WEAR, 3 encoders, seed 42)

Continuation of probe_why_gain.py. For windows that went wrong->correct compared with raw:
  1. neighbour purity in the stage-1 graph: raw -> after removal
  2. in the raw graph, top pairs of the most frequent other class surrounding the window (true class -> most frequent other class among neighbours)
  3. fraction where the raw (wrong) prediction matches that "surrounding class" (= fraction wrong because pulled by neighbour appearance)
Conditions: top-30 removed / subtract per-activity mean (diagnostic).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
ENC_LIST = os.environ.get("ENCS", "VideoMAEv2,CLIP,DINOv2").split(",")
import collections
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES
from postprocess import prior_correct, tune_tau
from probe_limbpred import oof
from probe_why_gain import pipeline
from probe_wear_encoders import eval_tiles
from refine_gpu import _t, drop_top_pcs


def main():
    d = oof(None); pos_of = {t: i for i, t in enumerate(d["tile"])}
    tile0, P, meta = eval_tiles(42)
    use = np.array([t in pos_of for t in tile0]); tile, P = tile0[use], P[use]
    pos = np.array([pos_of[t] for t in tile])
    y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]; Y1 = np.eye(N_CLASSES)[y]
    for e in ENC_LIST:
        Yc = d[f"Yc_{e}"][pos]
        C = np.zeros_like(Yc)
        for s in np.unique(sbj):
            m = sbj == s
            C[m] = Ridge(alpha=1e-3).fit(Y1[m], Yc[m]).predict(Y1[m]) - Yc[m].mean(0)
        R = {}
        for nm in ["raw", "top-30 removed", "subtract per-activity mean"]:
            Q = np.empty_like(P); pur = np.zeros(len(y)); dom = np.zeros(len(y), int); domf = np.zeros(len(y))
            for s in np.unique(sbj):
                m = np.where(sbj == s)[0]
                X = Yc[m] - (C[m] if nm == "subtract per-activity mean" else 0)
                f = _t(X); f = drop_top_pcs(f, 30) if nm == "top-30 removed" else f
                f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
                Qs, g1, _ = pipeline(_t(P[m]), f, 2)
                Q[m] = Qs.cpu().numpy()
                nn = g1[0].cpu().numpy(); keep = (g1[1] > -1e8).cpu().numpy()
                Hc = np.zeros((len(m), N_CLASSES))
                np.add.at(Hc, (np.repeat(np.arange(len(m)), nn.shape[1])[keep.ravel()], y[m][nn][keep]), 1)
                pur[m] = Hc[np.arange(len(m)), y[m]] / Hc.sum(1).clip(1)
                Hw = Hc.copy(); Hw[np.arange(len(m)), y[m]] = -1
                dom[m] = Hw.argmax(1); domf[m] = Hw.max(1) / Hc.sum(1).clip(1)
            f1, tau = tune_tau(Q, y)
            R[nm] = dict(pred=prior_correct(Q, tau).argmax(1), pur=pur, dom=dom, domf=domf)
        b = R["raw"]
        print(f"\n===== {e} =====")
        for nm in ["top-30 removed", "subtract per-activity mean"]:
            o = R[nm]
            fixed = (b["pred"] != y) & (o["pred"] == y)
            print(f"  [{nm}] fixed windows {fixed.sum()}: neighbour purity raw {b['pur'][fixed].mean():.3f} -> after removal {o['pur'][fixed].mean():.3f}"
                  f" | fraction of surrounding other class raw {b['domf'][fixed].mean():.3f} -> {o['domf'][fixed].mean():.3f}"
                  f" | raw wrong prediction = surrounding class {(b['pred'][fixed] == b['dom'][fixed]).mean():.3f}")
            cnt = collections.Counter(zip(y[fixed], b["dom"][fixed]))
            print("    true class <- surrounding class, top: " + "; ".join(
                f"{CLASS_NAMES[a]} <- {CLASS_NAMES[c]} {n}" for (a, c), n in cnt.most_common(8)))


if __name__ == "__main__":
    main()
