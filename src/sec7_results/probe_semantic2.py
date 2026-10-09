"""Continuation of probe_semantic.py.
1. Overlap without cross-validation (activities form one set within a recording, so time-block CV cannot be used):
   R^2 of explaining H (four-limb prediction) by the true labels (one-hot), and R^2 of explaining the per-activity mean component C by H (within participant)
2. Do neighbours come from the same set: in the stage-1 graph (features only, mutual kNN 30), fraction of neighbours from the same set (contiguous
   segment of the same label in the same recording), fraction with the same label (purity), median time gap to neighbours (s, within the same recording).
   raw / top-30 removed / subtract H / subtract per-activity mean (diagnostic). 3 encoders, seed 42.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
ENC_LIST = os.environ.get("ENCS", "VideoMAEv2,CLIP,DINOv2").split(",")
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from probe_limbpred import oof
from probe_wear_encoders import eval_tiles
from refine_gpu import _t, drop_top_pcs, knn_graph


def r2_in(X, Y):
    pr = Ridge(alpha=1e-3).fit(X, Y).predict(X)
    Yc = Y - Y.mean(0)
    return ((Yc - (pr - Y.mean(0))) ** 2).sum(), (Yc ** 2).sum()


def main():
    d = oof(None); pos_of = {t: i for i, t in enumerate(d["tile"])}
    tile0, P, meta = eval_tiles(42)
    use = np.array([t in pos_of for t in tile0]); tile = tile0[use]; pos = np.array([pos_of[t] for t in tile])
    y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
    rec = meta["rec"].to_numpy()[tile]; sec = meta["start"].to_numpy()[tile] / 50.0
    # set id: sort by time per recording and split where the label changes (or at gaps over 5 s)
    setid = np.zeros(len(y), int); nid = 0
    for r in np.unique(rec):
        ii = np.where(rec == r)[0]; ii = ii[np.argsort(sec[ii])]
        for k, i in enumerate(ii):
            if k == 0 or y[i] != y[ii[k - 1]] or sec[i] - sec[ii[k - 1]] > 5:
                nid += 1
            setid[i] = nid
    Y1 = np.eye(N_CLASSES)[y]
    for e in ENC_LIST:
        Yc, H = d[f"Yc_{e}"][pos], d[f"H_{e}"][pos]
        a = b = c = dd = 0.0
        C = np.zeros_like(Yc)
        for s in np.unique(sbj):
            m = sbj == s
            e1, t1 = r2_in(Y1[m], H[m]); a += e1; b += t1
            C[m] = Ridge(alpha=1e-3).fit(Y1[m], Yc[m]).predict(Y1[m]) - Yc[m].mean(0)
            e2, t2 = r2_in(H[m], C[m]); c += e2; dd += t2
        print(f"\n== {e}: H explained by true labels R^2 {1 - a / b:.3f} / per-activity mean component C explained by H R^2 {1 - c / dd:.3f}"
              f" / fraction of video variance in C {(C ** 2).sum() / (Yc ** 2).sum():.3f} / fraction in H {(H ** 2).sum() / (Yc ** 2).sum():.3f}")
        print(f"  {'':18s} {'purity':>6s} {'same set':>8s} {'median dt (s)':>14s}")
        for nm in ["raw", "top-30 removed", "subtract H", "subtract per-activity mean"]:
            acc = {"pur": [], "set": [], "dt": []}
            for s in np.unique(sbj):
                m = np.where(sbj == s)[0]
                X = Yc[m] - (H[m] if nm == "subtract H" else C[m] if nm == "subtract per-activity mean" else 0)
                f = _t(X); f = drop_top_pcs(f, 30) if nm == "top-30 removed" else f
                f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
                nn, w = knn_graph(f, 30); keep = (w > -1e8).cpu().numpy(); nn = nn.cpu().numpy()
                cnt = keep.sum(1).clip(1)
                acc["pur"].append(((y[m][nn] == y[m][:, None]) & keep).sum(1) / cnt)
                acc["set"].append(((setid[m][nn] == setid[m][:, None]) & keep).sum(1) / cnt)
                same_rec = (rec[m][nn] == rec[m][:, None]) & keep
                dt = np.abs(sec[m][nn] - sec[m][:, None]); dt[~same_rec] = np.nan
                acc["dt"].append(np.nanmedian(np.where(keep, dt, np.nan), 1))
            print(f"  {nm:18s} {np.mean(np.concatenate(acc['pur'])):6.3f} {np.mean(np.concatenate(acc['set'])):8.3f} "
                  f"{np.nanmedian(np.concatenate(acc['dt'])):14.1f}")


if __name__ == "__main__":
    main()
