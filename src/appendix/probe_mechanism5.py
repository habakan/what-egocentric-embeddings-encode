"""Mechanism of the removal gain, part 5: are the graph's wrong edges replaced from "partners hard for the classifier to tell apart" to "easy-to-tell partners"? (H5)

Attach to each edge joining different activities (i -> j, y_i != y_j) the pairwise confusability C[y_i, y_j].
C is the symmetrised confusion rate of P over all participants (evaluation windows of this seed): (rate a-windows predicted as b + rate b-windows predicted as a) / 2.
What is measured (stage-1 graph and rebuilt graph, class mean excluding NULL):
  hard edges: per window, Σ_{neighbours of a different activity} C / number of neighbours (weight of hard-to-tell wrong edges)
  easy edges: no quantity equivalent to (fraction of different-activity edges - hard edges) is reported; instead the mean C of different-activity edges is reported
H5 prediction: with both top-30 removal and residual, the stage-1 "mean C of different-activity edges" goes down (not with the permutation control).
  The per-class reduction in "hard edges" predicts the F1 gain.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from postprocess import prior_correct, tune_tau
from probe_limbpred import oof
from probe_mechanism4 import cls_mean, per_class_f1
from probe_wear_encoders import IMG, PUB2REC, eval_tiles
from refine_gpu import _t, drop_top_pcs, knn_graph, propagate, sharpen


def confusion(P, y):
    pa = P.argmax(1)
    M = np.zeros((N_CLASSES, N_CLASSES))
    for a in np.unique(y):
        M[a] = np.bincount(pa[y == a], minlength=N_CLASSES) / (y == a).sum()
    C = (M + M.T) / 2
    np.fill_diagonal(C, 0)
    return C


def edge_stats(nn, w, y, C):
    keep = (w > -1e8).cpu().numpy(); nn = nn.cpu().numpy()
    yn = y[nn]
    crossm = (yn != y[:, None]) & keep
    cnt = keep.sum(1).clip(1)
    Ce = C[y[:, None], yn] * crossm
    hard = Ce.sum(1) / cnt
    meanC = np.where(crossm.sum(1) > 0, Ce.sum(1) / crossm.sum(1).clip(1), np.nan)
    return hard, meanC, crossm.sum(1) / cnt


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower() / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2"]}
    d = oof(img)
    pos_of = {t: i for i, t in enumerate(d["tile"])}
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        tile, P, meta = eval_tiles(sd)
        use = np.array([t in pos_of for t in tile]); pos = np.array([pos_of[t] for t in tile[use]])
        y = meta["label"].to_numpy()[tile][use]; sbj = meta["sbj_id"].to_numpy()[tile][use]; P = P[use]
        C = confusion(P, y)
        rng = np.random.RandomState(sd)
        for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
            Yc, H = d[f"Yc_{e}"][pos], d[f"H_{e}"][pos]
            for nm in ["raw", "top-30 removed", "residual", "control"]:
                Q = np.empty_like(P); acc = collections.defaultdict(list); yy = []
                for s in np.unique(sbj):
                    ms = sbj == s
                    X = Yc[ms] if nm in ("raw", "top-30 removed") else Yc[ms] - (H[ms] if nm == "residual" else H[ms][rng.permutation(ms.sum())])
                    f = _t(X); f = drop_top_pcs(f, 30) if nm == "top-30 removed" else f
                    f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
                    nn, w = knn_graph(f, 30)
                    for k, v in zip(["hard1", "meanC1", "diff1"], edge_stats(nn, w, y[ms], C)): acc[k].append(v)
                    Qs = sharpen(propagate(_t(P[ms]), nn, w, 0.75, 5), 1.4)
                    for _ in range(2):
                        nn, w = knn_graph(torch.cat([f, 4.0 * torch.sqrt(Qs)], 1), 30)
                        Qs = sharpen(propagate(Qs, nn, w, 0.75, 5), 1.4)
                    for k, v in zip(["hard3", "meanC3", "diff3"], edge_stats(nn, w, y[ms], C)): acc[k].append(v)
                    Q[ms] = Qs.cpu().numpy(); yy.append(y[ms])
                yy = np.concatenate(yy)
                f1, tau = tune_tau(Q, y)
                row = {k: cls_mean(np.concatenate(v), yy) for k, v in acc.items()}
                row["F1"] = f1; row["clsF1"] = per_class_f1(y, prior_correct(Q, tau).argmax(1))
                R[(e, nm)].append(row)
        print(f"seed {sd} done", flush=True)
    act = list(range(1, N_CLASSES))
    avg = lambda e, nm, k: np.nanmean([r[k] for r in R[(e, nm)]], 0)
    ks = ["diff1", "meanC1", "hard1", "diff3", "meanC3", "hard3"]
    print("\n=== Class mean excluding NULL (2 seeds) ===")
    print(f"  {'':10s} {'':8s} {'F1':>7s} " + " ".join(f"{k:>7s}" for k in ks))
    for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
        for nm in ["raw", "top-30 removed", "residual", "control"]:
            print(f"  {e:10s} {nm:8s} {avg(e, nm, 'F1'):7.4f} " + " ".join(f"{np.nanmean(avg(e, nm, k)[act]):7.4f}" for k in ks))
    print("\n=== Per class: Spearman with the F1 gain (top-30 removed - raw / residual - raw) ===")
    for nm in ["top-30 removed", "residual"]:
        for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
            df = (avg(e, nm, "clsF1") - avg(e, "raw", "clsF1"))[act]
            out = []
            for k in ["diff1", "hard1", "diff3", "hard3"]:
                dk = -(avg(e, nm, k) - avg(e, "raw", k))[act]
                ok = np.isfinite(dk) & np.isfinite(df)
                out.append(f"{k} drop {spearmanr(dk[ok], df[ok])[0]:+.2f}")
            print(f"  {nm:8s} {e:10s} " + " | ".join(out))


if __name__ == "__main__":
    main()
