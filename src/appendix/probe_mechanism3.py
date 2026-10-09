"""Mechanism of the removal gain, part 3: does it prevent wrong neighbours from "concentrating on a single rival class"? (H4)

The propagation decision is set by the margin between the correct class and the strongest rival class. Purity only looks at the correct side.
H4: the top components link specific activities sharing the same posture, so wrong neighbours concentrate on a single rival class.
    Removing them scatters wrong neighbours over many classes, so the margin rises even if purity falls.
Measured on the stage-1 graph (features only), sweeping m (per-window values averaged per class excluding NULL, then averaged):
  purity        : fraction of neighbours in the correct class
  rival         : fraction of neighbours in the most frequent wrong class
  concentration : fraction of wrong neighbours taken by the most frequent wrong class
  maj_correct   : fraction where the most frequent label among neighbours is correct
  P_maj_correct : fraction where the argmax of the neighbours' mean P is correct (self excluded)
Also lists macro-F1 with 1 stage and no rebuilding.
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
from config import N_CLASSES
from postprocess import tune_tau
from probe_mechanism import load
from probe_wear_encoders import IMG, PREP, PUB2REC
from refine_gpu import _t, drop_top_pcs, knn_graph, propagate, sharpen

MS = [0, 10, 20, 30, 50, 80, 120]


def stats(nn, w, y, P):
    keep = (w > -1e8).cpu().numpy(); nn = nn.cpu().numpy()
    n = len(y)
    H = np.zeros((n, N_CLASSES))
    np.add.at(H, (np.repeat(np.arange(n), nn.shape[1])[keep.ravel()], y[nn][keep]), 1)
    cnt = H.sum(1).clip(1)
    pur = H[np.arange(n), y] / cnt
    Hw = H.copy(); Hw[np.arange(n), y] = 0
    riv = Hw.max(1) / cnt
    conc = np.where(Hw.sum(1) > 0, Hw.max(1) / Hw.sum(1).clip(1), np.nan)
    maj = (H.argmax(1) == y) & (H.sum(1) > 0)
    Pm = np.zeros((n, N_CLASSES))
    for j in range(nn.shape[1]):
        Pm += P[nn[:, j]] * keep[:, [j]]
    pmaj = (Pm.argmax(1) == y) & (keep.sum(1) > 0)
    return dict(purity=pur, rival=riv, concentration=conc, maj_correct=maj.astype(float), P_maj_correct=pmaj.astype(float))


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower() / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2"]}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        P, y, sbj, F = load(sd, img, video)
        for e in F:
            for m in MS:
                acc = collections.defaultdict(list); Q1 = np.empty_like(P); ys = []
                for s in np.unique(sbj):
                    ms = sbj == s
                    f = _t(F[e][s]); f = drop_top_pcs(f, m) if m else f
                    f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
                    nn, w = knn_graph(f, 30)
                    Q1[ms] = sharpen(propagate(_t(P[ms]), nn, w, 0.75, 5), 1.4).cpu().numpy()
                    for k, v in stats(nn, w, y[ms], P[ms]).items():
                        acc[k].append(v)
                yy = np.concatenate([y[sbj == s] for s in np.unique(sbj)])
                row = {}
                for k, v in acc.items():
                    v = np.concatenate(v)
                    row[k] = np.nanmean([np.nanmean(v[yy == c]) for c in np.unique(yy) if c != 0])
                row["F1_1stage"] = tune_tau(Q1, y)[0]
                R[(e, m)].append(row)
        print(f"seed {sd} done", flush=True)
    keys = ["F1_1stage", "purity", "rival", "concentration", "maj_correct", "P_maj_correct"]
    for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
        print(f"\n== {e} (class mean excluding NULL, 2 seeds)")
        print("     m " + " ".join(f"{k:>10s}" for k in keys))
        for m in MS:
            rs = R[(e, m)]
            print(f"  {m:4d} " + " ".join(f"{np.mean([r[k] for r in rs]):10.3f}" for k in keys))


if __name__ == "__main__":
    main()
