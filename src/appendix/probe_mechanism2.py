"""Mechanism of the removal gain, part 2 (WEAR, 3 encoders, real P and within-class shuffled P).

H1 (correlated errors) was rejected in probe_mechanism.py (the gain remains/increases even when P is shuffled).
H2 interaction with graph rebuilding: if the top components (77% of variance) remain, the spread of feature similarities is large, and the
   label block 4 sqrt(Q) added at rebuilding barely moves the neighbours. With removal, the labels drive the rebuilding.
   Predictions: (a) without rebuilding (1 stage) the removal gain disappears (b) raising λ without removal gives the same gain
         (c) the fraction of "edges linking different activities" in the rebuilt graph drops with removal (although it rises in the first stage)
H3 purity is dominated by NULL: purity averaged per class (excluding NULL) rises with removal.
The system is the same as probe_mechanism (graph from features only, g=0).
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
from postprocess import tune_tau
from probe_mechanism import load, shuffle_within_class
from probe_wear_encoders import IMG, PREP, PUB2REC
from refine_gpu import _t, drop_top_pcs, knn_graph, propagate, sharpen


def cross(nn, w, y):
    keep = (w > -1e8)
    return (((y[nn] != y[:, None]) & keep).sum() / keep.sum().clip(1)).item()


def cls_purity(nn, w, y):
    keep = (w > -1e8)
    pur = (((y[nn] == y[:, None]) & keep).sum(1).double() / keep.sum(1).clamp(min=1))
    cs = [c for c in torch.unique(y).tolist() if c != 0]
    return float(np.mean([pur[y == c].mean().item() for c in cs]))


def run(P, y, F, m, stages, lam):
    """One participant. Returns: Q and, for each stage's graph, (fraction of edges linking different activities, class-mean purity excluding NULL)."""
    f = _t(F); f = drop_top_pcs(f, m) if m else f
    f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
    yy = torch.tensor(y.astype(np.int64), device=f.device)
    P = _t(P)
    nn, w = knn_graph(f, 30)
    st = [(cross(nn, w, yy), cls_purity(nn, w, yy))]
    Q = sharpen(propagate(P, nn, w, 0.75, 5), 1.4)
    for _ in range(stages):
        nn, w = knn_graph(torch.cat([f, lam * torch.sqrt(Q)], 1), 30)
        st.append((cross(nn, w, yy), cls_purity(nn, w, yy)))
        Q = sharpen(propagate(Q, nn, w, 0.75, 5), 1.4)
    return Q.cpu().numpy(), st


CONFIGS = [("3-stage λ4 m0", 2, 4.0, 0), ("3-stage λ4 m30", 2, 4.0, 30),
           ("1-stage m0", 0, 4.0, 0), ("1-stage m30", 0, 4.0, 30),
           ("3-stage λ8 m0", 2, 8.0, 0), ("3-stage λ16 m0", 2, 16.0, 0), ("3-stage λ32 m0", 2, 32.0, 0)]


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower() / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2"]}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        P, y, sbj, F = load(sd, img, video)
        Ps = {"real": P, "shuf": shuffle_within_class(P, y, sbj, np.random.RandomState(sd))}
        for e in F:
            for kind, PP in Ps.items():
                for nm, stg, lam, m in CONFIGS:
                    Q = np.empty_like(PP); sts = []
                    for s in np.unique(sbj):
                        ms = sbj == s
                        Q[ms], st = run(PP[ms], y[ms], F[e][s], m, stg, lam)
                        sts.append((st, ms.sum()))
                    L = len(sts[0][0]); wts = np.array([n for _, n in sts], float)
                    agg = [tuple(np.average([st[i][j] for st, _ in sts], weights=wts) for j in range(2)) for i in range(L)]
                    R[(e, kind, nm)].append((tune_tau(Q, y)[0], agg))
        print(f"seed {sd} done", flush=True)
    for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
        for kind in ["real", "shuf"]:
            print(f"\n== {e} / {'real P' if kind == 'real' else 'P shuffled within class'}")
            for nm, *_ in CONFIGS:
                rs = R[(e, kind, nm)]
                f1 = np.mean([r[0] for r in rs])
                L = len(rs[0][1])
                cr = " / ".join(f"{np.mean([r[1][i][0] for r in rs]):.3f}" for i in range(L))
                cp = " / ".join(f"{np.mean([r[1][i][1] for r in rs]):.3f}" for i in range(L))
                print(f"  {nm:12s} F1 {f1:.4f} | cross-activity edges (per stage) {cr:23s} | class-mean purity excl. NULL {cp}")


if __name__ == "__main__":
    main()
