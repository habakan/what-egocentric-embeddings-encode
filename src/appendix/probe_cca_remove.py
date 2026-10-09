"""Linear version of direction B: remove from the video the "directions correlated with the four-limb inertial data" (video side of CCA).

No inertial data is needed at test time (directions are learned on the training subjects' four limbs and simply projected out of the video).
  Split: same subject-level 5-fold split as probe_decomp. Directions for an evaluated subject are learned on a split excluding that subject.
  CCA: whiten each block with a ridge (r = 0.1 x mean variance), orthonormalize the top k video-side directions with QR and project them out.
Compare: k in {5, 10, 20} alone (m=0) / combined with removal of the top 30 principal components (m=30).
By-product: overlap between the CCA directions and the within-subject top 30 principal components.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_decomp import PREP, csub, final_with, zsub
from probe_invariance import load_all
from probe_prf import TAGS, W, build

d = dict(np.load(WORK / "decomp" / "oof.npz"))
Yc, sbjA, tileA = d["Yc"], d["sbj"], d["tile"]
n = len(pd.read_parquet(PREP / "win_meta.parquet"))
fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
X4 = np.column_stack([zsub(np.nan_to_num(np.asarray(fi[tileA + l * n], np.float32)), sbjA) for l in range(4)])
us = np.random.RandomState(0).permutation(np.unique(sbjA))
fA = {s: i % 5 for i, s in enumerate(us)}
foldA = np.array([fA[s] for s in sbjA])
KS = [5, 10, 20]


def isqrt(C, r):
    C = C + r * np.mean(np.diag(C)) * np.eye(len(C))
    w, U = np.linalg.eigh(C)
    return U @ np.diag(w ** -0.5) @ U.T


Q = {}
canon = []
for k in range(5):
    tr = foldA != k
    V = Yc[tr].astype(np.float64); X = X4[tr].astype(np.float64)
    Cvv, Cxx, Cvx = V.T @ V / len(V), X.T @ X / len(X), V.T @ X / len(V)
    Wv, Wx = isqrt(Cvv, 0.1), isqrt(Cxx, 0.1)
    U, S, _ = np.linalg.svd(Wv @ Cvx @ Wx)
    A = Wv @ U                                            # video-side canonical directions (columns)
    canon.append(S[:20])
    Q[k] = {kk: np.linalg.qr(A[:, :kk])[0] for kk in KS}
print("canonical correlations (mean over training splits): " + " ".join(f"{v:.3f}" for v in np.mean(canon, 0)[[0, 1, 2, 4, 9, 19]]) + "  (1st,2nd,3rd,5th,10th,20th)")
ov = {kk: [] for kk in KS}
for s in np.unique(sbjA):
    _, _, Vt = np.linalg.svd(Yc[sbjA == s], full_matrices=False)
    for kk in KS:
        ov[kk].append(((Vt[:30] @ Q[fA[s]][kk]) ** 2).sum() / kk)
print("fraction of CCA directions lying in the within-subject top 30 principal components: " + " ".join(f"k={kk} {np.mean(v):.3f}" for kk, v in ov.items())
      + f"  (if unrelated 30/768={30/768:.3f})", flush=True)

res = {}
for sd in [42, 7]:
    out, yy, sb = build(sd); P = out["base"]
    _, _, _, _, sens, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    pos = np.searchsorted(tileA, tile)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(yy), N_CLASSES), np.float32)
        for s in np.unique(sb): M[sb == s] = dd[s]
        src.append(M)
    Vv = np.sqrt(src[0] * src[1]); Vv /= Vv.sum(1, keepdims=True)
    Yr = csub(d["Y"][pos], sb)
    cfgs = [("raw m=30 (current)", {s: Yr[sb == s] for s in np.unique(sb)}, 30),
            ("raw m=0", {s: Yr[sb == s] for s in np.unique(sb)}, 0)]
    for kk in KS:
        F = {s: Yr[sb == s] - (Yr[sb == s] @ Q[fA[s]][kk]) @ Q[fA[s]][kk].T for s in np.unique(sb)}
        cfgs += [(f"CCA removal k={kk} m=0", F, 0), (f"CCA removal k={kk} m=30", F, 30)]
    for nm, F, mm in cfgs:
        res.setdefault(nm, []).append(tune_tau(final_with(P, F, Vg, Vv, sb, mm), yy)[0])
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["raw m=30 (current)"])
for nm, v in res.items():
    v = np.array(v)
    print(f"  {nm:20s} {v.mean():.4f}  vs current {np.mean(v - ref):+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]")
