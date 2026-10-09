"""Neutralise the motion component with a linear Gaussian factor model split into shared / specific parts (group factor analysis, inter-battery FA).

Generative model (centred and scaled within subject):
  v   = W_v z + U_v u + e_v            z ~ N(0, I_k)   shared (body state)
  x_l = W_l z + U_l w_l + e_l          u ~ N(0, I_p)   video-specific (viewpoint, scene)
                                        w_l ~ N(0, I_q) specific to limb l
  e: diagonal Gaussian
Stacking all latents gives x_all = L f + e, f = [z, u, w_1..w_4], where L has structural zeros (learned by EM).
Neutralisation: v' = v - W_v E[z | observed blocks]  (observed = video + one limb; oracle is video + four limbs)
The subtracted amount always lies in span(W_v) (k-dim), so the subspace does not change with wearing position.
Comparison: m=0 / m=30, the test-usable "one limb+video" and the upper bound "four limbs+video", 2 seeds.
By-product: overlap between span(U_v) and the top 30 within-subject principal components.
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
from config import N_CLASSES, WORK
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_decomp import PREP, csub, final_with, zsub
from probe_invariance import load_all
from probe_prf import TAGS, W, build

DV, DX = 768, 78


def structure(k, p, q):
    """Mask of the non-zero positions of L (1080 x (k+p+4q))."""
    F = k + p + 4 * q
    M = np.zeros((DV + 4 * DX, F), bool)
    M[:, :k] = True                                  # shared
    M[:DV, k:k + p] = True                           # video-specific
    for l in range(4):
        r = slice(DV + l * DX, DV + (l + 1) * DX)
        M[r, k + p + l * q:k + p + (l + 1) * q] = True
    return M


def fit_em(X, M, iters, seed):
    """EM for structured factor analysis. X: (n, D), already centred."""
    rng = np.random.RandomState(seed)
    n, D = X.shape; F = M.shape[1]
    L = rng.randn(D, F) * 0.1 * M
    psi = X.var(0) * 0.5 + 1e-3
    S = X.T @ X / n
    for it in range(iters):
        # E: posterior f | x
        Lp = L / psi[:, None]
        Sig = np.linalg.inv(np.eye(F) + L.T @ Lp)            # posterior covariance
        B = Sig @ Lp.T                                        # E[f|x] = B x
        Eff = Sig + B @ S @ B.T                               # E[f f^T] (mean)
        SxF = S @ B.T                                         # E[x f^T] (mean)
        # M: per row, solve using only the masked columns
        for i in range(D):
            c = M[i]
            L[i, :] = 0
            L[i, c] = np.linalg.solve(Eff[np.ix_(c, c)], SxF[i, c])
        psi = np.maximum(np.diag(S) - np.einsum("ij,ij->i", L, SxF), 1e-4)
    return L, psi


def post_z(L, psi, Xo, obs, k):
    """E[z|x_obs] from the observed rows obs only (returns the shared factors only)."""
    Lo = L[obs]; Lp = Lo / psi[obs][:, None]
    Sig = np.linalg.inv(np.eye(L.shape[1]) + Lo.T @ Lp)
    return (Xo @ Lp @ Sig)[:, :k]


def main(a):
    d = dict(np.load(WORK / "decomp" / "oof.npz"))
    Yc, sbjA, tileA = d["Yc"], d["sbj"], d["tile"]
    n = len(pd.read_parquet(PREP / "win_meta.parquet"))
    fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    X4 = np.column_stack([zsub(np.nan_to_num(np.asarray(fi[tileA + l * n], np.float32)), sbjA) for l in range(4)])
    sv = Yc.std() + 1e-6                                       # one scale for all of video (keeps the directional structure)
    Xall = np.column_stack([Yc / sv, X4]).astype(np.float64)
    us = np.random.RandomState(0).permutation(np.unique(sbjA))
    fA = {s: i % 5 for i, s in enumerate(us)}
    foldA = np.array([fA[s] for s in sbjA])
    M = structure(a.k, a.p, a.q)
    tag = f"k{a.k}_p{a.p}_q{a.q}"
    cache = WORK / "decomp" / f"gfa_{tag}.npz"
    if cache.exists():
        H = dict(np.load(cache))
    else:
        H = {"one": np.zeros((4, *Yc.shape), np.float32), "four": np.zeros_like(Yc)}
        ov = []
        for kf in range(5):
            tr, te = foldA != kf, foldA == kf
            L, psi = fit_em(Xall[tr][::a.sub], M, a.iters, kf)
            Wv = L[:DV, :a.k] * sv
            vi = np.arange(DV)
            z4 = post_z(L, psi, Xall[te], np.arange(DV + 4 * DX), a.k)
            H["four"][te] = z4 @ Wv.T
            for l in range(4):
                obs = np.concatenate([vi, DV + l * DX + np.arange(DX)])
                H["one"][l, te] = post_z(L, psi, Xall[te][:, obs], obs, a.k) @ Wv.T
            # overlap between the video-specific subspace and the top 30 within-subject PCs (mean over training subjects)
            Uv = np.linalg.qr(L[:DV, a.k:a.k + a.p])[0]; Ws = np.linalg.qr(L[:DV, :a.k])[0]
            for s in np.unique(sbjA[te]):
                _, _, Vt = np.linalg.svd(Yc[sbjA == s], full_matrices=False)
                ov.append([((Vt[:30] @ Ws) ** 2).sum(1).mean(), ((Vt[:30] @ Uv) ** 2).sum(1).mean()])
            print(f"  fold {kf} done", flush=True)
        ov = np.array(ov)
        H["ov"] = ov.mean(0)
        np.savez(cache, **H)
    r2 = lambda h: 1 - ((Yc - h) ** 2).sum() / (Yc ** 2).sum()
    print(f"[{tag}] R^2 all video: four limbs+video {r2(H['four']):.3f} / one limb+video " +
          " ".join(f"{r2(H['one'][l]):.3f}" for l in range(4)))
    print(f"  fraction of top 30 within-subject PCs in the shared subspace {H['ov'][0]:.3f} (dim ratio {a.k/DV:.3f}), "
          f"in the video-specific subspace {H['ov'][1]:.3f} (dim ratio {a.p/DV:.3f})", flush=True)

    res = {}
    for sd in a.seeds:
        out, yy, sb = build(sd); P = out["base"]
        _, _, _, _, sens, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
        pos = np.searchsorted(tileA, tile)
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
            Mx = np.empty((len(yy), N_CLASSES), np.float32)
            for s in np.unique(sb): Mx[sb == s] = dd[s]
            src.append(Mx)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        Yr = csub(d["Y"][pos], sb)
        per = lambda A: {s: A[sb == s] for s in np.unique(sb)}
        cfgs = [("raw m=30 (current)", Yr, 30),
                ("one limb+video m=0", Yr - H["one"][sens, pos], 0), ("one limb+video m=30", Yr - H["one"][sens, pos], 30),
                ("four limbs+video m=0", Yr - H["four"][pos], 0), ("four limbs+video m=30", Yr - H["four"][pos], 30)]
        for nm, F, mm in cfgs:
            res.setdefault(nm, []).append(tune_tau(final_with(P, per(F), Vg, V, sb, mm), yy)[0])
        print(f"  seed {sd} done", flush=True)
    ref = np.array(res["raw m=30 (current)"])
    for nm, v in res.items():
        v = np.array(v)
        print(f"  {nm:16s} {v.mean():.4f}  vs current {np.mean(v - ref):+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--p", type=int, default=30)
    p.add_argument("--q", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--sub", type=int, default=2)
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7])
    main(p.parse_args())
