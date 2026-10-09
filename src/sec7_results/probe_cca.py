"""Correlation structure between the feature spaces of the two modalities. Tests hypothesis 8 at the feature level.

So far redundancy has only been measured at the **prediction level** (argmax agreement). Measuring at the feature level
"which video PCs can be predicted from inertial features" directly answers the mechanism of PC removal:

  top PCs = predictable from inertial (shared component), lower PCs = not predictable (video-specific)
  -> removing the top is an operation that "discards what inertial already has and keeps the video-specific part"
  -> propagation can only correct with information the base lacks, so this could be the mechanism

Conversely, if the top and lower PCs are equally (un)predictable, this explanation also fails.

What we measure:
  (1) Explained fraction R^2 per component (inertial 78-dim -> video PC_k). Subject-level CV.
  (2) Top canonical correlations (CCA). How much the two spaces share.
  (3) For comparison, also the reverse direction: predicting inertial PCs from video.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, SENSORS, WORK
from cv import subject_folds

PREP = WORK / "prep"


def ridge_oof(X, Y, groups, lam):
    pred = np.zeros_like(Y)
    folds, _ = subject_folds(groups, n_splits=5)
    for tr, va in folds:
        Xt = X[tr]; mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        Xt = (Xt - mu) / sd
        ym = Y[tr].mean(0)
        Wt = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xt.shape[1], dtype=np.float32),
                             Xt.T @ (Y[tr] - ym))
        pred[va] = ((X[va] - mu) / sd) @ Wt + ym
    return pred


def r2(Y, P):
    return 1.0 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def main(a):
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]; rows = tile + sens * n
    y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]

    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    Xv = np.asarray(vid[tile], np.float32).mean(1)
    fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    Xi = np.nan_to_num(np.asarray(fi[rows], np.float32))
    Xi = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-6)
    # also put the sensor-position one-hot on the inertial side (information known at test time)
    Xi = np.concatenate([Xi, np.eye(4, dtype=np.float32)[sens]], 1)

    # centre within subject, then global PCA (same treatment as the graph)
    Xc = Xv.copy()
    for s in np.unique(sbj):
        m = sbj == s
        Xc[m] -= Xc[m].mean(0)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    ncomp = min(a.ncomp, Vt.shape[0])
    Z = Xc @ Vt[:ncomp].T                         # video PC scores
    varr = sv[:ncomp] ** 2 / (sv ** 2).sum()

    print("=== (1) How well video PCs are predicted from inertial features (subject-level CV) ===")
    print(f"  {'comp':>6s}{'var ratio':>9s}{'R^2 from inert':>15s}{'activity eta^2':>12s}")
    P = ridge_oof(Xi, Z, sbj, a.lam)
    rr = r2(Z, P)
    def eta2(v, lab):
        tot = v.var()
        if tot < 1e-12: return 0.0
        g = np.unique(lab)
        gm = np.array([v[lab == c].mean() for c in g]); gn = np.array([(lab == c).sum() for c in g], float)
        return float((gn * (gm - v.mean()) ** 2).sum() / len(v) / tot)
    for i in [0,1,2,4,9,19,29,49,99,199]:
        if i < ncomp:
            print(f"  {i+1:6d}{varr[i]:9.4f}{rr[i]:15.4f}{eta2(Z[:,i], y):12.4f}")
    for lo,hi,nm in [(0,30,"top 1-30 (removed)"),(30,100,"31-100"),(100,ncomp,f"101-{ncomp}")]:
        if lo < ncomp:
            sl = slice(lo, min(hi,ncomp))
            print(f"  {nm:22s} variance {varr[sl].sum():.3f}  mean R^2 from inertial {rr[sl].mean():.4f}")

    print("\n=== (2) Canonical correlations (top 10) ===")
    def whiten(A):
        A = A - A.mean(0)
        U, s, _ = np.linalg.svd(A, full_matrices=False)
        k = (s > s[0] * 1e-6).sum()
        return U[:, :k]
    Uv, Ui = whiten(Z[:, :a.cca_dim]), whiten(Xi)
    cc = np.linalg.svd(Uv.T @ Ui, compute_uv=False)
    print("  " + "  ".join(f"{c:.3f}" for c in cc[:10]))
    print(f"  mean of top 10 {cc[:10].mean():.3f}   mean of all {len(cc)} {cc.mean():.3f}")
    print("  (note: this is in-sample. The subject-CV R^2 is the value that generalises)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ncomp", type=int, default=200)
    p.add_argument("--lam", type=float, default=1e3)
    p.add_argument("--cca-dim", type=int, default=100)
    main(p.parse_args())
