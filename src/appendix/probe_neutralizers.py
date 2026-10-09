"""Compare video-feature "neutralizers" (four-limb inertial -> video prediction models): MLP / linear / GBDT.

residual = video (centred within subject) - neutralizer's OOF prediction. Split is the same 5-fold-by-subject as probe_decomp.
  MLP   : H4 from probe_decomp (2 layers of 512, all 768 dims)
  linear: Ridge (input is within-subject z, alpha swept and best taken)
  GBDT  : LightGBM, regress each of the top 32 video principal components (global PCA) separately, then map back to 768 dims
For each neutralizer:
  a. R^2 per within-subject principal-component band
  b. residual spectrum, subspace overlap with raw, eta^2 per band
  c. cross-subject neighbour purity (m=0, 30)
  d. graph performance (residual m=0 / m=30, shuffled-prediction control m=0), 2 seeds
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
from probe_decomp import PREP, csub, final_with, r2_by_pc, zsub
from probe_invariance import load_all
from probe_prf import TAGS, W, build

d = dict(np.load(WORK / "decomp" / "oof.npz"))
Yc, sbj, y, tileA = d["Yc"], d["sbj"], d["y"], d["tile"]
subs = np.unique(sbj)
us = np.random.RandomState(0).permutation(subs)
fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
n = len(pd.read_parquet(PREP / "win_meta.parquet"))
fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
X4 = np.column_stack([zsub(np.nan_to_num(np.asarray(fi[tileA + l * n], np.float32)), sbj) for l in range(4)])

cache = WORK / "decomp" / "neutralizers.npz"
if cache.exists():
    H = dict(np.load(cache))
else:
    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    H = {"mlp": d["H4"]}
    best = None
    for al in [1, 10, 100, 1000]:
        h = np.zeros_like(Yc)
        for k in range(5):
            tr = fold != k
            h[~tr] = Ridge(alpha=al).fit(X4[tr], Yc[tr]).predict(X4[~tr])
        r2 = 1 - ((Yc - h) ** 2).sum() / (Yc ** 2).sum()
        print(f"  ridge alpha={al}: R^2 {r2:.4f}", flush=True)
        if best is None or r2 > best[0]: best = (r2, al, h)
    H["linear"] = best[2]; print(f"  ridge chosen alpha={best[1]}", flush=True)
    _, _, Vt = np.linalg.svd(Yc[::3], full_matrices=False); U = Vt[:32].T
    Z = Yc @ U; hz = np.zeros_like(Z)
    for j in range(32):
        for k in range(5):
            tr = fold != k
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.5, min_child_samples=50,
                                  verbose=-1, n_jobs=8, random_state=k).fit(X4[tr], Z[tr, j])
            hz[~tr, j] = m.predict(X4[~tr])
        if j % 8 == 7: print(f"  gbdt PC{j + 1} done", flush=True)
    H["gbdt"] = hz @ U.T
    np.savez(cache, **H)

bands = [(1, 2), (3, 10), (11, 30), (31, 100)]


def pcs(X):
    _, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return S ** 2, Vt


def eta2(Z, lab):
    tot = ((Z - Z.mean(0)) ** 2).sum(0)
    btw = sum((lab == c).sum() * (Z[lab == c].mean(0) - Z.mean(0)) ** 2 for c in np.unique(lab))
    return btw / (tot + 1e-12)


def purity(X, m, rng):
    F = {}
    for s in subs:
        Z = X[sbj == s] - X[sbj == s].mean(0)
        if m:
            _, _, Vt = np.linalg.svd(Z, full_matrices=False); Z = Z - (Z @ Vt[:m].T) @ Vt[:m]
        F[s] = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    pur = []
    for s in subs:
        idx = rng.choice(len(F[s]), min(300, len(F[s])), replace=False)
        G = np.vstack([F[t] for t in subs if t != s]); gy = np.concatenate([y[sbj == t] for t in subs if t != s])
        nn = np.argpartition(-(F[s][idx] @ G.T), 30, axis=1)[:, :30]
        pur.append((gy[nn] == y[sbj == s][idx][:, None]).mean())
    return float(np.mean(pur))


print("\n=== a-c. Analysis per neutralizer (mean over subjects) ===")
tot = lambda h: 1 - ((Yc - h) ** 2).sum() / (Yc ** 2).sum()
for nm in ["raw", "mlp", "linear", "gbdt"]:
    R = Yc if nm == "raw" else Yc - H[nm]
    line = f"  [{nm:6s}] "
    if nm != "raw":
        r = r2_by_pc(Yc, H[nm], sbj, bands + [(101, 768)])
        line += f"total R2 {tot(H[nm]):.3f} | " + " ".join(f"PC{b[0]}-{b[1]} {v:+.3f}" for b, v in r.items()) + "\n           "
    sh10, sh30, ov, et = [], [], [], {b: [] for b in bands}
    for s in subs:
        m = sbj == s
        ev, V = pcs(R[m]); _, Vr = pcs(Yc[m])
        sh10.append(ev[:10].sum() / ev.sum()); sh30.append(ev[:30].sum() / ev.sum())
        ov.append(((Vr[:30] @ V[:30].T) ** 2).sum() / 30)
        e = eta2((R[m] - R[m].mean(0)) @ V[:100].T, y[m])
        for b in bands: et[b].append(e[b[0] - 1:b[1]].mean())
    rng = np.random.RandomState(0)
    line += (f"top10 share {np.mean(sh10):.3f} top30 share {np.mean(sh30):.3f} overlap with raw (top30) {np.mean(ov):.3f} | eta2 "
             + " ".join(f"PC{b[0]}-{b[1]} {np.mean(et[b]):.3f}" for b in bands)
             + f" | cross-subject purity m=0 {purity(R, 0, rng):.3f} m=30 {purity(R, 30, rng):.3f}")
    print(line, flush=True)

print("\n=== d. Graph performance (final configuration, 2 seeds) ===", flush=True)
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
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    Yr = csub(d["Y"][pos], sb)
    per = lambda A: {s: A[sb == s] for s in np.unique(sb)}
    rng = np.random.RandomState(sd)
    cfgs = [("raw m=0", Yr, 0), ("raw m=30", Yr, 30)]
    for nm in ["mlp", "linear", "gbdt"]:
        h = H[nm][pos]; hs = h.copy()
        for s in np.unique(sb):
            m = np.where(sb == s)[0]; hs[m] = h[rng.permutation(m)]
        cfgs += [(f"{nm} residual m=0", Yr - h, 0), (f"{nm} residual m=30", Yr - h, 30), (f"{nm} shuffled m=0", Yr - hs, 0)]
        if nm == "mlp":
            cfgs += [("mlp residual m=10", Yr - h, 10), ("mlp residual m=50", Yr - h, 50)]
    for nm, F, mm in cfgs:
        res.setdefault(nm, []).append(tune_tau(final_with(P, per(F), Vg, V, sb, mm), yy)[0])
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["raw m=30"])
for nm, v in res.items():
    v = np.array(v)
    print(f"  {nm:18s} {v.mean():.4f}  vs raw m=30 {np.mean(v - ref):+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]")
