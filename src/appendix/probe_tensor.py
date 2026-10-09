"""Can the features of missing limbs be filled by tensor completion with low rank + video-graph smoothing?

Per subject, a window x limb(4) x feature(78) tensor. In the test format only one limb is observed per window (75% missing, fibre-wise).
Model: x_i (312) ~ W z_i. W (312 x r) is the principal components of fully observed training subjects (low rank of the window-mode unfolding).
For evaluation subjects, fit only the observed blocks and smooth the window factors with the video kNN graph (drop30, mutual k=30):
  min_z sum_i ||x_i,o - W_o z_i||^2 + lam * sum_(ij) a_ij ||z_i - z_j||^2 + mu ||z||^2
Solve with conjugate gradients (GPU) and fill the missing limbs with W z.
Baselines: same-limb mean over video neighbours (stage-2 pooling) / Ridge from the observed limb (trained on training subjects).
Evaluation: R^2 of missing-limb features (z-scored within subject; overall, partner, other group). 5 folds by subject, limb assignment with seed 42.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK
from probe_decomp import PREP, zsub
from refine_gpu import drop_top_pcs, knn_graph

dev = "cuda"
MATE = {0: 3, 3: 0, 1: 2, 2: 1}
d = dict(np.load(WORK / "decomp" / "oof.npz"))
tile, sbj, Y = d["tile"], d["sbj"], d["Y"]
n = len(pd.read_parquet(PREP / "win_meta.parquet"))
fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
Xl = np.stack([zsub(np.nan_to_num(np.asarray(fi[tile + l * n], np.float32)), sbj) for l in range(4)], 1)  # (N,4,78)
N, D = len(tile), 78
limb = np.random.RandomState(42).randint(0, 4, N)
us = np.random.RandomState(0).permutation(np.unique(sbj))
fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])


def graph(s_mask):
    F = torch.tensor(Y[s_mask], device=dev, dtype=torch.float32)
    F = drop_top_pcs(F, 30); F = F - F.mean(0); F = F / (F.norm(dim=1, keepdim=True) + 1e-8)
    nn, w = knn_graph(F, 30)
    keep = w > -1e8
    return nn, keep


def complete(W, xo, lo, nn, keep, lam, prior, sig2, iters=600):
    """W: (4,78,r) torch, xo: (n,78) observed, lo: (n,) observed limb. Returns z (n,r).
    PPCA-style: observation noise sig2, prior on z N(0, diag(prior)). The objective is multiplied by sig2."""
    n_, r = xo.shape[0], W.shape[2]
    Wo = W[lo]                                            # (n,78,r)
    G = torch.einsum("ndr,nds->nrs", Wo, Wo) + torch.diag(sig2 / prior)[None]   # (n,r,r)
    b = torch.einsum("ndr,nd->nr", Wo, xo)
    A = keep.float()                                      # adjacency of (mutual) neighbours, weight 1
    deg = A.sum(1)
    def op(z):
        nb = (z[nn] * A[:, :, None]).sum(1)
        # symmetrise: if j is a neighbour of i, i also contributes as a neighbour of j (nearly symmetric since mutual kNN)
        Lz = deg[:, None] * z - nb
        return torch.einsum("nrs,ns->nr", G, z) + lam * 2 * Lz
    # conjugate gradients with block-diagonal (r x r per window) preconditioner
    Minv = torch.linalg.inv(G + (lam * 2 * deg)[:, None, None] * torch.eye(r, device=dev)[None])
    pre = lambda v: torch.einsum("nrs,ns->nr", Minv, v)
    z = torch.zeros(n_, r, device=dev); res = b - op(z); y = pre(res); p = y.clone(); rs = (res * y).sum()
    for _ in range(iters):
        Ap = op(p); a = rs / ((p * Ap).sum() + 1e-12)
        z = z + a * p; res = res - a * Ap
        if res.norm() < 1e-6 * b.norm(): break
        y = pre(res); rn = (res * y).sum(); p = y + (rn / rs) * p; rs = rn
    return z


acc = {}
def add(nm, true, pred, rel):
    acc.setdefault(nm, {}).setdefault(rel, [0.0, 0.0])
    acc[nm][rel][0] += ((true - pred) ** 2).sum(); acc[nm][rel][1] += (true ** 2).sum()

for k in range(5):
    tr, te = fold != k, fold == k
    X4tr = Xl[tr].reshape(-1, 4 * D)
    # loadings W: principal components of training subjects
    U, S, Vt = np.linalg.svd(X4tr - X4tr.mean(0), full_matrices=False)
    ev = S ** 2 / len(X4tr)                                   # variance of principal components
    Ws = {r: (torch.tensor(Vt[:r].T.reshape(4, D, r), device=dev, dtype=torch.float32),
              torch.tensor(ev[:r], device=dev, dtype=torch.float32),
              float(ev[r:].mean())) for r in [60, 120]}
    ridge = {(o, t): Ridge(alpha=10.0).fit(Xl[tr][:, o], Xl[tr][:, t]) for o in range(4) for t in range(4) if o != t}
    for s in np.unique(sbj[te]):
        m = np.where(sbj == s)[0]
        lo = limb[m]; xo_np = Xl[m, lo]; truth = Xl[m]
        nn, keep = graph(sbj == s)
        xo = torch.tensor(xo_np, device=dev); lo_t = torch.tensor(lo, device=dev)
        preds = {}
        # baseline 1: same-limb mean over video neighbours (mean over the neighbours, including self, that observe that limb)
        nn_np = nn.cpu().numpy(); kp = keep.cpu().numpy()
        nb_mean = np.zeros((len(m), 4, D), np.float32)
        for t in range(4):
            has = (lo[nn_np] == t) & kp
            v = xo_np[nn_np] * has[:, :, None]
            nb_mean[:, t] = v.sum(1) / np.maximum(has.sum(1), 1)[:, None]
        preds["baseline1 video-nbr mean"] = nb_mean
        # baseline 2: Ridge from the observed limb
        rp = np.zeros((len(m), 4, D), np.float32)
        for o in range(4):
            mo = lo == o
            for t in range(4):
                if t != o and mo.any(): rp[mo, t] = ridge[(o, t)].predict(xo_np[mo])
        preds["baseline2 Ridge"] = rp
        preds["mean of baselines 1,2"] = 0.5 * (nb_mean + rp)
        for r, (W, prior, sig2) in Ws.items():
            for lam in [0.0, 0.001, 0.003, 0.01]:
                z = complete(W, xo, lo_t, nn, keep, lam * sig2 / 0.1, prior, sig2)
                preds[f"completion r={r} lam={lam}"] = torch.einsum("ldr,nr->nld", W, z).cpu().numpy()
            # upper bound: project the true four limbs onto W (z by least squares from full observation)
            Wf = W.reshape(4 * D, -1); zt = torch.tensor(truth.reshape(len(m), -1), device=dev) @ Wf
            preds[f"upper r={r} (true 4 limbs proj.)"] = (zt @ Wf.T).reshape(len(m), 4, D).cpu().numpy()
        for nm, P in preds.items():
            for t in range(4):
                miss = lo != t
                if not miss.any(): continue
                rel = np.where(np.vectorize(MATE.get)(lo[miss]) == t, "partner", "other")
                for rr in ["partner", "other"]:
                    mm = rel == rr
                    add(nm, truth[miss][mm, t], P[miss][mm, t], rr)
    print(f"  fold {k} done", flush=True)

print("\n=== R^2 of missing-limb features (subject-wise OOF) ===")
for nm, v in acc.items():
    tot = 1 - (v["partner"][0] + v["other"][0]) / (v["partner"][1] + v["other"][1])
    print(f"  {nm:24s} overall {tot:+.3f}  partner {1 - v['partner'][0]/v['partner'][1]:+.3f}  other {1 - v['other'][0]/v['other'][1]:+.3f}")
