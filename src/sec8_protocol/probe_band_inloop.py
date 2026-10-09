"""Put the count band between stages (09-30 ideate hypothesis 1).

After each stage of the paper's system (remove top 30, [4,4], per-stage video injection), compute per-class biases b with the per-recording count band [78,126],
set Q ∝ Q·exp(beta·b), and then build the next stage's graph [f‖λ√Q]. At the end, tune_tau + band as before.
Baseline for comparison is system+band (0.8688 from probe_countband_paper.py). 3 seeds.

  uv run python src/probe_band_inloop.py
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import tune_tau
from probe_countband import band_calibrate
from probe_invariance import load_all
from probe_prf import TAGS, W
from refine_gpu import _t, drop_top_pcs, gate_blend, knn_graph, propagate, sharpen

LO, HI = 78, 126


def band_bias(L, lo=LO, hi=HI, step=0.1, iters=200):
    """Same iteration as band_calibrate; returns the per-class biases for one recording."""
    b = np.zeros(L.shape[1])
    for _ in range(iters):
        cnt = np.bincount((L + b).argmax(1), minlength=L.shape[1])
        up, dn = cnt[1:] < lo, cnt[1:] > hi
        if not (up.any() or dn.any()):
            break
        b[1:] += step * up - step * dn
    return b


def apply_band(Q, rec, beta):
    """Q: (n, C) torch. Compute the band biases per recording and apply them to Q."""
    if beta == 0:
        return Q
    L = torch.log(torch.clamp(Q, min=1e-9)).cpu().numpy()
    B = np.zeros_like(L)
    for r in np.unique(rec):
        m = rec == r
        B[m] = band_bias(L[m])
    R = Q * torch.exp(beta * _t(B, Q.dtype))
    return R / R.sum(1, keepdim=True)


def system(P, G, V, rec, beta, after, stages=(4.0, 4.0), k=30, alpha=0.75, iters=5, temp=1.4, g=0.2):
    """refine_gpu.multistage_at (where="every") plus applying the band after the stages in after. Stage 0 is the first propagation, -1 is before propagation."""
    gi = g / (len(stages) + 1)
    if -1 in after:                      # before the first propagation, also apply to the classifier's P
        P = apply_band(P, rec, beta)
    Fn = G - G.mean(0)
    Fn = Fn / (torch.linalg.norm(Fn, dim=1, keepdim=True) + 1e-8)
    nn, w = knn_graph(Fn, k)
    Q = gate_blend(sharpen(propagate(P, nn, w, alpha, iters), temp), V, gi)
    if 0 in after:
        Q = apply_band(Q, rec, beta)
    for i, lam in enumerate(stages, 1):
        nn, w = knn_graph(torch.cat([Fn, lam * torch.sqrt(Q)], 1), k)
        Q = gate_blend(sharpen(propagate(Q, nn, w, alpha, iters), temp), V, gi)
        if i in after:
            Q = apply_band(Q, rec, beta)
    return Q


def load(seed):
    P, y, sbj, F0 = load_eval_set(TAGS, W, seed=seed)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=seed)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        d = load_vgraph_oof(tg, TAGS[0], seed=seed)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj):
            M[sbj == s] = d[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    _, _, _, _, _, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=seed)
    rec = pd.read_parquet(WORK / "prep" / "win_meta.parquet")["rec"].to_numpy()[tile]
    return P, y, sbj, F0, Vg, V, rec


def run(P, sbj, F0, Vg, V, rec, beta, after, **kw):
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = drop_top_pcs(_t(F0[s]), 30)
        f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1)
        Q[m] = system(_t(P[m]), G, _t(V[m]), rec[m], beta, after, **kw).cpu().numpy()
    return Q


def score(Q, y, rec):
    f0, tau = tune_tau(Q, y)
    L = np.log(np.clip(Q, 1e-9, None)) - tau * np.log(Q.mean(0) + 1e-9)
    return f0, macro_f1(y, band_calibrate(L, rec, LO, HI))


def main():
    configs = [(0.0, ())] + [(b, a) for a in [(0,), (0, 1), (0, 1, 2)] for b in [0.5, 1.0]]
    res = {}
    for sd in [42, 7, 1337]:
        P, y, sbj, F0, Vg, V, rec = load(sd)
        for beta, after in configs:
            res.setdefault((beta, after), []).append(score(run(P, sbj, F0, Vg, V, rec, beta, after), y, rec))
        print(f"  seed {sd} done", flush=True)
    ref = np.array([v[1] for v in res[(0.0, ())]])
    for (beta, after), v in res.items():
        v = np.array(v); d = v[:, 1] - ref
        print(f"  beta {beta:.1f} after {str(after):10s} tau only {v[:, 0].mean():.4f}  +band {v[:, 1].mean():.4f}  "
              f"vs system+band {d.mean():+.4f} [" + " ".join(f"{x:+.4f}" for x in d) + "]")


if __name__ == "__main__":
    main()
