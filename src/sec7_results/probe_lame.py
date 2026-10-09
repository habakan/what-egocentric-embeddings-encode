"""Comparison with LAME (Boudiaf et al., CVPR 2022): is it the propagation rule that helps, or where the graph comes from?

LAME: iterate z_i ∝ p_i ⊙ exp(beta * Σ_j W_ij z_j) to convergence (concave-convex procedure, no labels, at test time).
  W: row-normalized mutual kNN (k=30) (same exp(10 s) weights as ours).
Compared (all on GPU, 2 seeds, OOF macro-F1, tau tuned on OOF for each configuration):
  A. single stage, video graph (same G as stage 1 of the final configuration):  our propagation vs LAME (beta sweep)
  B. single stage, inertial graph (the classifier's own features = LAME's original setting):  our propagation vs LAME
  C. whole system (3 stages, rebuilding, sharpen and per-stage video injection kept as is), with only the propagation rule swapped for LAME
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from refine_gpu import _t, drop_top_pcs, gate_blend, knn_graph, propagate, sharpen


def rownorm(F):
    return F / (torch.linalg.norm(F, dim=1, keepdim=True) + 1e-8)


def weights(nn, w, sharp=10.0):
    ww = torch.exp((w - w.max(1, keepdim=True).values) * sharp)
    return ww / (ww.sum(1, keepdim=True) + 1e-12)


def lame(P, nn, w, beta, iters=100):
    ww = weights(nn, w)
    logp = torch.log(torch.clamp(P, min=1e-9))
    z = P.clone()
    for _ in range(iters):
        agg = (z[nn] * ww[:, :, None]).sum(1)
        zn = torch.softmax(logp + beta * agg, 1)
        if (zn - z).abs().max() < 1e-6:
            z = zn; break
        z = zn
    return z


def multistage(P, G, V, rule, stages=(4.0, 4.0), k=30, temp=1.4, g=0.2):
    gi = g / (len(stages) + 1)
    Fn = rownorm(G - G.mean(0))
    nn, w = knn_graph(Fn, k)
    Q = gate_blend(sharpen(rule(P, nn, w), temp), V, gi)
    for lam in stages:
        nn, w = knn_graph(torch.cat([Fn, lam * torch.sqrt(Q)], 1), k)
        Q = gate_blend(sharpen(rule(Q, nn, w), temp), V, gi)
    return Q


ours = lambda P, nn, w: propagate(P, nn, w, 0.75, 5)
BETAS = [0.5, 1.0, 2.0, 4.0, 8.0]
res = {}
for sd in [42, 7]:
    out, y, sbj = build(sd); P0 = out["base"]
    _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
    _, _, y2, _, sens, _, _, Fi, _ = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    assert (y2 == y).all()
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    outs = {}
    for s in np.unique(sbj):
        m = sbj == s
        P = _t(P0[m]); Vs = _t(V[m])
        f = drop_top_pcs(_t(F0[s]), 30); f = rownorm(f - f.mean(0))
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1)
        Gn = rownorm(G - G.mean(0))
        nnv, wv = knn_graph(Gn, 30)
        fi = _t(np.nan_to_num(Fi[s])); fi = (fi - fi.mean(0)) / (fi.std(0) + 1e-6)
        nni, wi = knn_graph(rownorm(fi - fi.mean(0)), 30)
        cfg = {"A video graph 1 stage ours": ours(P, nnv, wv),
               "B inertial graph 1 stage ours": ours(P, nni, wi),
               "C whole system ours": multistage(P, G, Vs, ours)}
        for b in BETAS:
            cfg[f"A video graph 1 stage LAME b={b}"] = lame(P, nnv, wv, b)
            cfg[f"B inertial graph 1 stage LAME b={b}"] = lame(P, nni, wi, b)
            cfg[f"C whole system LAME b={b}"] = multistage(P, G, Vs, lambda Q, nn, w, b=b: lame(Q, nn, w, b))
        for nm, Q in cfg.items():
            outs.setdefault(nm, np.empty_like(P0))[m] = Q.cpu().numpy()
    res.setdefault("base P (no propagation)", []).append(tune_tau(P0, y)[0])
    for nm, Q in outs.items():
        res.setdefault(nm, []).append(tune_tau(Q, y)[0])
    print(f"  seed {sd} done", flush=True)
for nm in sorted(res, key=lambda k: (k[0], k)):
    v = np.array(res[nm])
    print(f"  {nm:30s} {v.mean():.4f}  [" + " ".join(f"{x:.4f}" for x in v) + "]")
