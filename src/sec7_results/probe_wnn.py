"""Can fusing the video and inertial graphs with WNN-style (Hao et al., Cell 2021) per-window weights beat the division of roles?

The only previous refutation of fusion was "a single global weight" (automatic weighting, -0.068). WNN varies the weight per window:
  r_vv = cos(f_i, mean f of video neighbours), r_vi = cos(f_i, mean f of inertial neighbours)
  r_ii = cos(x_i, mean x of inertial neighbours), r_iv = cos(x_i, mean x of video neighbours)
  (theta_v, theta_i) = softmax([r_vv - r_vi, r_ii - r_iv] / T)
  fused similarity S_ij = theta_v(i) s^v_ij + theta_i(i) s^i_ij (per-row weights, symmetrized afterwards) -> mutual kNN (k=30)
Only the stage-1 graph is replaced; stage 2-3 re-linking ([G || 4 sqrt(Q)]), video injection and sharpen stay as in the final configuration.
Compare: video only (current) / inertial only / global half-half / WNN (sweep T). 2 seeds, OOF macro-F1.
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


def knn_from_S(S, k=30):
    n = S.shape[0]
    S = S.clone(); S.fill_diagonal_(-float("inf"))
    w, nn = torch.topk(S, min(k, n - 2), dim=1)
    r = torch.zeros((n, n), dtype=torch.bool, device=S.device); r.scatter_(1, nn, True)
    keep = r[nn, torch.arange(n, device=S.device)[:, None].expand_as(nn)]
    return nn, torch.where(keep, w, torch.full_like(w, -1e9))


def run(P, Gn, V, first, stages=(4.0, 4.0), temp=1.4, g=0.2):
    gi = g / (len(stages) + 1)
    nn, w = first
    Q = gate_blend(sharpen(propagate(P, nn, w, 0.75, 5), temp), V, gi)
    for lam in stages:
        nn, w = knn_graph(torch.cat([Gn, lam * torch.sqrt(Q)], 1), 30)
        Q = gate_blend(sharpen(propagate(Q, nn, w, 0.75, 5), temp), V, gi)
    return Q


res, wstat = {}, {}
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
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1); Gn = rownorm(G - G.mean(0))
        x = _t(np.nan_to_num(Fi[s])); x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        x = torch.cat([x, 3.0 * torch.nn.functional.one_hot(_t(sens[m], torch.long), 4).double()], 1)  # also include sensor placement
        xn = rownorm(x - x.mean(0))
        Sv = Gn @ Gn.T; Si = xn @ xn.T
        nnv, _ = knn_from_S(Sv); nni, _ = knn_from_S(Si)
        fv_mean = lambda nn_: rownorm(Gn[nn_].mean(1)); fi_mean = lambda nn_: rownorm(xn[nn_].mean(1))
        r_vv = (Gn * fv_mean(nnv)).sum(1); r_vi = (Gn * fv_mean(nni)).sum(1)
        r_ii = (xn * fi_mean(nni)).sum(1); r_iv = (xn * fi_mean(nnv)).sum(1)
        cfg = {"video only (current)": knn_from_S(Sv), "inertial only": knn_from_S(Si), "global half-half": knn_from_S(0.5 * Sv + 0.5 * Si)}
        for T in [0.02, 0.05, 0.1, 0.3]:
            th = torch.softmax(torch.stack([r_vv - r_vi, r_ii - r_iv], 1) / T, 1)
            Sw = th[:, :1] * Sv + th[:, 1:] * Si
            cfg[f"WNN T={T}"] = knn_from_S(0.5 * (Sw + Sw.T))
            wstat.setdefault(f"WNN T={T}", []).append(th[:, 0].cpu().numpy())
        for nm, first in cfg.items():
            outs.setdefault(nm, np.empty_like(P0))[m] = run(P, Gn, Vs, first).cpu().numpy()
    for nm, Q in outs.items():
        res.setdefault(nm, []).append(tune_tau(Q, y)[0])
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["video only (current)"])
for nm, v in res.items():
    v = np.array(v); d = v - ref
    extra = ""
    if nm in wstat:
        th = np.concatenate(wstat[nm]); extra = f"  video weight: mean {th.mean():.2f}, 10-90% [{np.percentile(th,10):.2f}, {np.percentile(th,90):.2f}]"
    print(f"  {nm:14s} {v.mean():.4f}  vs current {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d) + "]" + extra)
