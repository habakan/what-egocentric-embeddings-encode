"""Switch the graph construction by placement (sensor location) so that the single-limb neutralization acts only "between the same placement".

A. Separate graph per placement: split a subject's windows into 4 by placement and run the final configuration within each
   (drop30 without neutralization / drop30 with neutralization). Separates out the effect of losing cross-placement neighbours.
B. Switch the distance by placement pair: pairs of windows with the same placement use the neutralized features, pairs with different placements use the current (drop30) features
   for similarity, and propagate over a single graph. B-cal rescales the neutralized similarities per subject so that the mean/std over same-placement pairs
   match the current side (prevents neighbours from concentrating on the same placement).
Neutralization: linear prediction of the limb the window is worn on (xl_only of probe_bayes_impute). Usable on test.
All on GPU (refine_gpu), 2 seeds. Also reports the fraction of neighbours with the same placement.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_decomp import csub
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from refine_gpu import _t, drop_top_pcs, final_with, gate_blend, propagate, sharpen

d = dict(np.load(WORK / "decomp" / "oof.npz"))
xl = np.load(WORK / "decomp" / "bayes.npz")["xl_only"]
tileA = d["tile"]


def rownorm(F):
    return F / (torch.linalg.norm(F, dim=1, keepdim=True) + 1e-8)


def graph_feat(F, Vg, m):
    f = drop_top_pcs(F, m) if m else F
    f = rownorm(f - f.mean(0))
    G = torch.cat([f, 0.5 * torch.sqrt(Vg)], 1)
    G = G - G.mean(0)
    return rownorm(G)


def knn_from_S(S, k=30):
    n = S.shape[0]
    S = S.clone(); S.fill_diagonal_(-float("inf"))
    w, nn = torch.topk(S, min(k, n - 2), dim=1)
    r = torch.zeros((n, n), dtype=torch.bool, device=S.device); r.scatter_(1, nn, True)
    keep = r[nn, torch.arange(n, device=S.device)[:, None].expand_as(nn)]
    return nn, torch.where(keep, w, torch.full_like(w, -1e9))


def switched_S(Ga, Gb, same, cal):
    """Similarity from Gb (neutralized) for same pairs, Ga (current) otherwise."""
    Sa = Ga @ Ga.T; Sb = Gb @ Gb.T
    if cal:
        ma, sa = Sa[same].mean(), Sa[same].std(); mb, sb = Sb[same].mean(), Sb[same].std()
        Sb = (Sb - mb) / sb * sa + ma
    return torch.where(same, Sb, Sa)


def multistage_switched(P, Ga, Gb, V, same, cal, stages=(4.0, 4.0), alpha=0.75, iters=5, temp=1.4, g=0.2):
    gi = g / (len(stages) + 1)
    nn, w = knn_from_S(switched_S(Ga, Gb, same, cal))
    frac = same.gather(1, nn).float().mean().item()
    Q = gate_blend(sharpen(propagate(P, nn, w, alpha, iters), temp), V, gi)
    for lam in stages:
        Ca = rownorm(torch.cat([Ga, lam * torch.sqrt(Q)], 1)); Cb = rownorm(torch.cat([Gb, lam * torch.sqrt(Q)], 1))
        nn, w = knn_from_S(switched_S(Ca, Cb, same, cal))
        Q = gate_blend(sharpen(propagate(Q, nn, w, alpha, iters), temp), V, gi)
    return Q, frac


res, fracs = {}, {}
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
    Yr = csub(d["Y"][pos], sb); h = xl[sens, pos]; Rn = Yr - h
    per = lambda A, msk=None: {s: A[(sb == s) if msk is None else ((sb == s) & msk)] for s in np.unique(sb)}

    # reference
    res.setdefault("current (drop30)", []).append(tune_tau(final_with(P, per(Yr), Vg, V, sb, 30), yy)[0])

    # A: separate graph per placement
    for nm, F in [("A per-placement graph, no neutralization", Yr), ("A per-placement graph, neutralized", Rn)]:
        Q = np.empty_like(P)
        for l in range(4):
            msk = sens == l
            Vg_l = {s: Vg[s][sens[sb == s] == l] for s in np.unique(sb)}
            Q[msk] = final_with(P[msk], per(F, msk), Vg_l, V[msk], sb[msk], 30)
        res.setdefault(nm, []).append(tune_tau(Q, yy)[0])

    # B: switch distance by placement pair
    for m_neu in [30, 10]:
        for cal in [False, True]:
            nm = f"B switch neutralized m={m_neu}" + (" cal" if cal else "")
            Q = np.empty_like(P); fr = []
            for s in np.unique(sb):
                msk = sb == s
                vg = _t(Vg[s]); Ga = graph_feat(_t(Yr[msk]), vg, 30); Gb = graph_feat(_t(Rn[msk]), vg, m_neu)
                ss = _t(sens[msk], torch.long); same = ss[:, None] == ss[None, :]
                q, f = multistage_switched(_t(P[msk]), Ga, Gb, _t(V[msk]), same, cal)
                Q[msk] = q.cpu().numpy(); fr.append(f)
            res.setdefault(nm, []).append(tune_tau(Q, yy)[0]); fracs.setdefault(nm, []).append(np.mean(fr))
    # reference: fraction of same-placement neighbours in the current graph
    fr = []
    for s in np.unique(sb):
        msk = sb == s
        G = graph_feat(_t(Yr[msk]), _t(Vg[s]), 30); ss = _t(sens[msk], torch.long)
        nn, _ = knn_from_S(G @ G.T); fr.append((ss[nn] == ss[:, None]).float().mean().item())
    fracs.setdefault("current (drop30)", []).append(np.mean(fr))
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["current (drop30)"])
for nm, v in res.items():
    v = np.array(v); f = f"  same-placement nbrs {np.mean(fracs[nm]):.3f}" if nm in fracs else ""
    print(f"  {nm:26s} {v.mean():.4f}  vs current {np.mean(v - ref):+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]" + f)
