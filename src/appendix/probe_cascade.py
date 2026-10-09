"""Cascade: propagate the distribution Q' corrected by the second stage (exp017) over the graph once more.

The first propagation accounted for most of the system's gain. The second stage corrects Q per window, but those corrections do not spread to neighbours.
Check whether propagating the corrected Q' again on the same (drop30) graph amplifies the corrections over whole segments.
  exp017: Q' = Q^0.7 S^0.3 (S is the OOF of the second-stage LightGBM)
  C1: propagate Q' for one stage only (sweep alpha, no sharpening)
  C2: use Q' as P and run the whole multi-stage refinement of the final configuration (with video injection)
2 seeds, GPU. Criterion: beats exp017 on both seeds, mean +0.005 or more.
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
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from refine_gpu import _t, drop_top_pcs, knn_graph, multistage_at, propagate
from stack_q import blend
from submit_stack import feats, lgbm

res = {}
for sd in [42, 7]:
    out, y, sbj = build(sd); Q0 = out["final"]
    _, _, y2, _, sens, _, Fv, Fi, _ = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    assert (y2 == y).all()
    cache = WORK / "decomp" / f"stage2_oof_s{sd}.npz"
    if cache.exists():
        S = np.load(cache)["S"]
    else:
        X = feats(Q0, Fi, Fv, sens, sbj, 30)
        us = np.random.RandomState(sd).permutation(np.unique(sbj))
        fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
        S = np.zeros((len(y), N_CLASSES))
        for k in range(5):
            tr = fold != k
            m = lgbm(sd, 8).fit(X[tr], y[tr])
            pr = np.zeros(((~tr).sum(), N_CLASSES)); pr[:, m.classes_] = m.predict_proba(X[~tr]); S[~tr] = pr
        np.savez(cache, S=S)
    Qp = blend(Q0, S, 0.3)
    res.setdefault("current Q (exp015)", []).append(tune_tau(Q0, y)[0])
    res.setdefault("exp017 Q'", []).append(tune_tau(Qp, y)[0])

    _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    C1 = {(a, it): np.empty_like(Qp) for a in [0.75, 0.85, 0.9, 0.95] for it in [5, 10, 20]}
    C2 = {}
    for s in np.unique(sbj):
        msk = sbj == s
        f = drop_top_pcs(_t(F0[s]), 30); f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1); G = G - G.mean(0)
        Gq = torch.cat([G / (torch.linalg.norm(G, dim=1, keepdim=True) + 1e-8), 4.0 * torch.sqrt(_t(Qp[msk]))], 1)
        nn, w = knn_graph(Gq, 30)
        for (a, it) in C1:
            C1[(a, it)][msk] = propagate(_t(Qp[msk]), nn, w, a, it).cpu().numpy()
        for g in C2:
            C2[g][msk] = multistage_at(_t(Qp[msk]), G, _t(V[msk]), [4.0, 4.0], k=30, alpha=0.75, iters=5,
                                       temp=1.4, g=g, where="every").cpu().numpy()
    for (a, it), Q in C1.items():
        res.setdefault(f"C1 alpha={a} iters={it}", []).append(tune_tau(Q, y)[0])
    for g, Q in C2.items():
        res.setdefault(f"C2 full multi-stage g={g}", []).append(tune_tau(Q, y)[0])
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["exp017 Q'"])
for nm, v in res.items():
    d = np.array(v) - ref
    print(f"  {nm:24s} {np.mean(v):.4f}  vs exp017 {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d) + "]")
