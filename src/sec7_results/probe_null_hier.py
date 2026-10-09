"""Does a hierarchical decision that judges null separately help? Is it better to decide null from video alone?

Current: 19-class argmax Q_c / pi_c^tau.
Hierarchical: decide "null or activity" with a separate score and threshold; windows judged as activity take the non-null argmax of Q (with prior correction).
  H1: null probability of the video-only model V / H2: null probability of the final Q / H3: geometric mean of Q and V /
  H4: V's null probability propagated over the stage-1 video graph of the final config / H5: null probability of the base P (inertial)
The threshold maximises macro-F1 over the whole OOF (1 parameter, treated the same in every variant).
2 seeds. Binary AUC / F1, and 19-class macro-F1 (overall and hard participants).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_prf import TAGS, W, build
from refine_gpu import _t, drop_top_pcs, knn_graph, propagate

HARD = [10, 4, 5, 9]
res, auc, bf1, sub = {}, {}, {}, {}
for sd in [42, 7]:
    out, y, sbj = build(sd); P, Q = out["base"], out["final"]
    _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    # propagate V's null over the video graph (H4)
    Vp = np.empty(len(y))
    for s in np.unique(sbj):
        m = sbj == s
        f = drop_top_pcs(_t(F0[s]), 30); f = f - f.mean(0); f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1); G = G - G.mean(0)
        nn, w = knn_graph(G, 30)
        v2 = _t(np.stack([V[m, 0], 1 - V[m, 0]], 1))
        Vp[m] = propagate(v2, nn, w, 0.75, 5)[:, 0].cpu().numpy()
    tau = tune_tau(Q, y)[1]
    cur = prior_correct(Q, tau).argmax(1)
    Qa = Q.copy(); Qa[:, 0] = 0
    act = prior_correct(Qa, tau).argmax(1)              # class when judged as activity
    scores = {"H1 video V": V[:, 0], "H2 final Q": Q[:, 0], "H3 geo-mean of Q and V": np.sqrt(Q[:, 0] * V[:, 0]),
              "H4 V propagated on graph": Vp, "H5 base P (inertial)": P[:, 0]}
    isn = (y == 0).astype(int)
    res.setdefault("current (19-class argmax)", []).append(macro_f1(y, cur))
    bf1.setdefault("current (19-class argmax)", []).append(f1_score(isn, (cur == 0).astype(int)))
    for s in HARD:
        m = sbj == s; sub.setdefault(("current (19-class argmax)", s), []).append(macro_f1(y[m], cur[m]))
    for nm, sc in scores.items():
        auc.setdefault(nm, []).append(roc_auc_score(isn, sc))
        best = (-1, None)
        for th in np.quantile(sc, np.linspace(0.3, 0.8, 51)):
            pred = np.where(sc >= th, 0, act)
            f = macro_f1(y, pred)
            if f > best[0]: best = (f, th, pred)
        res.setdefault(nm, []).append(best[0])
        bf1.setdefault(nm, []).append(f1_score(isn, (best[2] == 0).astype(int)))
        for s in HARD:
            m = sbj == s; sub.setdefault((nm, s), []).append(macro_f1(y[m], best[2][m]))
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["current (19-class argmax)"])
print("\n=== 19-class macro-F1 / null-vs-activity F1 / AUC of the null score (2 seeds) ===")
for nm in res:
    v = np.array(res[nm]); d = v - ref
    a = f"AUC {np.mean(auc[nm]):.3f}" if nm in auc else ""
    print(f"  {nm:22s} {v.mean():.4f}  vs current {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d) + f"]  binary F1 {np.mean(bf1[nm]):.3f}  {a}")
print("\n=== macro-F1 of hard participants ===")
for nm in res:
    print(f"  {nm:22s} " + "  ".join(f"sbj_{s} {np.mean(sub[(nm, s)]):.3f}" for s in HARD))
