"""Why do stages help: is the rebuilding between stages "re-estimating the graph from predictions", or just more propagation?

Final configuration: stage 1 is mutual kNN on G, propagation (alpha 0.75, 5 iters) -> sharpen -> video injection. Between stages the graph is rebuilt on [G || 4 sqrt(Q)] (twice).
Compared (sharpen/injection schedule kept identical. 2 seeds, OOF macro-F1):
  current           : rebuild with the current Q
  no stages         : once only
  (1) no rebuild    : keep the same G graph, propagate->sharpen->inject 3 times
  (1') more propagation: a single propagation with 15 iterations
  (2) shuffled Q    : rebuild with Q shuffled across windows (propagation uses the real Q)
  (3) initial P     : rebuild with P before propagation
  (4) ground truth (upper bound): rebuild with one-hot ground-truth labels
  (5) edge metric   : fraction of edges in each stage's graph linking windows with different true classes
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
from probe_prf import TAGS, W, build
from refine_gpu import _t, drop_top_pcs, gate_blend, knn_graph, propagate, sharpen


def rownorm(F):
    return F / (torch.linalg.norm(F, dim=1, keepdim=True) + 1e-8)


def cross_frac(nn, w, yy):
    keep = w > -1e8
    return ((yy[nn] != yy[:, None]) & keep).sum().item() / max(keep.sum().item(), 1)


def run(P, Gn, V, yy, mode, rng, lam=4.0, temp=1.4, g=0.2):
    n_rounds = 1 if mode == "no stages" else 3
    gi = g / n_rounds
    nn, w = knn_graph(Gn, 30)
    stats = [cross_frac(nn, w, yy)]
    it = 15 if mode == "(1') more propagation" else 5
    Q = gate_blend(sharpen(propagate(P, nn, w, 0.75, it), temp), V, gi if mode != "(1') more propagation" else g)
    if mode in ("no stages", "(1') more propagation"):
        return Q, stats
    for _ in range(2):
        if mode == "(1) no rebuild":
            pass
        else:
            if mode == "current": R = Q
            elif mode == "(2) shuffled Q": R = Q[torch.tensor(rng.permutation(len(Q)), device=Q.device)]
            elif mode == "(3) initial P": R = P
            elif mode == "(4) ground truth (upper bound)": R = torch.nn.functional.one_hot(yy, N_CLASSES).double()
            nn, w = knn_graph(torch.cat([Gn, lam * torch.sqrt(R)], 1), 30)
        stats.append(cross_frac(nn, w, yy))
        Q = gate_blend(sharpen(propagate(Q, nn, w, 0.75, 5), temp), V, gi)
    return Q, stats


MODES = ["current", "no stages", "(1) no rebuild", "(1') more propagation", "(2) shuffled Q", "(3) initial P", "(4) ground truth (upper bound)"]
res, st = {}, {}
for sd in [42, 7]:
    out, y, sbj = build(sd); P0 = out["base"]
    _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    rng = np.random.RandomState(sd)
    outs = {md: np.empty_like(P0) for md in MODES}
    for s in np.unique(sbj):
        m = sbj == s
        f = drop_top_pcs(_t(F0[s]), 30); f = rownorm(f - f.mean(0))
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1); Gn = rownorm(G - G.mean(0))
        yy = torch.tensor(y[m].astype(np.int64), device=Gn.device)
        for md in MODES:
            Q, sts = run(_t(P0[m]), Gn, _t(V[m]), yy, md, rng)
            outs[md][m] = Q.cpu().numpy()
            st.setdefault(md, []).append((sts, m.sum()))
    for md in MODES:
        res.setdefault(md, []).append(tune_tau(outs[md], y)[0])
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["current"])
print("\n=== OOF macro-F1 ===")
for md in MODES:
    v = np.array(res[md]); d = v - ref
    print(f"  {md:18s} {v.mean():.4f}  vs current {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d) + "]")
print("\n=== (5) fraction of edges linking windows with different true classes (window-weighted mean) stage 1 / stage 2 / stage 3 ===")
for md in MODES:
    L = max(len(a) for a, _ in st[md])
    rows = [a for a, _ in st[md]]; wts = np.array([nw for _, nw in st[md]], float)
    means = [np.average([r[i] for r in rows], weights=wts) for i in range(L)]
    print(f"  {md:18s} " + " / ".join(f"{x:.3f}" for x in means))
