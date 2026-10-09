"""What happens if the per-subject graph is shared across subjects?

As in the test, group subjects into sets of 4 and compute similarities over all windows of the group. Subtract a penalty delta from similarities of cross-subject pairs
and build mutual kNN (k=30). delta = inf is the current setup (per subject), 0 is fully shared. The same penalty is applied in multi-stage rebuilding.
Feature preprocessing (top-30 PC removal, centring, l2) stays per subject (same as current).
2 seeds x 2 groupings. OOF macro-F1, fraction of cross-subject edges, and fraction of those edges whose true activities differ.
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
from refine_gpu import _t, drop_top_pcs, gate_blend, propagate, sharpen

DEV = "cuda"


def rownorm(F):
    return F / (torch.linalg.norm(F, dim=1, keepdim=True) + 1e-8)


def knn_pen(F, grp, delta, k=30):
    F = rownorm(F)
    S = F @ F.T
    if np.isfinite(delta):
        S = S - delta * (grp[:, None] != grp[None, :]).float()
    else:
        S = torch.where(grp[:, None] == grp[None, :], S, torch.full_like(S, -1e4))
    S.fill_diagonal_(-float("inf"))
    w, nn = torch.topk(S, k, dim=1)
    n = S.shape[0]
    r = torch.zeros((n, n), dtype=torch.bool, device=S.device); r.scatter_(1, nn, True)
    keep = r[nn, torch.arange(n, device=S.device)[:, None].expand_as(nn)] & (w > -1e3)
    return nn, torch.where(keep, w, torch.full_like(w, -1e9))


def run(P, G, V, grp, delta, yy, stages=(4.0, 4.0), temp=1.4, g=0.2):
    gi = g / (len(stages) + 1)
    nn, w = knn_pen(G, grp, delta)
    keep = w > -1e8
    cross = (grp[nn] != grp[:, None]) & keep
    stat = (cross.sum().item() / keep.sum().item(),
            ((yy[nn] != yy[:, None]) & cross).sum().item() / max(cross.sum().item(), 1))
    Q = gate_blend(sharpen(propagate(P, nn, w, 0.75, 5), temp), V, gi)
    for lam in stages:
        nn, w = knn_pen(torch.cat([G, lam * torch.sqrt(Q)], 1), grp, delta)
        Q = gate_blend(sharpen(propagate(Q, nn, w, 0.75, 5), temp), V, gi)
    return Q, stat


DELTAS = [np.inf, 0.3, 0.2, 0.1, 0.05, 0.0]
res, stats = {}, {}
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
    # per-subject preprocessed graph features
    Gs = {}
    for s in np.unique(sbj):
        f = drop_top_pcs(_t(F0[s]), 30); f = rownorm(f - f.mean(0))
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1)
        Gs[s] = rownorm(G - G.mean(0)).float()
    for rep_ in range(2):
        subs = np.random.RandomState(100 * sd + rep_).permutation(np.unique(sbj))
        groups = [subs[i:i + 4] for i in range(0, len(subs), 4)]
        outs = {d: np.empty_like(P0) for d in DELTAS}
        for gp in groups:
            idx = np.concatenate([np.where(sbj == s)[0] for s in gp])
            G = torch.cat([Gs[s] for s in gp]); grp = torch.tensor(np.concatenate([[i] * (sbj == s).sum() for i, s in enumerate(gp)]), device=DEV)
            yy = torch.tensor(y[idx].astype(np.int64), device=DEV)
            P = torch.tensor(P0[idx], device=DEV, dtype=torch.float32); Vt = torch.tensor(V[idx], device=DEV, dtype=torch.float32)
            for d in DELTAS:
                Q, stt = run(P, G, Vt, grp, d, yy)
                outs[d][idx] = Q.cpu().numpy()
                stats.setdefault(d, []).append((stt, len(idx)))
        for d in DELTAS:
            res.setdefault(d, []).append(tune_tau(outs[d], y)[0])
        print(f"  seed {sd} rep {rep_} done", flush=True)
ref = np.array(res[np.inf])
for d in DELTAS:
    v = np.array(res[d]); dd = v - ref
    sts = stats[d]; wts = np.array([n for _, n in sts], float)
    cf = np.average([a[0] for a, _ in sts], weights=wts); cw = np.average([a[1] for a, _ in sts], weights=wts)
    nm = "per subject (current)" if not np.isfinite(d) else ("fully shared" if d == 0 else f"penalty delta={d}")
    print(f"  {nm:16s} {v.mean():.4f}  vs current {dd.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in dd) + f"]  "
          f"cross-subject edges {cf:.3f}, of which different activity {cw:.3f}")
