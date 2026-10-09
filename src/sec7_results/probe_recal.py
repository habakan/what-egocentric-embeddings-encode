"""Recalibration of structural parameters (backlog #2). Sweep k / number of stages / alpha / iterations / temperature / video injection g / number removed m
one at a time around the current configuration (drop30, injection at every stage), and combine the promising ones. GPU (refine_gpu).

Current: k=30, stages [4,4] (3 propagations in total), alpha=0.75, iters=5, temp=1.4, g=0.2, m=30, neighbour-feature concatenation 0.5*sqrt(Vg).
These were set with the 09-03 configuration (OOF 0.78) and left unchanged. The graph geometry has changed since.
Criterion: only those beating current on both seeds with mean +0.005 or more (a size verifiable on the LB) are re-tested with the second stage.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import itertools
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
from refine_gpu import _t, drop_top_pcs, multistage_at

BASE = dict(k=30, stages=(4.0, 4.0), alpha=0.75, iters=5, temp=1.4, g=0.2, m=30, gam=0.5)


def run(P, F0, Vg, V, sbj, c):
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = _t(F0[s]); f = drop_top_pcs(f, c["m"]) if c["m"] else f
        f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
        G = torch.cat([f, c["gam"] * torch.sqrt(_t(Vg[s]))], 1)
        Q[m] = multistage_at(_t(P[m]), G, _t(V[m]), list(c["stages"]), k=c["k"], alpha=c["alpha"],
                             iters=c["iters"], temp=c["temp"], g=c["g"], where="every").cpu().numpy()
    return Q


grid = [("current", {})]
for key, vals in [("k", [15, 20, 45, 60, 90]), ("stages", [(), (4.0,), (4.0, 4.0, 4.0), (2.0, 2.0), (8.0, 8.0)]),
                  ("alpha", [0.6, 0.7, 0.8, 0.85]), ("iters", [3, 8, 12]), ("temp", [1.0, 1.2, 1.6, 2.0]),
                  ("g", [0.1, 0.15, 0.3, 0.4]), ("m", [20, 40, 50]), ("gam", [0.0, 0.25, 1.0])]:
    for v in vals:
        grid.append((f"{key}={v}", {key: v}))

data = {}
for sd in [42, 7]:
    out, y, sbj = build(sd)
    _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    data[sd] = (out["base"], F0, Vg, V, sbj, y, tune_tau(out["final"], y)[0])
    print(f"  seed {sd} loaded (build final {data[sd][6]:.4f})", flush=True)

res = {}
for nm, ch in grid:
    c = dict(BASE, **ch)
    res[nm] = [tune_tau(run(*data[sd][:5], c), data[sd][5])[0] for sd in data]
ref = np.array(res["current"])
print(f"\ncurrent {ref.mean():.4f} [" + " ".join(f"{x:.4f}" for x in ref) + "]")
for nm, v in res.items():
    d = np.array(v) - ref
    flag = " ◎" if (d > 0).all() and d.mean() >= 0.005 else (" ○" if (d > 0).all() else "")
    print(f"  {nm:22s} {np.mean(v):.4f}  vs current {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d) + "]" + flag, flush=True)

# combine the single changes that were positive on both seeds
good = [(nm, ch) for nm, ch in grid[1:] if (np.array(res[nm]) - ref > 0).all()]
keys = {}
for nm, ch in good:
    (k, v), = ch.items()
    if k not in keys or np.mean(res[nm]) > np.mean(res[keys[k][0]]): keys[k] = (nm, v)
if len(keys) >= 2:
    print("\nCombinations:", {k: v for k, (_, v) in keys.items()})
    items = list(keys.items())
    for r in range(2, len(items) + 1):
        for comb in itertools.combinations(items, r):
            c = dict(BASE, **{k: v for k, (_, v) in comb})
            v = [tune_tau(run(*data[sd][:5], c), data[sd][5])[0] for sd in data]
            d = np.array(v) - ref
            print(f"  {'+'.join(f'{k}={vv}' for k, (_, vv) in comb):40s} {np.mean(v):.4f}  vs current {d.mean():+.4f}  ["
                  + " ".join(f"{x:+.4f}" for x in d) + "]", flush=True)
