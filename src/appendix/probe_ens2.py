"""Graph ensemble — rebuilt using **only perturbations that also exist on the test side**.

Flaw in the previous measurement:
  the members used two video arrays, head / sync. But that difference is
  "how the continuous per-subject npy is cut into windows", which **exists only on the training side**.
  The test is distributed with the organisers having already selected the 15 frames, so there is only one way to cut windows.
  So at submission time the sync member becomes identical to the head member, and one axis of diversity disappears.
  This is exactly the kind of construction where OOF mispredicts LB, so the measurement is rebuilt.

Replacement:
  **which range of the 15 distributed frames to use for mean pooling** (all 15 / first 10 / last 10).
  It exists on both the training and test sides, and is a temporal perturbation of the same kind as head/sync.

Members (all g=0.2, where=every, alpha=0.75, iters=5):
  1. rows=0:15 m=30 k=30 stages[4,4] temp1.4   <- current production config
  2. rows=0:15 m=0  k=30
  3. rows=0:15 m=30 k=15
  4. rows=0:10 m=30 k=30
  5. rows=5:15 m=30 k=30
  6. rows=0:15 m=30 k=30 stages[4,4,4]
  7. rows=0:15 m=30 k=30 temp1.0

Decision:
  check, with paired differences (same seed vs same seed), whether it beats the current config on every seed.
  The marginal sigma is not used because it is mostly noise shared across seeds (which sensor is drawn).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, WORK
from final_submit import load_vgraph_oof
from postprocess import gate_blend, tune_tau
from refine import drop_top_pcs, knn_graph, propagate, sharpen

PREP = WORK / "prep"

MEMBERS = [
    ("m30 k30 all15",    (0, 15), 30, 30, [4.0, 4.0], 1.4),
    ("m0  k30 all15",    (0, 15), 0, 30, [4.0, 4.0], 1.4),
    ("m30 k15 all15",    (0, 15), 30, 15, [4.0, 4.0], 1.4),
    ("m30 k30 first10",    (0, 10), 30, 30, [4.0, 4.0], 1.4),
    ("m30 k30 last10",    (5, 15), 30, 30, [4.0, 4.0], 1.4),
    ("m30 k30 3stage",     (0, 15), 30, 30, [4.0, 4.0, 4.0], 1.4),
    ("m30 k30 t1.0",    (0, 15), 30, 30, [4.0, 4.0], 1.0),
]


def multistage(P, F, V, stages, k, alpha, iters, temp, g):
    gi = g / (len(stages) + 1)
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    nn, w = knn_graph(Fn, k, center=False)
    Q = gate_blend(sharpen(propagate(P, nn, w, alpha, iters), temp), V, gi)
    for lam in stages:
        G = np.concatenate([Fn, lam * np.sqrt(Q)], 1)
        nn, w = knn_graph(G, k, center=False)
        Q = gate_blend(sharpen(propagate(Q, nn, w, alpha, iters), temp), V, gi)
    return Q


def load_oof(tags, weights, vbase, seed):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile))
    ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    rows = tile + sens * n
    P = None
    for t, w in zip(tags, weights):
        o = np.load(WORK / t / "oof.npy")[rows]
        s = o.sum(1, keepdims=True)
        o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(weights)
    keep = P.sum(1) > 1e-6
    tile, P = tile[keep], P[keep] / P[keep].sum(1, keepdims=True)
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    Vd = load_vgraph_oof(vbase, tags[0], seed=seed)
    V = np.empty_like(P)
    for s in np.unique(sbj):
        V[sbj == s] = Vd[s]
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    raw = {s: np.asarray(vid[tile[sbj == s]], np.float32) for s in np.unique(sbj)}
    return P, V, y, sbj, raw, Vd


def graph_feat(raw_s, r0, r1, m, vgraph_s, gamma):
    """Build graph features in the same order as production (final_submit --vgraph).

    mean-pool -> remove top PCs -> row-normalise -> concatenate gamma*sqrt(video model probabilities).
    Without this concatenated term the absolute OOF comes out about 0.006 lower than production.
    """
    F = raw_s[:, r0:r1].mean(1)
    if m:
        F = drop_top_pcs(F, m)
    if vgraph_s is None:
        return F
    F = F - F.mean(0)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    return np.concatenate([F, gamma * np.sqrt(vgraph_s)], 1)


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    res = {}
    for sd in a.seeds:
        P, V, y, sbj, raw, Vd = load_oof(a.tags, w, a.vbase, sd)
        Qs = {}
        for nm, (r0, r1), m, k, stg, tp in MEMBERS:
            Q = np.empty_like(P)
            for s in np.unique(sbj):
                msk = sbj == s
                F = graph_feat(raw[s], r0, r1, m, Vd[s] if a.gamma else None, a.gamma)
                Q[msk] = multistage(P[msk], F, V[msk], stg, k, a.alpha, a.iters, tp, a.g)
            Qs[nm] = Q
            res.setdefault(f"single {nm}", []).append(tune_tau(Q, y)[0])
        keys = list(Qs)
        combos = [tuple(keys)] + [c for r in (4, 5) for c in itertools.combinations(keys, r)][:a.max_combos]
        for c in combos:
            lg = np.mean([np.log(Qs[x] + 1e-9) for x in c], 0)
            Qg = np.exp(lg); Qg /= Qg.sum(1, keepdims=True)
            res.setdefault("geo " + "+".join(x.split()[0] + x.split()[-1] for x in c), []).append(
                tune_tau(Qg, y)[0])
        print(f"  seed {sd} done", flush=True)

    ref = res["single m30 k30 all15"]
    print(f"\n=== Ensemble of perturbations that also exist on the test side ({len(a.seeds)} seeds) ===")
    for nm, v in sorted(res.items(), key=lambda x: -np.mean(x[1]))[:12]:
        line = f"  {np.mean(v):.4f}  {nm}"
        if nm != "single m30 k30 all15":
            d = np.array(v) - np.array(ref)
            line += (f"\n      vs current {d.mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in d)
                     + f"]  all same sign {bool(np.all(d > 0) or np.all(d < 0))}")
        else:
            line += "   <- current production config"
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--vbase", required=True)
    p.add_argument("--g", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.5,
                   help="weight of video probabilities concatenated to graph features (0.5, same as production). 0 disables")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--max-combos", type=int, default=12)
    p.add_argument("--seeds", nargs="*", type=int, default=[SEED, 7, 123, 2024, 31337])
    main(p.parse_args())
