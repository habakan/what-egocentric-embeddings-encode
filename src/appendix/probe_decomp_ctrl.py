"""Control for probe_decomp: shuffle the four-limb prediction within subject before subtracting it.

This adds a perturbation with the same energy in the same subspace, without inertial information.
If the shuffled version works just as well, we cannot say "it worked because the result of motion was subtracted".
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_decomp import csub, final_with
from probe_invariance import load_all
from probe_prf import TAGS, W, build

d = dict(np.load(WORK / "decomp" / "oof.npz"))
res = {}
for sd in [42, 7]:
    out, y, sbj = build(sd); P = out["base"]
    _, _, y2, _, sens, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    pos = np.searchsorted(d["tile"], tile)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    Yrc = csub(d["Y"][pos], sbj); h4 = d["H4"][pos]
    rng = np.random.RandomState(sd)
    hs = h4.copy()
    for s in np.unique(sbj):
        m = np.where(sbj == s)[0]; hs[m] = h4[rng.permutation(m)]
    per = lambda A: {s: A[sbj == s] for s in np.unique(sbj)}
    for nm, F, dr in [("drop30 (current)", Yrc, 30), ("residual four limbs", Yrc - h4, 0),
                      ("residual four limbs shuffled", Yrc - hs, 0), ("raw", Yrc, 0)]:
        res.setdefault(nm, []).append(tune_tau(final_with(P, per(F), Vg, V, sbj, dr), y)[0])
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["drop30 (current)"])
for nm, v in res.items():
    v = np.array(v)
    print(f"  {nm:22s} {v.mean():.4f}  vs current {np.mean(v - ref):+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]")
