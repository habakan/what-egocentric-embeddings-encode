"""Is the degradation from subtracting a single-limb prediction caused by "per-placement offsets"?

Use the single-limb linear prediction (xl_only of probe_bayes_impute: E[x_miss|x_l] fed into the Ridge four-limb neutralizer) and
the single-limb MLP (H1 of probe_decomp).
  fixed limb (diagnostic)        : subtract the right-arm prediction on all windows (not usable on test; removes only the placement mismatch)
  mixed (current)                : subtract the prediction of the limb worn in each window
  mixed+placement centering      : subtract the mean prediction per subject x placement before use (usable on test)
  mixed+placement standardizing  : additionally equalize prediction norms per subject x placement (usable on test)
m=0 / m=30, 2 seeds. Reference: linear four-limb oracle m=30 +0.0029, mixed m=30 -0.0013.
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
xl = np.load(WORK / "decomp" / "bayes.npz")["xl_only"]
tileA = d["tile"]


def by_group(h, sb, sens, scale):
    h = h.copy()
    ref = np.linalg.norm(h, axis=1).mean()
    for s in np.unique(sb):
        for l in range(4):
            m = (sb == s) & (sens == l)
            if m.sum() == 0: continue
            h[m] -= h[m].mean(0)
            if scale:
                h[m] *= ref / (np.linalg.norm(h[m], axis=1).mean() + 1e-8)
    return h


res = {}
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
    Yr = csub(d["Y"][pos], sb)
    per = lambda A: {s: A[sb == s] for s in np.unique(sb)}
    lin_fix = xl[0, pos]; lin_mix = xl[sens, pos]; mlp_mix = d["H1"][sens, pos]
    hs = {"linear fixed limb (right arm)": lin_fix, "linear mixed": lin_mix,
          "linear mixed+placement centering": by_group(lin_mix, sb, sens, False),
          "linear mixed+placement standardizing": by_group(lin_mix, sb, sens, True),
          "MLP mixed+placement centering": by_group(mlp_mix, sb, sens, False)}
    cfgs = [("raw m=30 (current)", Yr, 30)]
    for nm, h in hs.items():
        cfgs += [(f"{nm} m=0", Yr - h, 0), (f"{nm} m=30", Yr - h, 30)]
    for nm, F, mm in cfgs:
        res.setdefault(nm, []).append(tune_tau(final_with(P, per(F), Vg, V, sb, mm), yy)[0])
    print(f"  seed {sd} done", flush=True)
ref = np.array(res["raw m=30 (current)"])
for nm, v in res.items():
    v = np.array(v)
    print(f"  {nm:24s} {v.mean():.4f}  vs current {np.mean(v - ref):+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]")
