"""Reproduce the paper's figure fig:pcdrop (b) (sweep over the number m of top PCs removed) and Appendix D explanations 4--6 (alpha / iterations / gamma at m=0).

With the final system (same config as "final" in probe_prf.build), vary only the number m of PCs removed from the graph's video features.
At m=0, vary one at a time the propagation strength alpha, the iterations per stage, and the weight gamma of the probability block in the graph features.
Mean and standard deviation over 3 seeds (42/7/123). Results are written back to "prod" and "knobs" in paper/figdata/pcdrop.json.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_prf import TAGS, W
from refine import drop_top_pcs, multistage_at

SEEDS = [42, 7, 123]
FIGDATA = Path(__file__).parents[2] / "paper" / "figdata" / "pcdrop.json"


def system(P, sbj, F0, Vg, V, m=30, alpha=0.75, iters=5, gamma=0.5):
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        msk = sbj == s
        f = drop_top_pcs(F0[s], m) if m else F0[s]
        f = f - f.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G = np.concatenate([f, gamma * np.sqrt(Vg[s])], 1)
        Q[msk] = multistage_at(P[msk], G, V[msk], [4.0, 4.0], k=30, alpha=alpha, iters=iters,
                               temp=1.4, g=0.2, where="every")
    return Q


def main():
    data = []
    for sd in SEEDS:
        P, y, sbj, F0 = load_eval_set(TAGS, W, seed=sd)
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            d = load_vgraph_oof(tg, TAGS[0], seed=sd)
            M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj):
                M[sbj == s] = d[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        data.append((P, y, sbj, F0, Vg, V))

    def score(**kw):
        v = [tune_tau(system(P, sbj, F0, Vg, V, **kw), y)[0] for P, y, sbj, F0, Vg, V in data]
        return float(np.mean(v)), float(np.std(v))

    ms = [0, 5, 10, 20, 30, 50, 80, 120]
    prod = {"m": ms, "f1": [], "std": []}
    for m in ms:
        f, sdv = score(m=m); prod["f1"].append(round(f, 4)); prod["std"].append(round(sdv, 4))
        print(f"m={m:3d}  {f:.4f} ± {sdv:.4f}", flush=True)
    knobs = {"alpha": {"x": [0.45, 0.55, 0.65, 0.75]}, "iters": {"x": [1, 2, 3, 5]},
             "gamma": {"x": [0.25, 0.5, 1.0, 2.0, 4.0]}}
    for name, d in knobs.items():
        d["f1"] = [round(score(m=0, **{name: x})[0], 4) for x in d["x"]]
        print(f"m=0, {name}: " + "  ".join(f"{x}: {f:.4f}" for x, f in zip(d["x"], d["f1"])), flush=True)
    knobs["target"] = prod["f1"][ms.index(30)]
    js = json.loads(FIGDATA.read_text())
    js["prod"], js["knobs"] = prod, knobs
    js["_source"] = "src/probe_factor.py (comp/purity), src/repro_msweep.py (prod, knobs)"
    FIGDATA.write_text(json.dumps(js, indent=1))
    print("->", FIGDATA)


if __name__ == "__main__":
    main()
