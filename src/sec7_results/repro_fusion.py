"""Re-measure, with the current final system, paper §7.3 "fusion loses in every form" and "propagation already fuses the sensors".

Final system = same configuration as "final" in probe_prf.build (top 30 component removal, 3 stages, pooling of video evidence after each stage).
Only the base (the propagated prediction) is swapped. Mean of 3 seeds (42/7/123). One tau per configuration over the whole CV.
  inertial       : probabilities P of the inertial classifier (the paper's system)
  video+inertial : probabilities U of a network taking both as input (exp013, fusion in the classifier)
  mixture        : geometric mean of P and U (mixture of the 2 bases)
  teacher        : teacher model that sees all four limbs (teacher01, privileged information)
  distilled      : P with exp017 among the inertial classifiers replaced by exp021, distilled from the teacher
Report macro-F1 standalone (no propagation) and after passing through the system.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W
from repro_msweep import system

SEEDS = [42, 7, 123]
TAGS_DISTILL = [t if t != "exp017_nn_aux" else "exp021_distill_w05" for t in TAGS]


def norm(X):
    s = X.sum(1, keepdims=True)
    return np.divide(X, s, out=np.full_like(X, 1.0 / N_CLASSES), where=s > 1e-9)


def main():
    res = {}
    for sd in SEEDS:
        P, y, sbj, F0 = load_eval_set(TAGS, W, seed=sd)
        Pd = load_eval_set(TAGS_DISTILL, W, seed=sd)[0]
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        U = np.empty_like(P)
        for s in np.unique(sbj):
            U[sbj == s] = Vg[s]
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            d = load_vgraph_oof(tg, TAGS[0], seed=sd)
            M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj):
                M[sbj == s] = d[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)[-1]
        T = norm(np.load(WORK / "teacher01" / "oof_win.npy")[tile].astype(np.float32))
        bases = {"inertial P (paper)": norm(P), "video+inertial U": norm(U), "mixture sqrt(P U)": norm(np.sqrt(norm(P) * norm(U))),
                 "teacher (four limbs)": T, "distilled (exp017->exp021)": norm(Pd)}
        for nm, B in bases.items():
            res.setdefault((nm, "standalone"), []).append(tune_tau(B, y)[0])
            res.setdefault((nm, "system"), []).append(tune_tau(system(B, sbj, F0, Vg, V), y)[0])
        print(f"seed {sd} done", flush=True)
    print(f"\n{'base':24s} {'standalone':>8s} {'system':>8s}")
    for nm in bases:
        print(f"{nm:24s} {np.mean(res[(nm, 'standalone')]):8.4f} {np.mean(res[(nm, 'system')]):8.4f}")


if __name__ == "__main__":
    main()
