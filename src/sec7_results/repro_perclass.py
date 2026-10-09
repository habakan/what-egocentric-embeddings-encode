"""Reproduce the paper's table tab:perclass (Appendix B) and the numbers in the main text.

On the seed-42 evaluation clips, report per-class F1 for the inertial classifier / the video-only classifier / the final system,
and report the inertial classifier's recall separately for arm-sensor (left/right wrist) and leg-sensor (left/right ankle) clips.
Also report the values for "choosing the better modality per class takes macro-F1 from X to Y".
tau is chosen once over the whole CV for each predictor (same procedure as the paper).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES
from final_submit import load_vgraph_oof
from postprocess import prior_correct, tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build

SEED = 42
ARM, LEG = (0, 3), (1, 2)          # 0 R wrist, 1 R ankle, 2 L ankle, 3 L wrist


def decide(Q, y):
    Q = Q / Q.sum(1, keepdims=True)
    return prior_correct(Q, tune_tau(Q, y)[1]).argmax(1)


def main():
    out, y, sbj = build(SEED)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        d = load_vgraph_oof(tg, TAGS[0], seed=SEED)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj):
            M[sbj == s] = d[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    sens = load_all(TAGS, W, "exp013_nn_video_5fold", seed=SEED)[4]
    pi, pv, pf = decide(out["base"], y), decide(V, y), decide(out["final"], y)
    lab = list(range(N_CLASSES))
    fi, fv, ff = (f1_score(y, p, labels=lab, average=None, zero_division=0) for p in (pi, pv, pf))
    arm = np.isin(sens, ARM)
    ra = recall_score(y[arm], pi[arm], labels=lab, average=None, zero_division=0)
    rl = recall_score(y[~arm], pi[~arm], labels=lab, average=None, zero_division=0)
    order = np.argsort(fi - fv)
    print(f"{'class':30s} {'inert':>6s} {'video':>6s} {'final':>6s} {'arm':>6s} {'leg':>6s}")
    for c in order:
        print(f"{CLASS_NAMES[c]:30s} {fi[c]:6.2f} {fv[c]:6.2f} {ff[c]:6.2f} {ra[c]:6.2f} {rl[c]:6.2f}")
    print(f"{'macro-F1':30s} {fi.mean():6.3f} {fv.mean():6.3f} {ff.mean():6.3f}")
    print(f"better modality per class: {fi.mean():.3f} -> {np.maximum(fi, fv).mean():.3f}")


if __name__ == "__main__":
    main()
