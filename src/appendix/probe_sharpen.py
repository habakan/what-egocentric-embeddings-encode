import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_pipeline import load_eval_set, run
from metric import macro_f1
from postprocess import tune_tau
from sinkhorn import apply_per_subject

TAGS = ["exp001_lgb_inertial", "exp014_nn_inertial_rot",
        "exp015_nn_inertial_s1337", "exp016_nn_inertial_rot30"]
W = [1.5, 1.0, 1.0, 1.0]


def main():
    P, y, sbj, F = load_eval_set(TAGS, W)
    for stages, temp in [([4.0, 4.0], 1.2), ([4.0, 4.0], 1.4), ([4.0, 4.0], 1.5),
                         ([4.0, 4.0], 1.6), ([4.0, 4.0], 1.8), ([4.0], 1.5),
                         ([4.0, 4.0, 4.0], 1.4)]:
        Q = run(P, y, sbj, F, stages, 30, 0.75, 5, temp=temp)
        f_t, tau = tune_tau(Q, y)
        best = max((macro_f1(y, apply_per_subject(Q, sbj, q).argmax(1)), q)
                   for q in np.arange(0.14, 0.50, 0.03))
        print(f"  stages={len(stages)} temp={temp}: tau -> {f_t:.4f} (tau={tau:.2f}) | "
              f"Sinkhorn -> {best[0]:.4f} (q={best[1]:.2f})", flush=True)


if __name__ == "__main__":
    main()
