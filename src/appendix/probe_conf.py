import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_pipeline import load_eval_set, run
from postprocess import tune_tau

TAGS = ["exp001_lgb_inertial", "exp014_nn_inertial_rot",
        "exp015_nn_inertial_s1337", "exp016_nn_inertial_rot30"]
W = [1.5, 1.0, 1.0, 1.0]


def main():
    P, y, sbj, F = load_eval_set(TAGS, W)
    for cp, ca in [(0, 0), (1, 0), (2, 0), (4, 0), (0, 0.5), (0, 1.0), (2, 0.5), (2, 1.0)]:
        Q = run(P, y, sbj, F, [4.0, 4.0], 30, 0.75, 5, temp=1.4, conf_pow=cp, conf_alpha=ca)
        f, tau = tune_tau(Q, y)
        print(f"  conf_pow={cp} conf_alpha={ca} -> {f:.4f} (tau={tau:.2f})", flush=True)


if __name__ == "__main__":
    main()
