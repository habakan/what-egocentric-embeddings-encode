"""Build a table, per clip x limb, of whether the acceleration has missing values (NaN/inf).

Output: experiments/prep/valid_mask.npy  (n_clip, 4) bool. (clip, limb) pairs with missing values are excluded from the validation and training assignment.
A script version of a command that was run interactively once during development (same content).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK


def main():
    a = np.load(WORK / "prep" / "inertial.npy", mmap_mode="r")          # (n_clip, 4, 50, 3)
    n = a.shape[0]
    valid = np.ones((n, 4), bool)
    for i in range(0, n, 20000):
        b = np.asarray(a[i:i + 20000], np.float32)
        valid[i:i + 20000] = np.isfinite(b).all(axis=(2, 3))
    np.save(WORK / "prep" / "valid_mask.npy", valid)
    print("valid per limb", valid.sum(0), "of", n)


if __name__ == "__main__":
    main()
