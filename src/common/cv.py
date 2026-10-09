"""Single source of truth for CV.
test is 4 unseen subjects (sbj_22..25), so split by subject.
sbj_X and sbj_X_2 are the same person, so they always go in the same fold (group by sbj_id, not recording)."""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import numpy as np
from sklearn.model_selection import GroupKFold

from config import SEED


def subject_folds(sbj_ids, n_splits: int = 5, seed: int = SEED):
    """Subject-level GroupKFold. With 5 folds, valid has 4-5 subjects/fold, close to the granularity of test (4 subjects).

    Returns: list of (train_idx, valid_idx)
    """
    sbj_ids = np.asarray(sbj_ids)
    uniq = np.unique(sbj_ids)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    # to shuffle subjects before passing to GroupKFold, map groups to permuted IDs
    remap = {s: perm[i] for i, s in enumerate(uniq)}
    groups = np.array([remap[s] for s in sbj_ids])
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(sbj_ids)), groups=groups)), groups
