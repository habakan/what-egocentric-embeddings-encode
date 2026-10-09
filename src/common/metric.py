"""Own implementation of the official metric. Host description "average of the F1-scores of each activity class" = macro-F1.
⚠ Whether the null class is included in the average is unconfirmed, so compute both and look at the difference."""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import numpy as np
from sklearn.metrics import f1_score

from config import N_CLASSES


def macro_f1(y_true, y_pred, include_null: bool = True) -> float:
    labels = list(range(N_CLASSES)) if include_null else list(range(1, N_CLASSES))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def per_class_f1(y_true, y_pred) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=list(range(N_CLASSES)),
                    average=None, zero_division=0)
