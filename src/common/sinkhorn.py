"""Imposes, per subject, the structural prior "the 18 non-null classes are uniform" via Sinkhorn.

In the WEAR protocol each subject performs the 18 activities for roughly equal durations.
Measured on the 22 train subjects, the proportions of the 18 non-null classes have a median coefficient of variation of 0.09,
i.e. nearly uniform (each class is about 1/18 = 5.6% of non-null windows).
The only free parameter is the null ratio, so unlike per-class weights (19 of them) it is hard to overfit.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import numpy as np


def sinkhorn_balance(P, q_null, n_iter=50, eps=1e-9):
    """P: (n, 19) probabilities. Scaled so the column sums become [q_null, (1-q_null)/18 x 18]."""
    n, C = P.shape
    target = np.full(C, (1.0 - q_null) / (C - 1))
    target[0] = q_null
    target = target * n
    Q = P.copy() + eps
    for _ in range(n_iter):
        Q /= Q.sum(1, keepdims=True)            # rows: normalize to probabilities
        col = Q.sum(0)
        Q *= (target / (col + eps))[None, :]    # columns: match the target marginal
    Q /= Q.sum(1, keepdims=True)
    return Q


def apply_per_subject(P, sbj, q_null, n_iter=50):
    Q = P.copy()
    for s in np.unique(sbj):
        m = sbj == s
        Q[m] = sinkhorn_balance(P[m], q_null, n_iter)
    return Q
