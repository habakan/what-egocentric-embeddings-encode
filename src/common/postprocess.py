"""Post-processing for macro-F1: class-prior correction and search for per-class weights.

macro-F1 weights minority classes equally, so a plain argmax leans too much towards the large class (null).
The correction is always decided on OOF (= test-equivalent conditions), never on LB.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import numpy as np

from metric import macro_f1


def prior_correct(P, tau, prior=None):
    """p / prior^tau. tau=0 means no correction, tau=1 removes the prior completely. One parameter, so hard to overfit."""
    pr = P.mean(0) if prior is None else prior
    return P / np.power(pr + 1e-9, tau)


def tune_tau(P, y, taus=np.linspace(0, 2.0, 41)):
    best = (-1, 0.0)
    for t in taus:
        f = macro_f1(y, prior_correct(P, t).argmax(1))
        if f > best[0]:
            best = (f, t)
    return best              # (f1, tau)


def tune_weights(P, y, n_iter=3, grid=np.linspace(-1.0, 1.0, 21)):
    """Coordinate ascent on per-class log weights. Stronger than tau but has 19 parameters and overfits easily."""
    logw = np.zeros(P.shape[1])
    logP = np.log(P + 1e-9)
    best = macro_f1(y, (logP + logw).argmax(1))
    for _ in range(n_iter):
        for c in range(P.shape[1]):
            cur = logw[c]
            for g in grid:
                logw[c] = cur + g
                f = macro_f1(y, (logP + logw).argmax(1))
                if f > best:
                    best, cur = f, logw[c]
            logw[c] = cur
    return best, logw


def gate_blend(Q, V, g0, gated=False):
    """Mix the independent video probabilities V into the refined probabilities Q by log-linear pooling.

        Q' ∝ Q^(1-g) · V^g

    If gated=True, g = g0·(1 - max_c Q_ic) varies per window (the lower the confidence, the more weight on video).
    Measured, the **uniform version (0.8420)** is stronger than the gated one (0.8365); gating was not essential.
    The qualifier that matters is not "uniform or gated" but **"before or after refinement"**.
    The independence principle forbids the earlier stage (blending the bases); at the later stage an independent view can be added.
    One parameter, g0. The dose response has an interior optimum (peak at g0=0.2, breaks down at 0.5).
    """
    if g0 == 0:
        return Q
    g = g0 * (1.0 - Q.max(1, keepdims=True)) if gated else g0
    lg = (1 - g) * np.log(np.clip(Q, 1e-9, None)) + g * np.log(np.clip(V, 1e-9, None))
    lg -= lg.max(1, keepdims=True)
    R = np.exp(lg)
    return R / R.sum(1, keepdims=True)
