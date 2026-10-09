import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import numpy as np

from postprocess import gate_blend


def knn_graph(F, k, mutual=True, center=True):
    if center:
        F = F - F.mean(0)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    S = F @ F.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(F) - 2)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    w = np.take_along_axis(S, nn, 1)
    if mutual:
        r = np.zeros(S.shape, bool)
        r[np.arange(len(nn))[:, None], nn] = True
        keep = r[np.arange(len(nn))[:, None], nn] & r[nn, np.arange(len(nn))[:, None]]
        w = np.where(keep, w, -1e9)
    return nn, w


def propagate(P, nn, w, alpha, iters, sharp=10.0, conf_pow=0.0, conf_alpha=0.0,
              node_u=None, u_beta=0.0):
    """Propagate probabilities along neighbours.

    conf_pow > 0: weight each neighbour's contribution by "that window's confidence^conf_pow".
    conf_alpha > 0: the less confident a window, the more it discards its own prediction and relies on neighbours (per-node alpha).

    Both address the fact that "windows from sensors on irrelevant limbs carry no information".
    Empirically, classes 6/8 had an accuracy of exactly 0.00 on leg-sensor windows.

    node_u / u_beta: drive per-node alpha with an arbitrary **external uncertainty** (normalized to [0,1]).
    Whereas conf_alpha is fixed to P.max(1), here the driving quantity can be swapped.
    The 09-03 rejection was of the max-prob **implementation**, not of the mechanism
    (a flat distribution already acts as a weak vote in the propagation arithmetic, so it was a double correction).
    Epistemic uncertainty points to windows where "the distribution is sharp but the model does not know",
    which lies outside that argument.
    """
    ww = np.exp((w - w.max(1, keepdims=True)) * sharp)
    if conf_pow > 0:
        c = P.max(1)
        ww = ww * np.power(c[nn], conf_pow)
    ww /= ww.sum(1, keepdims=True) + 1e-12
    if node_u is not None and u_beta != 0.0:
        u = np.asarray(node_u, np.float64)
        u = (u - u.min()) / (u.max() - u.min() + 1e-9)
        a = np.clip(alpha + (1 - alpha) * u_beta * u, 0, 0.99)[:, None]
    elif conf_alpha > 0:
        c = P.max(1)
        cn = (c - c.min()) / (c.max() - c.min() + 1e-9)
        a = alpha + (1 - alpha) * conf_alpha * (1 - cn)      # raise alpha for lower confidence
        a = np.clip(a, 0, 0.99)[:, None]
    else:
        a = alpha
    Q = P.copy()
    for _ in range(iters):
        Q = (1 - a) * P + a * (Q[nn] * ww[:, :, None]).sum(1)
    return Q


def sharpen(Q, temp):
    """Sharpen probabilities. The aim is to suppress within-segment fluctuations (purity 0.85 in the error diagnosis)."""
    if temp == 1.0:
        return Q
    R = np.power(np.clip(Q, 1e-9, None), temp)
    return R / R.sum(1, keepdims=True)


def multistage(P, F, stages, k=30, alpha=0.75, iters=5, temp=1.0,
               conf_pow=0.0, conf_alpha=0.0):
    """stages: lam for each stage (weight of the probability features relative to the video features). Stage 1 corresponds to lam=0.

    Running stages too strongly (alpha 0.85 / iters 10) over-smooths the second stage and hurts.
    Empirically alpha=0.75, iters=5 is stable."""
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    nn, w = knn_graph(Fn, k, center=False)
    Q = sharpen(propagate(P, nn, w, alpha, iters, conf_pow=conf_pow,
                          conf_alpha=conf_alpha), temp)
    for lam in stages:
        G = np.concatenate([Fn, lam * np.sqrt(Q)], 1)
        nn, w = knn_graph(G, k, center=False)
        Q = sharpen(propagate(Q, nn, w, alpha, iters, conf_pow=conf_pow,
                              conf_alpha=conf_alpha), temp)
    return Q


def multistage_at(P, F, V, stages, k=30, alpha=0.75, iters=5, temp=1.0,
                  g=0.0, where="none"):
    """Multi-stage refinement while injecting the video probabilities V.

    where: 'none' | 'last' (adopted in exp013, LB +0.0142) | 'every' (every stage) | 'once' (right after stage 1)
    The total video weight is fixed to g (every uses g/n per stage). Without equalizing the weight,
    "effect of position" and "excess weight" cannot be distinguished (a confound hit in probe_blend_post).

    Injecting between stages makes the video information **also enter the next stage's graph construction** (the graph features are
    [Fn, lam*sqrt(Q)], so when Q changes the edges change). Empirically every > last.
    """
    n_inject = len(stages) + 1 if where == "every" else 1
    gi = g / n_inject if where == "every" else g

    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    nn, w = knn_graph(Fn, k, center=False)
    Q = sharpen(propagate(P, nn, w, alpha, iters), temp)
    if where in ("every", "once"):
        Q = gate_blend(Q, V, gi)
    for lam in stages:
        G = np.concatenate([Fn, lam * np.sqrt(Q)], 1)
        nn, w = knn_graph(G, k, center=False)
        Q = sharpen(propagate(Q, nn, w, alpha, iters), temp)
        if where == "every":
            Q = gate_blend(Q, V, gi)
    if where == "last":
        Q = gate_blend(Q, V, g)
    return Q


def drop_top_pcs(F, m):
    """Return the features with the top m components of within-subject PCA removed.

    Preprocessing for the affinity graph. It does not use ordering, so it can run on test.
    Empirically, increasing m monotonically raises neighbour activity purity and the final score
    (0.8151 -> 0.8277 at m=10, plain configuration, 3 seeds).
    Note: what is being removed is not understood. The temporal-drift hypothesis was wrong
    (the time gap to neighbours does not widen; it shrinks). A preprocessing step whose effect alone has been confirmed.
    """
    if m <= 0:
        return F
    c = F - F.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    B = Vt[:min(m, Vt.shape[0])]
    return c - (c @ B.T) @ B
