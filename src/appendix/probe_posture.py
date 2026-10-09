"""Test a hypothesis formed by looking at raw video frames: are the top principal components "head orientation"?

Observation (checked on WEAR raw video sbj_0, frames taken at the middle of segments of the 19 classes):
  The head-mounted camera **never shows the exercising wearer**. What it shows is the ground (grass and fallen leaves),
  occasionally hands and feet, and the sky and trees. Push-ups/burpees/lunges "look down at the nearby ground",
  sit-ups/hip rotations "look up at the sky and trees", jogging is "grass ahead flowing past".
  So what represents the activity on the video side is not the subject's motion but **head orientation and how the ground looks**.

Hypothesis that follows:
  top PCs = head orientation (+ overall brightness). This correlates strongly with activity (hence the high eta^2), but
  **it links different activities that share a posture**. Push-ups, burpees, and lunges are all "looking down".
  And that confusion happens **in the same pairs as the inertial side's confusion**, because an accelerometer that sees
  only one limb also mixes up exercises with the same posture.
  If so, removing the top PCs cuts "the correlation between graph errors and base-classifier errors".
  Propagation helps when neighbours **err differently from oneself**, so this is a gain even if purity drops.

Test (if this is false, the hypothesis is discarded):
  N_m[c,c'] = in the graph after removing m principal components, the fraction of neighbours of windows with true label c that have true label c' (row-normalized)
  B[c,c']   = confusion matrix of the inertial base classifier (row-normalized)
  Compare the correlations of the off-diagonal entries, corr(N_0, B) and corr(N_30, B).
    hypothesis true  -> corr(N_0, B) > corr(N_30, B)  (removal cuts the error correlation)
    hypothesis false -> no difference, or the opposite direction

  Also directly measure "the conditional probability that, when a neighbour is wrong, the base makes the same error".
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES, SEED
from probe_invariance import drop_top_pcs, load_all


def neigh_mat(F, y, k):
    """Row-normalized neighbour label matrix N[c,c']."""
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(Fn) - 2)
    nn = np.argpartition(-S, kk, axis=1)[:, :kk]
    M = np.zeros((N_CLASSES, N_CLASSES))
    for c in range(N_CLASSES):
        m = y == c
        if m.sum() == 0:
            continue
        lab, cnt = np.unique(y[nn[m]], return_counts=True)
        M[c, lab] = cnt
    return M, nn


def null_mat(y, purity, k, rng):
    """N for random neighbours matched only in purity. A control to see whether corr(N,B) moves only with
    "how diffuse the graph is". Draw the same label with probability purity, and the rest uniformly from other labels."""
    M = np.zeros((N_CLASSES, N_CLASSES))
    for c in range(N_CLASSES):
        n = int((y == c).sum())
        if n == 0:
            continue
        same = rng.binomial(n * k, purity)
        M[c, c] += same
        others = np.array([o for o in range(N_CLASSES) if o != c and (y == o).sum() > 0])
        if len(others) == 0:
            continue
        # draw other labels in proportion to their counts (the limit where the graph carries no information)
        w = np.array([(y == o).sum() for o in others], float)
        w /= w.sum()
        M[c, others] += (n * k - same) * w
    return M


def rownorm(M):
    s = M.sum(1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


def offdiag(M):
    return M[~np.eye(N_CLASSES, dtype=bool)]


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    agg = {}
    for sd in a.seeds:
        P, V, y, sbj, sens, start, Fv, _, _ = load_all(a.tags, w, a.vbase, seed=sd)
        us = np.unique(sbj)
        pred = P.argmax(1)

        B = np.zeros((N_CLASSES, N_CLASSES))
        for t, p in zip(y, pred):
            B[t, p] += 1
        Bn = rownorm(B)

        for m in a.drop_pcs:
            N = np.zeros((N_CLASSES, N_CLASSES))
            pure_num = pure_den = 0
            same_err = tot_err = 0
            for s in us:
                Fs = drop_top_pcs(Fv[s], m) if m else Fv[s]
                ms = sbj == s
                Nm, nn = neigh_mat(Fs, y[ms], a.k)
                N += Nm
                pure_num += np.trace(Nm); pure_den += Nm.sum()
                # when a neighbour has a wrong label, the fraction where the base prediction matches "that wrong label"
                ys, ps = y[ms], pred[ms]
                wrong = ys[nn] != ys[:, None]
                tot_err += wrong.sum()
                same_err += ((ys[nn] == ps[:, None]) & wrong).sum()
            Nn = rownorm(N)
            r = np.corrcoef(offdiag(Nn), offdiag(Bn))[0, 1]
            pur = pure_num / max(pure_den, 1)
            rng2 = np.random.RandomState(sd)
            Nnull = rownorm(sum(null_mat(y[sbj == s], pur, a.k, rng2) for s in us))
            rn = np.corrcoef(offdiag(Nnull), offdiag(Bn))[0, 1]
            agg.setdefault(m, []).append((r, rn, same_err / max(tot_err, 1), pur))
        print(f"  seed {sd} done", flush=True)

    print(f"\n=== Top-PC removal and the correlation of graph errors x base errors ({len(a.seeds)} seeds) ===")
    print(f"  {'PCs removed':>7s}{'corr(N,B)':>12s}{'same-purity random':>15s}{'diff':>9s}"
          f"{'wrong nbr=base pred':>19s}{'purity':>9s}")
    for m in a.drop_pcs:
        v = np.array(agg[m])
        print(f"  {m:7d}{v[:,0].mean():12.4f}{v[:,1].mean():15.4f}"
              f"{v[:,0].mean()-v[:,1].mean():9.4f}{v[:,2].mean():19.4f}{v[:,3].mean():9.4f}")
    print("\n  Hypothesis predicts: removal lowers corr (graph errors become unrelated to base errors)")
    print("  Control: if corr also rises with same-purity random, the rise is an artifact of 'the graph just becoming diffuse'")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--vbase", required=True)
    p.add_argument("--drop-pcs", nargs="*", type=int, default=[0, 10, 30, 50])
    p.add_argument("--seeds", nargs="*", type=int, default=[SEED, 7])
    p.add_argument("--k", type=int, default=30)
    main(p.parse_args())
