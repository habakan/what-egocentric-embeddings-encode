"""Measure the effect of adding "inertial similarity between windows of the same sensor placement" edges to the graph.

Inertial data should separate activities that look similar in video but differ (e.g. the various stretching ones).
On test each window has only 1 sensor, so comparisons are limited to windows of the same sensor placement
(within subject, about 25% per sensor = 700-1300 windows).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from eval_pipeline import load_eval_set
from postprocess import tune_tau
from refine import knn_graph, propagate

PREP = WORK / "prep"


def combined_graph(Fv, Xi, sens, k_v, k_i):
    """Use the union of video kNN and same-sensor inertial kNN as edges"""
    n = len(Fv)
    nn_v, w_v = knn_graph(Fv, k_v)
    nn_list = [nn_v]; w_list = [w_v]
    if k_i > 0:
        nn_i = np.full((n, k_i), 0, np.int64)
        w_i = np.full((n, k_i), -1e9, np.float32)
        Z = Xi - Xi.mean(0)
        Z /= Z.std(0) + 1e-6
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        for s in np.unique(sens):
            idx = np.where(sens == s)[0]
            if len(idx) < k_i + 3:
                continue
            S = Z[idx] @ Z[idx].T
            np.fill_diagonal(S, -np.inf)
            top = np.argpartition(-S, k_i, axis=1)[:, :k_i]
            nn_i[idx] = idx[top]
            w_i[idx] = np.take_along_axis(S, top, 1)
        nn_list.append(nn_i); w_list.append(w_i)
    return np.concatenate(nn_list, 1), np.concatenate(w_list, 1)


def main(a):
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    P, y, sbj, F = load_eval_set(a.tags, [1.0] * len(a.tags))
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile2, sens2 = tile[ok], sens[ok]
    rows = tile2 + sens2 * n
    oof0 = np.load(WORK / a.tags[0] / "oof.npy")[rows]
    keep = oof0.sum(1) > 1e-6
    rows, sens2 = rows[keep], sens2[keep]
    Xi = np.asarray(np.load(PREP / "feat_inertial.npy", mmap_mode="r")[rows], np.float32)

    def pipeline(k_v, k_i, stages=(4.0, 4.0)):
        Q = P.copy()
        for s in np.unique(sbj):
            m = sbj == s
            Fv = F[s] - F[s].mean(0)
            Fv /= np.linalg.norm(Fv, axis=1, keepdims=True) + 1e-8
            nn, w = combined_graph(Fv, Xi[m], sens2[m], k_v, k_i)
            q = propagate(P[m], nn, w, 0.75, 5)
            for lam in stages:
                G = np.concatenate([Fv, lam * np.sqrt(q)], 1)
                nn2, w2 = knn_graph(G, k_v, center=False)
                if k_i > 0:
                    nn2 = np.concatenate([nn2, nn[:, k_v:]], 1)
                    w2 = np.concatenate([w2, w[:, k_v:]], 1)
                q = propagate(q, nn2, w2, 0.75, 5)
            Q[m] = q
        return tune_tau(Q, y)

    print(f"video only k=30          : {pipeline(30, 0)[0]:.4f}")
    for k_i in [5, 10, 20]:
        f1, tau = pipeline(30, k_i)
        print(f"video k=30 + inertial k={k_i:<2d}   : {f1:.4f} (tau={tau:.2f})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["exp001_lgb_inertial", "exp014_nn_inertial_rot"])
    main(p.parse_args())
