"""Measure the effect of consolidating with "cluster means" after graph refinement.

The oracle (averaging over true segments) is 0.927, so a hard grouping equivalent to segments should
beat iterative propagation. Run hierarchical clustering within subject (average linkage, cosine) and
sweep the number of clusters. Uses scipy's linkage, so it finishes in seconds even for n≈3000.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK
from eval_pipeline import load_eval_set, run
from postprocess import tune_tau

PREP = WORK / "prep"


def cluster_labels(F, n_clusters, method="average"):
    X = F - F.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0, None)
    Z = linkage(squareform(D, checks=False), method=method)
    return fcluster(Z, n_clusters, criterion="maxclust")


def main(a):
    P, y, sbj, F = load_eval_set(a.tags, a.weights or [1.0] * len(a.tags))
    Q = run(P, y, sbj, F, [4.0, 4.0], 30, 0.75, 5)
    print(f"3-stage refine (current) : {tune_tau(Q, y)[0]:.4f}")

    for src_name, src in [("from base probs", P), ("from refined", Q)]:
        for per in a.per:                       # mean number of windows per cluster
            for blend in a.blend:
                R = Q.copy()
                for s in np.unique(sbj):
                    m = sbj == s
                    n_c = max(2, int(m.sum() / per))
                    lab = cluster_labels(F[s], n_c)
                    df = pd.DataFrame(src[m]); df["g"] = lab
                    mean = df.groupby("g").transform("mean").to_numpy()
                    R[m] = blend * mean + (1 - blend) * Q[m]
                f1, tau = tune_tau(R, y)
                print(f"  {src_name} {per:3d} win/cluster blend={blend} -> {f1:.4f} (tau={tau:.2f})",
                      flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+",
                   default=["exp001_lgb_inertial", "exp014_nn_inertial_rot",
                            "exp015_nn_inertial_s1337", "exp016_nn_inertial_rot30"])
    p.add_argument("--weights", nargs="*", type=float, default=[1.5, 1.0, 1.0, 1.0])
    p.add_argument("--per", nargs="*", type=int, default=[15, 30, 50])
    p.add_argument("--blend", nargs="*", type=float, default=[0.5, 1.0])
    main(p.parse_args())
