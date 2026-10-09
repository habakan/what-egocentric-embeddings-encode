"""Apply transductive kNN smoothing to the test predictions and write the submission CSV.

Build a kNN graph of video features within each subject and propagate probabilities. Statistics are computed from test itself only.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import TEST_DIR, WORK
from refine import multistage
from postprocess import prior_correct

PREP = WORK / "prep"


def main(tags, weights, k, alpha, iters, out_name, no_smooth, tau, stages):
    te = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    P = None
    for t, w in zip(tags, weights):
        p = np.load(WORK / t / "test_pred.npy")
        p = p / p.sum(1, keepdims=True)
        P = w * p if P is None else P + w * p
    P /= sum(weights)

    if no_smooth:
        Q = P
    else:
        video = np.load(PREP / "video_raw_test.npy", mmap_mode="r")
        Q = P.copy()
        for s in te["sbj_id"].unique():
            m = (te["sbj_id"] == s).to_numpy()
            F = np.asarray(video[np.where(m)[0]], np.float32).mean(1)
            Q[m] = multistage(P[m], F, stages, k=k, alpha=alpha, iters=iters)
        print("changed by smoothing:", round(float((Q.argmax(1) != P.argmax(1)).mean()), 4))

    if tau > 0:
        Q = prior_correct(Q / Q.sum(1, keepdims=True), tau)
    pred = Q.argmax(1)
    out = Path(__file__).parents[2] / "submissions" / f"{out_name}.csv"
    pd.DataFrame({"id": te["id"], "target_feature": pred}).to_csv(out, index=False)
    print("wrote", out, "class dist:", np.bincount(pred, minlength=19).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", required=True)
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--tau", type=float, default=0.4, help="strength of the prior correction. Decided on OOF")
    p.add_argument("--stages", nargs="*", type=float, default=[2.0],
                   help="lambda of each multi-stage refinement stage. Empty means a single stage")
    a = p.parse_args()
    main(a.tags, a.weights or [1.0] * len(a.tags), a.k, a.alpha, a.iters, a.out,
         a.no_smooth, a.tau, a.stages)
