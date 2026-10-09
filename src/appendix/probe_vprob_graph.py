"""Add "the video model's predicted probabilities" to the graph features.

The video model alone is weak on unseen subjects (exp013 OOF 0.588), but
for the purpose of **placing windows of the same activity close together within one subject**,
it should be a more activity-specific representation than raw VideoMAE features.
It is a view independent of the inertial base, so it also fits the "principle of independence".
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from eval_pipeline import load_eval_set, run
from postprocess import tune_tau

PREP = WORK / "prep"
TAGS = ["exp001_lgb_inertial", "exp014_nn_inertial_rot", "exp015_nn_inertial_s1337",
        "exp016_nn_inertial_rot30", "exp017_nn_aux"]
W = [1.0, 0.7, 0.7, 0.7, 3.0]


def main():
    P, y, sbj, F = load_eval_set(TAGS, W)
    # Look up the video model (exp013) OOF probabilities corresponding to the evaluation rows
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile2, sens2 = tile[ok], sens[ok]
    rows = tile2 + sens2 * n
    base0 = np.load(WORK / TAGS[0] / "oof.npy")[rows]
    keep = base0.sum(1) > 1e-6
    rows, tile2 = rows[keep], tile2[keep]
    V = np.load(WORK / "exp013_nn_video_5fold" / "oof.npy")[rows]
    s = V.sum(1, keepdims=True)
    V = np.divide(V, s, out=np.full_like(V, 1 / 19), where=s > 0)

    Vs = {ss: V[sbj == ss] for ss in np.unique(sbj)}
    print(f"  video features only (current)  -> {tune_tau(run(P, y, sbj, F, [4.0,4.0],30,0.75,5,temp=1.4), y)[0]:.4f}")
    for gam in [0.2, 0.3, 0.4, 0.5, 0.7]:
        G = {}
        for ss in np.unique(sbj):
            f = F[ss] - F[ss].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            G[ss] = np.concatenate([f, gam * np.sqrt(Vs[ss])], 1)
        Q = run(P, y, sbj, G, [4.0, 4.0], 30, 0.75, 5, temp=1.4)
        f1, tau = tune_tau(Q, y)
        print(f"  + video model probs gamma={gam:<4} -> {f1:.4f} (tau={tau:.2f})", flush=True)
    # Re-check temp and the stage configuration at the best gamma
    for gam, temp, stages in [(0.4, 1.0, [4.0, 4.0]), (0.4, 1.6, [4.0, 4.0]),
                              (0.4, 1.4, [4.0]), (0.4, 1.4, [4.0, 4.0, 4.0])]:
        G = {}
        for ss in np.unique(sbj):
            f = F[ss] - F[ss].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            G[ss] = np.concatenate([f, gam * np.sqrt(Vs[ss])], 1)
        Q = run(P, y, sbj, G, stages, 30, 0.75, 5, temp=temp)
        print(f"  gamma={gam} temp={temp} stages={len(stages)} -> {tune_tau(Q, y)[0]:.4f}", flush=True)


if __name__ == "__main__":
    main()
