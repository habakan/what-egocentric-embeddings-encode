"""Compare macro-F1 of graphs built on embeddings from metric learning against plain VideoMAE features."""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from metric import macro_f1
from postprocess import tune_tau
from probe_graph import graph_smooth

PREP = WORK / "prep"


def main(tag, fold, base_tag):
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    oof = np.load(WORK / base_tag / "oof.npy")
    valid = np.load(PREP / "valid_mask.npy")
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    out = WORK / f"metric_{tag}"
    va_w = np.load(out / f"emb_idx_fold{fold}.npy")
    emb = np.load(out / f"emb_fold{fold}.npy")

    # use only the "1 s tile x one random sensor" windows among the fold's valid windows
    rng = np.random.RandomState(SEED)
    is_tile = meta["start"].to_numpy()[va_w] % 50 == 0
    sel = np.where(is_tile)[0]
    w = va_w[sel]; e = emb[sel]
    sens = rng.randint(0, 4, len(w)); ok = valid[w, sens]
    w, e, sens = w[ok], e[ok], sens[ok]
    P = oof[w + sens * n]; keep = P.sum(1) > 0
    w, e, P = w[keep], e[keep], P[keep]
    P = P / P.sum(1, keepdims=True)
    y = meta["label"].to_numpy()[w]
    sbj = meta["sbj_id"].to_numpy()[w]
    raw = {s: np.asarray(video[w[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}

    print(f"fold{fold} n={len(y)}  base = {tune_tau(P, y)[0]:.4f}")
    for name, feats, center in [("plain VideoMAE mean", raw, True),
                                ("metric-learning embedding", {s: e[sbj == s] for s in np.unique(sbj)}, False)]:
        for k, mut in [(8, False), (15, True), (25, True)]:
            Q = P.copy()
            for s in np.unique(sbj):
                m = sbj == s
                Q[m] = graph_smooth(P[m], feats[s], k, 0.75, 5, mutual=mut, center=center)
            f1, tau = tune_tau(Q, y)
            print(f"  {name:18s} k={k:<3d} mutual={str(mut):5s} -> {f1:.4f} (tau={tau:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="v1")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--base-tag", default="exp001_lgb_inertial")
    a = p.parse_args()
    main(a.tag, a.fold, a.base_tag)
