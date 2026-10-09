"""Directly measure whether the time synchronization between video and inertial data is off.

Hypothesis: if the 0.5 s video windows are systematically shifted relative to the 1 s inertial windows, the video features
partly describe the activity of the neighbouring window. The most activity-discriminative top components would encode that "shifted activity"
most strongly, so removing them would help; that storyline is possible.

Direct test: see for **which shifted labels** the video graph's neighbours are purest.
  If the video at window i describes activity i+s, windows i,j with similar video will have y[i+s]==y[j+s].
  purity(s) = fraction of neighbours j with y[i+s]==y[j+s]. If s=0 is the maximum, synchronization is fine.

Test windows are non-overlapping 1 s tiles, so shift 1 = 1 s.
The same test is also run with the activity eta^2 of the top PCs (to see whether the shift differs per component).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from refine import drop_top_pcs

PREP = WORK / "prep"


def load(offset="head", seed=SEED):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile = tile[ok]
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    rec = meta["rec"].to_numpy()[tile]
    sec = (meta["start"].to_numpy()[tile] // 50).astype(int)
    f = f"video_raw_{offset}.npy" if offset != "head" else "video_raw_head.npy"
    vid = np.load(PREP / f, mmap_mode="r")
    F = {s: np.asarray(vid[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    return F, y, sbj, rec, sec


def shifted_purity(F, y, rec, sec, k, shifts):
    """Return how pure the neighbours are with respect to labels shifted by s seconds."""
    Fn = F - F.mean(0)
    Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(Fn) - 2)
    nn = np.argpartition(-S, kk, axis=1)[:, :kk]
    # look up s seconds ahead in a (rec, sec) -> label dict
    idx = {(r, t): i for i, (r, t) in enumerate(zip(rec, sec))}
    out = {}
    for s in shifts:
        tgt = np.array([idx.get((rec[i], sec[i] + s), -1) for i in range(len(y))])
        ok = tgt >= 0
        ys = np.where(ok, y[np.clip(tgt, 0, len(y) - 1)], -1)
        m = ok[:, None] & ok[nn]
        out[s] = float((ys[nn] == ys[:, None])[m].mean()) if m.sum() else float("nan")
    return out


def main(a):
    for off in a.offsets:
        try:
            F, y, sbj, rec, sec = load(off)
        except FileNotFoundError:
            print(f"{off}: feature file missing, skipping"); continue
        print(f"\n=== offset={off} : neighbour purity against shifted labels (k={a.k}) ===")
        for m in a.drop_pcs:
            agg = {}
            for s in np.unique(sbj):
                msk = sbj == s
                r = shifted_purity(drop_top_pcs(F[s], m), y[msk], rec[msk], sec[msk],
                                   a.k, a.shifts)
                for kk, v in r.items():
                    agg.setdefault(kk, []).append(v)
            row = {kk: np.nanmean(v) for kk, v in agg.items()}
            best = max(row, key=lambda z: row[z])
            print(f"  PCs removed={m:3d}  " +
                  "  ".join(f"s={kk:+d}:{row[kk]:.4f}" for kk in a.shifts) +
                  f"   max s={best:+d}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--shifts", nargs="*", type=int, default=[-2, -1, 0, 1, 2])
    p.add_argument("--drop-pcs", nargs="*", type=int, default=[0, 30])
    p.add_argument("--offsets", nargs="*", default=["head"])
    main(p.parse_args())
