"""Reproduce Appendix A, "the public video file numbers differ from the competition for four recordings. Each video was matched to a recording by
cross-correlating the frame-to-frame change of its embeddings with that of the competition features (all at lag 0, r = 0.71--0.83, next candidate about 0.4)",
and verify the mapping table probe_wear_encoders.PUB2REC.

Public-video side: CLIP features made by wear_encoders.py (mean of 8 frames per second).
Competition side: the distributed VideoMAEv2 features (prep/video_raw_head.npy) averaged per 1 s training tile.
Both are turned into series of "magnitude of embedding change between adjacent seconds", and the lag-0 correlation is computed for all pairs.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from probe_wear_encoders import IMG, PREP, PUB2REC


def change(F):
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    return np.r_[0.0, 1 - (F[1:] * F[:-1]).sum(1)]


def main():
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    t = meta[meta.start % 50 == 0]
    comp = {}
    for rec, g in t.groupby("rec"):
        g = g.sort_values("start")
        sec = g.start.to_numpy() // 50
        F = np.zeros((sec.max() + 1, video.shape[-1]), np.float32); ok = np.zeros(sec.max() + 1, bool)
        F[sec] = np.asarray(video[g.index.to_numpy()], np.float32).mean(1); ok[sec] = True
        comp[rec] = (change(F), ok)
    pubs = sorted(p.stem for p in (IMG / "clip").glob("sbj_*.npy"))
    wrong = 0
    for pub in pubs:
        c = change(np.load(IMG / "clip" / f"{pub}.npy").astype(np.float32).mean(1))
        rs = {}
        for rec, (d, ok) in comp.items():
            n = min(len(c), len(d)); m = ok[:n] & np.r_[False, ok[1:n] & ok[:n - 1]]
            rs[rec] = np.corrcoef(c[:n][m], d[:n][m])[0, 1] if m.sum() > 100 else np.nan
        best = sorted(rs, key=lambda r: -np.nan_to_num(rs[r], nan=-1))
        exp = PUB2REC.get(pub, "(excluded)")
        flag = "" if best[0] == exp else "  <- mismatch with PUB2REC"
        wrong += bool(flag) and exp != "(excluded)"
        print(f"{pub:7s} best {best[0]:9s} r={rs[best[0]]:.2f}  next {best[1]:9s} r={rs[best[1]]:.2f}   PUB2REC: {exp}{flag}")
    print(f"mappings disagreeing with PUB2REC: {wrong}")


if __name__ == "__main__":
    main()
