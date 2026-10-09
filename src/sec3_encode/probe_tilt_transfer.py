"""Does the "direction that reads head tilt out of the video embedding" learned on Ego-Exo4D transfer to WEAR (a different camera)?

Training: 6 Ego-Exo4D participants, embeddings centered within each take -> head tilt (a common axis obtained by linearly regressing the measured gravity direction onto GeoCalib pitch, in degrees).
Application: apply to each WEAR recording (centered within the recording) and read out the per-window head tilt.
Comparison: (1) correlation with WEAR GeoCalib pitch (per recording, median) (2) ordering of per-activity means (rank correlation with GeoCalib activity means)
      (3) R^2 of explaining the read-out tilt with within-recording PC1/PC2. 3 encoders.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES
from egoexo_tilt_view import load
from probe_wear_encoders import IMG, PREP, PUB2REC

DIRS = {"VideoMAEv2": "features", "CLIP": "features_clip", "DINOv2": "features_dinov2"}
REC2PUB = {v: k for k, v in PUB2REC.items()}


def ego_train(d):
    by = load(d)
    Fs, Gs, Ps = [], [], []
    for p in by:
        for a in by[p]:
            if len(a[1]) < 40:
                continue
            Fs.append(a[1] - a[1].mean(0)); Gs.append(a[2] - a[2].mean(0)); Ps.append(a[3] - a[3].mean())
    F, G, pit = np.concatenate(Fs), np.concatenate(Gs), np.concatenate(Ps)
    w, *_ = np.linalg.lstsq(G, pit, rcond=None)             # common axis: measured gravity -> pitch (degrees)
    tilt = G @ w
    return RidgeCV(alphas=np.logspace(0, 5, 12)).fit(F, tilt)


def main():
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    for enc, d in DIRS.items():
        m = ego_train(d)
        cors, r2s, acts, gcs = [], [], [], []
        for rec, pub in sorted(REC2PUB.items()):
            gf = Path(f"{DATA_DIR}/geocalib/wear/{pub}.npz")
            if not gf.exists():
                continue
            t = meta[(meta.start % 50 == 0) & (meta.rec == rec)]
            sec = t.start.to_numpy() // 50
            g = np.load(gf)
            if enc == "VideoMAEv2":
                F = np.asarray(video[t.index.to_numpy()], np.float32).mean(1)
                ok = sec < len(g["rp"])
            else:
                A = np.load(IMG / enc.lower() / f"{pub}.npy", mmap_mode="r")
                ok = (sec < len(g["rp"])) & (sec < len(A))
                F = np.asarray(A[sec[ok]], np.float32).mean(1)
                t = t[ok]; sec = sec[ok]; ok = np.ones(len(sec), bool)
            F = F[ok]; sec = sec[ok]; y = t.label.to_numpy()[ok]
            F = F - F.mean(0)
            tilt = m.predict(F)
            gp = np.degrees(g["rp"][sec, 1])
            cors.append(np.corrcoef(tilt, gp)[0, 1])
            _, _, Vt = np.linalg.svd(F, full_matrices=False); Z = F @ Vt[:2].T
            X = np.column_stack([Z, np.ones(len(Z))]); b, *_ = np.linalg.lstsq(X, tilt, rcond=None)
            r2s.append(1 - ((tilt - X @ b) ** 2).sum() / ((tilt - tilt.mean()) ** 2).sum())
            acts.append(pd.Series(tilt - tilt.mean(), index=y)); gcs.append(pd.Series(gp - gp.mean(), index=y))
        A = pd.concat(acts).groupby(level=0).mean(); B = pd.concat(gcs).groupby(level=0).mean()
        rho = spearmanr(A.values, B.loc[A.index].values)[0]
        order = A.sort_values()
        print(f"\n== {enc}: read-out tilt vs GeoCalib pitch, median correlation {np.median(cors):+.2f} (range {min(cors):+.2f} to {max(cors):+.2f})")
        print(f"   rank correlation of per-activity means (read-out vs GeoCalib) {rho:+.2f}; median R^2 of read-out tilt explained by PC1-2 {np.median(r2s):.2f}")
        print("   most downward: " + ", ".join(f"{CLASS_NAMES[c]} {v:+.1f}" for c, v in order.head(3).items())
              + " / most upward: " + ", ".join(f"{CLASS_NAMES[c]} {v:+.1f}" for c, v in order.tail(3).items()))


if __name__ == "__main__":
    main()
