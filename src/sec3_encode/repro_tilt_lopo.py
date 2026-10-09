"""Reproduce paper Section 3.2: "the directions that carry head tilt are shared across people. A linear readout learned on 5 people predicts the 6th person's head tilt
with correlation 0.70--0.95".

Linearly read out (RidgeCV) the measured head tilt from Ego-Exo4D first-person embeddings (centred per take).
Tilt uses the same definition as probe_tilt_transfer.ego_train: the per-take-centred gravity direction projected onto a common axis
fitted by least squares to GeoCalib's pitch. Leave one participant out, learn on the other 5, and correlate on the held-out one. Per encoder.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from egoexo_tilt_view import load

DIRS = {"VideoMAEv2": "features", "CLIP": "features_clip", "DINOv2": "features_dinov2", "VC-1": "features_vc1"}


def per_participant(by):
    out = {}
    for p in by:
        Fs, Gs, Ps = [], [], []
        for a in by[p]:
            if len(a[1]) < 40:
                continue
            Fs.append(a[1] - a[1].mean(0)); Gs.append(a[2] - a[2].mean(0)); Ps.append(a[3] - a[3].mean())
        if Fs:
            out[p] = (np.concatenate(Fs), np.concatenate(Gs), np.concatenate(Ps))
    return out


def main():
    for enc, d in DIRS.items():
        if not (Path(DATA_DIR + "/egoexo4d") / d).exists():
            continue
        pp = per_participant(load(d))
        # gravity direction -> common pitch axis (one for everyone; defined on the label side, so independent of training the readout)
        G = np.concatenate([v[1] for v in pp.values()]); Pi = np.concatenate([v[2] for v in pp.values()])
        w, *_ = np.linalg.lstsq(G, Pi, rcond=None)
        rs = []
        for p in sorted(pp):
            tr = [q for q in pp if q != p]
            X = np.concatenate([pp[q][0] for q in tr]); t = np.concatenate([pp[q][1] for q in tr]) @ w
            m = RidgeCV(alphas=np.logspace(0, 5, 12)).fit(X, t)
            pred = m.predict(pp[p][0]); rs.append(np.corrcoef(pred, pp[p][1] @ w)[0, 1])
        print(f"{enc:10s} correlation on held-out participant: " + "  ".join(f"{p}: {r:.2f}" for p, r in zip(sorted(pp), rs))
              + f"   range {min(rs):.2f}--{max(rs):.2f}", flush=True)


if __name__ == "__main__":
    main()
