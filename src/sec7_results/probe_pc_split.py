"""WEAR: split the gain of removing the top 30 components into "components 1-2 (first-person gaze direction)" and "components 3-30".

In Ego-Exo4D, first-person PC1-2 are explained by head tilt, height and heading with within-take R^2 around 0.5, while PC3-30 are barely explained
(egoexo_viewpose.py). Compare which carries the removal gain, using a features-only graph + 3 stages (final_with). 3 encoders, 2 seeds.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
ENC_LIST = os.environ.get("ENCS", "VideoMAEv2,CLIP,DINOv2").split(",")
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_prf import TAGS
from probe_mechanism import load
from probe_wear_encoders import IMG, PREP, PUB2REC, eval_tiles
from refine_gpu import final_with


def drop_band(F, lo, hi):
    X = F - F.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    B = Vt[lo:hi]
    return X - (X @ B.T) @ B


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower().replace("-", "") / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2", "VC-1"]}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    R = collections.defaultdict(list)
    bands = {"none": None, "PC1-2": (0, 2), "PC3-30": (2, 30), "PC1-30": (0, 30), "PC1-10": (0, 10), "PC11-30": (10, 30)}
    for sd in [42, 7]:
        P, y, sbj, F = load(sd, img, video)
        tile_all, _, meta = eval_tiles(sd)
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        # reproduce the same window filtering as load() to align V
        rec = meta["rec"].to_numpy()[tile_all]; sec = meta["start"].to_numpy()[tile_all] // 50
        use = np.array([r in img["CLIP"] and s < len(img["CLIP"][r]) and s < len(img["DINOv2"][r]) for r, s in zip(rec, sec)])
        sbj_all = meta["sbj_id"].to_numpy()[tile_all]
        V = np.empty((len(y), N_CLASSES)); Vg0 = {}
        for s in np.unique(sbj):
            u = use[sbj_all == s]
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
            Vg0[s] = np.zeros((u.sum(), N_CLASSES))
        for e in [e for e in F if e in ENC_LIST]:
            for nm, b in bands.items():
                Fv = {s: (F[e][s] if b is None else drop_band(F[e][s], *b)) for s in F[e]}
                R[(e, nm)].append(tune_tau(final_with(P, Fv, Vg0, V, sbj, 0), y)[0])
        print(f"seed {sd} done", flush=True)
    for e in ENC_LIST:
        b = np.array(R[(e, "none")])
        print(f"  {e:10s} " + "  ".join(f"{nm} {(np.array(R[(e, nm)]) - b).mean():+.4f}" for nm in bands if nm != "none"))


if __name__ == "__main__":
    main()
