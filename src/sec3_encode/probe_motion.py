"""Is "the video component predictable from the four limbs" (which carries 60-100% of the removal gain) camera shake from body motion? (WEAR)

Quantities measured (per window):
  shake (pixels)    : phase correlation from wear_motion.py [mean|dx|, mean|dy|, std dx, std dy (vertical bounce), max displacement, frame difference]
  change (features) : change of the embedding within the window. Over 15 rows for VideoMAEv2 and 8 frames for CLIP/DINOv2: mean cosine distance between neighbours, and
                      mean distance from the window mean (each encoder's own values)
  scene/body/tilt   : same as probe_body_scene (for comparison)
A. Shake per activity (vertical bounce, displacement)
B. R^2 of predicting the four-limb-predictable component H (OOF MLP from probe_limbpred) from each group (within participant, 5-time-block CV, relative to the total variance of H)
C. Removal: within each participant, subtract the part of the video explained by regression on each group, and compare the gain of the system (feature-only graph + 3 stages).
   Also list subtracting the four-limb prediction (H) and top-30 removal. 3 encoders, 2 seeds.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
ENC_LIST = os.environ.get("ENCS", "VideoMAEv2,CLIP,DINOv2").split(",")
import collections
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_limbpred import oof
from probe_prf import TAGS
from probe_wear_encoders import IMG, PREP, PUB2REC, eval_tiles
from refine_gpu import final_with

SEG = Path(DATA_DIR + "/wear_seg"); GC = Path(DATA_DIR + "/geocalib"); MOT = Path(DATA_DIR + "/wear_motion")
PERSON = 12


def speed(X):
    """X: (n, k, d) k embeddings within a window -> (n, 2) [mean cosine distance between neighbours, mean cosine distance from the window mean]"""
    Xn = X / (np.linalg.norm(X, axis=2, keepdims=True) + 1e-9)
    adj = 1 - (Xn[:, 1:] * Xn[:, :-1]).sum(2)
    m = Xn.mean(1, keepdims=True); m /= np.linalg.norm(m, axis=2, keepdims=True) + 1e-9
    return np.column_stack([adj.mean(1), (1 - (Xn * m).sum(2)).mean(1)])


def cv_r2_total(X, Y, blk):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9); Yc = Y - Y.mean(0); pr = np.zeros_like(Yc)
    for b in range(5):
        tr, te = blk != b, blk == b
        pr[te] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Xz[tr], Yc[tr]).predict(Xz[te])
    return ((Yc - pr) ** 2).sum(), (Yc ** 2).sum()


def explained(X, Y, alpha=10.0):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return Ridge(alpha=alpha).fit(Xz, Y).predict(Xz)


def main():
    load = lambda d, k: {PUB2REC[Path(f).stem]: np.load(f)[k] for f in glob.glob(str(d / "*.npz")) if Path(f).stem in PUB2REC}
    mot, seg = load(MOT, "M"), load(SEG, "H")
    gc = {PUB2REC[Path(f).stem]: np.load(f) for f in glob.glob(str(GC / "wear" / "*.npz")) if Path(f).stem in PUB2REC}
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    raw = {e: {PUB2REC[p]: np.load(IMG / e.lower().replace("-", "") / f"{p}.npy", mmap_mode="r") for p in pubs} for e in ["CLIP", "DINOv2", "VC-1"]}
    d = oof({e: {r: v for r, v in raw[e].items()} for e in raw})
    pos_of = {t: i for i, t in enumerate(d["tile"])}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        tile0, P, meta = eval_tiles(sd)
        rec = meta["rec"].to_numpy()[tile0]; sec = meta["start"].to_numpy()[tile0] // 50
        use = np.array([t in pos_of and r in mot and r in seg and r in gc and s < len(mot[r]) and s < len(seg[r])
                        and s < len(gc[r]["g"]) for t, r, s in zip(tile0, rec, sec)])
        tile, P, rec, sec = tile0[use], P[use], rec[use], sec[use]
        pos = np.array([pos_of[t] for t in tile])
        y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
        Mp = np.stack([mot[r][s] for r, s in zip(rec, sec)])
        Hs = np.stack([seg[r][s].astype(np.float32) for r, s in zip(rec, sec)])
        keep = [k for k in range(150) if k != PERSON and Hs[:, :, k].mean() > 0.002]
        sbt = np.column_stack([Hs[:, :, PERSON], Hs[:, :, keep].reshape(len(Hs), -1),
                               np.stack([gc[r]["g"][s] for r, s in zip(rec, sec)]), np.stack([gc[r]["rp"][s] for r, s in zip(rec, sec)])])
        if sd == 42:
            print(f"windows {len(y)}")
            print("\n=== A. Shake per activity (pixels; mean|dx|, mean|dy|, vertical bounce std dy, frame difference) ===")
            for c in np.unique(y):
                m = y == c
                print(f"  {CLASS_NAMES[c]:30s} |dx| {Mp[m, 0].mean():5.2f}  |dy| {Mp[m, 1].mean():5.2f}  bounce {Mp[m, 3].mean():5.2f}  diff {Mp[m, 5].mean():5.1f}")
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        sbj_all = meta["sbj_id"].to_numpy()[tile0]
        V = np.empty((len(y), N_CLASSES)); Vg0 = {}
        for s in np.unique(sbj):
            u = use[sbj_all == s]
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
            Vg0[s] = np.zeros((u.sum(), N_CLASSES))
        for e in ENC_LIST:
            Yc, H = d[f"Yc_{e}"][pos], d[f"H_{e}"][pos]
            if e == "VideoMAEv2":
                Mf = speed(np.asarray(video[tile], np.float32))
            else:
                Mf = speed(np.stack([np.asarray(raw[e][r][s], np.float32) for r, s in zip(rec, sec)]))
            groups = {"shake (pixels)": Mp, "change (features)": Mf, "shake+change": np.column_stack([Mp, Mf]),
                      "scene/body/tilt": sbt, "all": np.column_stack([Mp, Mf, sbt])}
            if sd == 42:
                err = collections.defaultdict(float); tot = 0.0
                for s in np.unique(sbj):
                    m = sbj == s; blk = np.arange(m.sum()) * 5 // m.sum()
                    for g, X in groups.items():
                        e_, t_ = cv_r2_total(X[m], H[m], blk); err[g] += e_
                    tot += t_
                print(f"\n=== B. [{e}] R^2 of predicting the four-limb-predictable component H from each group (within-participant CV) ===")
                print("  " + "  ".join(f"{g} {1 - err[g] / tot:+.3f}" for g in groups))
            feats = {"raw": {}, "subtract four limbs prediction": {}}
            for g in groups:
                feats[g] = {}
            for s in np.unique(sbj):
                m = sbj == s
                feats["raw"][s] = Yc[m]; feats["subtract four limbs prediction"][s] = Yc[m] - H[m]
                for g, X in groups.items():
                    feats[g][s] = Yc[m] - explained(X[m], Yc[m])
            for g, Fv in feats.items():
                R[(e, g)].append(tune_tau(final_with(P, Fv, Vg0, V, sbj, 0), y)[0])
            R[(e, "remove top 30")].append(tune_tau(final_with(P, feats["raw"], Vg0, V, sbj, 30), y)[0])
        print(f"seed {sd} done", flush=True)
    print("\n=== C. Removal (feature-only graph, 3 stages, mean of 2 seeds, diff from raw) ===")
    for e in ENC_LIST:
        b = np.array(R[(e, "raw")])
        print(f"  {e}: raw {b.mean():.4f}  " + "  ".join(
            f"{g} {(np.array(R[(e, g)]) - b).mean():+.4f}" for g in ["remove top 30", "subtract four limbs prediction", "shake (pixels)", "change (features)", "shake+change", "scene/body/tilt", "all"]))


if __name__ == "__main__":
    main()
