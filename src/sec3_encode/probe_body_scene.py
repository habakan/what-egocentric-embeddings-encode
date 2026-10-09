"""How much do the top components of egocentric embeddings represent "how one's own body appears" and "place/background"? Does that explain the removal gain? (WEAR)

Quantities measured (SegFormer ADE20K, 1 frame per second, pixel fraction per cell of a 3x3 screen grid; wear_segment.py):
  body       : fraction of 'person' (9 cells). In egocentric video, the visible person is almost always one's own limbs
  background : fractions of non-person classes whose overall mean fraction is >= 0.2% (9 cells x K)
  tilt       : GeoCalib camera gravity direction (3) + roll/pitch (for comparison)
1. Body fraction and main background classes per activity (check against the impression from samples)
2. R^2 of predicting principal-component bands from body / background / body+background / all (per participant, 5 time-block cross-validation)
3. Removal: within participant, subtract from the features the component explained by regressing on body / background / body+background / all, and compare
   the system's (feature-only graph + 3 stages) macro-F1. Control: subtract with random regressors of the same dimension (the effect of subtracting anything with a high-dimensional ridge).
   Also top 30 removal. 3 encoders, 2 seeds.
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
from probe_prf import TAGS
from probe_wear_encoders import IMG, PREP, PUB2REC, eval_tiles
from refine_gpu import final_with

SEG = Path(DATA_DIR + "/wear_seg"); GC = Path(DATA_DIR + "/geocalib")
PERSON = 12


def ade_names():
    from transformers import SegformerConfig
    return SegformerConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512").id2label


def cv_r2(X, Z, blk):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9); Zc = Z - Z.mean(0); pr = np.zeros_like(Zc)
    for b in range(5):
        tr, te = blk != b, blk == b
        pr[te] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Xz[tr], Zc[tr]).predict(Xz[te])
    return ((Zc - pr) ** 2).sum(0), (Zc ** 2).sum(0)


def explained(X, Y, alpha=10.0):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return Ridge(alpha=alpha).fit(Xz, Y).predict(Xz)


def main():
    names = ade_names()
    seg = {PUB2REC[Path(f).stem]: np.load(f)["H"].astype(np.float32) for f in glob.glob(str(SEG / "*.npz")) if Path(f).stem in PUB2REC}
    gc = {PUB2REC[Path(f).stem]: np.load(f) for f in glob.glob(str(GC / "wear" / "*.npz")) if Path(f).stem in PUB2REC}
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower().replace("-", "") / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2", "VC-1"]}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        tile0, P, meta = eval_tiles(sd)
        rec = meta["rec"].to_numpy()[tile0]; sec = meta["start"].to_numpy()[tile0] // 50
        use = np.array([r in seg and r in gc and r in img["CLIP"] and s < len(seg[r]) and s < len(gc[r]["g"])
                        and s < len(img["CLIP"][r]) and s < len(img["DINOv2"][r]) for r, s in zip(rec, sec)])
        tile, P, rec, sec = tile0[use], P[use], rec[use], sec[use]
        y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
        H = np.stack([seg[r][s] for r, s in zip(rec, sec)])                       # (n, 9, 150)
        body = H[:, :, PERSON]
        keep = [k for k in range(150) if k != PERSON and H[:, :, k].mean() > 0.002]
        scene = H[:, :, keep].reshape(len(H), -1)
        tilt = np.column_stack([np.stack([gc[r]["g"][s] for r, s in zip(rec, sec)]),
                                np.stack([gc[r]["rp"][s] for r, s in zip(rec, sec)])])
        groups = {"body": body, "background": scene, "body+background": np.column_stack([body, scene]),
                  "body+background+tilt": np.column_stack([body, scene, tilt])}
        if sd == 42:
            print(f"windows {len(y)}, background classes {len(keep)}: " + ", ".join(names[k] for k in keep))
            print("\n=== 1. Per-activity pixel fraction of body ('person') (whole screen / bottom row) and most common background classes ===")
            for c in np.unique(y):
                m = y == c
                sc = H[m][:, :, keep].mean((0, 1)); top = np.argsort(-sc)[:3]
                print(f"  {CLASS_NAMES[c]:30s} body {body[m].mean():.3f} / bottom {body[m][:, 6:].mean():.3f}   "
                      + ", ".join(f"{names[keep[t]]} {sc[t]:.2f}" for t in top))
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        sbj_all = meta["sbj_id"].to_numpy()[tile0]
        V = np.empty((len(y), N_CLASSES)); Vg0 = {}
        for s in np.unique(sbj):
            u = use[sbj_all == s]
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
            Vg0[s] = np.zeros((u.sum(), N_CLASSES))
        rng = np.random.RandomState(sd)
        for e in ENC_LIST:
            Y = np.asarray(video[tile], np.float32).mean(1) if e == "VideoMAEv2" else np.stack([img[e][r][s] for r, s in zip(rec, sec)])
            Yc = Y.copy()
            for s in np.unique(sbj):
                Yc[sbj == s] -= Yc[sbj == s].mean(0)
            if sd == 42:
                bands = [(1, 2), (3, 10), (11, 30), (31, 100)]
                err = collections.defaultdict(lambda: np.zeros(100)); tot = np.zeros(100)
                for s in np.unique(sbj):
                    m = sbj == s
                    _, _, Vt = np.linalg.svd(Yc[m], full_matrices=False)
                    Z = Yc[m] @ Vt[:100].T
                    blk = np.arange(m.sum()) * 5 // m.sum()
                    for g, X in groups.items():
                        e_, t_ = cv_r2(X[m], Z, blk); err[g] += e_
                    tot += (Z - Z.mean(0)).__pow__(2).sum(0)
                print(f"\n=== 2. [{e}] R^2 of principal-component bands (within participant, 5 time-block CV) ===")
                for g in groups:
                    r2 = 1 - err[g] / tot
                    print(f"  {g:10s} " + "  ".join(f"PC{b[0]}-{b[1]} {r2[b[0] - 1:b[1]].mean():+.3f}" for b in bands))
            feats = {"raw": {}}
            for g, X in list(groups.items()) + [("control random (same dim as body+background)", None)]:
                feats[g] = {}
            for s in np.unique(sbj):
                m = sbj == s
                feats["raw"][s] = Yc[m]
                for g, X in groups.items():
                    feats[g][s] = Yc[m] - explained(X[m], Yc[m])
                Xr = rng.randn(m.sum(), groups["body+background"].shape[1])
                feats["control random (same dim as body+background)"][s] = Yc[m] - explained(Xr, Yc[m])
            for g, Fv in feats.items():
                R[(e, g)].append(tune_tau(final_with(P, Fv, Vg0, V, sbj, 0), y)[0])
            R[(e, "top30 removed")].append(tune_tau(final_with(P, feats["raw"], Vg0, V, sbj, 30), y)[0])
            if sd == 42:
                tv = sum((f ** 2).sum() for f in feats["raw"].values())
                print(f"  [{e}] fraction of variance subtracted: " + ", ".join(
                    f"{g} {1 - sum((f ** 2).sum() for f in feats[g].values()) / tv:.3f}" for g in feats if g != "raw"))
        print(f"seed {sd} done", flush=True)
    print("\n=== 3. Removal (feature-only graph, 3 stages, mean of 2 seeds, diff vs raw) ===")
    for e in ENC_LIST:
        b = np.array(R[(e, "raw")])
        print(f"  {e}: raw {b.mean():.4f}")
        for g in ["top30 removed", "body", "background", "body+background", "body+background+tilt", "control random (same dim as body+background)"]:
            v = np.array(R[(e, g)])
            print(f"    {g:30s} {v.mean():.4f} ({(v - b).mean():+.4f})")


if __name__ == "__main__":
    main()
