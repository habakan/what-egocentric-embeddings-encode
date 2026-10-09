"""How far does the egocentric camera orientation (estimated from the video with GeoCalib) determine the top components of the WEAR embeddings?

0. Validity (Ego-Exo4D): compare GeoCalib's gravity in camera coordinates, g_cam, against Aria's measurement (gravity in device coordinates).
   To absorb a fixed rotation difference, linearly regress the measurement on g_cam (and 2nd-order terms); R^2 by cross-validation over takes.
WEAR (20 participants; 3 encoders):
1. R^2 of predicting principal-component bands from the camera tilt (g_cam, up to 2nd order) (per participant, 5 time-block folds)
   Reverse: R^2 of recovering the camera tilt from the top 30 / the complement
2. Mean camera pitch (degrees) per activity: does it split by lying / prone / standing, and stay the same between an exercise and its variants?
3. R^2 of predicting the camera tilt from the four-limb acceleration's static component (window mean = limb gravity direction, 12 dims) / dynamic component (mean-removed
   std, range, magnitude of successive differences, 36 dims) (5 folds by subject)
4. Removal: remove "the component explained by the camera tilt" from the graph features (within participant, in-sample ridge).
   Compare: raw / top-30 removed / camera-tilt component removed / its permutation control / four-limb static prediction removed / dynamic prediction removed
   (four-limb predictions are OOF ridge over 5 folds by subject). System: feature-only graph + 3 stages (final_with), 2 seeds.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
ENC_LIST = os.environ.get("ENCS", "VideoMAEv2,CLIP,DINOv2").split(",")
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_prf import TAGS
from probe_wear_encoders import IMG, PREP, PUB2REC, eval_tiles
from refine_gpu import final_with

GC = Path(DATA_DIR + "/geocalib")
EGO = Path(DATA_DIR + "/egoexo4d")


def quad(G):
    X = np.column_stack([G, G ** 2, G[:, [0]] * G[:, [1]], G[:, [0]] * G[:, [2]], G[:, [1]] * G[:, [2]]])
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def cv_r2(X, Y, grp, alphas=np.logspace(-2, 3, 12)):
    Y = Y if Y.ndim == 2 else Y[:, None]
    pr = np.zeros_like(Y)
    for tr, te in GroupKFold(len(np.unique(grp)) if len(np.unique(grp)) < 5 else 5).split(X, Y, grp):
        pr[te] = RidgeCV(alphas=alphas).fit(X[tr], Y[tr]).predict(X[te]).reshape(len(te), -1)
    return 1 - ((Y - pr) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def validity():
    print("=== 0. Validity: GeoCalib estimate vs Aria measurement (Ego-Exo4D, cross-validation over takes) ===")
    by = collections.defaultdict(list)
    for f in sorted((EGO / "features").glob("*.npz")):
        g = GC / "ego" / f.name
        if not g.exists():
            continue
        z = np.load(f); e = np.load(g)
        n = min(len(z["pose"]), len(e["g"]))
        ok = z["ok"][:n]
        by[int(z["participant"])].append((z["pose"][:n][ok, :3], e["g"][:n][ok], e["rp"][:n][ok]))
    r2s = []
    for p, v in by.items():
        Gm = np.concatenate([a for a, _, _ in v]); Gc = np.concatenate([b for _, b, _ in v])
        grp = np.concatenate([[i] * len(a) for i, (a, _, _) in enumerate(v)])
        r2 = cv_r2(quad(Gc), Gm, grp)
        j = int(np.argmax(Gm.std(0)))
        r2s.append(r2[j])
        print(f"  participant {p}: measured gravity (max-variance component) from GeoCalib R^2 {r2[j]:.3f}  (3 components {np.round(r2, 2).tolist()})")
    print(f"  mean {np.mean(r2s):.3f}")


def wear_rows(sd):
    tile, P, meta = eval_tiles(sd)
    rec = meta["rec"].to_numpy()[tile]; sec = meta["start"].to_numpy()[tile] // 50
    gc = {PUB2REC[Path(f).stem]: np.load(f) for f in glob.glob(str(GC / "wear" / "*.npz")) if Path(f).stem in PUB2REC}
    return tile, P, meta, rec, sec, gc


def main():
    validity()
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower().replace("-", "") / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2", "VC-1"]}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    inert = np.load(PREP / "inertial.npy", mmap_mode="r")
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        tile, P, meta, rec, sec, gc = wear_rows(sd)
        use = np.array([r in gc and r in img["CLIP"] and s < len(gc[r]["g"]) and s < len(img["CLIP"][r])
                        and s < len(img["DINOv2"][r]) for r, s in zip(rec, sec)])
        tile, P, rec, sec = tile[use], P[use], rec[use], sec[use]
        y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
        G = np.stack([gc[r]["g"][s] for r, s in zip(rec, sec)])
        pitch = np.degrees(np.stack([gc[r]["rp"][s] for r, s in zip(rec, sec)])[:, 1])
        W = np.nan_to_num(np.asarray(inert[tile], np.float32))                    # (n, 4, 50, 3) missing = 0
        stat = W.mean(2).reshape(len(W), -1)
        Wd = W - W.mean(2, keepdims=True)
        dyn = np.column_stack([Wd.std(2).reshape(len(W), -1), np.ptp(Wd, 2).reshape(len(W), -1),
                               np.abs(np.diff(Wd, axis=2)).mean(2).reshape(len(W), -1)])
        us = np.random.RandomState(0).permutation(np.unique(sbj))
        fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
        # 3. limbs -> camera tilt
        if sd == 42:
            print("\n=== 2. Camera pitch per activity (degrees, mean of raw values not centred within participant) ===")
            mu = sorted(((CLASS_NAMES[c], pitch[y == c].mean(), pitch[y == c].std()) for c in np.unique(y)), key=lambda t: t[1])
            for nm, m_, s_ in mu:
                print(f"  {nm:30s} {m_:+6.1f} ± {s_:4.1f}")
            print("\n=== 3. Predicting the camera tilt from four-limb acceleration (5 folds by subject) ===")
            def csub(A):
                A = A.copy()
                for s_ in np.unique(sbj):
                    A[sbj == s_] -= A[sbj == s_].mean(0)
                return A
            for nm, X in [("static component (limb gravity direction)", stat), ("dynamic component", dyn), ("both", np.column_stack([stat, dyn]))]:
                Xz = csub(X); Xz = Xz / (Xz.std(0) + 1e-9)
                r2 = cv_r2(Xz, csub(G), fold)
                print(f"  {nm:26s} R^2 (3 components) {np.round(r2, 3).tolist()}")
        # four-limb predictions (OOF ridge) are built per feature
        Vg_d = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        sbj_full = meta["sbj_id"].to_numpy()[eval_tiles(sd)[0]]
        Vg0, V = {}, np.empty((len(y), N_CLASSES))
        for s in np.unique(sbj):
            u = use[sbj_full == s]
            Vg0[s] = np.zeros((u.sum(), N_CLASSES))
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
        rng = np.random.RandomState(sd)
        for e in ENC_LIST:
            if e == "VideoMAEv2":
                Y = np.asarray(video[tile], np.float32).mean(1)
            else:
                Y = np.stack([img[e][r][s] for r, s in zip(rec, sec)])
            Yc = Y.copy()
            for s in np.unique(sbj):
                Yc[sbj == s] -= Yc[sbj == s].mean(0)
            if sd == 42:
                print(f"\n=== 1. [{e}] camera tilt -> PC band R^2 / R^2 of recovering the tilt (mean over participants) ===")
                bands = [(1, 2), (3, 10), (11, 30), (31, 100)]; acc = collections.defaultdict(list)
                for s in np.unique(sbj):
                    m = sbj == s
                    _, _, Vt = np.linalg.svd(Yc[m], full_matrices=False)
                    Z = Yc[m] @ Vt[:100].T
                    blk = np.arange(m.sum()) * 5 // m.sum()
                    r2 = cv_r2(quad(G[m]), Z, blk)
                    for b in bands:
                        acc[b].append(r2[b[0] - 1:b[1]].mean())
                    Gz = G[m] - G[m].mean(0)
                    j = int(np.argmax(Gz.std(0)))
                    top = Yc[m] @ Vt[:30].T; rest = Yc[m] - top @ Vt[:30]
                    acc["top"].append(cv_r2(top, Gz[:, j], blk)[0]); acc["rest"].append(cv_r2(rest @ Vt[30:130].T, Gz[:, j], blk)[0])
                print("  tilt -> PC " + "  ".join(f"{b[0]}-{b[1]}: {np.mean(acc[b]):.3f}" for b in bands))
                print(f"  tilt <- top 30 {np.mean(acc['top']):.3f}   <- complement {np.mean(acc['rest']):.3f}")
            # removal variants
            feats = {"raw": {}, "top-30 removed": {}, "camera-tilt component removed": {}, "control (permuted)": {},
                     "four-limb static prediction removed": {}, "four-limb dynamic prediction removed": {}}
            Hs = {}
            for nm, X in [("static", stat), ("dynamic", dyn)]:
                Xz = (X - X.mean(0)) / (X.std(0) + 1e-9); H = np.zeros_like(Yc)
                for k in range(5):
                    tr, te = fold != k, fold == k
                    H[te] = Ridge(alpha=100.0).fit(Xz[tr], Yc[tr]).predict(Xz[te])
                Hs[nm] = H
            for s in np.unique(sbj):
                m = sbj == s
                Xg = quad(G[m]); Hg = Ridge(alpha=1.0).fit(Xg, Yc[m]).predict(Xg)
                feats["raw"][s] = Yc[m]; feats["top-30 removed"][s] = Yc[m]
                feats["camera-tilt component removed"][s] = Yc[m] - Hg
                feats["control (permuted)"][s] = Yc[m] - Hg[rng.permutation(m.sum())]
                feats["four-limb static prediction removed"][s] = Yc[m] - Hs["static"][m]
                feats["four-limb dynamic prediction removed"][s] = Yc[m] - Hs["dynamic"][m]
                if sd == 42 and s == np.unique(sbj)[0]:
                    pass
            for nm, Fv in feats.items():
                Q = final_with(P, Fv, Vg0, V, sbj, 30 if nm == "top-30 removed" else 0)
                R[(e, nm)].append(tune_tau(Q, y)[0])
            # size of the removed component (fraction of variance)
            if sd == 42:
                tot = (Yc ** 2).sum()
                print(f"  [{e}] fraction of variance removed: camera-tilt component {1 - sum(((feats['camera-tilt component removed'][s]) ** 2).sum() for s in feats['raw']) / tot:.3f}, "
                      f"four-limb static {1 - ((Yc - Hs['static']) ** 2).sum() / tot:.3f}, dynamic {1 - ((Yc - Hs['dynamic']) ** 2).sum() / tot:.3f}")
        print(f"seed {sd} done", flush=True)
    print("\n=== 4. Removal variants (feature-only graph, 3 stages, mean of 2 seeds, difference from raw) ===")
    for e in ENC_LIST:
        b = np.array(R[(e, "raw")])
        print(f"  {e}")
        for nm in ["raw", "top-30 removed", "camera-tilt component removed", "control (permuted)", "four-limb static prediction removed", "four-limb dynamic prediction removed"]:
            v = np.array(R[(e, nm)])
            print(f"    {nm:28s} {v.mean():.4f} ({(v - b).mean():+.4f})")


if __name__ == "__main__":
    main()
