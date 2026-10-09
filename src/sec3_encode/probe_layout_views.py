"""Compare on the same metric: how much of the 1st and 2nd PCs of the embeddings is explained by the layout of what is on screen?
Conditions: WEAR egocentric / Ego-Exo4D egocentric / Ego-Exo4D exocentric, each with VideoMAEv2, CLIP, DINOv2.
Regressors (SegFormer-B2 ADE20K, class fractions per 3x3 cell; classes with mean fraction > 0.2% in that condition):
  background  : excluding 'person' (egocentric: one's own limbs; exocentric: the performer, are labelled person)
  full frame  : including 'person'
Procedure (same for all conditions): centre per participant and PCA, order windows by time (Ego-Exo4D by take), 5-block cross-validation
ridge regression, mean R^2 of components 1-2 (weighted by the participant's window count).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from probe_wear_encoders import IMG, PREP, PUB2REC

EGO = Path(DATA_DIR + "/egoexo4d"); SEG = Path(DATA_DIR + "/wear_seg")
PERSON = 12
EDIR = {"ego": {"VideoMAEv2": "features", "CLIP": "features_clip", "DINOv2": "features_dinov2", "VC-1": "features_vc1"},
        "exo": {"VideoMAEv2": "features_exo_videomae", "CLIP": "features_exo_clip", "DINOv2": "features_exo_dinov2",
                "VC-1": "features_exo_vc1"}}
ENCS = ["VideoMAEv2", "CLIP", "DINOv2", "VC-1"]


def layout(H, keep_person):
    cls = [k for k in range(150) if H[:, :, k].mean() > 0.002 and (keep_person or k != PERSON)]
    return H[:, :, cls].reshape(len(H), -1)


def r2_pc12(groups, takes):
    """groups: [(F (n,d), X (n,p))] per participant, takes: take ids of each participant (n,).
    PCs are per participant. Comparison is within a take (recording for WEAR): centre Z and X per take, and
    cross-validate over 5 time blocks within the take (takes with fewer than 40 windows are excluded). R^2 of components 1-2 (weighted by within-take variance)."""
    err = tot = 0.0
    for (F, X), tk in zip(groups, takes):
        F = F - F.mean(0)
        _, _, Vt = np.linalg.svd(F, full_matrices=False)
        Zall = F @ Vt[:2].T
        for t in np.unique(tk):
            m = tk == t
            if m.sum() < 40:
                continue
            Z = Zall[m] - Zall[m].mean(0); Xt = X[m]
            blk = np.arange(m.sum()) * 5 // m.sum()
            Xz = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9); pr = np.zeros_like(Z)
            for b in range(5):
                tr, te = blk != b, blk == b
                pr[te] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Xz[tr], Z[tr]).predict(Xz[te])
            err += ((Z - pr) ** 2).sum(); tot += (Z ** 2).sum()
    return 1 - err / tot


def wear_groups(enc, keep_person):
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    t = meta[meta.start % 50 == 0]
    seg = {PUB2REC[Path(f).stem]: np.load(f)["H"].astype(np.float32) for f in glob.glob(str(SEG / "*.npz")) if Path(f).stem in PUB2REC}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    img = None
    if enc != "VideoMAEv2":
        img = {PUB2REC[p.stem]: np.load(p, mmap_mode="r") for p in (IMG / enc.lower().replace("-", "")).glob("sbj_*.npy") if p.stem in PUB2REC}
    by = collections.defaultdict(lambda: ([], [], []))
    for rec, g in t.groupby("rec"):
        if rec not in seg:
            continue
        sec = (g["start"].to_numpy() // 50)
        n = len(seg[rec]) if img is None else min(len(seg[rec]), len(img[rec]))
        ok = sec < n; idx = g.index.to_numpy()[ok]; sec = sec[ok]
        F = np.asarray(video[idx], np.float32).mean(1) if img is None else np.asarray(img[rec][sec], np.float32).mean(1)
        s = int(g["sbj_id"].iloc[0])
        by[s][0].append(F); by[s][1].append(seg[rec][sec]); by[s][2].append(np.full(len(F), hash(rec) % 10 ** 6))
    out = [(np.concatenate(a), np.concatenate(b)) for a, b, _ in by.values()]
    tks = [np.concatenate(c) for _, _, c in by.values()]
    Hall = np.concatenate([H for _, H in out])
    cls = [k for k in range(150) if Hall[:, :, k].mean() > 0.002 and (keep_person or k != PERSON)]
    return [(F, H[:, :, cls].reshape(len(H), -1)) for F, H in out], tks


def ego_groups(view, enc, keep_person):
    by = collections.defaultdict(lambda: ([], [], []))
    for i, f in enumerate(sorted((EGO / EDIR[view][enc]).glob("*.npz"))):
        sg = EGO / f"seg_{view}" / f.name
        if not sg.exists():
            continue
        z = np.load(f, allow_pickle=True); H = np.load(sg)["H"].astype(np.float32)
        n = min(len(z["ok"]), len(H)); ok = z["ok"][:n]
        p = int(z["participant"])
        by[p][0].append(z["F"][:n].mean(1)[ok]); by[p][1].append(H[:n][ok]); by[p][2].append(np.full(ok.sum(), i))
    out = [(np.concatenate(a), np.concatenate(b)) for a, b, _ in by.values()]
    tks = [np.concatenate(c) for _, _, c in by.values()]
    Hall = np.concatenate([H for _, H in out])
    cls = [k for k in range(150) if Hall[:, :, k].mean() > 0.002 and (keep_person or k != PERSON)]
    return [(F, H[:, :, cls].reshape(len(H), -1)) for F, H in out], tks


if __name__ == "__main__":
    res = {}
    for kp, lab in [(False, "background"), (True, "full frame")]:
        for enc in ENCS:
            res[("WEAR ego", lab, enc)] = r2_pc12(*wear_groups(enc, kp))
            res[("Ego-Exo4D ego", lab, enc)] = r2_pc12(*ego_groups("ego", enc, kp))
            res[("Ego-Exo4D exo", lab, enc)] = r2_pc12(*ego_groups("exo", enc, kp))
            print(f"  {lab} {enc} done", flush=True)
    for lab in ["background", "full frame"]:
        print(f"\n=== Screen layout ({lab}) -> PC1-2 R^2 (within take, 5 time-block CV) ===")
        for cond in ["WEAR ego", "Ego-Exo4D ego", "Ego-Exo4D exo"]:
            print(f"  {cond:18s} " + "  ".join(f"{e} {res[(cond, lab, e)]:+.3f}" for e in ENCS))
