"""Does "subtracting the video component predicted by the four limbs recovers most of the removal gain" (72% with VideoMAEv2) also replicate with CLIP / DINOv2?

For each encoder, predict the video features of all tiled windows (centred within participant) from the four limbs' inertial features with OOF over 5 subject-level folds
(same MLP and same splits as probe_decomp). Build the graph on
  raw / top-30 removed / residual (video - four-limb prediction, no components removed) / control: subtract the prediction permuted across windows within participant
and look at the recovered fraction = (residual - raw) / (top-30 removed - raw). The system is the same as probe_wear_encoders.
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_decomp import csub, fit_mlp, predict, zsub
from probe_prf import TAGS
from probe_wear_encoders import IMG, PREP, PUB2REC, eval_tiles
from refine_gpu import final_with

CACHE = Path(DATA_DIR + "/wear_img_feats/limbpred.npz")


def all_tiles(img):
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    rec = meta["rec"].to_numpy()[tile]; sec = meta["start"].to_numpy()[tile] // 50
    ok = np.array([r in img["CLIP"] and s < len(img["CLIP"][r]) and s < len(img["DINOv2"][r]) for r, s in zip(rec, sec)])
    return meta, n, tile[ok], rec[ok], sec[ok]


def oof(img):
    prev = {}
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True); prev = {k: d[k] for k in d.files}
        if img is None or all(f"H_{e}" in prev for e in ["VideoMAEv2"] + list(img)):
            return prev
    meta, n, tile, rec, sec = all_tiles(img)
    if prev:
        assert np.array_equal(prev["tile"], tile), "window order in the cache has changed"
    sbj = meta["sbj_id"].to_numpy()[tile]
    fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    X4 = np.column_stack([zsub(np.nan_to_num(np.asarray(fi[tile + l * n], np.float32)), sbj) for l in range(4)])
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    Ys = {} if "H_VideoMAEv2" in prev else {"VideoMAEv2": np.concatenate([np.asarray(vid[tile[i:i + 4096]], np.float32).mean(1) for i in range(0, len(tile), 4096)])}
    for e in img:
        Ys[e] = np.stack([img[e][r][s] for r, s in zip(rec, sec)])
    us = np.random.RandomState(0).permutation(np.unique(sbj))
    fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
    out = {"tile": tile}
    out.update(prev)
    for e, Y in Ys.items():
        if f"H_{e}" in prev:
            continue
        Yc = csub(Y, sbj); sd = Yc.std(0) + 1e-6
        H = np.zeros_like(Yc)
        for k in range(5):
            tr, te = fold != k, fold == k
            H[te] = predict(fit_mlp(X4[tr], Yc[tr] / sd, k, 30), X4[te]) * sd
        r2 = 1 - ((Yc - H) ** 2).sum() / (Yc ** 2).sum()
        print(f"  {e}: four limbs -> video OOF R^2 {r2:.3f}", flush=True)
        out[f"Yc_{e}"] = Yc; out[f"H_{e}"] = H
    np.savez(CACHE, **out)
    return out


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower().replace("-", "") / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2", "VC-1"]}
    d = oof(img)
    pos_of = {t: i for i, t in enumerate(d["tile"])}
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        tile, P, meta = eval_tiles(sd)
        y_all = meta["label"].to_numpy()[tile]; sbj_all = meta["sbj_id"].to_numpy()[tile]
        use = np.array([t in pos_of for t in tile])
        pos = np.array([pos_of[t] for t in tile[use]])
        y, sbj, Pu = y_all[use], sbj_all[use], P[use]
        Vg_d = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        Vg, Vg0, V = {}, {}, np.empty((use.sum(), N_CLASSES))
        for s in np.unique(sbj):
            u = use[sbj_all == s]
            Vg[s] = Vg_d[s][u]; Vg0[s] = np.zeros_like(Vg[s])
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
        rng = np.random.RandomState(sd)
        for e in ENC_LIST:
            Yc, H = d[f"Yc_{e}"][pos], d[f"H_{e}"][pos]
            Fs = {"raw": {}, "residual": {}, "control": {}}
            for s in np.unique(sbj):
                m = sbj == s
                Fs["raw"][s] = Yc[m]; Fs["residual"][s] = Yc[m] - H[m]
                Fs["control"][s] = Yc[m] - H[m][rng.permutation(m.sum())]
            for blk, VG in [("features only", Vg0), ("+probs", Vg)]:
                for nm, Fv, drop in [("raw", Fs["raw"], 0), ("top-30 removed", Fs["raw"], 30),
                                     ("residual", Fs["residual"], 0), ("control", Fs["control"], 0)]:
                    R[(e, blk, nm)].append(tune_tau(final_with(Pu, Fv, VG, V, sbj, drop), y)[0])
        print(f"seed {sd} done", flush=True)
    for blk in ["features only", "+probs"]:
        print(f"\n=== {blk} (mean of 2 seeds) ===")
        for e in ENC_LIST:
            g = {nm: np.mean(R[(e, blk, nm)]) for nm in ["raw", "top-30 removed", "residual", "control"]}
            frac = (g["residual"] - g["raw"]) / (g["top-30 removed"] - g["raw"])
            print(f"  {e:10s} raw {g['raw']:.4f}  top-30 removed {g['top-30 removed']:.4f}  residual {g['residual']:.4f}  control {g['control']:.4f}"
                  f"  -> recovered fraction {frac:.0%}")


if __name__ == "__main__":
    main()
