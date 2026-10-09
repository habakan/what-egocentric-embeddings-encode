"""macro-F1 of the system + count band (per recording) when the mirrored LightGBM (exp025_lgb_mirror) is put into the system's bases.
current: TAGS (exp001 with weight 1.5) / replace: exp001 -> exp025 / add: add exp025 with weight 1.5. seeds 42/7/1.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import tune_tau
from probe_countband import band_calibrate
from probe_prf import TAGS, W
from refine_gpu import final_with

PREP = WORK / "prep"
SETS = {"current": (TAGS, W),
        "replace": (["exp025_lgb_mirror"] + TAGS[1:], W),
        "add": (TAGS + ["exp025_lgb_mirror"], W + [1.5])}


def rows(seed, tags, ws):
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    r = tile + sens * n
    P = None
    for t, w in zip(tags, ws):
        o = np.load(WORK / t / "oof.npy")[r]
        s = o.sum(1, keepdims=True); o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(ws)
    keep = np.load(WORK / TAGS[0] / "oof.npy")[r].sum(1) > 1e-6          # same windows as load_vgraph_oof
    return tile[keep], P[keep] / P[keep].sum(1, keepdims=True), meta


video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
res = {k: [] for k in SETS}; base_f = {k: [] for k in SETS}
for sd in [42, 7, 1]:
    for name, (tags, ws) in SETS.items():
        tile, P, meta = rows(sd, tags, ws)
        y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]; rec = meta["rec"].to_numpy()[tile]
        F = {s: np.asarray(video[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
            M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj): M[sbj == s] = dd[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        Q = final_with(P, F, Vg, V, sbj, 30)
        _, tau = tune_tau(Q, y)
        L = np.log(np.clip(Q, 1e-9, None)) - tau * np.log(Q.mean(0) + 1e-9)
        res[name].append(macro_f1(y, band_calibrate(L, rec, 78, 126)))
        base_f[name].append(tune_tau(P, y)[0])
    print(f"seed {sd}: " + "  ".join(f"{k} {v[-1]:.4f} (P {base_f[k][-1]:.4f})" for k, v in res.items()), flush=True)
ref = np.array(res["current"])
for k, v in res.items():
    v = np.array(v)
    print(f"  {k:6s} sys+band {v.mean():.4f}  diff {(v - ref).mean():+.4f} [" + " ".join(f"{x:+.4f}" for x in v - ref) + f"]   base P {np.mean(base_f[k]):.4f}")
