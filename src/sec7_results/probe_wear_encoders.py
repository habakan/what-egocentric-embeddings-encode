"""WEAR: swap only the graph's video features for VideoMAEv2 / CLIP / DINOv2 and compare the effect of removing the top principal components.

The classifiers (P, V, Vg) are kept as in the current setup; only the graph features F change. The number of removed components m is swept.
Windows of recordings with no matching public video (sbj_14, sbj_20, sbj_21) are excluded identically for all three encoders. Evaluation is subject-level OOF macro-F1 (tune_tau), 2 seeds.
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
from postprocess import tune_tau
from probe_prf import TAGS, W
from refine_gpu import final_with

PREP = WORK / "prep"
IMG = Path(DATA_DIR + "/wear_img_feats")
MS = [0, 10, 30, 50]
# public video file name -> competition recording. Identified by cross-correlating the rate of change of the video embeddings (all at time offset 0, r 0.71-0.83).
# the public sbj_14 matches the competition sbj_14 only weakly (r 0.39), so it is not used. The videos corresponding to competition sbj_20/21 were not downloaded.
PUB2REC = {f"sbj_{i}": f"sbj_{i}" for i in list(range(0, 14)) + [15, 16, 17]}
PUB2REC.update({"sbj_18": "sbj_0_2", "sbj_19": "sbj_14_2", "sbj_20": "sbj_18", "sbj_21": "sbj_19"})


def eval_tiles(seed):
    """Returns the evaluation windows (tile) and P with the same random state as eval_pipeline.load_eval_set."""
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    rows = tile + sens * n
    P = None
    for t, w in zip(TAGS, W):
        o = np.load(WORK / t / "oof.npy")[rows]
        s = o.sum(1, keepdims=True)
        o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(W)
    keep = P.sum(1) > 1e-6
    tile, P = tile[keep], P[keep]
    return tile, P / P.sum(1, keepdims=True), meta


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if (IMG / "clip" / p.name).exists() and p.stem in PUB2REC)
    recs = [PUB2REC[p] for p in pubs]
    print("recordings with image features:", recs, flush=True)
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    IE = ["clip", "dinov2"] + (["vc1"] if all((IMG / "vc1" / f"{p}.npy").exists() for p in pubs) else [])
    img = {e: {PUB2REC[p]: np.load(IMG / e / f"{p}.npy").astype(np.float32).mean(1) for p in pubs} for e in IE}
    NAME = {"clip": "CLIP", "dinov2": "DINOv2", "vc1": "VC-1"}
    ENCS = ["VideoMAEv2"] + [NAME[e] for e in IE]
    res = {}
    for sd in [42, 7]:
        tile, P, meta = eval_tiles(sd)
        y_all = meta["label"].to_numpy()[tile]; sbj_all = meta["sbj_id"].to_numpy()[tile]
        rec_all = meta["rec"].to_numpy()[tile]; sec_all = meta["start"].to_numpy()[tile] // 50
        Vg_d = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        # also exclude windows beyond the length (in seconds) of the image features
        nsec = {r: len(img["clip"][r]) for r in recs}
        use = np.array([r in nsec and all(s < len(img[e][r]) for e in IE) for r, s in zip(rec_all, sec_all)])
        # filter the per-participant arrays (load_vgraph_oof is ordered by sbj==s) with use
        Vg, V = {}, np.empty((use.sum(), N_CLASSES))
        y, sbj = y_all[use], sbj_all[use]
        for s in np.unique(sbj):
            ms = sbj_all == s
            u = use[ms]
            assert len(Vg_d[s]) == ms.sum(), "evaluation window order is misaligned with load_vgraph_oof"
            Vg[s] = Vg_d[s][u]
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
        Pu = P[use]
        F = {e: {} for e in ENCS}
        for s in np.unique(sbj):
            idx = np.where(use & (sbj_all == s))[0]
            F["VideoMAEv2"][s] = np.asarray(video[tile[idx]], np.float32).mean(1)
            for e in IE:
                F[NAME[e]][s] = np.stack([img[e][rec_all[i]][sec_all[i]] for i in idx])
        res.setdefault("P (no graph)", []).append(tune_tau(Pu, y)[0])
        for e in F:
            for m in MS:
                Q = final_with(Pu, F[e], Vg, V, sbj, m)
                res.setdefault((e, m), []).append(tune_tau(Q, y)[0])
            # drop the VideoMAEv2-derived probability block from the graph and build it from the encoder features only
            Vg0 = {s: np.zeros_like(v) for s, v in Vg.items()}
            for m in [0, 30]:
                Q = final_with(Pu, F[e], Vg0, V, sbj, m)
                res.setdefault((e, "noVg", m), []).append(tune_tau(Q, y)[0])
        print(f"seed {sd}: windows {use.sum()} / {len(use)}, participants {len(np.unique(sbj))}", flush=True)
    print("\n=== OOF macro-F1 (mean of 2 seeds) ===")
    print(f"  P (no graph) {np.mean(res['P (no graph)']):.4f}")
    for e in ENCS:
        base = np.array(res[(e, 0)])
        row = "  ".join(f"m={m}: {np.mean(res[(e, m)]):.4f} ({np.mean(np.array(res[(e, m)]) - base):+.4f})" for m in MS)
        print(f"  {e:10s} {row}")
    print("  -- no probability block in the graph (features only) --")
    for e in ENCS:
        a, b = np.array(res[(e, "noVg", 0)]), np.array(res[(e, "noVg", 30)])
        print(f"  {e:10s} m=0: {a.mean():.4f}  m=30: {b.mean():.4f} ({(b - a).mean():+.4f})  seeds [" + " ".join(f"{x:+.4f}" for x in b - a) + "]")


if __name__ == "__main__":
    main()
