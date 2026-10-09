"""What if, instead of the top 30 principal components, only the head-tilt directions are orthogonalized and removed? (WEAR)

WEAR has no head sensor, so the directions head tilt creates in the embedding are learned from Ego-Exo4D measurements:
  regress the per-participant-centred embedding F on head tilt g (3-D gravity direction in device coordinates; also 2nd-order terms),
  and orthonormalize the coefficients (the directions tilt creates in F). Project that subspace out of the WEAR features.
All three encoders encode both datasets the same way (VideoMAEv2 follows the same spec as WEAR's released features).
Compared: raw / remove top 30 / tilt-direction removal (linear 3, up to 2nd order 9) / removal of random directions of the same dimension / tilt removal + remove top 30
Sanity check: apply the tilt reader learned on Ego to WEAR and look at the estimated tilt per activity (lying and standing exercises should separate).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES
from postprocess import tune_tau
from probe_wear_encoders import IMG, PREP, PUB2REC, eval_tiles
from refine_gpu import final_with
from final_submit import load_vgraph_oof
from probe_prf import TAGS
from config import N_CLASSES

EGO = Path(DATA_DIR + "/egoexo4d")
EDIR = {"VideoMAEv2": "features", "CLIP": "features_clip", "DINOv2": "features_dinov2"}


def ego_tilt(enc):
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(str(EGO / EDIR[enc] / "*.npz"))):
        z = np.load(f); ok = z["ok"]
        by[int(z["participant"])].append((z["F"].mean(1)[ok], z["pose"][ok, :3]))
    Fs, Gs = [], []
    for v in by.values():
        F = np.concatenate([a for a, _ in v]); G = np.concatenate([b for _, b in v])
        Fs.append(F - F.mean(0)); Gs.append(G - G.mean(0))
    return np.concatenate(Fs), np.concatenate(Gs)


def tilt_basis(F, G, quad):
    X = np.column_stack([G] + ([G ** 2, G[:, [0]] * G[:, [1]], G[:, [0]] * G[:, [2]], G[:, [1]] * G[:, [2]]] if quad else []))
    X = (X - X.mean(0)) / X.std(0)
    C = Ridge(alpha=1.0).fit(X, F).coef_            # (768, p): directions tilt creates in F
    Q, _ = np.linalg.qr(C)
    return Q.T                                        # (p, 768) orthonormal


def project_out(F, B):
    return F - (F @ B.T) @ B


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower() / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2"]}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    res = collections.defaultdict(list)
    bases = {}
    for e in EDIR:
        Fe, Ge = ego_tilt(e)
        bases[e] = {"lin": tilt_basis(Fe, Ge, False), "quad": tilt_basis(Fe, Ge, True),
                    "dec": RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Fe, Ge)}
    for sd in [42, 7]:
        tile, P, meta = eval_tiles(sd)
        y_all = meta["label"].to_numpy()[tile]; sbj_all = meta["sbj_id"].to_numpy()[tile]
        rec_all = meta["rec"].to_numpy()[tile]; sec_all = meta["start"].to_numpy()[tile] // 50
        recs = set(img["CLIP"])
        use = np.array([r in recs and s < len(img["CLIP"][r]) and s < len(img["DINOv2"][r]) for r, s in zip(rec_all, sec_all)])
        Vg_d = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        y, sbj = y_all[use], sbj_all[use]
        Vg, V = {}, np.empty((use.sum(), N_CLASSES))
        for s in np.unique(sbj):
            u = use[sbj_all == s]
            Vg[s] = np.zeros_like(Vg_d[s][u])          # build the graph from features only
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
        Pu = P[use]
        rng = np.random.RandomState(sd)
        for e in EDIR:
            F = {}
            for s in np.unique(sbj):
                idx = np.where(use & (sbj_all == s))[0]
                if e == "VideoMAEv2":
                    X = np.asarray(video[tile[idx]], np.float32).mean(1)
                else:
                    X = np.stack([img[e][rec_all[i]][sec_all[i]] for i in idx])
                F[s] = X - X.mean(0)
            if sd == 42:                               # sanity: estimated tilt per activity
                Gh = bases[e]["dec"].predict(np.concatenate([F[s] for s in np.unique(sbj)]))
                yy = np.concatenate([y[sbj == s] for s in np.unique(sbj)])
                j = int(np.argmax(Gh.std(0)))
                mu = {CLASS_NAMES[c]: Gh[yy == c, j].mean() for c in np.unique(yy)}
                lo = sorted(mu.items(), key=lambda t: t[1])
                print(f"[{e}] estimated tilt (component {j}) activity means, ascending: " + ", ".join(f"{k} {v:+.2f}" for k, v in lo[:4])
                      + " ... " + ", ".join(f"{k} {v:+.2f}" for k, v in lo[-4:]), flush=True)
            dim = F[next(iter(F))].shape[1]
            variants = {
                "raw": ({s: f for s, f in F.items()}, 0),
                "remove top 30": ({s: f for s, f in F.items()}, 30),
                "tilt removal linear 3": ({s: project_out(f, bases[e]["lin"]) for s, f in F.items()}, 0),
                "tilt removal quad 9": ({s: project_out(f, bases[e]["quad"]) for s, f in F.items()}, 0),
                "control random 9": ({s: project_out(f, np.linalg.qr(rng.randn(dim, 9))[0].T) for s, f in F.items()}, 0),
                "tilt removal 9 + remove top 30": ({s: project_out(f, bases[e]["quad"]) for s, f in F.items()}, 30),
            }
            for nm, (Fv, m) in variants.items():
                Q = final_with(Pu, Fv, Vg, V, sbj, m)
                res[(e, nm)].append(tune_tau(Q, y)[0])
        print(f"seed {sd} done", flush=True)
    print("\n=== OOF macro-F1 (feature-only graph, mean of 2 seeds, diff from raw) ===")
    for e in EDIR:
        base = np.array(res[(e, "raw")])
        print(f"  {e}")
        for nm in ["raw", "remove top 30", "tilt removal linear 3", "tilt removal quad 9", "control random 9", "tilt removal 9 + remove top 30"]:
            v = np.array(res[(e, nm)])
            print(f"    {nm:22s} {v.mean():.4f}  ({(v - base).mean():+.4f})")


if __name__ == "__main__":
    main()
