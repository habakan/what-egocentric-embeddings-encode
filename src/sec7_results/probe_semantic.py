"""Is the video component H predictable from the four limbs a component of "what the activity is"? (WEAR)

H cannot be explained by background, body, tilt, or camera shake (probe_motion.py).
Hypothesis: H is the "looks like this activity" direction in the video (the average appearance of each activity).
  B. R^2 of predicting H from the true activity labels (one-hot) / the inertial classifier's prediction P (within participant, 5 time-block CV)
  C. Removal: gain when subtracting the video component explained by regressing, within participant, on true labels (diagnostic; not usable on test) / P / H
System is the feature-only graph + 3 stages, 3 encoders, 2 seeds.
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
from sklearn.linear_model import Ridge, RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_limbpred import oof
from probe_prf import TAGS
from probe_wear_encoders import eval_tiles
from refine_gpu import final_with


def cv_r2_total(X, Y, blk):
    Yc = Y - Y.mean(0); pr = np.zeros_like(Yc)
    for b in range(5):
        tr, te = blk != b, blk == b
        pr[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], Yc[tr]).predict(X[te])
    return ((Yc - pr) ** 2).sum(), (Yc ** 2).sum()


def explained(X, Y, alpha=1.0):
    return Ridge(alpha=alpha).fit(X, Y).predict(X)


def main():
    d = oof(None)
    pos_of = {t: i for i, t in enumerate(d["tile"])}
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        tile0, P, meta = eval_tiles(sd)
        use = np.array([t in pos_of for t in tile0])
        tile, P = tile0[use], P[use]; pos = np.array([pos_of[t] for t in tile])
        y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
        Y1 = np.eye(N_CLASSES)[y]
        Vs = [load_vgraph_oof(tg, TAGS[0], seed=sd) for tg in ["exp020_video_only", "exp022_vonly_center_5f"]]
        sbj_all = meta["sbj_id"].to_numpy()[tile0]
        V = np.empty((len(y), N_CLASSES)); Vg0 = {}
        for s in np.unique(sbj):
            u = use[sbj_all == s]
            v = np.sqrt(Vs[0][s][u] * Vs[1][s][u]); V[sbj == s] = v / v.sum(1, keepdims=True)
            Vg0[s] = np.zeros((u.sum(), N_CLASSES))
        for e in ENC_LIST:
            Yc, H = d[f"Yc_{e}"][pos], d[f"H_{e}"][pos]
            groups = {"true labels": Y1, "P (inertial prediction)": np.log(P + 1e-4)}
            if sd == 42:
                err = collections.defaultdict(float); tot = 0.0; errY = collections.defaultdict(float); totY = 0.0
                for s in np.unique(sbj):
                    m = sbj == s; blk = np.arange(m.sum()) * 5 // m.sum()
                    for g, X in groups.items():
                        a, b = cv_r2_total(X[m], H[m], blk); err[g] += a
                        a2, b2 = cv_r2_total(X[m], Yc[m], blk); errY[g] += a2
                    tot += b; totY += b2
                print(f"[{e}] R^2 predicting H: " + "  ".join(f"{g} {1 - err[g] / tot:+.3f}" for g in groups)
                      + "   | R^2 predicting all video: " + "  ".join(f"{g} {1 - errY[g] / totY:+.3f}" for g in groups), flush=True)
            feats = {"raw": {}, "subtract H": {}}
            for g in groups:
                feats[f"subtract component explained by {g}"] = {}
            for s in np.unique(sbj):
                m = sbj == s
                feats["raw"][s] = Yc[m]; feats["subtract H"][s] = Yc[m] - H[m]
                for g, X in groups.items():
                    feats[f"subtract component explained by {g}"][s] = Yc[m] - explained(X[m], Yc[m])
            for g, Fv in feats.items():
                R[(e, g)].append(tune_tau(final_with(P, Fv, Vg0, V, sbj, 0), y)[0])
            R[(e, "top30 removed")].append(tune_tau(final_with(P, feats["raw"], Vg0, V, sbj, 30), y)[0])
        print(f"seed {sd} done", flush=True)
    print("\n=== Removal (feature-only graph, 3 stages, mean of 2 seeds, diff vs raw) ===")
    for e in ENC_LIST:
        b = np.array(R[(e, "raw")])
        print(f"  {e}: raw {b.mean():.4f}  " + "  ".join(f"{g} {(np.array(v) - b).mean():+.4f}" for (ee, g), v in R.items() if ee == e and g != "raw"))


if __name__ == "__main__":
    main()
