"""Mechanism of the removal gain, part 4: check per class "scatter concentrated wrong neighbourhoods so re-linking can prune them" (H4).

For each encoder, with the graph features set to raw (m=0) / top 30 removed / residual after subtracting the four-limb prediction / permutation control,
  stage-1 graph: class-mean (excluding NULL) purity and concentration (share of the most common wrong class among wrong neighbours)
  graph after re-linking (stage 3): share of edges linking different activities, and concentration
  final macro-F1 (features only, 3 stages with g=0) and per-class F1
H4 predictions:
  (i)  concentration also drops for the residual (but not for the permutation control)
  (ii) the per-class drop in concentration correlates positively with that class's F1 gain (across 3 encoders)
  (iii) edges removed by the removal concentrate on class pairs that share posture (e.g. sit-ups and its variant, standing stretches)
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_limbpred import oof
from probe_wear_encoders import IMG, PUB2REC, eval_tiles
from refine_gpu import _t, drop_top_pcs, knn_graph, propagate, sharpen


def graph_stats(nn, w, y):
    keep = (w > -1e8).cpu().numpy(); nn = nn.cpu().numpy(); n = len(y)
    H = np.zeros((n, N_CLASSES))
    np.add.at(H, (np.repeat(np.arange(n), nn.shape[1])[keep.ravel()], y[nn][keep]), 1)
    cnt = H.sum(1).clip(1)
    pur = H[np.arange(n), y] / cnt
    Hw = H.copy(); Hw[np.arange(n), y] = 0
    conc = np.where(Hw.sum(1) > 0, Hw.max(1) / Hw.sum(1).clip(1), np.nan)
    cross = Hw.sum(1) / cnt
    # Number of wrong edges per class pair (true class -> neighbour class)
    E = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(E, (y, slice(None)), 0)
    for c in range(N_CLASSES):
        E[c] = Hw[y == c].sum(0)
    return pur, conc, cross, E


def pipeline(P, y, f):
    f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
    Pt = _t(P)
    nn, w = knn_graph(f, 30); s1 = graph_stats(nn, w, y)
    Q = sharpen(propagate(Pt, nn, w, 0.75, 5), 1.4)
    for _ in range(2):
        nn, w = knn_graph(torch.cat([f, 4.0 * torch.sqrt(Q)], 1), 30)
        Q = sharpen(propagate(Q, nn, w, 0.75, 5), 1.4)
    s3 = graph_stats(nn, w, y)
    return Q.cpu().numpy(), s1, s3


def cls_mean(v, y):
    return np.array([np.nanmean(v[y == c]) if (y == c).any() else np.nan for c in range(N_CLASSES)])


def per_class_f1(y, pred):
    out = np.full(N_CLASSES, np.nan)
    for c in np.unique(y):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        out[c] = 2 * tp / max(2 * tp + fp + fn, 1)
    return out


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower() / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2"]}
    d = oof(img)
    pos_of = {t: i for i, t in enumerate(d["tile"])}
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        tile, P, meta = eval_tiles(sd)
        use = np.array([t in pos_of for t in tile]); pos = np.array([pos_of[t] for t in tile[use]])
        y = meta["label"].to_numpy()[tile][use]; sbj = meta["sbj_id"].to_numpy()[tile][use]; P = P[use]
        rng = np.random.RandomState(sd)
        for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
            Yc, H = d[f"Yc_{e}"][pos], d[f"H_{e}"][pos]
            for nm in ["raw", "top30 removed", "residual", "control"]:
                Q = np.empty_like(P); acc = collections.defaultdict(list); E1 = 0; E3 = 0; yy = []
                for s in np.unique(sbj):
                    ms = sbj == s
                    X = Yc[ms] if nm in ("raw", "top30 removed") else Yc[ms] - (H[ms] if nm == "residual" else H[ms][rng.permutation(ms.sum())])
                    f = _t(X); f = drop_top_pcs(f, 30) if nm == "top30 removed" else f
                    Q[ms], s1, s3 = pipeline(P[ms], y[ms], f)
                    for k, v in zip(["purity1", "conc1", "diffedge1"], s1[:3]): acc[k].append(v)
                    for k, v in zip(["purity3", "conc3", "diffedge3"], s3[:3]): acc[k].append(v)
                    E1 = E1 + s1[3]; E3 = E3 + s3[3]; yy.append(y[ms])
                yy = np.concatenate(yy)
                f1, tau = tune_tau(Q, y)
                pred = prior_correct(Q, tau).argmax(1)
                row = {k: cls_mean(np.concatenate(v), yy) for k, v in acc.items()}
                row["F1"] = f1; row["clsF1"] = per_class_f1(y, pred); row["E1"] = E1; row["E3"] = E3
                R[(e, nm)].append(row)
        print(f"seed {sd} done", flush=True)

    act = [c for c in range(1, N_CLASSES)]
    def avg(e, nm, k):
        return np.nanmean([r[k] for r in R[(e, nm)]], 0)
    print("\n=== Class mean excluding NULL (2 seeds) ===")
    print(f"  {'':10s} {'':8s} {'F1':>7s} {'purity1':>7s} {'conc1':>7s} {'diffedge3':>7s} {'conc3':>7s}")
    for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
        for nm in ["raw", "top30 removed", "residual", "control"]:
            print(f"  {e:10s} {nm:8s} {avg(e, nm, 'F1'):7.4f} {np.nanmean(avg(e, nm, 'purity1')[act]):7.3f} "
                  f"{np.nanmean(avg(e, nm, 'conc1')[act]):7.3f} {np.nanmean(avg(e, nm, 'diffedge3')[act]):7.3f} "
                  f"{np.nanmean(avg(e, nm, 'conc3')[act]):7.3f}")
    print("\n=== (ii) per class: drop in conc1 vs F1 gain (top30 removed - raw), Spearman ===")
    xs, ys_ = [], []
    for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
        dc = -(avg(e, "top30 removed", "conc1") - avg(e, "raw", "conc1"))[act]
        dp = -(avg(e, "top30 removed", "purity1") - avg(e, "raw", "purity1"))[act]
        df = (avg(e, "top30 removed", "clsF1") - avg(e, "raw", "clsF1"))[act]
        dx = -(avg(e, "top30 removed", "diffedge3") - avg(e, "raw", "diffedge3"))[act]
        ok = np.isfinite(dc) & np.isfinite(df)
        print(f"  {e:10s} concentration drop rho {spearmanr(dc[ok], df[ok])[0]:+.2f} | purity drop rho {spearmanr(dp[ok], df[ok])[0]:+.2f}"
              f" | drop in different-activity edges after re-linking rho {spearmanr(dx[ok], df[ok])[0]:+.2f}  (n={ok.sum()})")
        xs += list(dc[ok]); ys_ += list(df[ok])
    print(f"  3 encoders pooled: concentration drop rho {spearmanr(xs, ys_)[0]:+.2f} (n={len(xs)})")
    print("\n=== (iii) class pairs of wrong edges most reduced by the removal in the stage-1 graph (sum over 3 encoders) ===")
    D = sum(np.mean([r["E1"] for r in R[(e, "raw")]], 0) - np.mean([r["E1"] for r in R[(e, "top30 removed")]], 0)
            for e in ["VideoMAEv2", "CLIP", "DINOv2"])
    Ds = D + D.T; iu = np.triu_indices(N_CLASSES, 1)
    order = np.argsort(-Ds[iu])[:10]
    for o in order:
        a, b = iu[0][o], iu[1][o]
        print(f"  {CLASS_NAMES[a]:28s} <-> {CLASS_NAMES[b]:28s} decrease {Ds[a, b]:.0f}")
    order = np.argsort(Ds[iu])[:5]
    print("  (conversely, pairs that increase most)")
    for o in order:
        a, b = iu[0][o], iu[1][o]
        print(f"  {CLASS_NAMES[a]:28s} <-> {CLASS_NAMES[b]:28s} increase {-Ds[a, b]:.0f}")


if __name__ == "__main__":
    main()
