"""What do we learn by splitting video features = f(body motion) + residual?

Hypothesis: part of the head-camera video features is "a result of body motion" (egomotion, change of view),
  predictable from the limbs' inertial data. Removing the top 30 principal components works perhaps because it removes this "result of motion"
  from the video side, so the graph is built on the part that does not overlap with inertial.

Difference from before: extend the linear, one-limb-at-a-time measurement (CCA mean 0.101, only PC1-2 predictable)
  to all four limbs at once and non-linear (MLP).

What we measure:
  1. For each within-subject principal component, the fraction R^2 explained by inertial (OOF prediction, 5 folds by subject).
     If concentrated in the top 30, the explanation "removal = removing the result of motion" holds
  2. Per-class R^2 (should differ between whole-body and arm-only activities)
  3. Run the final configuration with different graph features:
       raw / drop30 (current) / residual(four limbs) / residual(one limb, usable at test) / drop30(residual one limb) / explained part(one limb)
kill: if the residual(one limb) graph is 0.005 or more worse than drop30 in OOF, and R^2 is not concentrated in the top 30,
  stop (record as the 18th refutation).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES, WORK
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from refine import drop_top_pcs, multistage_at

PREP = WORK / "prep"


def zsub(X, sbj):
    X = X.copy()
    for s in np.unique(sbj):
        m = sbj == s
        X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-6)
    return X


def csub(Y, sbj):
    Y = Y.copy()
    for s in np.unique(sbj):
        m = sbj == s; Y[m] -= Y[m].mean(0)
    return Y


def fit_mlp(X, Y, seed, epochs, dev="cuda"):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(512, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X).float().to(dev); Yt = torch.from_numpy(Y).float().to(dev)
    n = len(X)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def predict(net, X, dev="cuda"):
    return net(torch.from_numpy(X).float().to(dev)).cpu().numpy()


def oof_predictions(a):
    """For all tiled windows, build OOF predictions of video (within-subject centred) from four limbs and from one limb."""
    cache = WORK / "decomp" / "oof.npz"
    if cache.exists() and not a.refit:
        d = np.load(cache); return {k: d[k] for k in d.files}
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sbj = meta["sbj_id"].to_numpy()[tile]; y = meta["label"].to_numpy()[tile]
    fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    Xl = [np.nan_to_num(np.asarray(fi[tile + l * n], np.float32)) for l in range(4)]
    Xl = [zsub(x, sbj) for x in Xl]
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    Y = np.concatenate([np.asarray(vid[tile[i:i + 4096]], np.float32).mean(1)
                        for i in range(0, len(tile), 4096)])
    Yc = csub(Y, sbj); sd = Yc.std(0) + 1e-6; Yn = Yc / sd
    X4 = np.column_stack(Xl)
    us = np.random.RandomState(0).permutation(np.unique(sbj))
    fold = np.array([{s: i % 5 for i, s in enumerate(us)}[s] for s in sbj])
    H4 = np.zeros_like(Yc); H1 = np.zeros((4, *Yc.shape), np.float32)
    for k in range(5):
        tr, te = fold != k, fold == k
        net = fit_mlp(X4[tr], Yn[tr], k, a.epochs); H4[te] = predict(net, X4[te]) * sd
        X1 = np.vstack([np.column_stack([Xl[l], np.tile(np.eye(4)[l], (len(Xl[l]), 1))]) for l in range(4)])
        tr1 = np.tile(tr, 4); Y1 = np.tile(Yn, (4, 1))
        net = fit_mlp(X1[tr1], Y1[tr1], 100 + k, a.epochs)
        for l in range(4):
            H1[l, te] = predict(net, np.column_stack([Xl[l][te], np.tile(np.eye(4)[l], (te.sum(), 1))])) * sd
        print(f"  fold {k} done", flush=True)
    out = dict(tile=tile, sbj=sbj, y=y, Y=Y, Yc=Yc, H4=H4, H1=H1)
    cache.parent.mkdir(exist_ok=True); np.savez(cache, **out)
    return out


def r2_by_pc(Yc, H, sbj, bands):
    """On each subject's principal axes, average the fraction of explained variance per band."""
    acc = {b: [] for b in bands}
    for s in np.unique(sbj):
        m = sbj == s
        _, _, Vt = np.linalg.svd(Yc[m], full_matrices=False)
        tot = ((Yc[m] @ Vt.T) ** 2).sum(0); err = (((Yc[m] - H[m]) @ Vt.T) ** 2).sum(0)
        r2 = 1 - err / tot
        for b in bands:
            acc[b].append(r2[b[0] - 1:b[1]].mean())
    return {b: float(np.mean(v)) for b, v in acc.items()}


def proj_sub(Yc, H, sbj_rows, r):
    """Subtract the prediction only within the subspace of the top r within-subject principal components (no noise injected into low-variance directions)."""
    out = Yc.copy()
    for s in np.unique(sbj_rows):
        m = sbj_rows == s
        _, _, Vt = np.linalg.svd(Yc[m], full_matrices=False)
        U = Vt[:r].T
        out[m] = Yc[m] - (H[m] @ U) @ U.T
    return out


def imputed_h4(d, pos, sens, sbj, Fd, k=30):
    """Four limbs usable at test: own wearing position uses own features, other positions are imputed by the mean over video neighbours (drop30);
    then recompute the four-limb model's prediction. The four-limb model must be retrained per OOF split, so here
    the MLP is retrained per subject split."""
    from probe_nbfeat import pooled_features
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    tile = d["tile"][pos]
    fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
    Xl = [zsub(np.nan_to_num(np.asarray(fi[tile + l * n], np.float32)), sbj) for l in range(4)]
    own = np.stack(Xl, 1)[np.arange(len(tile)), sens]                  # (m, 78)
    X4 = np.zeros((len(tile), 4 * 78), np.float32)
    for s in np.unique(sbj):
        m = np.where(sbj == s)[0]
        pool = pooled_features(Fd[s], own[m], sens[m], k)[:, :4 * 78]
        pool = np.nan_to_num(pool)
        for l in range(4):
            blk = pool[:, l * 78:(l + 1) * 78]
            blk[sens[m] == l] = own[m][sens[m] == l]
            X4[m, l * 78:(l + 1) * 78] = blk
    return X4


def final_with(P, Fd, Vg, V, sbj, drop):
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = drop_top_pcs(Fd[s], drop) if drop else Fd[s]
        f = f - f.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G = np.concatenate([f, 0.5 * np.sqrt(Vg[s])], 1)
        Q[m] = multistage_at(P[m], G, V[m], [4.0, 4.0], k=30, alpha=0.75, iters=5,
                             temp=1.4, g=0.2, where="every")
    return Q


def main(a):
    d = oof_predictions(a)
    if a.only3:
        return part3(a, d)
    Yc, H4, H1, sbjA, yA = d["Yc"], d["H4"], d["H1"], d["sbj"], d["y"]
    bands = [(1, 2), (3, 10), (11, 30), (31, 100), (101, 768)]
    print("\n=== 1. R^2 per within-subject principal-component band (inertial -> video, subject-level OOF) ===")
    tot = lambda H: 1 - ((Yc - H) ** 2).sum() / (Yc ** 2).sum()
    print(f"  overall R^2: four limbs {tot(H4):.3f}  / one limb " + " ".join(f"{tot(H1[l]):.3f}" for l in range(4)))
    for nm, H in [("four limbs", H4), ("one limb (right arm)", H1[0]), ("one limb (right leg)", H1[1])]:
        r = r2_by_pc(Yc, H, sbjA, bands)
        print(f"  {nm:10s} " + "  ".join(f"PC{b[0]}-{b[1]}: {v:+.3f}" for b, v in r.items()))

    print("\n=== 2. Per-class R^2 (four limbs) ===")
    rows = []
    for c in range(N_CLASSES):
        m = yA == c
        rows.append((CLASS_NAMES[c], 1 - ((Yc[m] - H4[m]) ** 2).sum() / (Yc[m] ** 2).sum(), m.sum()))
    for nm, r, cnt in sorted(rows, key=lambda t: -t[1]):
        print(f"    {nm:30s} {r:+.3f}  (n={cnt})")

    part3(a, d)


def part3(a, d):
    Yc, H4, H1, sbjA = d["Yc"], d["H4"], d["H1"], d["sbj"]
    print("\n=== 3. Final configuration with different graph features ===")
    res = {}
    for sd in a.seeds:
        out, y, sbj = build(sd)
        P = out["base"]
        _, _, y2, _, sens, _, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
        assert (y2 == y).all()
        pos = np.searchsorted(d["tile"], tile); assert (d["tile"][pos] == tile).all()
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
            M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj): M[sbj == s] = dd[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        Yr = d["Y"][pos]; h4 = H4[pos]; h1 = H1[sens, pos]
        per = lambda A: {s: A[sbj == s] for s in np.unique(sbj)}
        Yrc = csub(Yr, sbj)
        # from the neighbour-imputed four limbs, predict with the four-limb model trained on the same subject split
        Xi = imputed_h4(d, pos, sens, sbj, per(Yr))
        tileA = d["tile"]; n = len(pd.read_parquet(PREP / "win_meta.parquet"))
        fi = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
        XA = np.column_stack([zsub(np.nan_to_num(np.asarray(fi[tileA + l * n], np.float32)), sbjA) for l in range(4)])
        sdY = Yc.std(0) + 1e-6
        usA = np.random.RandomState(0).permutation(np.unique(sbjA))
        fA = {s: i % 5 for i, s in enumerate(usA)}
        foldA = np.array([fA[s] for s in sbjA]); foldE = np.array([fA[s] for s in sbj])
        hI = np.zeros_like(Yr)
        for kf in range(5):
            net = fit_mlp(XA[foldA != kf], Yc[foldA != kf] / sdY, kf, a.epochs)
            hI[foldE == kf] = predict(net, Xi[foldE == kf]) * sdY
        variants = [("raw", per(Yr), 0), ("drop30 (current)", per(Yr), 30),
                    ("residual four limbs top10 projection", per(proj_sub(Yrc, h4, sbj, 10)), 0),
                    ("residual one limb top10 projection", per(proj_sub(Yrc, h1, sbj, 10)), 0),
                    ("residual imputed four limbs", per(Yrc - hI), 0),
                    ("residual imputed four limbs top10 projection", per(proj_sub(Yrc, hI, sbj, 10)), 0),
                    ("drop30(residual four limbs)", per(Yrc - h4), 30),
                    ("residual four limbs", per(Yrc - h4), 0), ("residual one limb", per(Yrc - h1), 0),
                    ("drop30(residual one limb)", per(Yrc - h1), 30)]
        for nm, Fd, dr in variants:
            f = tune_tau(final_with(P, Fd, Vg, V, sbj, dr), y)[0]
            res.setdefault(nm, []).append(f)
        print(f"  seed {sd} done", flush=True)
    ref = np.array(res["drop30 (current)"])
    for nm, v in res.items():
        v = np.array(v)
        print(f"  {nm:20s} {v.mean():.4f}  vs current {np.mean(v - ref):+.4f}  [" +
              " ".join(f"{x:+.4f}" for x in v - ref) + "]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--only3", action="store_true")
    p.add_argument("--refit", action="store_true")
    main(p.parse_args())
