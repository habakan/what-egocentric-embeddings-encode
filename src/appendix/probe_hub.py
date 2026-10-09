"""Deep dive into hypothesis 8: measure the four mechanisms pointed to by the literature at once.

 A. Hubness (Radovanović+ 2010): skewness of the in-degree (k-occurrence) of the kNN graph.
    Hubs "spread labels too widely". If PC removal reduces hubs, that is a candidate mechanism.
    Also check consistency with the plan's "mutual kNN (hub suppression) gives +0.014".
 B. Anisotropy (Mu & Viswanath 2018): mean pairwise cosine similarity. Higher means more anisotropic, and
    the similarity ranking is dominated by a few directions.
 C. Coarse/fine activities (refinement of hypothesis 8): do the top PCs encode "coarse activity groups" (null/jog/stretch/...)
    and the lower PCs "fine distinctions within a group"? If the base's errors concentrate within groups,
    the lower PCs that separate within a group are exactly what correction needs.
 D. Conditional rescue rate: rescue / P(neighbour same class ∧ self wrong). Correction headroom discounted by purity.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES, SEED, WORK
from refine import drop_top_pcs

PREP = WORK / "prep"

# coarse activity groups (physically coherent units)
COARSE = {0: "null"}
for c, nm in enumerate(CLASS_NAMES):
    if c == 0: continue
    COARSE[c] = nm.split(" ")[0].split("-")[0]      # jogging / stretching / push / sit / burpees / lunges / bench
GROUP = {g: i for i, g in enumerate(sorted(set(COARSE.values())))}
yc_of = np.array([GROUP[COARSE[c]] for c in range(N_CLASSES)])


def load(tags, weights, seed=SEED):
    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]; rows = tile + sens * n
    P = None
    for t, w in zip(tags, weights):
        o = np.load(WORK / t / "oof.npy")[rows]; s = o.sum(1, keepdims=True)
        o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(weights); keep = P.sum(1) > 1e-6
    tile, P = tile[keep], P[keep]; P /= P.sum(1, keepdims=True)
    y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    F = {s: np.asarray(vid[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    return P, y, sbj, F


def knn(F, k):
    Fn = F - F.mean(0); Fn = Fn / (np.linalg.norm(Fn, axis=1, keepdims=True) + 1e-8)
    S = Fn @ Fn.T
    tri = S[np.triu_indices(len(S), 1)]
    np.fill_diagonal(S, -np.inf)
    kk = min(k, len(Fn) - 2)
    return np.argpartition(-S, kk, axis=1)[:, :kk], tri


def eta2(proj, lab):
    tot = proj.var()
    if tot < 1e-12: return 0.0
    g = np.unique(lab)
    gm = np.array([proj[lab == c].mean() for c in g]); gn = np.array([(lab == c).sum() for c in g], float)
    return float((gn * (gm - proj.mean()) ** 2).sum() / len(proj) / tot)


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    P, y, sbj, F = load(a.tags, w)
    pred = P.argmax(1); us = np.unique(sbj)

    # ---------- C. are the base's errors within or between groups
    wrong = pred != y
    within = (yc_of[pred] == yc_of[y]) & wrong
    print("=== C1. Structure of the base classifier's errors ===")
    print(f"  error rate {wrong.mean():.3f}  of which **errors within the same coarse group** {within.sum()/wrong.sum():.3f}  "
          f"errors across groups {1-within.sum()/wrong.sum():.3f}")

    # ---------- C2. per-component coarse-group eta^2 and fine (within-group) eta^2
    print("\n=== C2. What each component separates (mean over subjects) ===")
    print(f"  {'comp':>6s}{'coarse eta^2':>14s}{'19-class eta^2':>15s}{'within-group':>12s}")
    idx = [0, 1, 2, 4, 9, 19, 29, 49, 99]
    acc = {i: [] for i in idx}
    for s in us:
        m = sbj == s
        c = F[s] - F[s].mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        pr = c @ Vt[:100].T
        for i in idx:
            if i < pr.shape[1]:
                e_fine = eta2(pr[:, i], y[m]); e_coarse = eta2(pr[:, i], yc_of[y[m]])
                acc[i].append((e_coarse, e_fine))
    for i in idx:
        if acc[i]:
            ec, ef = np.mean([x[0] for x in acc[i]]), np.mean([x[1] for x in acc[i]])
            print(f"  {i+1:6d}{ec:14.4f}{ef:15.4f}{ef-ec:12.4f}")

    # ---------- A, B, D: graph properties per m
    print(f"\n=== A/B/D. Graph properties (k={a.k}, mean over subjects) ===")
    print(f"  {'PCs off':>7s}{'hub skew':>10s}{'max indeg/k':>14s}{'mean cos':>9s}"
          f"{'purity':>8s}{'cond rescue':>13s}{'hub err rate':>13s}")
    for mdrop in a.ms:
        skew, maxdeg, mcos, pur, crescue, huberr = [], [], [], [], [], []
        for s in us:
            m = sbj == s
            Fs = drop_top_pcs(F[s], mdrop)
            nn, tri = knn(Fs, a.k)
            ys, ps = y[m], pred[m]
            # A. skewness of in-degree
            deg = np.bincount(nn.ravel(), minlength=len(Fs)).astype(float)
            sk = ((deg - deg.mean()) ** 3).mean() / (deg.std() ** 3 + 1e-9)
            skew.append(sk); maxdeg.append(deg.max() / a.k)
            # error rate of hubs (top 5% in-degree) vs overall
            hub = deg >= np.percentile(deg, 95)
            huberr.append((ps[hub] != ys[hub]).mean() - (ps != ys).mean())
            # B. anisotropy
            mcos.append(tri.mean())
            # D. purity and conditional rescue
            same = ys[nn] == ys[:, None]
            pur.append(same.mean())
            me_wrong = (ps != ys)[:, None]
            nb_right = (ps == ys)[nn]
            denom = (me_wrong & same).sum()
            crescue.append((me_wrong & same & nb_right).sum() / max(denom, 1))
        print(f"  {mdrop:7d}{np.mean(skew):10.3f}{np.mean(maxdeg):14.1f}{np.mean(mcos):9.4f}"
              f"{np.mean(pur):8.4f}{np.mean(crescue):13.4f}{np.mean(huberr):+13.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--ms", nargs="*", type=int, default=[0, 10, 30, 50])
    main(p.parse_args())
