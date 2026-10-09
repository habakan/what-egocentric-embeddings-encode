"""Additional experiment C for likely review comments (paper/review-openreview.md): check the implications of the causal diagram with a quasi-experiment.

Split the video embedding (centred within participant) into two parts (cache from probe_limbpred.py, OOF with 5-fold split by subject):
  appearance A = the part predictable from the inertial features of the four limbs (the part that should come from activity y)
  remainder  R = embedding - A (includes the part that should come from the situation c)
The unit is a "set" (a run of the same activity label within one recording, NULL excluded, 10 windows or more). Take the mean of A and R per set,
and compare cosine distances between pairs of sets.

  Hold the situation nearly fixed and change only the activity: adjacent sets in the same recording (gap of 90 s or less)
  Change the situation and hold the activity fixed   : sets of a different participant (different place, day, person)

Implications of the causal diagram:
  (1) For adjacent sets, R does not change when the activity changes -> R's "activity effect" AUC is near 0.5
  (2) A follows the activity whether the situation is the same or not -> A's "activity effect" AUC is above 0.5
  (3) With the same activity, R changes a lot when the situation changes -> R's "situation effect" AUC is above 0.5, and larger than A's
AUC = probability that "distance of a pair with different conditions > distance of a pair with the same condition". 3 encoders. Bootstrap intervals with participant as the unit.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK

CACHE = Path(DATA_DIR + "/wear_img_feats/limbpred.npz")
MIN_LEN, GAP = 10, 90


def cosd(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9); b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return 1 - (a * b).sum(-1)


def sets_of(meta, tile):
    rec = meta["rec"].to_numpy()[tile]; sec = meta["start"].to_numpy()[tile] // 50
    lab = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
    rows = []
    for r in np.unique(rec):
        ii = np.where(rec == r)[0]; ii = ii[np.argsort(sec[ii])]
        cur = [ii[0]]
        for a, b in zip(ii[:-1], ii[1:]):
            if lab[b] == lab[a] and sec[b] - sec[a] <= 2:
                cur.append(b)
            else:
                rows.append(cur); cur = [b]
        rows.append(cur)
    out = []
    for c in rows:
        c = np.array(c)
        if lab[c[0]] != 0 and len(c) >= MIN_LEN:
            out.append(dict(idx=c, rec=rec[c[0]], sbj=sbj[c[0]], y=lab[c[0]], t0=sec[c].min(), t1=sec[c].max()))
    return out


def pairs(S, rng):
    adj, cross = [], []
    by_rec = {}
    for i, s in enumerate(S):
        by_rec.setdefault(s["rec"], []).append(i)
    for r, ii in by_rec.items():
        ii = sorted(ii, key=lambda i: S[i]["t0"])
        for a, b in zip(ii[:-1], ii[1:]):
            if S[b]["t0"] - S[a]["t1"] <= GAP:
                adj.append((a, b))
    n = len(S)
    while len(cross) < 20000:
        a, b = rng.integers(n, size=2)
        if S[a]["sbj"] != S[b]["sbj"]:
            cross.append((a, b))
    return adj, cross


def auc(d_same, d_diff, w_same=None, w_diff=None):
    if w_same is not None:      # drop pairs with weight 0 (pairs involving participants not drawn in the bootstrap)
        d_same, d_diff = d_same[w_same > 0], d_diff[w_diff > 0]
        w_same, w_diff = w_same[w_same > 0], w_diff[w_diff > 0]
    if len(d_same) < 3 or len(d_diff) < 3:
        return np.nan
    sw = None if w_same is None else np.r_[w_same, w_diff]
    return roc_auc_score(np.r_[np.zeros(len(d_same)), np.ones(len(d_diff))], np.r_[d_same, d_diff], sample_weight=sw)


def stats(S, M, adj, cross, cnt=None):
    """cnt: participant -> number of times k drawn in the bootstrap. Pairs of different participants get weight k_i * k_j, pairs within the same participant get k
    (duplicating a participant k times only multiplies its within-participant pairs by k). Participants not drawn get 0."""
    def pw(i, j):
        si, sj = S[i]["sbj"], S[j]["sbj"]
        return cnt.get(si, 0) if si == sj else cnt.get(si, 0) * cnt.get(sj, 0)
    w = (lambda pr: None) if cnt is None else (lambda pr: np.array([pw(i, j) for i, j in pr], float))
    res = {}
    for nm, X in M.items():
        d = lambda pr: np.array([cosd(X[i], X[j]) for i, j in pr])
        same = lambda pr: [(i, j) for i, j in pr if S[i]["y"] == S[j]["y"]]
        diff = lambda pr: [(i, j) for i, j in pr if S[i]["y"] != S[j]["y"]]
        for key, a, b in [("act|adj", same(adj), diff(adj)), ("act|cross", same(cross), diff(cross)),
                          ("ctx|same act", same(adj), same(cross))]:
            res[(nm, key)] = auc(d(a), d(b), w(a), w(b))
    return res


def main():
    meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
    c = np.load(CACHE, allow_pickle=True)
    tile = c["tile"]
    S = sets_of(meta, tile)
    rng = np.random.default_rng(0)
    adj, cross = pairs(S, rng)
    na_s = sum(S[i]["y"] == S[j]["y"] for i, j in adj)
    print(f"sets {len(S)} (NULL excluded, {MIN_LEN} windows or more). adjacent pairs {len(adj)} (same activity {na_s}, different activity {len(adj) - na_s}), "
          f"pairs of different participants {len(cross)}")
    sbjs = np.unique([s["sbj"] for s in S])
    for enc in ["VideoMAEv2", "CLIP", "DINOv2", "VC-1"]:
        Yc, H = c[f"Yc_{enc}"], c[f"H_{enc}"]
        M = {"A (part predictable from four limbs)": np.stack([H[s["idx"]].mean(0) for s in S]),
             "R (remainder)": np.stack([(Yc - H)[s["idx"]].mean(0) for s in S])}
        pt = stats(S, M, adj, cross)
        bs = []
        for _ in range(200):         # use the number of draws with replacement as pair weights (deduplicating leaves only 12.8 of 20 participants on average)
            u, n = np.unique(rng.choice(sbjs, len(sbjs), replace=True), return_counts=True)
            bs.append(stats(S, M, adj, cross, dict(zip(u, n))))
        print(f"\n=== {enc} ===  AUC (0.5 = no effect), [participant bootstrap 95%]")
        for key in ["act|adj", "act|cross", "ctx|same act"]:
            lab = {"act|adj": "activity effect (situation nearly fixed: adjacent sets)",
                   "act|cross": "activity effect (different situation: different participant)",
                   "ctx|same act": "situation effect (activity fixed: adjacent vs different participant)"}[key]
            print(f"  {lab}")
            for nm in M:
                v = np.array([b[(nm, key)] for b in bs]); v = v[~np.isnan(v)]
                lo, hi = np.percentile(v, [2.5, 97.5])
                print(f"    {nm:22s} {pt[(nm, key)]:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
