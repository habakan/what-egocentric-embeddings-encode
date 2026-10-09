"""Following a public solution (Rudra), decide each window's successor with a learned link scorer + linear assignment, and smooth the system's output (slot-2 candidate, uses the leak).

Note: the adjacency relation is a strong assumption not acceptable for the task. Only for maximising the competition score; disclosed in the technical report if used.

1. Candidates: for each window A, the top K by video last-to-first distance + the top K by same-limb inertial extrapolation distance.
2. Pair features: v_last (1-cos last/first), v_ext (extrapolation distance), v_mean (1-cos mean), same limb or not, inertial extrapolation distance (same limb only;
   a different limb gets 0 and a missing flag), the rank of each within the other's candidates (dominance: B's rank as seen from A and A's rank as seen from B).
3. Scoring: logistic regression gives p = "probability of being the successor". Split subjects into 2 groups, train on one and score the other.
4. Assignment: per participant, linear assignment on a sparse cost matrix -log p with "no successor" (cost c0) added
   (scipy.optimize.linear_sum_assignment, non-candidates get a large cost).
5. Evaluation: for the assigned pairs, fraction that are true successors / label agreement / number accepted. Smooth the system's Q using only pairs with p > t, and report macro-F1 after the count band (per recording).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import tune_tau
from probe_chain_system import PREP, eval_rows, smooth
from probe_countband import band_calibrate
from probe_prf import TAGS
from refine_gpu import final_with

K = 20


def pair_feats(Vr, X, lm):
    """One participant. Returns: (a, b, feats) candidate pairs and features."""
    n = len(Vr); dev = "cuda"
    Vn = torch.nn.functional.normalize(Vr, dim=2)
    eye = torch.eye(n, dtype=torch.bool, device=dev); inf = torch.tensor(float("inf"), device=dev)
    vl = (1 - Vn[:, -1] @ Vn[:, 0].T).masked_fill(eye, inf)
    ext = torch.nn.functional.normalize(Vr[:, -1] + 16 * (Vr[:, -1] - Vr[:, -4]) / 3.0, dim=1)
    ve = torch.cdist(ext, Vn[:, 0]).masked_fill(eye, inf)
    mu = torch.nn.functional.normalize(Vr.mean(1), dim=1)
    vm = (1 - mu @ mu.T).masked_fill(eye, inf)
    same = (lm[:, None] == lm[None, :]) & ~eye
    ie = torch.where(same, torch.cdist(2 * X[:, -1] - X[:, -2], X[:, 0]), inf)
    cand = torch.zeros((n, n), dtype=torch.bool, device=dev)
    for D in [vl, ie]:
        idx = torch.topk(D, min(K, n - 1), dim=1, largest=False).indices
        cand.scatter_(1, idx, True)
    cand &= ~eye
    rk_row = lambda D: torch.argsort(torch.argsort(D, 1), 1).float()
    rk_col = lambda D: torch.argsort(torch.argsort(D, 0), 0).float()
    a, b = torch.nonzero(cand, as_tuple=True)
    iev = ie[a, b]; has = torch.isfinite(iev)
    F = torch.stack([vl[a, b], ve[a, b], vm[a, b], same[a, b].float(), torch.where(has, iev, torch.zeros_like(iev)),
                     torch.log1p(rk_row(vl)[a, b]), torch.log1p(rk_col(vl)[a, b]),
                     torch.where(has, torch.log1p(rk_row(ie)[a, b]), torch.full_like(iev, 8.0)),
                     torch.where(has, torch.log1p(rk_col(ie)[a, b]), torch.full_like(iev, 8.0))], 1)
    return a.cpu().numpy(), b.cpu().numpy(), F.cpu().numpy()


def assign(n, a, b, p, c0):
    """Linear assignment with sparse cost -log p (1e6 for non-candidates) + no successor (cost c0). Returns: accepted (a, b, p)."""
    C = np.full((n, n + n), 1e6, np.float32)
    C[a, b] = -np.log(np.clip(p, 1e-6, 1))
    C[np.arange(n), n + np.arange(n)] = c0
    r, c = linear_sum_assignment(C)
    keep = c < n
    r, c = r[keep], c[keep]
    pm = {(i, j): q for i, j, q in zip(a, b, p)}
    q = np.array([pm.get((i, j), 0.0) for i, j in zip(r, c)])
    return r, c, q


def main():
    xi = np.load(PREP / "inertial.npy", mmap_mode="r")
    vs = np.load(PREP / "video_raw_sync.npy", mmap_mode="r")
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    sd = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    tile, sens, P, meta = eval_rows(sd)
    y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
    rec = meta["rec"].to_numpy()[tile]; start = meta["start"].to_numpy()[tile]
    X = np.nan_to_num(np.asarray(xi[tile], np.float32))[np.arange(len(tile)), sens]
    subs = np.unique(sbj)
    data = {}
    for s in subs:
        m = np.where(sbj == s)[0]
        Vr = torch.tensor(np.asarray(vs[tile[m]], np.float32), device="cuda")
        a, b, F = pair_feats(Vr, torch.tensor(X[m], device="cuda"), torch.tensor(sens[m], device="cuda"))
        lab = (rec[m][a] == rec[m][b]) & (start[m][b] == start[m][a] + 50)
        data[s] = (m, a, b, F, lab)
        print(f"  sbj {s}: candidates {len(a)}, fraction with the true successor among candidates {lab.sum() / max(1, (np.isin(start[m] + 50, start[m])).sum()):.3f}", flush=True)
    grp = {s: i % 2 for i, s in enumerate(np.random.RandomState(0).permutation(subs))}
    prob = {}
    for g in [0, 1]:
        tr = [s for s in subs if grp[s] != g]
        Ftr = np.concatenate([data[s][3] for s in tr]); ytr = np.concatenate([data[s][4] for s in tr])
        mu, sdv = Ftr.mean(0), Ftr.std(0) + 1e-9
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight=None).fit((Ftr - mu) / sdv, ytr)
        for s in subs:
            if grp[s] == g:
                prob[s] = clf.predict_proba((data[s][3] - mu) / sdv)[:, 1]
    # the system's Q
    F0 = {s: np.asarray(video[tile[sbj == s]], np.float32).mean(1) for s in subs}
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in subs: M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    from probe_prf import W
    Q = final_with(P, F0, Vg, V, sbj, 30)
    _, tau = tune_tau(Q, y)
    L = lambda Z: np.log(np.clip(Z, 1e-9, None)) - tau * np.log(Z.mean(0) + 1e-9)
    base = macro_f1(y, band_calibrate(L(Q), rec, 78, 126))
    print(f"\nsystem + band (seed {sd}): {base:.4f}")
    for c0 in [1.6]:
        links = {}
        tot = corr = same_lab = 0
        for s in subs:
            m, a, b, F, lab = data[s]
            r, c, q = assign(len(m), a, b, prob[s], c0)
            links[s] = (r, c, q)
            ok = (rec[m][c] == rec[m][r]) & (start[m][c] == start[m][r] + 50)
            tot += len(r); corr += ok.sum(); same_lab += (y[m][r] == y[m][c]).sum()
        print(f"  c0={c0}: accepted {tot / len(y):.3f}/window, true successor {corr / max(tot, 1):.3f}, label match {same_lab / max(tot, 1):.3f}")
        for t in [0.0]:
            for aa in [0.2, 0.3, 0.5]:
                Z = Q.copy(); used = ok_n = 0
                for s in subs:
                    m = data[s][0]; r, c, q = links[s]; sel = q > t
                    used += sel.sum(); ok_n += ((rec[m][c[sel]] == rec[m][r[sel]]) & (start[m][c[sel]] == start[m][r[sel]] + 50)).sum()
                    Z[m] = smooth(Q[m], (r[sel], c[sel]), len(m), aa)
                f = macro_f1(y, band_calibrate(L(Z), rec, 78, 126))
                print(f"    p>{t} a={aa}: edges used {used / len(y):.3f}/window (true successor {ok_n / max(used, 1):.3f})  macro-F1 {f:.4f} ({f - base:+.4f})", flush=True)


if __name__ == "__main__":
    main()
