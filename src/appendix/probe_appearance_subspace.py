"""Idea 1: project out of the test participant's video features the "activity appearance subspace" learned from training labels (CV with the main system).

In the mechanism analysis (probe_semantic*.py), subtracting the per-activity means within participant (uses true labels, diagnostic only) helped more than removing the top 30.
Here the test participant's labels are not used: leave one participant out and estimate the subspace from the labels of the remaining participants only.
  CM      : subspace spanned by the per-activity means (classes weighted equally) of within-participant-centred features (at most 18 dims)
  CM-null : same, but only activities excluding NULL
  LDA-k   : top k discriminant-analysis directions (orthonormalized directions in feature space)
After removing that, combine with removal of the top m components. The system is final_with (probability block + video evidence injection) = same as the main one.
Evaluation: OOF macro-F1 (tune_tau), seeds 42/7/1.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import tune_tau
from probe_prf import TAGS, W
from refine_gpu import final_with


def csub(F):
    return F - F.mean(0)


def basis_cm(Ftr, ytr, drop_null):
    cs = [c for c in np.unique(ytr) if not (drop_null and c == 0)]
    M = np.stack([Ftr[ytr == c].mean(0) for c in cs])
    M = M - M.mean(0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = int((S > S[0] * 1e-6).sum())
    return Vt[:k]


def basis_lda(Ftr, ytr, k):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.1).fit(Ftr, ytr)
    Wd = lda.scalings_[:, :k]
    Q, _ = np.linalg.qr(Wd)
    return Q.T


def project_out(F, B):
    return F - (F @ B.T) @ B


def main():
    configs = [("none", None, 0), ("top 30 (current)", None, 30)]
    for bname in ["CM", "CM-null", "LDA-5", "LDA-10", "LDA-18"]:
        for m in [0, 10, 20, 30]:
            configs.append((f"{bname} + top {m}", bname, m))
    res = {c[0]: [] for c in configs}
    for sd in [42, 7, 1]:
        P, y, sbj, F0 = load_eval_set(TAGS, W, seed=sd)
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            dd = load_vgraph_oof(tg, TAGS[0], seed=sd)
            M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj): M[sbj == s] = dd[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        subs = np.unique(sbj)
        Fc = {s: csub(F0[s]) for s in subs}
        ys = {s: y[sbj == s] for s in subs}
        bases = {}
        for s in subs:
            Ftr = np.concatenate([Fc[t] for t in subs if t != s]); ytr = np.concatenate([ys[t] for t in subs if t != s])
            bases[s] = {"CM": basis_cm(Ftr, ytr, False), "CM-null": basis_cm(Ftr, ytr, True)}
            for k in [5, 10, 18]:
                bases[s][f"LDA-{k}"] = basis_lda(Ftr, ytr, k)
        for name, bname, m in configs:
            Fd = {s: (Fc[s] if bname is None else project_out(Fc[s], bases[s][bname])) for s in subs}
            res[name].append(tune_tau(final_with(P, Fd, Vg, V, sbj, m), y)[0])
        print(f"seed {sd} done: none {res['none'][-1]:.4f}  top 30 {res['top 30 (current)'][-1]:.4f}", flush=True)
    ref = np.array(res["top 30 (current)"])
    print("\n=== OOF macro-F1 (mean of 3 seeds, diff from current = top 30) ===")
    for name, *_ in configs:
        v = np.array(res[name])
        print(f"  {name:18s} {v.mean():.4f} ± {v.std():.4f}  vs current {(v - ref).mean():+.4f}  [" + " ".join(f"{x:+.4f}" for x in v - ref) + "]")


if __name__ == "__main__":
    main()
