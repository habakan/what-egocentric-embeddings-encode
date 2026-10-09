"""Response to the review (review.txt): sensitivity to the decision rule and to the definition of video rows, and an anisotropy check.

1. Where to average pi_c in the prior correction. The paper's results (probe_prf.py) average over all evaluated clips
   (in CV the pooled OOF of 22 people, on test all test clips).
   Compare with averaging per participant. tau uses the same search in both cases (same grid as tune_tau).
2. Video rows used for the graph. The paper's results use [f0, f0+15) (video_raw_head). The 15 test rows are
   [f0+8, f0+23), so compare with building on the same rows (video_raw_sync). The classifier is not retrained.
3. Anisotropy: eigenvalue skew of within-participant centred features, and whitening instead of top-component removal.
   Whitening keeps only the top n components, so a truncated PCA keeping the same n components without whitening is the control.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metric import macro_f1
from probe_prf import TAGS, W, build
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from refine import drop_top_pcs, multistage_at
from config import N_CLASSES

TAUS = np.linspace(0, 2.0, 41)


def decide(Q, sbj, tau, per_sbj):
    Q = Q / Q.sum(1, keepdims=True)
    if not per_sbj:
        return (Q / np.power(Q.mean(0) + 1e-9, tau)).argmax(1)
    out = np.empty(len(Q), int)
    for s in np.unique(sbj):
        m = sbj == s
        out[m] = (Q[m] / np.power(Q[m].mean(0) + 1e-9, tau)).argmax(1)
    return out


def best_f1(Q, y, sbj, per_sbj):
    return max(macro_f1(y, decide(Q, sbj, t, per_sbj)) for t in TAUS)


def whiten(F, n):
    """Project onto the top n components and equalize variance (drop beyond n: dimensionality reduction + variance equalization)."""
    f = F - F.mean(0)
    U, S, _ = np.linalg.svd(f, full_matrices=False)
    return U[:, :n] * np.sqrt(len(f))


def truncate(F, n):
    """Control for whiten: project onto the same top n components but do not equalize variance."""
    f = F - F.mean(0)
    U, S, _ = np.linalg.svd(f, full_matrices=False)
    return U[:, :n] * S[:n]


def spectrum(F):
    f = F - F.mean(0)
    lam = np.linalg.svd(f, compute_uv=False) ** 2
    lam = lam / lam.sum()
    fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    i = np.random.default_rng(0).choice(len(fn), size=(20000, 2))
    i = i[i[:, 0] != i[:, 1]]
    return dict(pc1=float(lam[0]), top10=float(lam[:10].sum()), top30=float(lam[:30].sum()),
                eff_dim=float(1 / (lam ** 2).sum()), mean_cos=float((fn[i[:, 0]] * fn[i[:, 1]]).sum(1).mean()))


def final_with(P, sbj, V, Vg, F, reduce):
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = reduce(F[s])
        f = f - f.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G = np.concatenate([f, 0.5 * np.sqrt(Vg[s])], 1)
        Q[m] = multistage_at(P[m], G, V[m], [4.0, 4.0], k=30, alpha=0.75, iters=5,
                             temp=1.4, g=0.2, where="every")
    return Q


def video_probs(y, sbj, seed):
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=seed)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        d = load_vgraph_oof(tg, TAGS[0], seed=seed)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = d[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    return V, Vg


def main(a):
    res = {"prior_offset": {}, "anisotropy": {}, "spectrum": {}}
    acc = {}
    for sd in a.seeds:
        for off in ["head", "sync"]:
            out, y, sbj = build(sd, video_offset=off)
            for k, Q in out.items():
                if off == "sync" and k == "base":
                    continue                       # does not use video
                for per in [False, True]:
                    acc.setdefault(f"{k}|{off}|{'per_sbj' if per else 'pooled'}", []).append(best_f1(Q, y, sbj, per))
        P, y, sbj, F = load_eval_set(TAGS, W, seed=sd)
        V, Vg = video_probs(y, sbj, sd)
        for nm, red in [("m0", lambda f: f), ("m30", lambda f: drop_top_pcs(f, 30)),
                        ("whiten128", lambda f: whiten(f, 128)), ("whiten256", lambda f: whiten(f, 256)),
                        ("pca128", lambda f: truncate(f, 128)), ("pca256", lambda f: truncate(f, 256))]:
            acc.setdefault(f"aniso|{nm}", []).append(best_f1(final_with(P, sbj, V, Vg, F, red), y, sbj, False))
        if sd == a.seeds[0]:
            sp = [spectrum(F[s]) for s in np.unique(sbj)]
            res["spectrum"] = {k: float(np.mean([d[k] for d in sp])) for k in sp[0]}
        print(f"seed {sd} done", flush=True)
    for k, v in acc.items():
        grp = "anisotropy" if k.startswith("aniso|") else "prior_offset"
        res[grp][k.split("|", 1)[1] if grp == "anisotropy" else k] = {"mean": float(np.mean(v)), "seeds": [float(x) for x in v]}
    print(json.dumps(res, indent=1))
    d = Path(__file__).parents[2] / "paper" / "figdata"
    (d / "review_sensitivity.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123])
    main(p.parse_args())
