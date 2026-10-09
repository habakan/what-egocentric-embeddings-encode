"""Additional experiments for likely reviewer comments (A and B in paper/review-openreview.md).

A: Participant-level bootstrap confidence intervals for the main differences (propagation, video evidence, component removal).
   Resample participants with replacement and compute macro-F1 over their pooled windows (same as the paper's metric: F1 pooled over all windows).
   95% intervals for the difference of the 3-seed means. Also a paired Wilcoxon test (per-participant F1 differences).
B: When only part of a participant's windows is available at once. Split each participant's windows into n equal parts in recording time order and run the system on each chunk separately
   (component removal and the graph are computed within the chunk only). n = 1 (the paper's setting), 2, 4, 10.
   Splitting into n parts corresponds to "running inference in a batch each time 1/n of the session ends".
tau is chosen once over the whole CV from each configuration's output (same procedure as the paper).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, WORK
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from postprocess import prior_correct, tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from refine import drop_top_pcs, multistage_at

SEEDS = [42, 7, 123]
meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")


def conf_by_sbj(y, p, sbj):
    out = {}
    for s in np.unique(sbj):
        m = sbj == s
        out[s] = np.bincount(y[m].astype(np.int64) * N_CLASSES + p[m].astype(np.int64), minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    return out


def f1_from_conf(C):
    tp = np.diag(C).astype(float); fp = C.sum(0) - tp; fn = C.sum(1) - tp
    d = 2 * tp + fp + fn
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)
    present = (C.sum(1) + C.sum(0)) > 0
    return f[present].mean()


def final_chunked(P, y, sbj, F0, Vg, V, rec, start, n):
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        idx = np.where(sbj == s)[0]
        order = idx[np.lexsort((start[idx], rec[idx]))]
        for part in np.array_split(order, n):
            loc = np.searchsorted(idx, np.sort(part))           # row indices within the participant
            f = drop_top_pcs(F0[s][loc], 30)
            f = f - f.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            G = np.concatenate([f, 0.5 * np.sqrt(Vg[s][loc])], 1)
            pp = np.sort(part)
            Q[pp] = multistage_at(P[pp], G, V[pp], [4.0, 4.0], k=min(30, len(pp) - 1), alpha=0.75, iters=5,
                                  temp=1.4, g=0.2, where="every")
    return Q


def main():
    confs = {}                   # (config, seed) -> {sbj: conf}
    for sd in SEEDS:
        out, y, sbj = build(sd)
        P = out["base"]
        _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
        Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
        src = []
        for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
            d = load_vgraph_oof(tg, TAGS[0], seed=sd)
            M = np.empty((len(y), N_CLASSES), np.float32)
            for s in np.unique(sbj):
                M[sbj == s] = d[s]
            src.append(M)
        V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
        _, _, _, _, _, start, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
        rec = meta["rec"].to_numpy()[tile]
        for n in [2, 4, 10]:
            out[f"chunk{n}"] = final_chunked(P, y, sbj, F0, Vg, V, rec, start, n)
        for k, Q in out.items():
            Q = Q / Q.sum(1, keepdims=True)
            _, tau = tune_tau(Q, y)
            confs[(k, sd)] = conf_by_sbj(y, prior_correct(Q, tau).argmax(1), sbj)
        print(f"seed {sd} done", flush=True)

    cfgs = ["base", "refine", "inject", "final", "chunk2", "chunk4", "chunk10"]
    sbjs = sorted(confs[("base", SEEDS[0])])
    def pooled(k, pick):
        return np.mean([f1_from_conf(sum(confs[(k, sd)][s] for s in pick)) for sd in SEEDS])
    point = {k: pooled(k, sbjs) for k in cfgs}
    per = {k: np.array([np.mean([f1_from_conf(confs[(k, sd)][s]) for sd in SEEDS]) for s in sbjs]) for k in cfgs}
    rng = np.random.default_rng(0)
    B = 2000
    boot = {k: [] for k in cfgs}
    for _ in range(B):
        pick = rng.choice(sbjs, len(sbjs), replace=True)
        for k in cfgs:
            boot[k].append(pooled(k, pick))
    boot = {k: np.array(v) for k, v in boot.items()}

    print(f"\n=== A: resampling {len(sbjs)} participants with replacement ({B} times), mean of 3 seeds, macro-F1 pooled over all windows ===")
    for k in cfgs[:4]:
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        print(f"  {k:8s} {point[k]:.4f}  95% [{lo:.4f}, {hi:.4f}]")
    print("  difference (after - before):")
    for a, b, nm in [("base", "refine", "propagation"), ("refine", "inject", "video evidence"), ("inject", "final", "component removal"),
                     ("base", "final", "overall")]:
        dlt = boot[b] - boot[a]
        lo, hi = np.percentile(dlt, [2.5, 97.5])
        dp = per[b] - per[a]
        pw = wilcoxon(dp).pvalue if np.any(dp != 0) else 1.0
        print(f"  {nm:8s} {point[b] - point[a]:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
              f"improved per participant {int((dp > 0).sum())}/{len(dp)}  Wilcoxon p = {pw:.2g}")

    print("\n=== B: split windows into n parts in time order and run the system per chunk ===")
    for k, n in [("final", 1), ("chunk2", 2), ("chunk4", 4), ("chunk10", 10)]:
        dlt = boot[k] - boot["final"]
        lo, hi = np.percentile(dlt, [2.5, 97.5])
        print(f"  n = {n:2d} (windows usable at once = 1/{n} of the participant): {point[k]:.4f}  "
              f"diff vs paper setting {point[k] - point['final']:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"diff vs inertial only {point[k] - point['base']:+.4f}")


if __name__ == "__main__":
    main()
