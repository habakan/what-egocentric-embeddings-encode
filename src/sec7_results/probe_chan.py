"""Can the top principal components be removed as "channels (dimensions)" rather than as a "subspace"?

So far we have only tried drop_top_pcs, i.e. **projecting out the subspace** spanned by the top 30 principal components.
Dropping specific channels out of the 768 dimensions entirely has **never been tried**. The two are different,
and their predictions differ too.

Hypothesis (many channels explain the scene, and they have large variance, so they become the top PCs):
  The head-mounted camera does not see the wearer; most of the frame is ground texture and sky.
  The scene's appearance changes constantly with head orientation, so channels that encode the scene have large variance.
  PCA orders by variance, so these make up the top PCs.
  -> If so, the energy of the top PCs should be **concentrated in a few channels**,
     and dropping just those channels should give the same gain as the removal.

Tests:
  (1) Concentration. Share of the top-30-PC energy carried by the top J channels.
      If it is spread flat over the 768 channels, "specific dimensions explain the scene"
      does not hold at the channel level.
  (2) Agreement across subjects. Do the channel rankings computed per subject agree (Spearman)?
      If not, the scene explanation is subject-specific and channel removal does not transfer.
  (3) Intervention. Drop the top J channels and measure macro-F1. As a control, also drop the "top J channels by variance".

Kill criterion:
  If no J of channel removal reaches the level of subspace removal (m=30),
  conclude that "what works is directions, not channels" and discard the hypothesis.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED
from postprocess import tune_tau
from probe_2x2 import neighbor_stats, run_cell
from probe_invariance import drop_top_pcs, load_all


def pc_channel_energy(F, m):
    """Weights distributing the energy of the top m principal components over channels (eigenvalue x squared loading)."""
    c = F - F.mean(0)
    _, sv, Vt = np.linalg.svd(c, full_matrices=False)
    k = min(m, len(sv))
    return ((sv[:k] ** 2)[:, None] * Vt[:k] ** 2).sum(0)


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    acc, pur, conc, agree = {}, {}, [], []
    for sd in a.seeds:
        P, V, y, sbj, sens, start, Fv, _, _ = load_all(a.tags, w, a.vbase, seed=sd)
        us = np.unique(sbj)

        E = {s: pc_channel_energy(Fv[s], a.m) for s in us}
        Var = {s: Fv[s].var(0) for s in us}

        # (1) concentration
        for s in us:
            e = np.sort(E[s])[::-1]
            conc.append([e[:j].sum() / e.sum() for j in a.drop_chans if j > 0])
        # (2) rank agreement across subjects
        from scipy.stats import spearmanr
        R = np.array([np.argsort(np.argsort(-E[s])) for s in us])
        rs = [spearmanr(R[i], R[j]).statistic
              for i in range(len(us)) for j in range(i + 1, len(us))]
        agree.append(np.mean(rs))

        # (3) intervention
        for nm, rank in [("top PC energy", E), ("top variance (control)", Var)]:
            for j in a.drop_chans:
                if j == 0 and nm != "top PC energy":
                    continue
                F = {}
                for s in us:
                    keep = np.argsort(-rank[s])[j:] if j else slice(None)
                    F[s] = Fv[s][:, keep] if j else Fv[s]
                st = [neighbor_stats(F[s], y[sbj == s], sens[sbj == s], a.k) for s in us]
                pur.setdefault((nm, j), []).append(np.mean([x[0] for x in st]))
                Q = run_cell(P, F, sbj, a.stages, a.k, a.alpha, a.iters, a.temp)
                acc.setdefault((nm, j), []).append(tune_tau(Q, y)[0])
        # reference: subspace removal
        for m in a.ref_pcs:
            F = {s: drop_top_pcs(Fv[s], m) for s in us}
            st = [neighbor_stats(F[s], y[sbj == s], sens[sbj == s], a.k) for s in us]
            pur.setdefault(("subspace removal", m), []).append(np.mean([x[0] for x in st]))
            Q = run_cell(P, F, sbj, a.stages, a.k, a.alpha, a.iters, a.temp)
            acc.setdefault(("subspace removal", m), []).append(tune_tau(Q, y)[0])
        print(f"  seed {sd} done", flush=True)

    js = [j for j in a.drop_chans if j > 0]
    print(f"\n=== (1) Share of the top-{a.m}-PC energy carried by the top J channels ===")
    c = np.array(conc).mean(0)
    for j, v in zip(js, c):
        print(f"  top {j:4d}ch ({j/768*100:4.1f}% of dims) -> {v:.3f}"
              f"   (uniform would be {j/768:.3f})")
    print(f"\n=== (2) Agreement of channel rankings across subjects (mean Spearman) === {np.mean(agree):.3f}")

    print(f"\n=== (3) macro-F1 ===")
    for nm in ["top PC energy", "top variance (control)"]:
        print(f"  {nm}")
        for j in a.drop_chans:
            if (nm, j) not in acc:
                continue
            print(f"    dropped ch {j:4d}  F1 {np.mean(acc[(nm,j)]):.4f}"
                  f"   purity {np.mean(pur[(nm,j)]):.4f}")
    print(f"  subspace removal (reference)")
    for m in a.ref_pcs:
        print(f"    removed PCs {m:4d}  F1 {np.mean(acc[('subspace removal',m)]):.4f}"
              f"   purity {np.mean(pur[('subspace removal',m)]):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float)
    p.add_argument("--vbase", required=True)
    p.add_argument("--m", type=int, default=30, help="number of principal components over which concentration is measured")
    p.add_argument("--drop-chans", nargs="*", type=int, default=[0, 30, 100, 200, 400])
    p.add_argument("--ref-pcs", nargs="*", type=int, default=[30])
    p.add_argument("--seeds", nargs="*", type=int, default=[SEED])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    main(p.parse_args())
