"""Reproduces Appendix A: "the transductive stages take about 1 second per participant (2,000--5,000 clips) on an 8-core desktop CPU without a GPU".

Measures the runtime of the final system's per-participant processing (removal of the top 30 components, graph features, 3-stage propagation) with the CPU version (refine.multistage_at).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES
from eval_pipeline import load_eval_set
from final_submit import load_vgraph_oof
from probe_prf import TAGS, W
from refine import drop_top_pcs, multistage_at


def main():
    P, y, sbj, F0 = load_eval_set(TAGS, W, seed=42)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=42)
    V = np.full_like(P, 1.0 / N_CLASSES)
    rows = []
    for rep, s in [(0, np.unique(sbj)[0])] + [(1, s) for s in np.unique(sbj)]:   # the first run is a warm-up and is discarded
        m = sbj == s
        t0 = time.perf_counter()
        f = drop_top_pcs(F0[s], 30); f = f - f.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G = np.concatenate([f, 0.5 * np.sqrt(Vg[s])], 1)
        multistage_at(P[m], G, V[m], [4.0, 4.0], k=30, alpha=0.75, iters=5, temp=1.4, g=0.2, where="every")
        if rep:
            rows.append((int(m.sum()), time.perf_counter() - t0))
    n, sec = np.array(rows).T
    print(f"CPU cores {os.cpu_count()}, participants {len(n)}, clips {int(n.min())}--{int(n.max())}")
    print(f"Runtime: median {np.median(sec):.2f} s, max {sec.max():.2f} s")


if __name__ == "__main__":
    main()
