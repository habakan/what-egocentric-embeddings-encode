"""Diagnosis of hard participants: is it the inertial model (base) or the video graph that breaks down?

Per participant (mean of seeds 42 and 7):
  base F1 (P) / F1 after propagation (final configuration) / graph purity (fraction of first-stage video-graph neighbours with the same true label) /
  segment upper bound (F1 of P averaged over each ground-truth segment = upper bound for a perfect graph) / null fraction /
  most-confused class pairs (after propagation)
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES, WORK
from eval_pipeline import load_eval_set
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_invariance import load_all
from probe_prf import TAGS, W, build
from final_submit import load_vgraph_oof
from refine_gpu import _t, drop_top_pcs, knn_graph

meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
rows = {}
conf = {}
for sd in [42, 7]:
    out, y, sbj = build(sd); P, Q = out["base"], out["final"]
    _, _, _, _, sens, start, _, _, tile = load_all(TAGS, W, "exp013_nn_video_5fold", seed=sd)
    rec = meta["rec"].to_numpy()[tile]
    _, _, _, F0 = load_eval_set(TAGS, W, seed=sd)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=sd)
    tb = tune_tau(P, y)[1]; tq = tune_tau(Q, y)[1]
    pb, pq = prior_correct(P, tb).argmax(1), prior_correct(Q, tq).argmax(1)
    # segment upper bound: average P over runs of the same label within each recording
    order = np.lexsort((start, rec))
    seg = np.zeros(len(y), int)
    rs, ys_ = rec[order], y[order]
    brk = np.r_[True, (rs[1:] != rs[:-1]) | (ys_[1:] != ys_[:-1])]
    seg[order] = np.cumsum(brk)
    Pb = np.zeros_like(P)
    for g in np.unique(seg):
        m = seg == g; Pb[m] = P[m].mean(0)
    to = tune_tau(Pb, y)[1]; po = prior_correct(Pb, to).argmax(1)
    for s in np.unique(sbj):
        m = sbj == s
        f = drop_top_pcs(_t(F0[s]), 30); f = f - f.mean(0); f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s]))], 1); G = G - G.mean(0)
        nn, w = knn_graph(G, 30)
        yy = torch.tensor(y[m].astype(np.int64), device=G.device)
        keep = w > -1e8
        pur = ((yy[nn] == yy[:, None]) & keep).sum().item() / keep.sum().item()
        r = rows.setdefault(s, {"base": [], "final": [], "purity": [], "oracle": [], "null": [], "n": m.sum()})
        r["base"].append(macro_f1(y[m], pb[m])); r["final"].append(macro_f1(y[m], pq[m]))
        r["purity"].append(pur); r["oracle"].append(macro_f1(y[m], po[m])); r["null"].append((y[m] == 0).mean())
        for a, b in zip(y[m], pq[m]):
            if a != b:
                conf.setdefault(s, {}).setdefault((a, b), 0)
                conf[s][(a, b)] += 1
    print(f"  seed {sd} done", flush=True)
tab = pd.DataFrame([{"sbj": f"sbj_{s}", "n": r["n"], "null": np.mean(r["null"]), "base": np.mean(r["base"]),
                     "final": np.mean(r["final"]), "graph_purity": np.mean(r["purity"]),
                     "bout_oracle": np.mean(r["oracle"])} for s, r in rows.items()]).sort_values("final")
pd.set_option("display.width", 200)
print(tab.round(3).to_string(index=False))
print("\nCorrelation (across participants): post-propagation F1 vs base %.2f / graph purity %.2f / segment upper bound %.2f / null fraction %.2f" % (
    np.corrcoef(tab.final, tab.base)[0, 1], np.corrcoef(tab.final, tab.graph_purity)[0, 1],
    np.corrcoef(tab.final, tab.bout_oracle)[0, 1], np.corrcoef(tab.final, tab["null"])[0, 1]))
print("\nMost-confused pairs for the 4 hard participants (true -> predicted, summed over 2 seeds):")
for s in [int(x.split("_")[1]) for x in tab.sbj[:4]]:
    top = sorted(conf[s].items(), key=lambda kv: -kv[1])[:4]
    print(f"  sbj_{s}: " + "; ".join(f"{CLASS_NAMES[a]} -> {CLASS_NAMES[b]} {c}" for (a, b), c in top))
