"""Do the 106 residual segments really "carry no information"? Re-measure with a video oracle added.

probe_base_ceiling.py found "of the 106 segments the student gets entirely wrong, the 4-sensor teacher also gets 60.4% wrong"
and concluded from that "1-second inertial windows carry no information".

But teacher01 is a **4-sensor inertial model**, and **video is not in the oracle**.
We could only say "the inertial data carries no information" but generalised it to "there is no information"
(when-stuck.md Step 7: did you try the version with the diagnosis's qualifier removed?).

Here we also apply video-side models to the same 106 segments and break them down:

  exp020_video_only   : video only (fully independent of the inertial base)
  exp013_nn_video_5fold : inertial+video
  teacher01           : 4-sensor inertial (privileged information)

If many segments are "missed by the inertial oracle but hit by video", the residual is not an absence of information but
**a problem of allocating modalities**, and moves that drop the qualifier of the independence principle
("harmful when mixed into the base classifier blend") —— for example
a gate that lets video act only on segments where inertial is uncertain —— become candidates.
when-stuck.md: "a signal worthless on its own can become valuable through a gate".
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK
from final_submit import load_vgraph_oof
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_base_ceiling import load_with_tile, seg_table
from eval_pipeline import run

PREP = WORK / "prep"


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    P, y, sbj, F, tile, meta = load_with_tile(a.tags, w)
    Fraw = {s: F[s].copy() for s in F}
    if a.vgraph:
        Vs = load_vgraph_oof(a.vgraph, a.tags[0])
        for s in F:
            f = F[s] - F[s].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            F[s] = np.concatenate([f, a.gamma * np.sqrt(Vs[s])], 1)
    Q = run(P, y, sbj, F, a.stages, a.k, a.alpha, a.iters, temp=a.temp)
    f_tau, tau = tune_tau(Q, y)
    student = prior_correct(Q, tau).argmax(1)
    print(f"pipeline macro-F1 = {f_tau:.4f}  n={len(y)}\n")

    preds = {"student": student}
    # 4-sensor inertial teacher (per window)
    T = np.load(WORK / a.teacher / "oof_win.npy")[tile].astype(np.float32)
    preds["inertial teacher"] = T.argmax(1)
    # video models (rows use the same selection as the student, so use load_vgraph_oof)
    for tag, nm in [(a.video_only, "video only"), (a.video_fusion, "inertial+video")]:
        if not tag:
            continue
        Vd = load_vgraph_oof(tag, a.tags[0])
        V = np.empty((len(y), 19), np.float32)
        for s in np.unique(sbj):
            V[sbj == s] = Vd[s]
        preds[nm] = V.argmax(1)

    print("per-window macro-F1 (reference):")
    for nm, pr in preds.items():
        print(f"  {nm:10s}: {macro_f1(y, pr):.4f}")

    r = seg_table(meta, tile, y, preds)
    ok = {nm: (r[nm] == r["true"]) for nm in preds}
    print(f"\nsegments (5+ windows) = {len(r)}")
    print("fraction of segments whose majority label is correct:")
    for nm in preds:
        print(f"  {nm:10s}: {ok[nm].mean():.3f}")

    bad = r[~ok["student"]]
    print(f"\n=== breakdown of the {len(bad)} segments the student gets entirely wrong ===")
    others = [nm for nm in preds if nm != "student"]
    for nm in others:
        print(f"  {nm:10s} rescues : {(bad[nm] == bad['true']).mean():.3f}")
    any_ok = np.zeros(len(bad), bool)
    for nm in others:
        any_ok |= (bad[nm] == bad["true"]).to_numpy()
    print(f"  {'any':10s} rescues : {any_ok.mean():.3f}   <- only the rest truly carry no information")
    if "inertial teacher" in preds and "video only" in preds:
        t_ok = (bad["inertial teacher"] == bad["true"]).to_numpy()
        v_ok = (bad["video only"] == bad["true"]).to_numpy()
        print(f"\n  only inertial teacher rescues : {(t_ok & ~v_ok).mean():.3f}")
        print(f"  only video only rescues      : {(~t_ok & v_ok).mean():.3f}  <- the part the inertial oracle cannot see")
        print(f"  both rescue                  : {(t_ok & v_ok).mean():.3f}")
        print(f"  neither rescues              : {(~t_ok & ~v_ok).mean():.3f}")

    print("\n  per class (wrong segments / inertial teacher / video only / any):")
    for c in range(19):
        bc = bad[bad["true"] == c]
        if len(bc) == 0:
            continue
        t_ = (bc["inertial teacher"] == bc["true"]).mean()
        v_ = (bc["video only"] == bc["true"]).mean() if "video only" in preds else float("nan")
        a_ = np.zeros(len(bc), bool)
        for nm in others:
            a_ |= (bc[nm] == bc["true"]).to_numpy()
        print(f"    class {c:2d}: n={len(bc):3d}  inertial {t_:.2f}  video {v_:.2f}  any {a_.mean():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--teacher", default="teacher01")
    p.add_argument("--video-only", default="exp020_video_only")
    p.add_argument("--video-fusion", default="exp013_nn_video_5fold")
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--vgraph", default="exp013_nn_video_5fold")
    p.add_argument("--gamma", type=float, default=0.5)
    main(p.parse_args())
