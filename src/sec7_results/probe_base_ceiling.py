"""VoI probe: if the base classifier gets stronger, how far does the whole pipeline go?

Background: the re-diagnosis showed the main residual cause is "7.7% of segments wrong as a whole = a base classifier problem".
But the ceiling of "strengthening the base" is unmeasured. Distillation (w=0.5) contributed nothing, pipeline 0.8296 vs 0.8297,
but that did not separate whether **the student could not catch up to the teacher**
or **it does not improve even at teacher level**.

So plug a privileged-information oracle directly into the base. teacher01 is a per-window prediction that saw all 4 sensors,
standalone OOF 0.7583 (1-sensor student 0.63). If the pipeline score with this as the base is

  - same as current  -> the ceiling of base strengthening is zero. Change where to invest (kill)
  - clearly higher   -> base strengthening has headroom. The question is how close the student can get

This single run determines whether the **whole family of moves** of base strengthening is worthwhile.

Also produce a segment-level breakdown: can the teacher get right the segments the student gets wrong as a whole?
  yes -> the information is in the other limbs (blind spot of a single sensor)
  no  -> 1 s windows of inertial data contain no information at all (base strengthening cannot recover it)
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
from config import SEED, WORK
from eval_pipeline import run
from final_submit import load_vgraph_oof
from metric import macro_f1, per_class_f1
from postprocess import prior_correct, tune_tau

PREP = WORK / "prep"


def load_with_tile(tags, weights, seed=SEED):
    """Same selection as load_eval_set, but also returns the window index (tile). Needed to look up teacher rows."""
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile))
    ok = valid[tile, sens]
    tile, sens = tile[ok], sens[ok]
    rows = tile + sens * n

    P = None
    for t, w in zip(tags, weights):
        o = np.load(WORK / t / "oof.npy")[rows]
        s = o.sum(1, keepdims=True)
        o = np.divide(o, s, out=np.zeros_like(o), where=s > 0)
        P = w * o if P is None else P + w * o
    P /= sum(weights)
    keep = P.sum(1) > 1e-6
    tile, P = tile[keep], P[keep]
    P /= P.sum(1, keepdims=True)
    y = meta["label"].to_numpy()[tile]
    sbj = meta["sbj_id"].to_numpy()[tile]
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    F = {s: np.asarray(video[tile[sbj == s]], np.float32).mean(1) for s in np.unique(sbj)}
    return P, y, sbj, F, tile, meta


def pipeline(P, y, sbj, F, stages, k, alpha, iters, temp):
    Q = run(P, y, sbj, F, stages, k, alpha, iters, temp=temp)
    f, tau = tune_tau(Q, y)
    return f, tau, prior_correct(Q, tau).argmax(1)


def seg_table(meta, tile, y, pred_map):
    """For each ground-truth segment, return the most frequent label of each prediction series."""
    m = meta.iloc[tile].reset_index(drop=True)
    seg = (m["label"].ne(m["label"].shift()) | m["rec"].ne(m["rec"].shift())).cumsum().to_numpy()
    df = pd.DataFrame({"seg": seg, "y": y, **pred_map})
    rows = []
    for sg, g in df.groupby("seg"):
        if len(g) < 5:
            continue
        r = {"seg": sg, "true": g["y"].iloc[0], "n": len(g)}
        for name in pred_map:
            r[name] = g[name].value_counts().index[0]
        rows.append(r)
    return pd.DataFrame(rows)


def main(a):
    w = a.weights or [1.0] * len(a.tags)
    P, y, sbj, F, tile, meta = load_with_tile(a.tags, w)
    if a.vgraph:
        Vs = load_vgraph_oof(a.vgraph, a.tags[0])
        for s in F:
            f = F[s] - F[s].mean(0)
            f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
            F[s] = np.concatenate([f, a.gamma * np.sqrt(Vs[s])], 1)

    T = np.load(WORK / a.teacher / "oof_win.npy")[tile].astype(np.float32)
    s = T.sum(1, keepdims=True)
    T = np.divide(T, s, out=np.full_like(T, 1 / 19), where=s > 1e-9)

    print(f"n={len(y)}  tags={a.tags}  teacher={a.teacher}\n")
    print("Base alone (before refinement, no tau correction):")
    print(f"  student (1 sensor) : {macro_f1(y, P.argmax(1)):.4f}")
    print(f"  teacher (4 sensors): {macro_f1(y, T.argmax(1)):.4f}")

    # Control: make the student a "4-sensor average". Standalone accuracy rises (ensemble), but
    # like the teacher it becomes **sensor-independent** = the decorrelation of errors across sensors within a segment disappears.
    # If this also drops, the culprit is not the teacher's content but "being sensor-independent" itself.
    n_all = len(meta)
    A = np.zeros_like(P)
    cnt = np.zeros((len(P), 1), np.float32)
    for si in range(4):
        o = np.load(WORK / a.tags[0] / "oof.npy")[tile + si * n_all]
        for t, ww in zip(a.tags[1:], w[1:]):
            o = o + ww * np.load(WORK / t / "oof.npy")[tile + si * n_all]
        s2 = o.sum(1, keepdims=True)
        m2 = (s2 > 1e-6).astype(np.float32)
        A += np.divide(o, s2, out=np.zeros_like(o), where=s2 > 1e-6)
        cnt += m2
    A = np.divide(A, cnt, out=P.copy(), where=cnt > 0)
    A /= A.sum(1, keepdims=True)

    # Dose-response: per window, draw k sensors and average. k=1 is current, k=4 is the control above.
    # If it rises monotonically, what helps is "averaging across sensors", not standalone accuracy.
    S4 = np.zeros((4, len(P), 19), np.float32)
    V4 = np.zeros((4, len(P)), bool)
    for si in range(4):
        o = None
        for t, ww in zip(a.tags, w):
            oo = np.load(WORK / t / "oof.npy")[tile + si * n_all]
            s2 = oo.sum(1, keepdims=True)
            oo = np.divide(oo, s2, out=np.zeros_like(oo), where=s2 > 1e-6)
            o = ww * oo if o is None else o + ww * oo
        V4[si] = o.sum(1) > 1e-6
        S4[si] = o / sum(w)

    def avg_k(kk, seed=0):
        rng = np.random.RandomState(seed)
        out = np.zeros_like(P)
        for i in range(len(P)):
            av = np.where(V4[:, i])[0]
            pick = rng.choice(av, min(kk, len(av)), replace=False)
            out[i] = S4[pick, i].mean(0)
        s2 = out.sum(1, keepdims=True)
        return np.divide(out, s2, out=P.copy(), where=s2 > 1e-6)

    kw = dict(stages=a.stages, k=a.k, alpha=a.alpha, iters=a.iters, temp=a.temp)
    res = {}
    print(f"  student 4-sensor avg : {macro_f1(y, A.argmax(1)):.4f}  <- control (sensor-independent)")
    # Control that softens the teacher's probabilities: only the teacher had an outlier tau of 0.30, so suspect overconfidence
    Tsoft = np.power(T, 1 / 3.0)
    Tsoft /= Tsoft.sum(1, keepdims=True)
    cands = [("student (current)", P), ("teacher (oracle)", T), ("teacher softened T=3", Tsoft),
             ("student 4-sensor avg", A), ("student:teacher = 1:1", (P + T) / 2)]
    cands += [(f"student {kk}-sensor avg", avg_k(kk)) for kk in (2, 3)]
    for name, B in cands:
        f, tau, pred = pipeline(B, y, sbj, F, **kw)
        res[name] = (f, pred)
        print(f"  {name:16s} -> pipeline {f:.4f} (tau={tau:.2f})")

    print("\n=== Segment-level breakdown ===")
    r = seg_table(meta, tile, y, {n: p for n, (_, p) in res.items()})
    ok_s = r["student (current)"] == r["true"]
    ok_t = r["teacher (oracle)"] == r["true"]
    print(f"  segments (>=5 windows) = {len(r)}")
    print(f"  segments student gets right = {ok_s.mean():.3f}   segments teacher gets right = {ok_t.mean():.3f}")
    bad = r[~ok_s]
    print(f"  segments student gets wrong as a whole = {len(bad)} ({len(bad)/len(r):.3f})")
    print(f"    of which teacher gets right = {(bad['teacher (oracle)'] == bad['true']).mean():.3f}"
          f"  <- higher means 'information is in the other limbs'")
    print(f"    of which teacher also wrong = {(bad['teacher (oracle)'] != bad['true']).mean():.3f}"
          f"  <- higher means '1 s inertial windows contain no information'")

    print("\n  per class (segments student gets wrong / fraction teacher gets right):")
    for c in range(19):
        bc = bad[bad["true"] == c]
        if len(bc) == 0:
            continue
        rec = (r["true"] == c).sum()
        print(f"    class {c:2d}: wrong {len(bc):3d}/{rec:3d}  teacher recovers {(bc['teacher (oracle)'] == bc['true']).mean():.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--weights", nargs="*", type=float, default=None)
    p.add_argument("--teacher", default="teacher01")
    p.add_argument("--stages", nargs="*", type=float, default=[4.0, 4.0])
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--temp", type=float, default=1.4)
    p.add_argument("--vgraph", default=None)
    p.add_argument("--gamma", type=float, default=0.5)
    main(p.parse_args())
