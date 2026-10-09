"""Why does removing the top principal components help: does it reduce "neighbours that make the same mistakes as the classifier"? (WEAR, 3 encoders)

Hypothesis H: the top components carry the body state the inertial classifier already sees (72% predictable from the four limbs). Windows close along those directions
  also err like the classifier, so errors remain after averaging. Removing them lowers label purity but raises the independence of errors.

1. Sweep m and list the stage-1 graph metrics next to the final macro-F1.
   purity      : fraction of neighbours whose true label equals the window's own
   signal      : mean probability that the neighbours' predictions P place on the window's true class
   same error  : among windows whose own P is wrong, fraction where the argmax of the neighbour's P is the same (wrong) class as the window's
2. Decisive control: shuffle P among windows of "the same participant and the same true class". Per-class error patterns are kept,
   and only the relation between errors and video is destroyed. If H is right, the removal gain vanishes or reverses.
System: the graph uses features only (no probability block), no video injection (g=0); we look only at the propagation of the classifier's predictions P.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from postprocess import tune_tau
from probe_wear_encoders import IMG, PREP, PUB2REC, eval_tiles
from refine_gpu import _t, drop_top_pcs, knn_graph, multistage_at

MS = [0, 5, 10, 20, 30, 50, 80, 120]


def load(sd, img, video):
    tile, P, meta = eval_tiles(sd)
    y = meta["label"].to_numpy()[tile]; sbj = meta["sbj_id"].to_numpy()[tile]
    rec = meta["rec"].to_numpy()[tile]; sec = meta["start"].to_numpy()[tile] // 50
    use = np.array([r in img["CLIP"] and s < len(img["CLIP"][r]) and s < len(img["DINOv2"][r]) for r, s in zip(rec, sec)])
    tile, P, y, sbj, rec, sec = tile[use], P[use], y[use], sbj[use], rec[use], sec[use]
    F = {"VideoMAEv2": {}, **{e: {} for e in img}}
    for s in np.unique(sbj):
        idx = np.where(sbj == s)[0]
        F["VideoMAEv2"][s] = np.asarray(video[tile[idx]], np.float32).mean(1)
        for e in img:
            F[e][s] = np.stack([img[e][rec[i]][sec[i]] for i in idx])
    return P, y, sbj, F


def shuffle_within_class(P, y, sbj, rng):
    Ps = P.copy()
    for s in np.unique(sbj):
        for c in np.unique(y[sbj == s]):
            ii = np.where((sbj == s) & (y == c))[0]
            Ps[ii] = P[rng.permutation(ii)]
    return Ps


def run(P, y, sbj, F, m):
    Q = np.empty_like(P)
    st = collections.defaultdict(list)
    for s in np.unique(sbj):
        ms = sbj == s
        f = _t(F[s]); f = drop_top_pcs(f, m) if m else f
        f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
        Ps = _t(P[ms])
        Q[ms] = multistage_at(Ps, f, Ps, [4.0, 4.0], k=30, alpha=0.75, iters=5, temp=1.4, g=0.0,
                              where="none").cpu().numpy()
        nn, w = knn_graph(f, 30)
        keep = (w > -1e8).cpu().numpy(); nn = nn.cpu().numpy()
        ys = y[ms]; pa = P[ms].argmax(1); Pm = P[ms]
        cnt = keep.sum(1).clip(1)
        st["purity"].append(((ys[nn] == ys[:, None]) & keep).sum(1) / cnt)
        st["signal"].append((Pm[nn, ys[:, None]] * keep).sum(1) / cnt)
        wrong = pa != ys
        st["same error"].append((((pa[nn] == pa[:, None]) & keep).sum(1) / cnt)[wrong])
    return tune_tau(Q, y)[0], {k: float(np.concatenate(v).mean()) for k, v in st.items()}


def main():
    pubs = sorted(p.stem for p in (IMG / "dinov2").glob("sbj_*.npy") if p.stem in PUB2REC)
    img = {e: {PUB2REC[p]: np.load(IMG / e.lower() / f"{p}.npy").astype(np.float32).mean(1) for p in pubs}
           for e in ["CLIP", "DINOv2"]}
    video = np.load(PREP / "video_raw_head.npy", mmap_mode="r")
    R = collections.defaultdict(list)
    for sd in [42, 7]:
        P, y, sbj, F = load(sd, img, video)
        Pshuf = shuffle_within_class(P, y, sbj, np.random.RandomState(sd))
        R["P alone"].append(tune_tau(P, y)[0])
        for e in F:
            for m in MS:
                f1, st = run(P, y, sbj, F[e], m)
                R[(e, "real", m)].append((f1, st))
                f1s, sts = run(Pshuf, y, sbj, F[e], m)
                R[(e, "shuf", m)].append((f1s, sts))
        print(f"seed {sd} done", flush=True)
    print(f"\nP alone {np.mean(R['P alone']):.4f}")
    for e in ["VideoMAEv2", "CLIP", "DINOv2"]:
        for kind, lab in [("real", "real P"), ("shuf", "P shuffled within class")]:
            print(f"\n== {e} / {lab}")
            print(f"  {'m':>4s} {'macro-F1':>9s} {'diff':>8s} {'purity':>7s} {'signal':>7s} {'same error':>8s}")
            base = np.mean([r[0] for r in R[(e, kind, 0)]])
            for m in MS:
                rs = R[(e, kind, m)]
                f1 = np.mean([r[0] for r in rs])
                st = {k: np.mean([r[1][k] for r in rs]) for k in rs[0][1]}
                print(f"  {m:4d} {f1:9.4f} {f1 - base:+8.4f} {st['purity']:7.3f} {st['signal']:7.3f} {st['same error']:8.3f}")


if __name__ == "__main__":
    main()
