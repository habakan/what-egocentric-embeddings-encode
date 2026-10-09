"""Band on the within-pair share of variants (09-30, 3 seeds).

In the training recordings, the complex share within each variant pair (push-ups / sit-ups / lunges and their complex) stays within 0.48-0.58 (5-95%)
(same experimental design, so each activity lasts about the same time). exp027's largest error pair is push-ups(complex) -> push-ups, 471 windows,
spread over 11 of 24 recordings with 20 or more windows each (sbj_10: plain 146 / complex 94).
The count band [78,126] allows this imbalance. Put the within-pair share into a band:
  per recording and per pair, add bias +d to complex and -d to plain, iterating until the argmax complex share falls within [lo, hi].
Where: after each stage, like the count band (right after the count band). The system is exp027+λ2.
kill: rejected unless it beats exp027+λ2 (tau only) on all 3 seeds.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import probe_band_inloop as pb
from probe_band_inloop import band_bias, load, run, score
from refine_gpu import _t

PAIRS = [(11, 12), (13, 14), (16, 17)]       # (plain, complex)


def pair_bias(L, lo, hi, step=0.05, iters=200):
    b = np.zeros(L.shape[1])
    for a, c in PAIRS:
        d = 0.0
        for _ in range(iters):
            p = (L + b).argmax(1)
            na, nc = (p == a).sum(), (p == c).sum()
            if na + nc == 0:
                break
            sh = nc / (na + nc)
            if lo <= sh <= hi:
                break
            d += step if sh < lo else -step
            b[a], b[c] = -d, d
    return b


def make_apply(lo, hi, beta_pair):
    def apply_band(Q, rec, beta):
        L = torch.log(torch.clamp(Q, min=1e-9)).cpu().numpy()
        B = np.zeros_like(L)
        for r in np.unique(rec):
            m = rec == r
            B[m] = beta * band_bias(L[m])
            if beta_pair > 0:
                B[m] += beta_pair * pair_bias(L[m] + B[m], lo, hi)
        R = Q * torch.exp(_t(B, Q.dtype))
        return R / R.sum(1, keepdim=True)
    return apply_band


def main():
    orig = pb.apply_band
    cfg = {"exp027+λ2": (0.45, 0.58, 0.0)}
    for lo, hi in [(0.45, 0.58), (0.47, 0.55)]:
        for bp in [0.5, 1.0]:
            cfg[f"pair [{lo},{hi}] β{bp}"] = (lo, hi, bp)
    res = {}
    for sd in [42, 7, 1337]:
        P, y, sbj, F0, Vg, V, rec = load(sd)
        for nm, (lo, hi, bp) in cfg.items():
            pb.apply_band = make_apply(lo, hi, bp) if bp > 0 else orig
            res.setdefault(nm, []).append(score(run(P, sbj, F0, Vg, V, rec, 0.5, (0, 1, 2), stages=(2.0, 2.0)), y, rec))
        pb.apply_band = orig
        print(f"  seed {sd} done", flush=True)
    ref = np.array([v[0] for v in res["exp027+λ2"]])
    for nm, v in res.items():
        v = np.array(v); d = v[:, 0] - ref
        print(f"  {nm:22s} tau {v[:, 0].mean():.4f} +band {v[:, 1].mean():.4f}  vs exp027+λ2 {d.mean():+.4f} ["
              + " ".join(f"{x:+.4f}" for x in d) + "]")


if __name__ == "__main__":
    main()
