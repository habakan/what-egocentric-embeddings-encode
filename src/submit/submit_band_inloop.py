"""Submission of the paper's system with the count band inserted between stages (probe_band_inloop.py beta=0.5, after each stage).

Each test participant has one recording (total activity windows before the band 1556-1985, within the range of one training recording), so the band is applied with recording = participant.
The prior-correction tau is set with tune_tau on Q of the OOF (seed 42) under the same settings. The final band is not applied (slightly negative on OOF).
First check that beta=0 gives the same labels as the exp015 submission.

  uv run python src/submit_band_inloop.py                          # exp027 (λ4)
  uv run python src/submit_band_inloop.py --lam 2 --pair --out exp028_band_inloop_pair   # + λ2 + band on the variant-pair ratio
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import TEST_DIR, WORK
from final_submit import vsource_test
from postprocess import prior_correct, tune_tau
import probe_band_inloop as pb
from probe_band_inloop import load, run, system
from probe_countband import band_calibrate
from probe_prf import TAGS, W
from refine_gpu import _t, drop_top_pcs

ROOT = Path(__file__).resolve().parents[2]
BETA, AFTER = 0.5, (0, 1, 2)


def test_inputs():
    te = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    Pt = None
    for t, ww in zip(TAGS, W):
        p = np.load(WORK / t / "test_pred.npy"); p = p / p.sum(1, keepdims=True)
        Pt = ww * p if Pt is None else Pt + ww * p
    Pt /= sum(W)
    Vt = vsource_test(["exp020_video_only", "exp022_vonly_center_5f"])
    vg = np.load(WORK / "exp013_nn_video_5fold" / "test_pred.npy"); vg = vg / vg.sum(1, keepdims=True)
    return te, Pt, Vt, vg


def test_system(te, Pt, Vt, vg, beta, after, **kw):
    video = np.load(WORK / "prep" / "video_raw_test.npy", mmap_mode="r")
    sbj = te["sbj_id"].to_numpy()
    Qt = np.empty_like(Pt)
    for s in np.unique(sbj):
        m = sbj == s
        f = drop_top_pcs(_t(np.asarray(video[np.where(m)[0]], np.float32).mean(1)), 30)
        f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
        G = torch.cat([f, 0.5 * torch.sqrt(_t(vg[m]))], 1)
        Qt[m] = system(_t(Pt[m]), G, _t(Vt[m]), sbj[m], beta, after, **kw).cpu().numpy()
    return Qt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=4.0)
    ap.add_argument("--pair", action="store_true", help="band on the variant-pair ratio [0.45,0.58] β0.5 (probe_family_balance)")
    ap.add_argument("--out", default="exp027_band_inloop")
    a = ap.parse_args()
    kw = dict(stages=(a.lam, a.lam))
    te, Pt, Vt, vg = test_inputs()
    sbj = te["sbj_id"].to_numpy()
    def old(name):                                       # compare with earlier local submissions (not in the public repository) only when present
        f = ROOT / "submissions" / f"{name}.csv"
        return pd.read_csv(f)["target_feature"].to_numpy() if f.exists() else None
    ref, b24, b27 = old("exp015_graphpc30"), old("exp024_graph_band"), old("exp027_band_inloop")

    P, y, sb, F0, Vg, V, rec = load(42)
    Q0 = run(P, sb, F0, Vg, V, rec, 0.0, ())
    _, tau0 = tune_tau(Q0, y)
    p0 = prior_correct(test_system(te, Pt, Vt, vg, 0.0, ()), tau0).argmax(1)
    if ref is not None:
        print(f"agreement of beta=0 with the exp015 submission {(p0 == ref).mean():.4f}")

    if a.pair:
        from probe_family_balance import make_apply
        pb.apply_band = make_apply(0.45, 0.58, 0.5)
    Q = run(P, sb, F0, Vg, V, rec, BETA, AFTER, **kw)
    f1, tau = tune_tau(Q, y)
    print(f"OOF (seed 42) beta {BETA} after {AFTER} λ {a.lam} pair {a.pair}: {f1:.4f} (tau {tau:.2f})")
    Qt = test_system(te, Pt, Vt, vg, BETA, AFTER, **kw)
    pred = prior_correct(Qt, tau).argmax(1)
    (ROOT / "experiments" / "decomp").mkdir(parents=True, exist_ok=True); (ROOT / "submissions").mkdir(exist_ok=True)
    np.savez(ROOT / "experiments" / "decomp" / f"{a.out}_test_probs.npz", Q=Qt)
    pd.DataFrame({"id": te["id"], "target_feature": pred}).to_csv(
        ROOT / "submissions" / f"{a.out}.csv", index=False)
    L = np.log(np.clip(Qt, 1e-9, None)) - tau * np.log(Qt.mean(0) + 1e-9)
    pband = band_calibrate(L, sbj, 78, 126)
    if b27 is not None:
        print(f"agreement vs exp027 {(pred == b27).mean():.4f}")
    print(f"wrote {a.out}.csv: label change if the band were also applied last {(pband != pred).mean():.4f}, null rate {(pred == 0).mean():.4f}")
    if b24 is not None:
        print(f"  agreement vs exp024 {(pred == b24).mean():.4f} (exp024 null rate {(b24 == 0).mean():.4f})")
    for s in np.unique(sbj):
        m = sbj == s
        c = np.bincount(pred[m], minlength=19)
        print(f"  sbj {s}: null {c[0] / m.sum():.3f}  activity-class window counts min {c[1:].min()} max {c[1:].max()}  outside band {((c[1:] < 78) | (c[1:] > 126)).sum()}  "
              f"complex fraction {c[12] / (c[11] + c[12]):.2f}/{c[14] / (c[13] + c[14]):.2f}/{c[17] / (c[16] + c[17]):.2f}")


if __name__ == "__main__":
    main()
