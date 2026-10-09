"""Evaluation following the WEAR paper's conventions: precision / recall / F1, confusion matrix, and timeline comparison.

The WEAR paper (Table 2) reports record-based P/R/F1 and segment-based mAP@tIoU.
mAP needs segments, so it cannot be computed under this protocol (no ordering). P/R/F1 can be computed.
The confusion matrix of Fig 4 and the colour-coded timeline comparison of Fig 5 are also produced in the same form.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from config import CLASS_NAMES, N_CLASSES, SEED, WORK
from eval_pipeline import load_eval_set, run
from final_submit import load_vgraph_oof
from postprocess import gate_blend, prior_correct, tune_tau
from refine import drop_top_pcs, multistage_at

PREP = WORK / "prep"
TAGS = ["exp001_lgb_inertial", "exp014_nn_inertial_rot", "exp015_nn_inertial_s1337",
        "exp016_nn_inertial_rot30", "exp017_nn_aux", "exp018_nn_aux_s99",
        "exp019_nn_aux_s555"]
W = [1.5, 1, 1, 1, 1, 1, 1]


def prf(y, p):
    lab = list(range(N_CLASSES))
    return dict(
        P=float(precision_score(y, p, labels=lab, average="macro", zero_division=0)),
        R=float(recall_score(y, p, labels=lab, average="macro", zero_division=0)),
        F1=float(f1_score(y, p, labels=lab, average="macro", zero_division=0)),
        F1_null=float(f1_score(y, p, labels=[0], average="macro", zero_division=0)),
        F1_act=float(f1_score(y, p, labels=lab[1:], average="macro", zero_division=0)),
    )


def build(seed, video_offset="head"):
    P, y, sbj, F0 = load_eval_set(TAGS, W, seed=seed, video_offset=video_offset)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=seed)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        d = load_vgraph_oof(tg, TAGS[0], seed=seed)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = d[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)

    out = {}
    out["base"] = P
    # + refinement (no video injection, no PC removal)
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = F0[s] - F0[s].mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G = np.concatenate([f, 0.5 * np.sqrt(Vg[s])], 1)
        Q[m] = multistage_at(P[m], G, V[m], [4.0, 4.0], k=30, alpha=0.75, iters=5,
                             temp=1.4, g=0.0, where="none")
    out["refine"] = Q
    # + video injection
    Q2 = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = F0[s] - F0[s].mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G = np.concatenate([f, 0.5 * np.sqrt(Vg[s])], 1)
        Q2[m] = multistage_at(P[m], G, V[m], [4.0, 4.0], k=30, alpha=0.75, iters=5,
                              temp=1.4, g=0.2, where="every")
    out["inject"] = Q2
    # + PC removal (final configuration)
    Q3 = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = drop_top_pcs(F0[s], 30)
        f = f - f.mean(0); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G = np.concatenate([f, 0.5 * np.sqrt(Vg[s])], 1)
        Q3[m] = multistage_at(P[m], G, V[m], [4.0, 4.0], k=30, alpha=0.75, iters=5,
                              temp=1.4, g=0.2, where="every")
    out["final"] = Q3
    return out, y, sbj


def main(a):
    acc = {}
    preds0 = None
    for sd in a.seeds:
        out, y, sbj = build(sd)
        for k, Q in out.items():
            _, tau = tune_tau(Q, y)
            p = prior_correct(Q / Q.sum(1, keepdims=True), tau).argmax(1)
            for m, v in prf(y, p).items():
                acc.setdefault((k, m), []).append(v)
            if sd == a.seeds[0]:
                preds0 = preds0 or {}
                preds0[k] = p
        if sd == a.seeds[0]:
            y0, sbj0 = y, sbj
        print(f"  seed {sd} done", flush=True)

    names = {"base": "Inertial ensemble (no refinement)",
             "refine": "+ transductive refinement",
             "inject": "+ video injection",
             "final": "+ top-30 PC removal (final)"}
    print(f"\n=== WEAR-style record-based evaluation ({len(a.seeds)} seeds) ===")
    hdr = f"  {'Configuration':34s}{'P':>8s}{'R':>8s}{'F1':>8s}{'F1 null':>9s}{'F1 act.':>9s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = {}
    for k in ["base", "refine", "inject", "final"]:
        r = {m: float(np.mean(acc[(k, m)])) for m in ["P", "R", "F1", "F1_null", "F1_act"]}
        rows[k] = r
        print(f"  {names[k]:34s}{r['P']:8.3f}{r['R']:8.3f}{r['F1']:8.3f}"
              f"{r['F1_null']:9.3f}{r['F1_act']:9.3f}")

    d = Path(__file__).parents[2] / "paper" / "figdata"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prf.json").write_text(json.dumps({"rows": rows, "names": names,
                                            "seeds": a.seeds}, indent=1))
    # Raw data for the confusion matrix and the timeline figure
    np.savez_compressed(d / "preds.npz", y=y0, sbj=sbj0,
                        **{k: v for k, v in preds0.items()})
    print(f"\n  -> {d/'prf.json'}, {d/'preds.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 7, 123])
    main(p.parse_args())
