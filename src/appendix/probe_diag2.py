"""Re-diagnosis with the current configuration: ceiling (mean over true segments), and segment purity / mode-label accuracy."""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import SEED, WORK
from eval_pipeline import load_eval_set, run
from final_submit import load_vgraph_oof
from metric import per_class_f1
from postprocess import prior_correct, tune_tau

PREP = WORK / "prep"
TAGS = ["exp001_lgb_inertial", "exp014_nn_inertial_rot", "exp015_nn_inertial_s1337",
        "exp016_nn_inertial_rot30", "exp017_nn_aux", "exp018_nn_aux_s99",
        "exp019_nn_aux_s555"]
W = [1.5, 0.7, 0.7, 0.7, 1.3, 1.3, 1.3]


def main():
    P, y, sbj, F = load_eval_set(TAGS, W)
    Vs = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0])
    G = {}
    for s in F:
        f = F[s] - F[s].mean(0)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        G[s] = np.concatenate([f, 0.5 * np.sqrt(Vs[s])], 1)
    Q = run(P, y, sbj, G, [4.0, 4.0], 30, 0.75, 5, temp=1.4)
    f, tau = tune_tau(Q, y)
    pred = prior_correct(Q, tau).argmax(1)
    print(f"current config macro-F1 = {f:.4f}")

    meta = pd.read_parquet(PREP / "win_meta.parquet"); n = len(meta)
    valid = np.load(PREP / "valid_mask.npy")
    rng = np.random.RandomState(SEED)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile)); ok = valid[tile, sens]
    rows = tile[ok] + sens[ok] * n
    keep = np.load(WORK / TAGS[0] / "oof.npy")[rows].sum(1) > 1e-6
    tl = tile[ok][keep]
    m = meta.iloc[tl].reset_index(drop=True)
    seg = (m["label"].ne(m["label"].shift()) | m["rec"].ne(m["rec"].shift())).cumsum().to_numpy()

    # Ceiling: average probabilities over the true segments
    df = pd.DataFrame(Q); df["g"] = seg
    Qo = df.groupby("g").transform("mean").to_numpy()
    print(f"ceiling (mean over true segments) = {tune_tau(Qo, y)[0]:.4f}")

    d = pd.DataFrame({"seg": seg, "y": y, "pred": pred})
    rows_ = []
    for sg, g in d.groupby("seg"):
        vc = g["pred"].value_counts()
        rows_.append({"true": g["y"].iloc[0], "n": len(g), "mode": vc.index[0],
                      "purity": vc.iloc[0] / len(g)})
    r = pd.DataFrame(rows_); r = r[r["n"] >= 5]
    print(f"segment purity median={r['purity'].median():.3f} mean={r['purity'].mean():.3f}  "
          f"mode label correct={(r['mode'] == r['true']).mean():.3f}  "
          f"split segments (purity<0.5)={(r['purity'] < 0.5).mean():.3f}")
    print("weak classes (lowest mode-label accuracy first):")
    agg = r.groupby("true").apply(
        lambda g: pd.Series({"n_seg": len(g), "purity": g["purity"].mean(),
                             "mode_acc": (g["mode"] == g["true"]).mean()}),
        include_groups=False).sort_values("mode_acc")
    print(agg.head(8).round(3).to_string())
    print("\nper-class F1:", np.round(per_class_f1(y, pred), 2).tolist())


if __name__ == "__main__":
    main()
