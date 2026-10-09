"""Measure how the training-label defects that a public notebook (goodpjw2008, WEAR@HASCA Learned Links + Counts) found from the WEAR
supplementary material (Table 2) affect the paper's numbers.

  sbj_10: 3rd session, from 2566 s to the end (1,419 s) is NULL, but the table lists 9 activities
  sbj_2 : 3rd session, from 3452 s to the end (112 s) is NULL, but the table lists 1 activity
  sbj_7 : in the 2nd session (from 1368 s) the labels lag the activity by 10 s

The paper's evaluation uses the labels as published. Here the same predictions are rescored in two ways: (a) exclude the first two segments from evaluation, (b) additionally shift sbj_7's labels
10 s earlier. Predictions and tau are the same as probe_prf.py (tau chosen on the published labels).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK
from metric import macro_f1
from postprocess import prior_correct, tune_tau
from probe_prf import build

SEEDS = [42, 7, 123]


def tiles(seed):
    """Same tile selection as load_eval_set (asserts the assumption that no row of P is 0)."""
    meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
    valid = np.load(WORK / "prep" / "valid_mask.npy")
    rng = np.random.RandomState(seed)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    sens = rng.randint(0, 4, len(tile))
    tile = tile[valid[tile, sens]]
    return meta["rec"].to_numpy()[tile], meta["start"].to_numpy()[tile] // 50, meta["label"].to_numpy()


def main():
    meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
    lab_at = {(r, s): l for r, s, l in zip(meta["rec"], meta["start"] // 50, meta["label"]) if True}
    res = {}
    for sd in SEEDS:
        out, y, sbj = build(sd)
        rec, sec, _ = tiles(sd)
        assert len(rec) == len(y)
        bad = ((rec == "sbj_10") & (sec >= 2566)) | ((rec == "sbj_2") & (sec >= 3452))
        y7 = y.copy()
        m7 = (rec == "sbj_7") & (sec >= 1368)
        for i in np.where(m7)[0]:
            y7[i] = lab_at.get(("sbj_7", sec[i] + 10), y[i])
        for k, Q in out.items():
            _, tau = tune_tau(Q, y)
            p = prior_correct(Q / Q.sum(1, keepdims=True), tau).argmax(1)
            r = res.setdefault(k, {"released": [], "excl": [], "excl_shift7": []})
            r["released"].append(macro_f1(y, p))
            r["excl"].append(macro_f1(y[~bad], p[~bad]))
            r["excl_shift7"].append(macro_f1(y7[~bad], p[~bad]))
            if k == "final":
                per = {int(s): (macro_f1(y[sbj == s], p[sbj == s]), macro_f1(y[(sbj == s) & ~bad], p[(sbj == s) & ~bad]))
                       for s in np.unique(sbj)}
                res.setdefault("per_sbj", []).append(per)
                res.setdefault("bad_pred_activity", []).append(float((p[bad] != 0).mean()))
        res.setdefault("n_bad", int(bad.sum()))
        print(f"seed {sd} done", flush=True)
    summ = {k: {m: float(np.mean(v)) for m, v in res[k].items()} for k in ["base", "refine", "inject", "final"]}
    per = {s: [float(np.mean([d[s][i] for d in res["per_sbj"]])) for i in (0, 1)] for s in res["per_sbj"][0]}
    summ["per_sbj_final"] = per
    summ["range_released"] = [min(v[0] for v in per.values()), max(v[0] for v in per.values())]
    summ["range_excl"] = [min(v[1] for v in per.values()), max(v[1] for v in per.values())]
    summ["bad_pred_activity"] = float(np.mean(res["bad_pred_activity"]))
    summ["n_bad"] = res["n_bad"]
    print(json.dumps(summ, indent=1))
    (Path(__file__).parents[2] / "paper" / "figdata" / "label_issues.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
