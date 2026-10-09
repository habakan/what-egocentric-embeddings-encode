import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import sys, collections; sys.path.insert(0, "src")
import numpy as np
from pathlib import Path
ROOT = Path(DATA_DIR + "/egoexo4d"); GC = Path(DATA_DIR + "/geocalib/ego")
def load(d):
    by = collections.defaultdict(list)
    for f in sorted((ROOT / d).glob("*.npz")):
        z = np.load(f, allow_pickle=True); g = np.load(GC / f.name)
        n = min(len(z["ok"]), len(g["rp"])); ok = z["ok"][:n]
        by[int(z["participant"])].append((f.stem, z["F"][:n].mean(1)[ok], z["pose"][:n][ok, :3], np.degrees(g["rp"][:n][ok, 1]), np.where(ok)[0]))
    return by
def within(by, p):
    v = [a for a in by[p] if len(a[1]) >= 40]
    F = np.concatenate([a[1] - a[1].mean(0) for a in v]); G = np.concatenate([a[2] - a[2].mean(0) for a in v])
    pit = np.concatenate([a[3] for a in v]); take = np.concatenate([[a[0]] * len(a[1]) for a in v]); win = np.concatenate([a[4] for a in v])
    _, _, Vt = np.linalg.svd(F, full_matrices=False); Z = F @ Vt[:2].T
    _, _, Vg = np.linalg.svd(G, full_matrices=False); tilt = G @ Vg[0]
    if np.corrcoef(tilt, pit)[0, 1] < 0: tilt = -tilt
    tilt = np.degrees(np.arcsin(np.clip(tilt, -1, 1)))
    return Z, tilt, take, win
if __name__ == "__main__":
    ego, exo = load("features"), load("features_exo_videomae")
    for p in ego:
        out = []
        for by in (ego, exo):
            Z, tilt, _, _ = within(by, p)
            X = np.column_stack([Z, np.ones(len(Z))]); b, *_ = np.linalg.lstsq(X, tilt, rcond=None)
            out.append(1 - ((tilt - X @ b) ** 2).sum() / ((tilt - tilt.mean()) ** 2).sum())
        print(f"participant {p}: centred within take  head tilt ~ PC1+PC2  ego R^2 {out[0]:.3f}  exo {out[1]:.3f}")
