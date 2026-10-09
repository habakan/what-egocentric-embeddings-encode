"""Check on Ego-Exo4D egocentric video whether the top PCs of VideoMAEv2 features represent "head orientation (viewpoint layout)".

WEAR has no ground truth for head pose, and the §5.3 reading "top directions = stable viewpoint layout" relied only on
internal measurements. Ego-Exo4D's Aria glasses have a 6-DoF head trajectory from SLAM, so we use it as external ground truth.

Extraction: per the Kaggle Data tab spec (30 fps, stretched to 224x224, 16-frame clips, the represented frame is
the 8th from the start). For each 1 s window take 3 clips centred at f0+8, f0+15, f0+22 and average them (approximating the 15 test rows).
Head pose: interpolate closed_loop_trajectory at the timesync Aria RGB capture times; per window,
  gravity direction in device coordinates (3; pitch/roll = head tilt), height, rotation speed, translation speed.
Analysis (per participant, within-subject PCA, cross-validation over takes):
  1. R^2 predictable from head pose, per PC band
  2. R^2 of predicting head tilt from [projection onto top 30 / remainder after removing top 30 / random 30 dims]
  3. Fraction of within-window variance (per PC)
Output: $WEAR_DATA/egoexo4d/features/<take>.npz
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", DATA_DIR + "/hf")
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from extract_videomae import DEV, load_model, to_tensor

ROOT = Path(DATA_DIR + "/egoexo4d")
OUT = ROOT / "features"; OUT.mkdir(exist_ok=True)


def decode_squash(src):
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(src), "-vf", "fps=30,scale=224:224",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    n = len(buf) // (224 * 224 * 3)
    return np.frombuffer(buf, np.uint8, n * 224 * 224 * 3).reshape(n, 224, 224, 3)


def quat_to_R(q):                      # q: (n,4) = x,y,z,w
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
                     np.stack([2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)], -1)], 1)


@torch.no_grad()
def extract(model, take, meta):
    out = OUT / f"{take['take_name']}.npz"
    if out.exists():
        return
    d = ROOT / take["root_dir"]
    vid = sorted((d / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4"))[0]
    aria = vid.name.split("_")[0]                          # aria01 / aria02 ...
    frames = decode_squash(vid)
    nwin = (len(frames) - 22 - 8) // 30
    if nwin < 5:
        return
    starts = np.array([30 * w + o for w in range(nwin) for o in (0, 7, 14)])   # clip start (centre = +8)
    x = to_tensor(frames)
    feats = []
    for i in range(0, len(starts), 48):
        b = torch.stack([x[:, s:s + 16] for s in starts[i:i + 48]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            feats.append(model.forward_features(b).float().cpu().numpy())
    F = np.concatenate(feats).reshape(nwin, 3, 768)
    # head pose
    cap = [c for c in (ROOT / "captures").iterdir() if take["take_name"].startswith(c.name)]
    col = f"{aria}_214-1_capture_timestamp_ns"
    ts = pd.read_csv(cap[0] / "timesync.csv", usecols=[col])[col].to_numpy()
    tr = pd.read_csv(d / "trajectory/closed_loop_trajectory.csv")
    t_us = tr["tracking_timestamp_us"].to_numpy()
    fidx = np.array([30 * w + o + 8 for w in range(nwin) for o in (0, 7, 14)])
    t_frame = ts[take["timesync_start_idx"] + fidx] / 1000.0
    ok = np.isfinite(t_frame)
    j = np.clip(np.searchsorted(t_us, t_frame), 0, len(t_us) - 1)
    q = tr[["qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device"]].to_numpy()[j]
    R = quat_to_R(q)
    g = tr[["gravity_x_world", "gravity_y_world", "gravity_z_world"]].to_numpy()[j]
    g_dev = np.einsum("nji,nj->ni", R, g); g_dev /= np.linalg.norm(g_dev, axis=1, keepdims=True) + 1e-9
    h = tr["tz_world_device"].to_numpy()[j]
    av = np.linalg.norm(tr[["angular_velocity_x_device", "angular_velocity_y_device", "angular_velocity_z_device"]].to_numpy()[j], axis=1)
    lv = np.linalg.norm(tr[["device_linear_velocity_x_device", "device_linear_velocity_y_device", "device_linear_velocity_z_device"]].to_numpy()[j], axis=1)
    pose = np.column_stack([g_dev, h, av, lv]).reshape(nwin, 3, 6).mean(1)
    np.savez(out, F=F.astype(np.float32), pose=pose.astype(np.float32), ok=ok.reshape(nwin, 3).all(1),
             participant=meta["participant_uid"], domain=meta["parent_task_name"])


def analyse(feat_dir=None):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    rows = []
    for f in sorted((feat_dir or OUT).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        rows.append((str(z["participant"]), str(z["domain"]), f.stem, z["F"], z["pose"], z["ok"]))
    parts = sorted({(r[0], r[1]) for r in rows})
    bands = [(1, 2), (3, 10), (11, 30), (31, 100)]
    res1 = {b: [] for b in bands}; res1r = []; res2 = {"top 30": [], "excluding top 30": [], "random 30": []}; stab = []
    rng = np.random.RandomState(0)
    for p, dom in parts:
        rs = [r for r in rows if r[0] == p and r[1] == dom]
        F = np.concatenate([r[3] for r in rs]); P = np.concatenate([r[4] for r in rs]); ok = np.concatenate([r[5] for r in rs])
        grp = np.concatenate([[i] * len(r[3]) for i, r in enumerate(rs)])
        F, P, grp = F[ok], P[ok], grp[ok]
        Fm = F.mean(1); Fm = Fm - Fm.mean(0)
        _, S, Vt = np.linalg.svd(Fm, full_matrices=False)
        Z = Fm @ Vt[:100].T
        ng = min(5, len(np.unique(grp)))
        if ng < 2:
            continue
        cv = GroupKFold(ng)
        Pz = (P - P.mean(0)) / (P.std(0) + 1e-9)
        Xp = np.column_stack([Pz, Pz ** 2, Pz[:, :3].prod(1, keepdims=True)])   # pose up to 2nd order
        def cvr2(X, Y):
            pred = np.zeros_like(Y)
            for tr_, te_ in cv.split(X, Y, grp):
                pred[te_] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr_], Y[tr_]).predict(X[te_])
            return 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        r2 = cvr2(Xp, Z)
        for b in bands:
            res1[b].append(r2[b[0] - 1:b[1]].mean())
        Rnd = Fm @ np.linalg.qr(rng.randn(Fm.shape[1], 10))[0]
        res1r.append(cvr2(Xp, Rnd).mean())
        # reverse: head tilt (gravity direction, 3) from video subspaces
        tilt = Pz[:, :3]
        top = Fm @ Vt[:30].T
        rest = Fm - (Fm @ Vt[:30].T) @ Vt[:30]
        rnd = Fm @ np.linalg.qr(rng.randn(Fm.shape[1], 30))[0]
        for nm, X in [("top 30", top), ("excluding top 30", rest), ("random 30", rnd)]:
            res2[nm].append(cvr2(X, tilt).mean())
        # fraction of within-window (3 clips) variance
        Fc = F - F.mean((0, 1))
        proj = np.einsum("nkd,jd->nkj", Fc, Vt[:100])
        within = ((proj - proj.mean(1, keepdims=True)) ** 2).mean((0, 1)); total = (proj ** 2).mean((0, 1))
        stab.append(within / total)
        print(f"  {dom:10s} {p:>5s}: windows {len(Fm)}, takes {len(rs)}", flush=True)
    print("\n=== 1. Predicting video PCs from head pose (take-wise CV R^2, mean over participants) ===")
    for b in bands:
        print(f"  PC{b[0]}-{b[1]}: {np.mean(res1[b]):+.3f}")
    print(f"  random directions (mean of 10): {np.mean(res1r):+.3f}")
    print("\n=== 2. Predicting head tilt (gravity direction) from video subspaces ===")
    for nm, v in res2.items():
        print(f"  {nm:10s}: {np.mean(v):+.3f}")
    st = np.mean(stab, 0)
    print("\n=== 3. Fraction of within-window variance (3 clips) ===")
    print("  PC1 {:.3f}  PC10 {:.3f}  PC30 {:.3f}  PC100 {:.3f}  | 1-30 mean {:.3f}  31-100 mean {:.3f}".format(
        st[0], st[9], st[29], st[99], st[:30].mean(), st[30:].mean()))


if __name__ == "__main__":
    takes = {t["take_uid"]: t for t in json.load(open(ROOT / "takes.json"))}
    uids = (ROOT / "selected_uids.txt").read_text().split()
    if "--analyse-only" not in sys.argv:
        model = load_model()
        for i, u in enumerate(uids):
            t = takes[u]
            if not list((ROOT / t["root_dir"] / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4")):
                continue
            try:
                extract(model, t, t)
            except Exception as e:
                print(f"  skip {t['take_name']}: {e}", flush=True)
            if i % 10 == 9:
                print(f"  {i + 1}/{len(uids)} extracted", flush=True)
    analyse()
