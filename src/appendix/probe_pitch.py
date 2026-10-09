"""Is head pitch (looking up / looking down) inside the top principal components that get removed?

Why this measurement:
  The top 30 PCs are the best part of the representation on four measures (largest variance, highest activity eta^2,
  strongest cross-subject transfer, most stable within a window), and physically they point to **the stable viewpoint layout of the window**;
  that is currently the only positive reading still standing.

  A head IMU would allow a direct check, but the GoPro telemetry (GPMF) has been stripped
  from the distributed videos (only 2 streams: audio + video). So we build a proxy from the pixels.

The v1 proxies (brightness difference, texture difference) had per-recording means scattered from +52.8 to -2.7,
measuring weather and ground type rather than pitch. v2 uses the **sky pixel fraction**:
  sky = pixels that are strongly blue (B > R, B > G), bright, and with little local gradient.
  Even if weather changes the sky brightness, the sky/non-sky decision changes little.
    sky_frac     = fraction of sky pixels (rises when looking up, zero when looking down)
    sky_centroid = vertical centroid of sky pixels (top of frame=1, bottom=0; 0 if no sky)

Old proxies (independent of VideoMAE features):
  looking up shows sky = bright, little texture. Looking down shows ground = dark, dense texture.
    pitch_bright  = mean brightness of top 1/3 - mean brightness of bottom 1/3
    pitch_texture = gradient energy of bottom 1/3 - gradient energy of top 1/3
  Both are oriented to be positive when looking up and negative when looking down.

What is at stake (unlike the previous two):
  visibility and ego-motion were measurements ruling out "is it something else?".
  This one targets **exactly what we are claiming**.
    if it hits  : the fraction in the top-30 subspace far exceeds 0.039 (random) -> the reading is externally confirmed
    if it misses: the only positive claim still standing collapses

  The target is standardised per recording (avoids scale confounding; we hit this with ego-motion).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import WORK

PREP = WORK / "prep"
VURL = "https://ubi29.informatik.uni-siegen.de/wear_dataset/raw/camera/{}.mp4"


def decode_rgb(rec, start, dur, fps, width=320):
    h = int(round(width * 9 / 16 / 2)) * 2
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start),
           "-i", VURL.format(rec), "-t", str(dur),
           "-vf", f"fps={fps},scale={width}:{h}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    n = len(buf) // (width * h * 3)
    return np.frombuffer(buf, np.uint8, n * width * h * 3).reshape(n, h, width, 3)


def sky_proxies(frames):
    """Sky pixels: blue-dominant, bright, little local gradient."""
    f = frames.astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    gray = f.mean(-1)
    gy = np.zeros_like(gray); gy[:, 1:] = np.abs(np.diff(gray, axis=1))
    gx = np.zeros_like(gray); gx[:, :, 1:] = np.abs(np.diff(gray, axis=2))
    grad = gx + gy
    sky = (B >= R) & (B >= G - 5) & (gray > 110) & (grad < 12)
    frac = sky.mean((1, 2))
    h = f.shape[1]
    rows = (1.0 - np.arange(h) / (h - 1))[None, :, None]    # top=1
    cnt = sky.sum((1, 2))
    cen = np.where(cnt > 0, (sky * rows).sum((1, 2)) / np.maximum(cnt, 1), 0.0)
    return frac, cen


def decode_gray(rec, start, dur, fps, width=320):
    h = int(round(width * 9 / 16 / 2)) * 2
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start),
           "-i", VURL.format(rec), "-t", str(dur),
           "-vf", f"fps={fps},scale={width}:{h},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    n = len(buf) // (width * h)
    return np.frombuffer(buf, np.uint8, n * width * h).reshape(n, h, width)


def pitch_proxies(frames):
    f = frames.astype(np.float32)
    h = f.shape[1]
    top, bot = f[:, :h // 3], f[:, 2 * h // 3:]
    bright = top.mean((1, 2)) - bot.mean((1, 2))          # positive when looking up
    gy = np.abs(np.diff(f, axis=1))
    gt, gb = gy[:, :h // 3], gy[:, 2 * h // 3 - 1:]
    tex = gb.mean((1, 2)) - gt.mean((1, 2))               # more positive the more ground is at the bottom
    return bright, tex


def main(a):
    from sklearn.linear_model import Ridge
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    vid = np.load(PREP / "video_raw_head.npy", mmap_mode="r")

    X, sb, ylab = [], [], []
    tg = {"sky fraction": [], "sky vertical centroid": [], "old: brightness (top-bottom)": []}
    for rec in a.recs:
        g = meta[meta["rec"] == rec]
        tmax = g["start"].max() / 50.0
        st = max((tmax - a.dur) / 2, 0)
        fr = decode_rgb(rec, st, a.dur, a.fps)
        sf, sc = sky_proxies(fr)
        br, _ = pitch_proxies(fr.mean(-1))
        tiles = g[g["start"] % 50 == 0]
        sec2row = {int(s_ // 50): r for s_, r in
                   zip(tiles["start"].to_numpy(), tiles.index.to_numpy())}
        secs = st + np.arange(len(br))
        rows = np.array([sec2row.get(int(s_), -1) for s_ in secs])
        ok = rows >= 0
        X.append(np.asarray(vid[rows[ok]], np.float32).mean(1))
        ylab.append(meta["label"].to_numpy()[rows[ok]])
        tg["sky fraction"].append(sf[ok]); tg["sky vertical centroid"].append(sc[ok])
        tg["old: brightness (top-bottom)"].append(br[ok])
        sb.append(np.full(int(ok.sum()), rec))
        print(f"  {rec}: {int(ok.sum())} s  sky fraction {sf[ok].mean():.3f}±{sf[ok].std():.3f}  "
              f"(fraction of seconds with sky {(sf[ok] > 0.01).mean():.2f})", flush=True)

    X = np.concatenate(X); sb = np.concatenate(sb)
    for r in np.unique(sb):
        X[sb == r] -= X[sb == r].mean(0)

    ys_all = np.concatenate(ylab)
    sf_all = np.concatenate(tg["sky fraction"])
    from config import CLASS_NAMES
    print("\n=== Sanity: sky fraction per activity (should be high for looking-up exercises, zero for looking-down) ===")
    order = sorted(np.unique(ys_all), key=lambda c: -sf_all[ys_all == c].mean())
    for c in order[:4] + order[-4:]:
        print(f"    {CLASS_NAMES[c]:28s} {sf_all[ys_all == c].mean():.3f}  (n={int((ys_all==c).sum())})")

    print(f"\n=== Linear prediction of pitch proxies (recording holdout, Ridge) ===")
    print(f"  {'proxy':22s}{'within-rec corr':>12s}{'fraction in top-30 PCs':>22s}")
    for nm, parts in tg.items():
        yv = np.concatenate(parts).astype(np.float64)
        for r in np.unique(sb):
            m = sb == r
            yv[m] = (yv[m] - yv[m].mean()) / (yv[m].std() + 1e-8)
        cors, ws = [], []
        for r in np.unique(sb):
            te = sb == r
            mdl = Ridge(alpha=10.0).fit(X[~te], yv[~te])
            cors.append(float(np.corrcoef(mdl.predict(X[te]), yv[te])[0, 1]))
            ws.append(mdl.coef_)
        w = np.mean(ws, 0); w /= np.linalg.norm(w) + 1e-12
        frac = []
        for r in np.unique(sb):
            m = sb == r
            _, _, Vt = np.linalg.svd(X[m] - X[m].mean(0), full_matrices=False)
            frac.append(float((Vt[:a.m] @ w) @ (Vt[:a.m] @ w)))
        print(f"  {nm:22s}{np.mean(cors):12.4f}{np.mean(frac):22.4f}")
    print(f"\n  A random dense direction gives {a.m/768:.4f}. "
          f"Reference: ego-motion 0.0392 / body visibility 0.025-0.042 (both at random level)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recs", nargs="*", default=["sbj_0", "sbj_3", "sbj_10", "sbj_18"])
    p.add_argument("--dur", type=float, default=400.0)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--m", type=int, default=30)
    main(p.parse_args())
