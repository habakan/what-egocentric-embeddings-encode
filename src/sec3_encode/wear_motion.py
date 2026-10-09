"""Measure pixel-level camera shake per window (1 s) from WEAR's public video (analysis only, sbj_0-21).

Read every 6th frame of the 60fps video (10fps) as 160x90 grayscale; from each pair of consecutive frames, get the global translation
(dx, dy) by phase correlation, and the frame difference as the mean absolute difference. From the 9 pairs within second t, build
  [mean|dx|, mean|dy|, std of dx, std of dy (vertical bounce), max displacement, mean frame difference]
Output: $WEAR_DATA/wear_motion/<public id>.npz  M: (n_seconds, 6)
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import subprocess
import sys
from pathlib import Path

import numpy as np

VID = Path(DATA_DIR + "/wear_raw_video"); OUT = Path(DATA_DIR + "/wear_motion"); OUT.mkdir(exist_ok=True)
W, H = 160, 90


def phase_shift(a, b, win):
    Fa = np.fft.rfft2(a * win); Fb = np.fft.rfft2(b * win)
    R = Fa * np.conj(Fb); R /= np.abs(R) + 1e-9
    r = np.fft.irfft2(R, s=a.shape)
    y, x = np.unravel_index(np.argmax(r), r.shape)
    if y > H // 2: y -= H
    if x > W // 2: x -= W
    return x, y


for i in map(int, sys.argv[1:]):
    assert 0 <= i <= 21
    o = OUT / f"sbj_{i}.npz"
    if o.exists():
        continue
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "2", "-i", str(VID / f"sbj_{i}.mp4"),
           "-vf", f"select='not(mod(n\\,6))',scale={W}:{H},format=gray", "-vsync", "0", "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    fr = np.frombuffer(buf, np.uint8).reshape(-1, H, W).astype(np.float32)
    win = np.outer(np.hanning(H), np.hanning(W))
    n = len(fr) // 10
    M = np.zeros((n, 6), np.float32)
    for t in range(n):
        f = fr[10 * t:10 * t + 10]
        sh = np.array([phase_shift(f[k], f[k + 1], win) for k in range(9)], np.float32)
        d = np.abs(np.diff(f, axis=0)).mean((1, 2))
        M[t] = [np.abs(sh[:, 0]).mean(), np.abs(sh[:, 1]).mean(), sh[:, 0].std(), sh[:, 1].std(),
                np.hypot(sh[:, 0], sh[:, 1]).max(), d.mean()]
    np.savez(o, M=M)
    print(f"sbj_{i}: {n} s", flush=True)
