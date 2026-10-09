"""Estimates the camera's gravity direction (= head tilt) from one frame per window with GeoCalib (ECCV 2024).

The aim is to test directly on WEAR (no head sensor) whether the egocentric camera orientation determines the top components of the embedding.
  ego   : Ego-Exo4D Aria video (448, 30fps). Center of window w is 30w+15. Validated against measured gravity.
  wear  : Public videos of the WEAR training participants (60fps). Center of VideoMAE rows 30t..30t+14 of tile t = frame 60t+14 at 60fps.
  exo   : Ego-Exo4D exocentric cameras (for comparison; not used here)
Fisheye/wide-angle, so the distorted weights and the simple_divisional model are used.
Output: $WEAR_DATA/geocalib/<src>/<name>.npz  (g: (N,3) gravity in camera coordinates, rp: (N,2) roll/pitch [rad], unc: (N,2))
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_HOME", DATA_DIR + "/hf/torch")
import numpy as np
import torch

OUT = Path(DATA_DIR + "/geocalib")
EGO = Path(DATA_DIR + "/egoexo4d")
WVID = Path(DATA_DIR + "/wear_raw_video")


def decode(src, sel, vf_scale):
    vf = f"select='{sel}',{vf_scale}"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "8", "-i", str(src), "-vf", vf, "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return buf


def frames_of(buf, h, w):
    n = len(buf) // (h * w * 3)
    return np.frombuffer(buf, np.uint8, n * h * w * 3).reshape(n, h, w, 3)


@torch.no_grad()
def estimate(model, fr, bs=32):
    g, rp, unc = [], [], []
    for i in range(0, len(fr), bs):
        x = torch.from_numpy(fr[i:i + bs]).cuda().permute(0, 3, 1, 2).float() / 255.0
        r = model.calibrate(x, camera_model="simple_divisional")
        grav = r["gravity"]
        g.append(grav.vec3d.cpu().numpy()); rp.append(grav.rp.cpu().numpy())
        u = torch.stack([r["roll_uncertainty"], r["pitch_uncertainty"]], -1) if "roll_uncertainty" in r else torch.zeros(len(x), 2)
        unc.append(u.cpu().numpy())
    return np.concatenate(g), np.concatenate(rp), np.concatenate(unc)


def run_ego(model):
    import json
    (OUT / "ego").mkdir(parents=True, exist_ok=True)
    takes = {t["take_name"]: t for t in json.load(open(EGO / "takes.json"))}
    for fb in sorted((EGO / "features").glob("*.npz")):
        o = OUT / "ego" / fb.name
        if o.exists():
            continue
        nwin = len(np.load(fb)["pose"])
        t = takes[fb.stem]
        vid = sorted((EGO / t["root_dir"] / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4"))[0]
        buf = decode(vid, "eq(mod(n\\,30)\\,15)", "scale=448:448")
        fr = frames_of(buf, 448, 448)[:nwin]
        g, rp, unc = estimate(model, fr)
        np.savez(o, g=g, rp=rp, unc=unc, n=nwin)
        print(f"  ego {fb.stem}: {len(g)}/{nwin}", flush=True)


def run_wear(model, ids):
    (OUT / "wear").mkdir(parents=True, exist_ok=True)
    for i in ids:
        assert 0 <= i <= 21
        src = WVID / f"sbj_{i}.mp4"; o = OUT / "wear" / f"sbj_{i}.npz"
        if o.exists() or not src.exists():
            continue
        buf = decode(src, "eq(mod(n\\,60)\\,14)", "scale=796:448")
        fr = frames_of(buf, 448, 796)
        g, rp, unc = estimate(model, fr)
        np.savez(o, g=g, rp=rp, unc=unc)
        print(f"  wear sbj_{i}: {len(g)} s", flush=True)


if __name__ == "__main__":
    from geocalib import GeoCalib
    model = GeoCalib(weights="distorted").cuda()
    if sys.argv[1] == "ego":
        run_ego(model)
    else:
        run_wear(model, [int(x) for x in sys.argv[2:]])
