"""Run the middle frame (30w+15) of each window of Ego-Exo4D first-person (Aria 214) and third-person (first fixed camera) video through
SegFormer-B2 (ADE20K) and save the per-class pixel fraction for each 3x3 cell (same format as wear_segment.py).
Output: $WEAR_DATA/egoexo4d/seg_{ego,exo}/<take>.npz  H: (n_windows, 9, 150) float16
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
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

sys.path.insert(0, str(Path(__file__).parent))
from egoexo_exo import exo_video
from egoexo_viewpoint import ROOT
from wear_segment import M, seg


def frames(vid, n):
    buf = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(vid), "-vf",
                          "select='eq(mod(n\\,30)\\,15)',scale=448:448", "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True).stdout
    fr = np.frombuffer(buf, np.uint8).reshape(-1, 448, 448, 3)[:n]
    return fr




def main():
    proc = SegformerImageProcessor.from_pretrained(M)
    net = SegformerForSemanticSegmentation.from_pretrained(M).cuda().eval()
    takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
    for view in ["ego", "exo"]:
        (ROOT / f"seg_{view}").mkdir(exist_ok=True)
    for fb in sorted((ROOT / "features").glob("*.npz")):
        n = len(np.load(fb)["pose"]); t = takes[fb.stem]
        d = ROOT / t["root_dir"] / "frame_aligned_videos/downscaled/448"
        for view, vid in [("ego", sorted(d.glob("aria*_214-1.mp4"))[0]), ("exo", exo_video(t))]:
            o = ROOT / f"seg_{view}" / fb.name
            if o.exists() or vid is None:
                continue
            fr = frames(vid, n)
            tmp = Path(DATA_DIR) / "tmp" / "egoseg"; tmp.mkdir(parents=True, exist_ok=True)
            paths = []
            for i, f in enumerate(fr):
                p = tmp / f"{i:05d}.jpg"; Image.fromarray(f).save(p, quality=90); paths.append(p)
            np.savez(o, H=seg(proc, net, paths))
            for p in paths:
                p.unlink()
        print(f"  {fb.stem}", flush=True)


if __name__ == "__main__":
    main()
