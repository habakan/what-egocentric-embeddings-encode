"""Runs SegFormer-B2 (ADE20K, 150 classes) on WEAR's one-frame-per-second frames and saves the per-class pixel fraction in each cell of a 3x3 grid over the image.

The aim is to measure the cues that distinguish activities in the egocentric view (how one's own body appears = 'person' pixels; place/background = grass, sky, trees, buildings, etc.).
Output: $WEAR_DATA/wear_seg/<public id>.npz  H: (n seconds, 9, 150) float16
Participants are processed in order as their frame extraction (wear_frames.py) finishes.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", DATA_DIR + "/hf")
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

FR = Path(DATA_DIR + "/wear_frames"); OUT = Path(DATA_DIR + "/wear_seg"); OUT.mkdir(exist_ok=True)
M = "nvidia/segformer-b2-finetuned-ade-512-512"


@torch.no_grad()
def seg(proc, net, paths):
    H = []
    for s in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + 16]]
        with torch.autocast("cuda", dtype=torch.float16):
            x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda()
            lab = net(pixel_values=x).logits.argmax(1)                         # (b, 128, 128)
        oh = torch.nn.functional.one_hot(lab, 150).float()                    # (b, 128, 128, 150)
        cells = []
        for r in range(3):
            for c in range(3):
                blk = oh[:, r * 128 // 3:(r + 1) * 128 // 3, c * 128 // 3:(c + 1) * 128 // 3]
                cells.append(blk.mean((1, 2)))
        H.append(torch.stack(cells, 1).half().cpu().numpy())
    return np.concatenate(H)


def main(ids):
    proc = SegformerImageProcessor.from_pretrained(M)
    net = SegformerForSemanticSegmentation.from_pretrained(M).cuda().eval()
    todo = [f"sbj_{i}" for i in ids]
    while todo:
        for r in list(todo):
            o = OUT / f"{r}.npz"
            if o.exists():
                todo.remove(r); continue
            if not (FR / r / "done").exists():
                continue
            paths = sorted((FR / r).glob("*.jpg"))
            np.savez(o, H=seg(proc, net, paths))
            print(f"{r}: {len(paths)} frames", flush=True)
            todo.remove(r)
        if todo:
            time.sleep(60)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
