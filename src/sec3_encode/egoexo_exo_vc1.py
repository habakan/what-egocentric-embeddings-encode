"""Encode the third-person cameras with VC-1 using the same procedure as egoexo_exo.py (for comparison with an encoder pretrained specifically for egocentric video).
Output: $WEAR_DATA/egoexo4d/features_exo_vc1/<take>.npz
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", DATA_DIR + "/hf")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from egoexo_encoders import load
from egoexo_exo import exo_video
from egoexo_viewpoint import ROOT, decode_squash


@torch.no_grad()
def main():
    takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
    proc, f = load("vc1")
    out = ROOT / "features_exo_vc1"; out.mkdir(exist_ok=True)
    base = sorted((ROOT / "features").glob("*.npz"))
    for i, fb in enumerate(base):
        if (out / fb.name).exists():
            continue
        z = np.load(fb, allow_pickle=True)
        nwin = len(z["pose"])
        vid = exo_video(takes[fb.stem])
        if vid is None:
            continue
        frames = decode_squash(vid)
        idx = [min(30 * w + dd + 8, len(frames) - 1) for w in range(nwin) for dd in (0, 7, 14)]
        fs = []
        for j in range(0, len(idx), 96):
            px = proc(images=list(frames[idx[j:j + 96]]), return_tensors="pt")["pixel_values"].to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                fs.append(f(px).float().cpu().numpy())
        np.savez(out / fb.name, F=np.concatenate(fs).reshape(nwin, 3, -1).astype(np.float32), pose=z["pose"], ok=z["ok"],
                 participant=z["participant"], domain=z["domain"], cam=vid.name)
        if i % 20 == 19:
            print(f"  exo vc1: {i + 1}/{len(base)}", flush=True)


if __name__ == "__main__":
    main()
