"""Control: do the top embedding components represent the wearer's head orientation also for Ego-Exo4D's third-person cameras (same takes, same windows)?

To check whether this is specific to first-person video, encode the first external camera (cam01 / gp01) of each take with the same procedure and the same 3 encoders as first-person.
  VideoMAEv2: 3 clips centred at f0+8, f0+15, f0+22 (16 frames). CLIP / DINOv2: the same 3 frames.
Windows, head pose and participants are taken as-is from the first-person features/<take>.npz (frame_aligned, so frames line up).
Output: $WEAR_DATA/egoexo4d/features_exo_<enc>/<take>.npz . Analysis is in egoexo_viewpoint.analyse.
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
from egoexo_encoders import load as load_img
from egoexo_viewpoint import ROOT, analyse, decode_squash
from extract_videomae import load_model, to_tensor


def exo_video(take):
    d = ROOT / take["root_dir"] / "frame_aligned_videos/downscaled/448"
    cams = sorted(list(d.glob("cam0*.mp4")) + list(d.glob("gp0*.mp4")))
    return cams[0] if cams else None


@torch.no_grad()
def main():
    takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
    vmae = load_model()
    imgs = {e: load_img(e) for e in ["clip", "dinov2"]}
    outs = {e: ROOT / f"features_exo_{e}" for e in ["videomae", "clip", "dinov2"]}
    for o in outs.values():
        o.mkdir(exist_ok=True)
    base = sorted((ROOT / "features").glob("*.npz"))
    for i, fb in enumerate(base):
        if all((o / fb.name).exists() for o in outs.values()):
            continue
        z = np.load(fb, allow_pickle=True)
        nwin = len(z["pose"])
        vid = exo_video(takes[fb.stem])
        if vid is None:
            print(f"  no exo: {fb.stem}", flush=True); continue
        frames = decode_squash(vid)
        meta = dict(pose=z["pose"], ok=z["ok"], participant=z["participant"], domain=z["domain"], cam=vid.name)
        # VideoMAEv2
        starts = [30 * w + o for w in range(nwin) for o in (0, 7, 14)]
        x = to_tensor(frames)
        feats = []
        for j in range(0, len(starts), 12):
            b = torch.stack([x[:, s:s + 16] for s in starts[j:j + 12]]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                feats.append(vmae.forward_features(b).float().cpu().numpy())
        np.savez(outs["videomae"] / fb.name, F=np.concatenate(feats).reshape(nwin, 3, -1).astype(np.float32), **meta)
        # CLIP / DINOv2
        idx = [30 * w + dd + 8 for w in range(nwin) for dd in (0, 7, 14)]
        for e, (proc, f) in imgs.items():
            fs = []
            for j in range(0, len(idx), 96):
                px = proc(images=list(frames[idx[j:j + 96]]), return_tensors="pt")["pixel_values"].cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    fs.append(f(px).float().cpu().numpy())
            np.savez(outs[e] / fb.name, F=np.concatenate(fs).reshape(nwin, 3, -1).astype(np.float32), **meta)
        if i % 10 == 9:
            print(f"  {i + 1}/{len(base)}", flush=True)
    for e, o in outs.items():
        print(f"\n########## exo {e} ##########", flush=True)
        analyse(o)


if __name__ == "__main__":
    main()
