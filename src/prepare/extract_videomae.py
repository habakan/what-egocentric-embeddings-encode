"""Run the raw WEAR videos through VideoMAEv2 ourselves and check whether we can reproduce the organisers' released features.

Why this is needed:
  The exo side of Ego-Exo4D has to be encoded by us. The ego side (WEAR) was encoded by the organisers, so
  if the extraction settings differ, the comparison itself is invalid. First check "can we hit the organisers' settings".

What we know:
  - the released features are 768-dim, **exactly 30 per second** (sbj_0: 83835 rows / 2794.5 s = 30.0)
  - the raw videos are 1080p / 60fps GoPro
  - VideoMAEv2-Base: img 224, patch 16, tubelet 2, num_frames 16, use_mean_pooling=True, num_classes=0
    -> one 768-dim vector per clip (16 frames)

  To get 30 per second, the 16-frame clip must slide in steps of **1 output = 1 input frame**.
  Decode at 30fps with stride 1, or at 60fps with stride 2. Both give the same output rate, but
  **the real time a clip covers differs by 2x** (0.53 s vs 1.07 s), so the features differ. This is what we pin down.

Unknown (searched exhaustively):
  decode fps (30 / 60), clip stride, temporal position within the clip (centre / start), layer to extract.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def decode(src, start, dur, fps, size=224, vf=None):
    """Extract [start, start+dur) at fps with ffmpeg, scale the short side to size and centre-crop size x size.
    Range seek also works on HTTP URLs, so no local download is needed."""
    vf = f"fps={fps}," + (vf or f"scale=-2:{size},crop={size}:{size}")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(start),
           "-i", src, "-t", str(dur), "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    buf = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    n = len(buf) // (size * size * 3)
    return np.frombuffer(buf, np.uint8, n * size * size * 3).reshape(n, size, size, 3)


def to_tensor(frames):
    x = frames.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x).permute(3, 0, 1, 2)          # (3, T, H, W)


MODEL_ID = "OpenGVLab/VideoMAEv2-Base"


def load_model():
    """Does not use HF from_pretrained.

    The released modeling code was written for transformers 4.38 and
    crashes on the current 5.x due to API changes in PreTrainedModel
    (`all_tied_weights_keys` is missing). The model itself is a plain VisionTransformer, so
    load the module directly and only load the weights. This is safer than downgrading
    the project's transformers and affecting other code.
    """
    import importlib.util
    import json

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    d = Path(snapshot_download(MODEL_ID))
    # modeling_videomaev2.py has a relative import `from .modeling_config import ...`, so
    # register the snapshot directory as a namespace package before loading.
    import types
    pkg = types.ModuleType("vmae2pkg")
    pkg.__path__ = [str(d)]
    sys.modules["vmae2pkg"] = pkg
    for name in ["modeling_config", "modeling_videomaev2"]:
        spec = importlib.util.spec_from_file_location(f"vmae2pkg.{name}", d / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[f"vmae2pkg.{name}"] = m
        spec.loader.exec_module(m)
    mod = sys.modules["vmae2pkg.modeling_videomaev2"]

    cfg = json.loads((d / "config.json").read_text())["model_config"]
    # norm_layer is eval()'d on their side, so pass it as a string
    model = mod.VisionTransformer(**dict(cfg))

    sd = load_file(d / "model.safetensors")
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    return model.eval().to(DEV)


@torch.no_grad()
def encode(model, frames, clip=16, stride=1, bs=32, layer=None):
    """frames (N,H,W,3) -> (M, 768). Clip i is frames[i*stride : i*stride+clip].

    layer=None gives the last layer (fc_norm(mean over tokens)); an integer gives that Block's output
    averaged over tokens. Which layer the organisers used is unknown, so this is kept searchable.
    """
    x = to_tensor(frames)                                    # (3, N, H, W)
    starts = list(range(0, max(frames.shape[0] - clip + 1, 0), stride))
    # Official description: "for frame i, input 16 frames: i plus 8 past + 7 future"
    # So clip [s, s+16) represents **frame s+8** (not the first frame s).
    feats = []
    if layer is not None:
        store = {}

        def hook(_m, _i, o):
            store["h"] = o
        h = model.blocks[layer].register_forward_hook(hook)
    for i in range(0, len(starts), bs):
        batch = torch.stack([x[:, s:s + clip] for s in starts[i:i + bs]]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.forward_features(batch)
        out = store["h"].mean(1) if layer is not None else out
        feats.append(out.float().cpu().numpy())
    if layer is not None:
        h.remove()
    return np.concatenate(feats) if feats else np.zeros((0, 768), np.float32)


def main(a):
    model = load_model()
    frames = decode(a.src, a.start, a.dur, a.fps)
    print(f"decoded {frames.shape} @ {a.fps}fps", flush=True)
    F = encode(model, frames, a.clip, a.stride, a.bs, a.layer)
    print(f"encoded {F.shape}")
    np.save(a.out, F.astype(np.float16))
    print("->", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--dur", type=float, default=60.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--clip", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--out", required=True)
    main(p.parse_args())
