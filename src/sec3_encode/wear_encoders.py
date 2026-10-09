"""Extract per-window CLIP / DINOv2 features from the raw video of the WEAR training participants (for the paper's analysis only).

The system's graph uses, for each 1-second tile t, the mean of VideoMAE rows [30t, 30t+15) (30fps).
For the image encoders, every other frame in that span, 8 frames (30fps rows 30t+0,2,...,14 = 60fps frames
60t+0,4,...,28), is resized to 224x224 and encoded. Averaging over the window is done on the analysis side.
Output: $WEAR_DATA/wear_img_feats/<enc>/<rec>.npy  (n_sec, 8, d) float16
Only sbj_0-21 (sbj_22 and later overlap in numbering with the competition's test participants, so they are not used).
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", DATA_DIR + "/hf")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from egoexo_encoders import load

VID = Path(DATA_DIR + "/wear_raw_video")
OUT = Path(DATA_DIR + "/wear_img_feats")
S = 224
PER_SEC = 8
ENCS = os.environ.get("WEAR_ENCS", "clip,dinov2").split(",")


def frames(src):
    """Yield, in order, the 60fps frames n with n mod 60 < 30 and n mod 4 == 0 (8 per second)."""
    vf = f"select='not(mod(n\\,4))*lt(mod(n\\,60)\\,30)',scale={S}:{S}"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "8", "-i", str(src), "-vf", vf,
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=S * S * 3 * PER_SEC * 16)
    sz = S * S * 3 * PER_SEC * 32                       # 32 seconds at a time
    while True:
        buf = p.stdout.read(sz)
        n = len(buf) // (S * S * 3)
        if n == 0:
            break
        yield np.frombuffer(buf, np.uint8, n * S * S * 3).reshape(n, S, S, 3)
    p.wait()


def gpu_pre(proc, x):
    """Run the HF image preprocessing (bicubic-resize short side to size -> centre-crop crop_size -> normalise) on the GPU.
    The input is already 224x224, so this is the identity for CLIP; for DINOv2 it upsizes to 256 and centre-crops 224."""
    se = proc.size["shortest_edge"]
    if se != x.shape[-1]:
        x = torch.nn.functional.interpolate(x, size=(se, se), mode="bicubic", align_corners=False).clamp(0, 1)
    c = proc.crop_size["height"]; o = (se - c) // 2
    x = x[..., o:o + c, o:o + c]
    m = torch.tensor(proc.image_mean, device=x.device).view(1, 3, 1, 1)
    sd = torch.tensor(proc.image_std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / sd


@torch.no_grad()
def main(ids):
    models = {e: load(e) for e in ENCS}
    for e in ENCS:
        (OUT / e).mkdir(parents=True, exist_ok=True)
    for i in ids:
        assert 0 <= i <= 21, "sbj_22 and later overlap in numbering with the test participants and are not used"
        rec = f"sbj_{i}"
        src = VID / f"{rec}.mp4"
        if all((OUT / e / f"{rec}.npy").exists() for e in ENCS) or not src.exists():
            continue
        feats = {e: [] for e in ENCS}
        for fr in frames(src):
            x = torch.from_numpy(fr).to("cuda").permute(0, 3, 1, 2).float() / 255.0
            for e, (proc, f) in models.items():
                for j in range(0, len(x), 256):
                    with torch.autocast("cuda", dtype=torch.float16):
                        feats[e].append(f(gpu_pre(proc, x[j:j + 256])).float().cpu().numpy().astype(np.float16))
        for e in ENCS:
            F = np.concatenate(feats[e])
            F = F[: len(F) // PER_SEC * PER_SEC].reshape(-1, PER_SEC, F.shape[1])
            np.save(OUT / e / f"{rec}.npy", F)
        print(f"{rec}: {F.shape[0]} s", flush=True)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]])
