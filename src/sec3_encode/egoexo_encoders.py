"""Does "main direction of the egocentric video embedding = head orientation" also hold for other encoders (Ego-Exo4D)?

Encoders: CLIP ViT-B/32 (image-text contrastive learning) / DINOv2 ViT-B/14 (self-supervised).
For each window, the same 3 frames as VideoMAE (f0+8, f0+15, f0+22) are resized to 224x224, encoded, and averaged over the window.
The head pose and window correspondence are taken as-is from features/<take>.npz (VideoMAE).
Analysis is the same as egoexo_viewpoint.analyse. Output: $WEAR_DATA/egoexo4d/features_<enc>/
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
from egoexo_viewpoint import ROOT, analyse, decode_squash

DEV = "cuda"
ENC = {"clip": "openai/clip-vit-base-patch32", "dinov2": "facebook/dinov2-base"}


class _ImgProc:
    """Minimal preprocessing with the same call signature as HF image preprocessing (VC-1: resize to 224, center crop, ImageNet normalization)."""
    size = {"shortest_edge": 224}; crop_size = {"height": 224}
    image_mean = [0.485, 0.456, 0.406]; image_std = [0.229, 0.224, 0.225]

    def __call__(self, images, return_tensors="pt"):
        x = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        m = torch.tensor(self.image_mean).view(1, 3, 1, 1); sd = torch.tensor(self.image_std).view(1, 3, 1, 1)
        return {"pixel_values": (x - m) / sd}


def load_vc1():
    """VC-1 ViT-B/16 (MAE, 4000+ hours of egocentric video + ImageNet; facebook/vc1-base). Output is the CLS token after the final normalization."""
    import timm
    from huggingface_hub import hf_hub_download
    sd = torch.load(hf_hub_download("facebook/vc1-base", "pytorch_model.bin"), map_location="cpu")["model"]
    sd = {k: v for k, v in sd.items() if not k.startswith("decoder") and k != "mask_token"}
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="token")
    m.load_state_dict(sd, strict=True)
    m = m.eval().to(DEV)
    return _ImgProc(), (lambda px: m(px))


def load(enc):
    if enc == "vc1":
        return load_vc1()
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection
    proc = AutoImageProcessor.from_pretrained(ENC[enc])
    if enc == "clip":
        m = CLIPVisionModelWithProjection.from_pretrained(ENC[enc])
        f = lambda px: m(pixel_values=px).image_embeds
    else:
        m = AutoModel.from_pretrained(ENC[enc])
        f = lambda px: m(pixel_values=px).pooler_output
    m = m.eval().to(DEV)
    return proc, f


@torch.no_grad()
def main(encs):
    takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
    base = sorted((ROOT / "features").glob("*.npz"))
    for enc in encs:
        out = ROOT / f"features_{enc}"; out.mkdir(exist_ok=True)
        proc, f = load(enc)
        for i, fb in enumerate(base):
            o = out / fb.name
            if o.exists():
                continue
            z = np.load(fb, allow_pickle=True)
            nwin = len(z["pose"])
            t = takes[fb.stem]
            vid = sorted((ROOT / t["root_dir"] / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4"))[0]
            frames = decode_squash(vid)
            idx = [30 * w + d + 8 for w in range(nwin) for d in (0, 7, 14)]
            feats = []
            for j in range(0, len(idx), 96):
                px = proc(images=list(frames[idx[j:j + 96]]), return_tensors="pt")["pixel_values"].to(DEV)
                with torch.autocast("cuda", dtype=torch.float16):
                    feats.append(f(px).float().cpu().numpy())
            F = np.concatenate(feats).reshape(nwin, 3, -1)
            np.savez(o, F=F.astype(np.float32), pose=z["pose"], ok=z["ok"], participant=z["participant"], domain=z["domain"])
            if i % 20 == 19:
                print(f"  {enc}: {i + 1}/{len(base)}", flush=True)
        del proc, f; torch.cuda.empty_cache()
    for enc in ["videomae"] + encs:
        print(f"\n########## {enc} ##########", flush=True)
        analyse(ROOT / ("features" if enc == "videomae" else f"features_{enc}"))


if __name__ == "__main__":
    main(sys.argv[1:] or ["clip", "dinov2"])
