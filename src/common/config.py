"""Constants, paths and seeds shared by all experiments. Per-experiment differences go here or in the experiment script's config dict."""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "input"
TRAIN_INERTIAL = INPUT / "train" / "inertial_feat"
TRAIN_VIDEOMAE = INPUT / "train" / "videomae_feat"
TEST_DIR = INPUT / "test"
WORK = ROOT / "experiments"
SUBS = ROOT / "submissions"

SEED = 42
SR = 50                 # inertial 50Hz
FPS = 30                # VideoMAEv2 frame features 30fps
WIN_SEC = 1
WIN_INERTIAL = SR * WIN_SEC     # 50
WIN_VIDEO = FPS * WIN_SEC       # 30
STEP_RATIO = SR // 10           # 5: inertial indices that are multiples of 5 give integer video indices

SENSORS = ["right_arm", "right_leg", "left_leg", "left_arm"]
AXES = ["x", "y", "z"]

# ⚠ Not verified against the primary source. Standard order from the WEAR paper (the public notebook uses the same order).
# If CV and LB diverge strongly on the first submission, suspect this first.
CLASS_NAMES = [
    "null",
    "jogging", "jogging (rotating arms)", "jogging (skipping)",
    "jogging (sidesteps)", "jogging (butt-kicks)",
    "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)",
    "stretching (hamstrings)", "stretching (lumbar rotation)",
    "push-ups", "push-ups (complex)",
    "sit-ups", "sit-ups (complex)",
    "burpees",
    "lunges", "lunges (complex)",
    "bench-dips",
]
N_CLASSES = len(CLASS_NAMES)
LABEL_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def seed_everything(seed: int = SEED) -> None:
    import os, random
    import numpy as np
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
