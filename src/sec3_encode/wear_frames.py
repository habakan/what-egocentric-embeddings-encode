"""From the public videos of the WEAR training participants, write the center frame of tile t (frame 60t+14 at 60fps) as a 512x288 JPEG.
Output: $WEAR_DATA/wear_frames/<public id>/<t:05d>.jpg (t is the 0-based second). Analysis only, sbj_0-21 only.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import subprocess
import sys
from pathlib import Path

VID = Path(DATA_DIR + "/wear_raw_video"); OUT = Path(DATA_DIR + "/wear_frames")

for i in map(int, sys.argv[1:]):
    assert 0 <= i <= 21
    d = OUT / f"sbj_{i}"
    if (d / "done").exists():
        continue
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-threads", "3", "-i", str(VID / f"sbj_{i}.mp4"),
                    "-vf", "select='eq(mod(n\\,60)\\,14)',scale=512:288", "-vsync", "0", "-q:v", "3",
                    "-start_number", "0", str(d / "%05d.jpg")], check=True)
    (d / "done").touch()
    print(f"sbj_{i}: {len(list(d.glob('*.jpg')))} frames", flush=True)
