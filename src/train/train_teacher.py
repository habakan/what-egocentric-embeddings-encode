"""Teacher model that sees all 4 sensors. Produces soft labels for distillation into the student (1 sensor).

test gives only one sensor per window, but train has all 4 sensors (privileged information).
Arm stretches carry no information from a leg sensor alone (measured accuracy 0.00), but
the teacher sees all four limbs and can tell them apart. By passing its soft labels to the student,
the student can learn the distribution "this leg-sensor window is one of the arm stretches".

Output: experiments/<tag>/oof_win.npy (n_win, 19)  * per window (sensor-independent)
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, SENSORS, WORK, seed_everything
from cv import subject_folds
from metric import macro_f1
from train_nn import conv_block, random_rotation

PREP = WORK / "prep"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class AllSensorDS(Dataset):
    def __init__(self, idx, inert, labels, train, rot_deg=0.0, valid=None):
        self.idx, self.inert, self.labels = idx, inert, labels
        self.train, self.rot_deg, self.valid = train, rot_deg, valid

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        w = self.idx[i]
        x = np.asarray(self.inert[w], np.float32)          # (4, 50, 3)
        if self.valid is not None:
            # missing sensor (left_arm of sbj_10) is zero-filled. 0 * NaN = NaN, so clear it with where
            x = np.where(self.valid[w][:, None, None] > 0, np.nan_to_num(x), 0.0).astype(np.float32)
        if self.train:
            for s in range(x.shape[0]):
                if self.rot_deg > 0:
                    x[s] = x[s] @ random_rotation(self.rot_deg)
            x = x * np.float32(np.random.normal(1.0, 0.1))
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
        x = x.transpose(0, 2, 1).reshape(-1, x.shape[1])   # (12, 50)
        return torch.from_numpy(x.copy()), torch.tensor(int(self.labels[w]))


class Teacher(nn.Module):
    def __init__(self, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(conv_block(12, 128, 7), conv_block(128, 256, 5, 2),
                                 conv_block(256, 256, 3), conv_block(256, 256, 3, 2))
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(drop), nn.Linear(512, 256),
                                  nn.GELU(), nn.Dropout(drop), nn.Linear(256, N_CLASSES))

    def forward(self, x):
        h = self.net(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], -1))


@torch.no_grad()
def predict(model, dl):
    model.eval(); out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(model(x.to(DEV)).float(), -1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main(a):
    seed_everything(a.seed)
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    n_win = len(meta)
    y = meta["label"].to_numpy(np.int64)
    inert = np.load(PREP / "inertial.npy", mmap_mode="r")
    valid = np.load(PREP / "valid_mask.npy").astype(np.float32)
    folds, _ = subject_folds(meta["sbj_id"].to_numpy(), n_splits=a.folds)

    oof = np.zeros((n_win, N_CLASSES), np.float32)
    scores = []
    for k, (tr, va) in enumerate(folds):
        dl_tr = DataLoader(AllSensorDS(tr, inert, y, True, a.rot, valid), batch_size=a.bs,
                           shuffle=True, num_workers=6, pin_memory=True, drop_last=True,
                           persistent_workers=True)
        dl_va = DataLoader(AllSensorDS(va, inert, y, False, valid=valid), batch_size=1024,
                           shuffle=False, num_workers=4, pin_memory=True)
        model = Teacher().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                    steps_per_epoch=len(dl_tr), pct_start=0.1)
        best, best_p = -1, None
        for ep in range(a.epochs):
            model.train(); t0 = time.time()
            for x, yb in dl_tr:
                x, yb = x.to(DEV, non_blocking=True), yb.to(DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x), yb, label_smoothing=0.05)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
            p = predict(model, dl_va)
            f1 = macro_f1(y[va], p.argmax(1))
            if f1 > best:
                best, best_p = f1, p
            print(f"  fold{k} ep{ep} valF1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        oof[va] = best_p; scores.append(best)
        print(f"fold{k} best={best:.4f}", flush=True)

    out = WORK / a.tag; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof_win.npy", oof)
    print(f"== teacher CV mean={np.mean(scores):.4f} std={np.std(scores):.4f}")
    print(f"== teacher OOF macro-F1 = {macro_f1(y, oof.argmax(1)):.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--rot", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default="teacher01")
    main(p.parse_args())
