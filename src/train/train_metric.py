"""Build video embeddings for the transductive graph with metric learning.

Raw VideoMAE features are dominated by "subject, location, clothing" differences, with activity differences on top.
The test graph is built **within the same subject**, so training also uses only
"same activity = positive / different activity = negative" **within the same recording**. This learns activity differences rather than subject differences.

Output: experiments/metric_<tag>/emb_fold{k}.npy (embeddings of valid windows), emb_test_fold{k}.npy

--loss selects the loss:
  supcon  : within-recording supervised contrastive (2026-09-03 implementation; rejected as on par with raw features)
  arcface : classification-based metric learning (ArcFace). Preferred in practice because training is stable
            without depending on hard-negative sampling. **The rejection was of the supcon implementation,
            not of metric learning as a mechanism**, so re-measure with a different family.

⚠ The biggest empirical fact of this system is "removing the top 30 principal components, the most activity-discriminative, from the video features raises the score".
   Metric learning maximises exactly that activity discriminability, so the two pull the same axis in opposite directions.
   So no conclusion can be drawn without measuring the 2x2 {raw/ArcFace} x {m=0/m=30} (probe_metric2.py).
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

sys.path.insert(0, str(Path(__file__).parent))
from config import N_CLASSES, SEED, TEST_DIR, WORK, seed_everything
from cv import subject_folds

PREP = WORK / "prep"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, dim=128, hid=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(768, hid, 1), nn.BatchNorm1d(hid), nn.GELU(),
            nn.Conv1d(hid, hid, 3, padding=1), nn.BatchNorm1d(hid), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(hid * 2, dim))

    def forward(self, v):                       # v: (B, 768, 15)
        h = self.net(v)
        h = torch.cat([h.mean(-1), h.amax(-1)], -1)
        return F.normalize(self.head(h), dim=-1)


class ArcFace(nn.Module):
    """Classification-based metric learning. L2-normalise both the embeddings and the class prototype vectors,
    add margin m to the angle of the true class, then softmax with temperature s."""

    def __init__(self, dim, n_cls, s=30.0, m=0.30):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_cls, dim))
        nn.init.xavier_uniform_(self.w)
        self.s, self.m = s, m

    def forward(self, z, y):
        cos = z @ F.normalize(self.w, dim=-1).T
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        tgt = F.one_hot(y.long(), self.w.shape[0]).bool()
        logit = torch.cos(torch.where(tgt, theta + self.m, theta)) * self.s
        return F.cross_entropy(logit, y.long())


def supcon(z, y, temp=0.1):
    """Supervised contrastive loss on a batch from a single recording."""
    sim = z @ z.T / temp
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    has = pos.any(1)
    if has.sum() == 0:
        return z.sum() * 0
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    return -(logp * pos).sum(1)[has].div(pos.sum(1)[has]).mean()


def sample_batch(meta_rec, video, rng, bs):
    """Sample bs windows from one recording (matches the graph's evaluation condition)"""
    rows = meta_rec["idx"].to_numpy()
    sel = rng.choice(rows, min(bs, len(rows)), replace=False)
    sel.sort()
    v = np.asarray(video[sel], np.float32).transpose(0, 2, 1)     # (B, 768, 15)
    return torch.from_numpy(v), torch.from_numpy(meta_rec.set_index("idx").loc[sel, "label"].to_numpy())


def main(a):
    seed_everything(a.seed)
    meta = pd.read_parquet(PREP / "win_meta.parquet").reset_index().rename(columns={"index": "idx"})
    video = np.load(PREP / f"video_raw_{a.offset}.npy", mmap_mode="r")
    vte = np.load(PREP / "video_raw_test.npy", mmap_mode="r")
    folds, _ = subject_folds(meta["sbj_id"].to_numpy(), n_splits=a.folds)
    out = WORK / f"metric_{a.tag}"; out.mkdir(parents=True, exist_ok=True)

    for k, (tr_w, va_w) in enumerate(folds):
        if a.only_fold is not None and k != a.only_fold:
            continue
        tr_meta = meta.iloc[tr_w]
        recs = [g for _, g in tr_meta.groupby("rec") if len(g) > a.bs]
        model = Encoder(a.dim).to(DEV)
        crit = ArcFace(a.dim, N_CLASSES, a.arc_s, a.arc_m).to(DEV) if a.loss == "arcface" else None
        params = list(model.parameters()) + (list(crit.parameters()) if crit else [])
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
        rng = np.random.RandomState(a.seed + k)
        model.train(); t0 = time.time(); run = 0.0
        for step in range(a.steps):
            g = recs[rng.randint(len(recs))]
            v, y = sample_batch(g, video, rng, a.bs)
            v, y = v.to(DEV), y.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(v).float()
                loss = crit(z, y) if crit else supcon(z, y, a.temp)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run += loss.item()
            if (step + 1) % 200 == 0:
                print(f"  fold{k} step{step+1} loss={run/200:.4f} ({time.time()-t0:.0f}s)", flush=True)
                run = 0.0

        model.eval()
        with torch.no_grad():
            def embed(arr, idx):
                res = []
                for i in range(0, len(idx), 2048):
                    v = np.asarray(arr[idx[i:i + 2048]], np.float32).transpose(0, 2, 1)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        res.append(model(torch.from_numpy(v).to(DEV)).float().cpu().numpy())
                return np.concatenate(res)
            np.save(out / f"emb_fold{k}.npy", embed(video, va_w))
            np.save(out / f"emb_idx_fold{k}.npy", va_w)
            np.save(out / f"emb_test_fold{k}.npy", embed(vte, np.arange(vte.shape[0])))
        print(f"fold{k} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--offset", default="head")
    p.add_argument("--loss", choices=["supcon", "arcface"], default="supcon")
    p.add_argument("--arc-s", type=float, default=30.0)
    p.add_argument("--arc-m", type=float, default=0.30)
    p.add_argument("--tag", default="v1")
    main(p.parse_args())
