"""Multimodal 1D-CNN (inertial + VideoMAE) — subject GroupKFold.

One test sample = "1 s of inertial data (50,3) x 1 sensor position" + "0.5 s of video features (15,768)", so
train is also expanded to window x sensor position, and the sensor position is conditioned on via an embedding.
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
from config import N_CLASSES, SEED, SENSORS, TEST_DIR, WORK, seed_everything
from cv import subject_folds
from metric import macro_f1, per_class_f1

PREP = WORK / "prep"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def random_rotation(max_deg: float) -> np.ndarray:
    """Small rotation matrix (3,3) about a random axis. Rodrigues' formula."""
    ax = np.random.normal(size=3)
    ax /= np.linalg.norm(ax) + 1e-8
    th = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)).astype(np.float32)


class WinDS(Dataset):
    """rows are indices "after sensor expansion". win = rows % n_win, sensor = rows // n_win"""

    def __init__(self, rows, inert, video, labels, n_win, train: bool, use_video: bool,
                 vnorm=None, rot_deg: float = 0.0, aux=None, soft=None):
        self.rows, self.inert, self.video = rows, inert, video
        self.labels, self.n_win, self.train, self.use_video = labels, n_win, train, use_video
        self.vnorm = vnorm      # (n_win, 2, 768): per-recording mean/std
        self.rot_deg = rot_deg
        self.aux = aux          # (n_win*4, 78) hand-crafted statistical/frequency features
        self.soft = soft        # (n_win, 19) soft labels from the 4-sensor teacher (per window)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        w, s = r % self.n_win, r // self.n_win
        x = np.asarray(self.inert[w, s], np.float32)            # (50, 3)
        if self.train:
            if self.rot_deg > 0:
                # 3D rotation mimicking per-person differences in mounting angle. Should help generalisation to unseen subjects
                x = x @ random_rotation(self.rot_deg)
            x = x * np.float32(np.random.normal(1.0, 0.1))       # scale
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)  # jitter
        out = [torch.from_numpy(x.T.copy()), torch.tensor(s)]
        if self.use_video:
            v = np.asarray(self.video[w], np.float32)            # (15, 768)
            if self.vnorm is not None:
                mu, sd = np.asarray(self.vnorm[w], np.float32)
                v = (v - mu) / (sd + 1e-3)
            if self.train:
                v = v + np.random.normal(0, 0.02, v.shape).astype(np.float32)
            out.append(torch.from_numpy(v.T.copy()))             # (768, 15)
        else:
            out.append(torch.zeros(1))
        out.append(torch.from_numpy(np.asarray(self.aux[r], np.float32))
                   if self.aux is not None else torch.zeros(1))
        out.append(torch.tensor(int(self.labels[w])))
        out.append(torch.from_numpy(np.asarray(self.soft[w], np.float32))
                   if self.soft is not None else torch.zeros(1))
        return out


def conv_block(cin, cout, k, pool=1):
    layers = [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU()]
    if pool > 1:
        layers.append(nn.MaxPool1d(pool))
    return nn.Sequential(*layers)


class Net(nn.Module):
    """Default is late fusion (pool both branches, then concat in the head).

    fusion="frame" switches to **frame-synchronous fusion**: the inertial conv output is
    interpolated to the number of video frames, concatenated along channels at each time step,
    and then passed through a fusion conv. Mixing after aligning the time axis corresponds to VSFF-style designs.
    Late fusion never aligns the time axis, so the two are different things.
    """

    def __init__(self, use_video: bool, vdim: int = 768, drop: float = 0.3, aux_dim: int = 0,
                 use_inertial: bool = True, fusion: str = "late", band: int = 0):
        super().__init__()
        self.use_video = use_video
        self.aux_dim = aux_dim
        self.use_inertial = use_inertial
        self.fusion = fusion if (use_video and use_inertial) else "late"
        self.band = band
        self.inert = nn.Sequential(conv_block(3, 64, 7), conv_block(64, 128, 5, 2),
                                   conv_block(128, 128, 3), conv_block(128, 128, 3, 2))
        self.sensor = nn.Embedding(len(SENSORS), 32)
        d = (256 if use_inertial else 0) + 32
        if aux_dim:
            self.aux_net = nn.Sequential(nn.LayerNorm(aux_dim), nn.Linear(aux_dim, 128),
                                         nn.GELU(), nn.Dropout(drop), nn.Linear(128, 128))
            d += 128
        if use_video:
            self.vproj = nn.Sequential(nn.Conv1d(vdim, 256, 1), nn.BatchNorm1d(256), nn.GELU(),
                                       conv_block(256, 256, 3))
            d += 512
        if self.fusion == "frame":
            # concat inertial (128ch) and video (256ch) at each time step, then convolve
            self.fuse = nn.Sequential(conv_block(128 + 256, 256, 3), conv_block(256, 256, 3))
            d = 512 + 32 + (128 if aux_dim else 0)     # fusion pool + sensor + aux
        if self.fusion == "xattn":
            # Sparse cross-attention with inertial as query and video as key/value.
            # Whereas frame fusion assumes "the indices are aligned", this one
            # **learns the alignment**. The video's effective receptive field is unknown and the spans differ,
            # so the benefit is not having to assume anything. Sequences are short (13x15), so sparsity acts
            # not for efficiency but as an inductive bias (alignment is local).
            self.dm = 256
            self.q = nn.Conv1d(128, self.dm, 1)
            self.kv = nn.Conv1d(256, self.dm * 2, 1)
            self.nh = 4
            self.xproj = nn.Linear(self.dm, self.dm)
            self.fuse = nn.Sequential(conv_block(128 + self.dm, 256, 3), conv_block(256, 256, 3))
            d = 512 + 32 + (128 if aux_dim else 0)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 256),
                                  nn.GELU(), nn.Dropout(drop), nn.Linear(256, N_CLASSES))

    @staticmethod
    def pool(h):
        return torch.cat([h.mean(-1), h.amax(-1)], -1)

    def cross_attend(self, hi, hv, band):
        """Inertial query x video key/value. If band>0, restrict to temporally close pairs."""
        B, _, Ti = hi.shape
        Tv = hv.shape[-1]
        q = self.q(hi).transpose(1, 2)                                  # (B,Ti,dm)
        k, val = self.kv(hv).transpose(1, 2).chunk(2, -1)                # (B,Tv,dm) x2
        H, dh = self.nh, self.dm // self.nh
        q = q.view(B, Ti, H, dh).transpose(1, 2)                        # (B,H,Ti,dh)
        k = k.view(B, Tv, H, dh).transpose(1, 2)
        val = val.view(B, Tv, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / dh ** 0.5                     # (B,H,Ti,Tv)
        if band > 0:
            ti = torch.arange(Ti, device=hi.device).float() * (Tv - 1) / max(Ti - 1, 1)
            tv = torch.arange(Tv, device=hi.device).float()
            mask = (ti[:, None] - tv[None, :]).abs() > band
            att = att.masked_fill(mask, float("-inf"))
        att = att.softmax(-1)
        out = (att @ val).transpose(1, 2).reshape(B, Ti, self.dm)
        return self.xproj(out).transpose(1, 2)                          # (B,dm,Ti)

    def forward(self, x, s, v, a=None):
        if self.fusion == "xattn":
            hi = self.inert(x)
            hv = self.vproj(v)
            ctx = self.cross_attend(hi, hv, self.band)
            feats = [self.pool(self.fuse(torch.cat([hi, ctx], 1))), self.sensor(s)]
            if self.aux_dim:
                feats.append(self.aux_net(a))
            return self.head(torch.cat(feats, -1))
        if self.fusion == "frame":
            hi = self.inert(x)                                   # (B,128,~13)
            hv = self.vproj(v)                                   # (B,256,15)
            # align the time axis to the video length, then concat along channels
            hi = F.interpolate(hi, size=hv.shape[-1], mode="linear", align_corners=False)
            feats = [self.pool(self.fuse(torch.cat([hi, hv], 1))), self.sensor(s)]
            if self.aux_dim:
                feats.append(self.aux_net(a))
            return self.head(torch.cat(feats, -1))
        feats = ([self.pool(self.inert(x))] if self.use_inertial else []) + [self.sensor(s)]
        if self.aux_dim:
            feats.append(self.aux_net(a))
        if self.use_video:
            feats.append(self.pool(self.vproj(v)))
        return self.head(torch.cat(feats, -1))


def expand(win_idx, n_win, valid):
    """Window index -> row index after sensor expansion. Missing sensors (sbj_10's left_arm) are excluded."""
    rows = np.concatenate([win_idx + s * n_win for s in range(len(SENSORS))])
    keep = valid[rows % n_win, rows // n_win]
    return rows[keep]


def run_fold(k, tr_w, va_w, inert, video, y_win, n_win, args, test_loader, valid, vnorm,
             aux, soft=None):
    rows_tr = expand(tr_w, n_win, valid)
    rows_va = expand(va_w, n_win, valid)
    ds_tr = WinDS(rows_tr, inert, video, y_win, n_win, True, args.video, vnorm, args.rot,
                  aux, soft)
    ds_va = WinDS(rows_va, inert, video, y_win, n_win, False, args.video, vnorm, 0.0, aux)
    dl_tr = DataLoader(ds_tr, batch_size=args.bs, shuffle=True, num_workers=6,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    model = Net(args.video, aux_dim=(aux.shape[1] if aux is not None else 0),
                use_inertial=not args.no_inertial, fusion=args.fusion,
                band=args.xattn_band).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(dl_tr), pct_start=0.1)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, s, v, ab, yb, sf in dl_tr:
            x, s, v, ab, yb = (x.to(DEV, non_blocking=True), s.to(DEV),
                               v.to(DEV, non_blocking=True), ab.to(DEV), yb.to(DEV))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, s, v, ab)
                loss = F.cross_entropy(logit, yb, label_smoothing=0.05)
                if soft is not None:
                    # distillation from the 4-sensor teacher. The T^2 factor matches the gradient scale to CE
                    T = args.distill_t
                    t = sf.to(DEV, non_blocking=True)
                    tsoft = torch.softmax(torch.log(t.float() + 1e-9) / T, -1)
                    kl = F.kl_div(F.log_softmax(logit.float() / T, -1), tsoft,
                                  reduction="batchmean") * (T * T)
                    loss = (1 - args.distill_w) * loss + args.distill_w * kl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        p = predict(model, dl_va)
        f1 = macro_f1(y_win[rows_va % n_win], p.argmax(1))
        if f1 > best:
            best, best_state = f1, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        print(f"  fold{k} ep{ep} loss={tot/len(dl_tr):.4f} valF1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model, rows_va, predict(model, dl_va), predict(model, test_loader), best


@torch.no_grad()
def predict(model, dl):
    model.eval(); out = []
    for batch in dl:
        x, s, v, a = (batch[0].to(DEV), batch[1].to(DEV), batch[2].to(DEV), batch[3].to(DEV))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(model(x, s, v, a).float(), -1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


class TestDS(Dataset):
    def __init__(self, x, v, s, use_video, vnorm=None, aux=None):
        self.x, self.v, self.s, self.use_video = x, v, s, use_video
        self.vnorm = vnorm
        self.aux = aux

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        xi = torch.from_numpy(np.asarray(self.x[i], np.float32).T.copy())
        if self.use_video:
            vv = np.asarray(self.v[i], np.float32)
            if self.vnorm is not None:
                mu, sd = np.asarray(self.vnorm[i], np.float32)
                vv = (vv - mu) / (sd + 1e-3)
            vi = torch.from_numpy(vv.T.copy())
        else:
            vi = torch.zeros(1)
        ai = (torch.from_numpy(np.asarray(self.aux[i], np.float32))
              if self.aux is not None else torch.zeros(1))
        return xi, torch.tensor(int(self.s[i])), vi, ai, torch.tensor(0), torch.zeros(1)


def main(args):
    seed_everything(args.seed)
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    n_win = len(meta)
    y_win = meta["label"].to_numpy(np.int64)
    inert = np.load(PREP / "inertial.npy", mmap_mode="r")                       # (n_win,4,50,3)
    valid = np.load(PREP / "valid_mask.npy")                                    # (n_win,4) missing-sensor mask
    vnorm = vnorm_te = None
    if args.video and args.vnorm:
        vnorm = np.load(PREP / f"vnorm_train_{args.offset}.npy", mmap_mode="r")
        vnorm_te = np.load(PREP / "vnorm_test.npy", mmap_mode="r")
    video = np.load(PREP / f"video_raw_{args.offset}.npy", mmap_mode="r") if args.video else None

    te_meta = pd.read_csv(TEST_DIR / "test_meta_data.csv")
    xte = np.load(TEST_DIR / "test_inertial_data.npy").astype(np.float32)
    vte = np.load(PREP / "video_raw_test.npy", mmap_mode="r") if args.video else None
    ste = te_meta["sensor_location"].map({s: i for i, s in enumerate(SENSORS)}).to_numpy()
    soft = None
    if args.distill:
        soft = np.load(WORK / args.distill / "oof_win.npy")
        print("distillation: using teacher OOF", args.distill, soft.shape)
    aux = aux_te = None
    if args.aux:
        aux = np.load(PREP / "feat_inertial.npy", mmap_mode="r")
        from train_lgb import load_test as _lt
        aux_te = _lt()[0][:, :aux.shape[1]]
    test_loader = DataLoader(TestDS(xte, vte, ste, args.video, vnorm_te, aux_te), batch_size=1024,
                             shuffle=False, num_workers=4)

    folds, _ = subject_folds(meta["sbj_id"].to_numpy(), n_splits=args.folds)
    oof = np.zeros((n_win * len(SENSORS), N_CLASSES), np.float32)
    used = np.zeros(n_win * len(SENSORS), bool)
    test_p = np.zeros((len(xte), N_CLASSES), np.float32)
    scores = []
    for k, (tr_w, va_w) in enumerate(folds):
        if args.only_fold is not None and k != args.only_fold:
            continue
        _, rows_va, pva, pte, best = run_fold(k, tr_w, va_w, inert, video, y_win,
                                              n_win, args, test_loader, valid, vnorm, aux, soft)
        oof[rows_va] = pva; used[rows_va] = True; test_p += pte
        scores.append(best)
        print(f"fold{k} best={best:.4f}", flush=True)

    test_p /= max(len(scores), 1)
    y4 = np.tile(y_win, len(SENSORS))
    yp = oof[used].argmax(1)
    print(f"== CV mean={np.mean(scores):.4f} std={np.std(scores):.4f}")
    print("== OOF macro-F1 (with null) =", round(macro_f1(y4[used], yp), 4))
    print("== OOF macro-F1 (without null) =", round(macro_f1(y4[used], yp, include_null=False), 4))
    print("per-class:", np.round(per_class_f1(y4[used], yp), 3).tolist())
    out = WORK / args.tag; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "oof.npy", oof); np.save(out / "used.npy", used)
    np.save(out / "test_pred.npy", test_p)
    if args.only_fold is None:
        pd.DataFrame({"id": te_meta["id"], "target_feature": test_p.argmax(1)}).to_csv(
            Path(__file__).parents[2] / "submissions" / f"{args.tag}.csv", index=False)
        print("wrote submission", args.tag)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only-fold", type=int, default=None)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--offset", default="head")
    p.add_argument("--video", action="store_true")
    p.add_argument("--vnorm", action="store_true", help="standardise video features per recording/subject")
    p.add_argument("--rot", type=float, default=0.0, help="max angle of rotation augmentation (degrees)")
    p.add_argument("--aux", action="store_true", help="use hand-crafted statistical/frequency features as auxiliary input")
    p.add_argument("--distill", default=None, help="experiment tag of the 4-sensor teacher")
    p.add_argument("--distill-w", type=float, default=0.5, help="weight of the distillation loss")
    p.add_argument("--distill-t", type=float, default=3.0, help="distillation temperature")
    p.add_argument("--xattn-band", type=int, default=0,
                   help="width restricting cross-attention to temporally close pairs (0=no restriction)")
    p.add_argument("--fusion", default="late", choices=["late", "frame", "xattn"],
                   help="late=concat after pooling (default) / frame=concat after aligning the time axis")
    p.add_argument("--no-inertial", action="store_true",
                   help="drop the inertial branch (when building a pure video model for the graph)")
    p.add_argument("--tag", default="exp010_nn")
    main(p.parse_args())
