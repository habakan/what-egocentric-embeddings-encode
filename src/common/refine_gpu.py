"""The multi-stage refinement of refine.py (knn_graph / propagate / sharpen / gate_blend / drop_top_pcs /
multistage_at) ported to PyTorch (GPU). test_equivalence checks that it returns the same results as the CPU version.

A speed-up because probe evaluations (dozens of similarity matrices, mutual kNN and propagations per subject) were slow due to CPU contention.
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _t(a, dt=torch.float64):
    return torch.as_tensor(np.asarray(a), dtype=dt, device=DEV)


def knn_graph(F, k, mutual=True):
    """F: (n, d) torch. As in the CPU version (center=False), rows are normalized here on every call."""
    n = F.shape[0]
    F = F / (torch.linalg.norm(F, dim=1, keepdim=True) + 1e-8)
    S = F @ F.T
    S.fill_diagonal_(-float("inf"))
    k = min(k, n - 2)
    w, nn = torch.topk(S, k, dim=1)
    if mutual:
        r = torch.zeros((n, n), dtype=torch.bool, device=F.device)
        r.scatter_(1, nn, True)
        keep = r[nn, torch.arange(n, device=F.device)[:, None].expand_as(nn)]
        w = torch.where(keep, w, torch.full_like(w, -1e9))
    return nn, w


def propagate(P, nn, w, alpha, iters, sharp=10.0):
    ww = torch.exp((w - w.max(1, keepdim=True).values) * sharp)
    ww = ww / (ww.sum(1, keepdim=True) + 1e-12)
    Q = P.clone()
    for _ in range(iters):
        Q = (1 - alpha) * P + alpha * (Q[nn] * ww[:, :, None]).sum(1)
    return Q


def sharpen(Q, temp):
    if temp == 1.0:
        return Q
    R = torch.clamp(Q, min=1e-9) ** temp
    return R / R.sum(1, keepdim=True)


def gate_blend(Q, V, g):
    if g == 0:
        return Q
    lg = (1 - g) * torch.log(torch.clamp(Q, min=1e-9)) + g * torch.log(torch.clamp(V, min=1e-9))
    lg = lg - lg.max(1, keepdim=True).values
    R = torch.exp(lg)
    return R / R.sum(1, keepdim=True)


def drop_top_pcs(F, m):
    if m <= 0:
        return F
    c = F - F.mean(0)
    _, _, Vh = torch.linalg.svd(c, full_matrices=False)
    B = Vh[:min(m, Vh.shape[0])]
    return c - (c @ B.T) @ B


def multistage_at(P, F, V, stages, k=30, alpha=0.75, iters=5, temp=1.0, g=0.0, where="none"):
    n_inject = len(stages) + 1 if where == "every" else 1
    gi = g / n_inject if where == "every" else g
    Fn = F - F.mean(0)
    Fn = Fn / (torch.linalg.norm(Fn, dim=1, keepdim=True) + 1e-8)
    nn, w = knn_graph(Fn, k)
    Q = sharpen(propagate(P, nn, w, alpha, iters), temp)
    if where in ("every", "once"):
        Q = gate_blend(Q, V, gi)
    for lam in stages:
        G = torch.cat([Fn, lam * torch.sqrt(Q)], 1)
        nn, w = knn_graph(G, k)
        Q = sharpen(propagate(Q, nn, w, alpha, iters), temp)
        if where == "every":
            Q = gate_blend(Q, V, gi)
    if where == "last":
        Q = gate_blend(Q, V, g)
    return Q


def final_with(P, Fd, Vg, V, sbj, drop, dt=torch.float64):
    """GPU version of probe_decomp.final_with. Inputs and outputs are numpy."""
    Q = np.empty_like(P)
    for s in np.unique(sbj):
        m = sbj == s
        f = _t(Fd[s], dt)
        f = drop_top_pcs(f, drop) if drop else f
        f = f - f.mean(0); f = f / (torch.linalg.norm(f, dim=1, keepdim=True) + 1e-8)
        G = torch.cat([f, 0.5 * torch.sqrt(_t(Vg[s], dt))], 1)
        Q[m] = multistage_at(_t(P[m], dt), G, _t(V[m], dt), [4.0, 4.0], k=30, alpha=0.75, iters=5,
                             temp=1.4, g=0.2, where="every").cpu().numpy()
    return Q


def test_equivalence(seed=42):
    """Whether macro-F1 and probabilities match the CPU probe_decomp.final_with on the same inputs."""
    import time
    from final_submit import load_vgraph_oof
    from postprocess import tune_tau
    from probe_decomp import final_with as final_cpu
    from probe_prf import TAGS, build
    from config import N_CLASSES
    from eval_pipeline import load_eval_set
    out, y, sbj = build(seed); P = out["base"]
    _, _, _, F0 = load_eval_set(TAGS, [1.5, 1, 1, 1, 1, 1, 1], seed=seed)
    Vg = load_vgraph_oof("exp013_nn_video_5fold", TAGS[0], seed=seed)
    src = []
    for tg in ["exp020_video_only", "exp022_vonly_center_5f"]:
        dd = load_vgraph_oof(tg, TAGS[0], seed=seed)
        M = np.empty((len(y), N_CLASSES), np.float32)
        for s in np.unique(sbj): M[sbj == s] = dd[s]
        src.append(M)
    V = np.sqrt(src[0] * src[1]); V /= V.sum(1, keepdims=True)
    t0 = time.time(); Qc = final_cpu(P, F0, Vg, V, sbj, 30); tc = time.time() - t0
    t0 = time.time(); Qg = final_with(P, F0, Vg, V, sbj, 30); tg = time.time() - t0
    fc, fg = tune_tau(Qc, y)[0], tune_tau(Qg, y)[0]
    print(f"CPU {fc:.4f} ({tc:.1f}s)  GPU {fg:.4f} ({tg:.1f}s)  argmax agreement {(Qc.argmax(1) == Qg.argmax(1)).mean():.4f}"
          f"  max abs diff {np.abs(Qc - Qg).max():.2e}  build final {tune_tau(out['final'], y)[0]:.4f}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    test_equivalence()
