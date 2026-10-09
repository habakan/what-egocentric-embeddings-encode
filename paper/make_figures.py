"""Generate the paper's figures from real data.

Principle: **all figures are generated from real data. No schematic drawings.**
Only one conceptual figure, showing the structural correspondence, is allowed.

Computed results are cached in paper/figdata/*.json, so redrawing is instant.
Use --recompute to recompute.

Outputs (paper/figs/):
  fig_position.pdf  dose-response of injection position × total weight  (core of the paper's claim)
  fig_seat.pdf      base seat: standalone accuracy vs pipeline accuracy (the teacher paradox)
  fig_residual.pdf  breakdown of rescues for the 106 residual segments (inertial / video)
  fig_pipeline.pdf  pipeline overview (the only conceptual figure)
"""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted((_pl.Path(__file__).resolve().parents[1] / "src").iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[1] / "data"))
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import CLASS_NAMES, WORK                      # noqa: E402
from eval_pipeline import load_eval_set, run              # noqa: E402
from final_submit import load_vgraph_oof                  # noqa: E402
from metric import macro_f1                               # noqa: E402
from postprocess import tune_tau                          # noqa: E402
from probe_base_ceiling import load_with_tile, seg_table  # noqa: E402
from refine import multistage_at                          # noqa: E402

PREP = WORK / "prep"
FIGD = Path(__file__).parent / "figs"
DATD = Path(__file__).parent / "figdata"
TAGS = ["exp001_lgb_inertial", "exp014_nn_inertial_rot", "exp015_nn_inertial_s1337",
        "exp016_nn_inertial_rot30", "exp017_nn_aux", "exp018_nn_aux_s99",
        "exp019_nn_aux_s555"]
W = [1.5, 1, 1, 1, 1, 1, 1]
VGRAPH = "exp013_nn_video_5fold"
VSRC = ["exp020_video_only", "exp022_vonly_center_5f"]
SEEDS = [42, 7, 123]
STAGES, K, ALPHA, ITERS, TEMP, GAMMA = [4.0, 4.0], 30, 0.75, 5, 1.4, 0.5

# acmart (acmsmall) text width is 395.8pt = 5.48 in. Figures drawn at 6.8-7.0 in for two-column layout are shrunk to the
# text width while keeping font pt (prevents text shrinking when scaled by includegraphics). Small 3.3 in figures are left as is.
TEXTW = 5.48
_orig_figure = plt.figure


def _figure(*args, figsize=None, **kw):
    if figsize is not None and figsize[0] > TEXTW:
        s_ = TEXTW / figsize[0]
        figsize = (TEXTW, figsize[1] * max(s_, 0.9))
    return _orig_figure(*args, figsize=figsize, **kw)


plt.figure = _figure

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C = {"every": "#1f4e79", "last": "#c0504d", "none": "#7f7f7f",
     "inertial": "#c0504d", "video": "#1f4e79", "both": "#7f9db9", "neither": "#d9d9d9"}
# validated categorical palette. Used in a fixed order, never cycled.
# the 8th and later are folded into "Other" (grey). validate_palette.js: all checks PASS,
# worst adjacent CVD ΔE 9.1 (protan). The contrast WARN is covered by legend + direct labels.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
OTHER = "#b8b7b2"
INK = "#0b0b0b"; INK2 = "#52514e"


def cached(name, fn, recompute):
    DATD.mkdir(parents=True, exist_ok=True)
    p = DATD / f"{name}.json"
    if p.exists() and not recompute:
        return json.loads(p.read_text())
    d = fn()
    p.write_text(json.dumps(d, indent=1))
    print(f"  computed -> {p.name}")
    return d


def prep_seed(sd):
    """Assemble the evaluation set, graph features and video sources for one seed."""
    P, y, sbj, F = load_eval_set(TAGS, W, seed=sd)
    Vs = load_vgraph_oof(VGRAPH, TAGS[0], seed=sd)
    for s in F:
        f = F[s] - F[s].mean(0)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        F[s] = np.concatenate([f, GAMMA * np.sqrt(Vs[s])], 1)
    acc = None
    for tg in VSRC:
        d = load_vgraph_oof(tg, TAGS[0], seed=sd)
        M = np.empty((len(y), 19), np.float32)
        for s in np.unique(sbj):
            M[sbj == s] = d[s]
        acc = M if acc is None else acc * M
    V = np.power(acc, 1.0 / len(VSRC))
    V /= V.sum(1, keepdims=True)
    return P, y, sbj, F, V


# ------------------------------------------------------------------ fig 1

def data_position():
    gs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    out = {"g": gs, "every": [], "last": []}
    per_seed = {"every": [[] for _ in gs], "last": [[] for _ in gs]}
    for sd in SEEDS:
        P, y, sbj, F, V = prep_seed(sd)
        for gi, g in enumerate(gs):
            for where in ("every", "last"):
                Q = np.empty_like(P)
                for s in np.unique(sbj):
                    m = sbj == s
                    Q[m] = multistage_at(P[m], F[s], V[m], STAGES, k=K, alpha=ALPHA,
                                         iters=ITERS, temp=TEMP, g=g, where=where)
                per_seed[where][gi].append(tune_tau(Q, y)[0])
        print(f"  seed {sd} done", flush=True)
    for where in ("every", "last"):
        out[where] = {"mean": [float(np.mean(v)) for v in per_seed[where]],
                      "std": [float(np.std(v)) for v in per_seed[where]]}
    return out


def fig_position(d):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    g = np.array(d["g"])
    base = d["every"]["mean"][0]
    ax.axhline(base, color=C["none"], ls=":", lw=1)
    ax.text(0.52, base, "no pooling", color=C["none"], va="bottom", ha="right", fontsize=6.5)
    for where, lbl, mk in [("every", "after every stage", "o"), ("last", "after last stage only", "s")]:
        m = np.array(d[where]["mean"])
        e = np.array(d[where]["std"])
        ax.errorbar(g, m, yerr=e, marker=mk, ms=3.5, lw=1.3, capsize=2,
                    color=C[where], label=lbl)
    ax.set_xlabel(r"total video weight $g$")
    ax.set_ylabel("OOF macro-F1")
    ax.legend(loc="lower left", frameon=False)
    ax.set_xlim(-0.02, 0.52)
    fig.savefig(FIGD / "fig_position.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2

def data_seat():
    P, y, sbj, F, tile, meta = load_with_tile(TAGS, W)
    Vs = load_vgraph_oof(VGRAPH, TAGS[0])
    for s in F:
        f = F[s] - F[s].mean(0)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        F[s] = np.concatenate([f, GAMMA * np.sqrt(Vs[s])], 1)
    T = np.load(WORK / "teacher01" / "oof_win.npy")[tile].astype(np.float32)
    s_ = T.sum(1, keepdims=True)
    T = np.divide(T, s_, out=np.full_like(T, 1 / 19), where=s_ > 1e-9)

    n_all = len(meta)
    S4 = np.zeros((4, len(P), 19), np.float32)
    V4 = np.zeros((4, len(P)), bool)
    for si in range(4):
        o = None
        for t, ww in zip(TAGS, W):
            oo = np.load(WORK / t / "oof.npy")[tile + si * n_all]
            s2 = oo.sum(1, keepdims=True)
            oo = np.divide(oo, s2, out=np.zeros_like(oo), where=s2 > 1e-6)
            o = ww * oo if o is None else o + ww * oo
        V4[si] = o.sum(1) > 1e-6
        S4[si] = o / sum(W)

    def avg_k(kk, seed=0):
        rng = np.random.RandomState(seed)
        out = np.zeros_like(P)
        for i in range(len(P)):
            av = np.where(V4[:, i])[0]
            pick = rng.choice(av, min(kk, len(av)), replace=False)
            out[i] = S4[pick, i].mean(0)
        s2 = out.sum(1, keepdims=True)
        return np.divide(out, s2, out=P.copy(), where=s2 > 1e-6)

    pts = [("student (1 sensor)", P), ("2-sensor avg", avg_k(2)),
           ("3-sensor avg", avg_k(3)), ("4-sensor avg", avg_k(4)),
           ("4-sensor teacher", T)]
    out = []
    for nm, B in pts:
        Q = run(B, y, sbj, F, STAGES, K, ALPHA, ITERS, temp=TEMP)
        out.append({"name": nm, "standalone": float(macro_f1(y, B.argmax(1))),
                    "pipeline": float(tune_tau(Q, y)[0])})
        print(f"  {nm}: {out[-1]}", flush=True)
    return {"points": out}


def fig_seat(d):
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    pts = d["points"]
    avg = [p for p in pts if "teacher" not in p["name"]]
    tea = [p for p in pts if "teacher" in p["name"]]
    ax.plot([p["standalone"] for p in avg], [p["pipeline"] for p in avg],
            "-o", ms=4, lw=1.3, color=C["video"],
            label="student, averaged over $k$ sensors")
    ax.plot([p["standalone"] for p in tea], [p["pipeline"] for p in tea],
            "D", ms=6, color=C["inertial"], label="privileged 4-sensor teacher")
    # label only the endpoints and the teacher (the $k$ legend suffices for the midpoints)
    off = {"student (1 sensor)": (5, -4, "left"),
           "4-sensor avg": (-3, 6, "right"),
           "4-sensor teacher": (-9, 2, "right")}
    for p in pts:
        if p["name"] not in off:
            continue
        dx, dy, ha = off[p["name"]]
        ax.annotate(p["name"], xy=(p["standalone"], p["pipeline"]),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=6.4)
    mid = [p for p in avg if p["name"] in ("2-sensor avg", "3-sensor avg")]
    ax.annotate("$k=2,3$", xy=(mid[0]["standalone"], mid[0]["pipeline"]),
                xytext=(6, -10), textcoords="offset points", ha="left", fontsize=6.4)
    xs = [p["standalone"] for p in pts]; ys = [p["pipeline"] for p in pts]
    ax.set_xlim(min(xs) - 0.018, max(xs) + 0.022)
    ax.set_ylim(min(ys) - 0.010, max(ys) + 0.022)
    # compare the teacher with "what it actually replaces" = the student (consistent with -0.039 in the text)
    tp = tea[0]
    st = [p for p in avg if "student" in p["name"]][0]
    ax.axhline(st["pipeline"], color="#bbbbbb", lw=0.7, ls=":", zorder=0)
    ax.annotate("", xy=(tp["standalone"], tp["pipeline"] + 0.002),
                xytext=(tp["standalone"], st["pipeline"] - 0.002),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#888888"))
    ax.text(tp["standalone"] - 0.003, (tp["pipeline"] + st["pipeline"]) / 2,
            f"$-{st['pipeline'] - tp['pipeline']:.3f}$\nvs. student",
            fontsize=6.3, color="#666666", ha="right", va="center")
    ax.set_xlabel("standalone clip macro-F1")
    ax.set_ylabel("pipeline macro-F1")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(-0.02, 1.02))
    fig.savefig(FIGD / "fig_seat.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 3

def data_residual():
    P, y, sbj, F, tile, meta = load_with_tile(TAGS, W)
    Vs = load_vgraph_oof(VGRAPH, TAGS[0])
    for s in F:
        f = F[s] - F[s].mean(0)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
        F[s] = np.concatenate([f, GAMMA * np.sqrt(Vs[s])], 1)
    Q = run(P, y, sbj, F, STAGES, K, ALPHA, ITERS, temp=TEMP)
    from postprocess import prior_correct
    _, tau = tune_tau(Q, y)
    preds = {"student": prior_correct(Q, tau).argmax(1)}
    T = np.load(WORK / "teacher01" / "oof_win.npy")[tile].astype(np.float32)
    preds["teacher"] = T.argmax(1)
    d = load_vgraph_oof("exp020_video_only", TAGS[0])
    Vm = np.empty((len(y), 19), np.float32)
    for s in np.unique(sbj):
        Vm[sbj == s] = d[s]
    preds["video"] = Vm.argmax(1)

    r = seg_table(meta, tile, y, preds)
    bad = r[r["student"] != r["true"]]
    t_ok = (bad["teacher"] == bad["true"]).to_numpy()
    v_ok = (bad["video"] == bad["true"]).to_numpy()
    per_class = []
    for c in range(19):
        bc = bad[bad["true"] == c]
        if len(bc) < 3:
            continue
        per_class.append({
            "cls": int(c), "n": int(len(bc)),
            "teacher": float((bc["teacher"] == bc["true"]).mean()),
            "video": float((bc["video"] == bc["true"]).mean())})
    return {"n_bad": int(len(bad)), "n_seg": int(len(r)),
            "inertial_only": float((t_ok & ~v_ok).mean()),
            "video_only": float((~t_ok & v_ok).mean()),
            "both": float((t_ok & v_ok).mean()),
            "neither": float((~t_ok & ~v_ok).mean()),
            "per_class": per_class}


CLS = [c.replace("stretching", "stretch").replace("jogging", "jog")
       for c in CLASS_NAMES]


def fig_residual(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5),
                                 gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.42})
    seg = [("inertial teacher only", d["inertial_only"], C["inertial"]),
           ("video-only model only", d["video_only"], C["video"]),
           ("both", d["both"], C["both"]),
           ("neither", d["neither"], C["neither"])]
    left = 0.0
    for lbl, v, col in seg:
        a1.barh([0], [v], left=left, color=col, height=0.42, edgecolor="white", lw=0.8)
        left += v
    a1.set_yticks([])
    a1.set_ylim(-0.55, 0.55)
    a1.set_xlim(0, 1)
    a1.set_xlabel("share of the 106 wrong sets")
    a1.set_title(f"{d['n_bad']} sets predicted entirely wrong\n(of {d["n_seg"]} sets)",
                 fontsize=7, pad=4)
    a1.spines["left"].set_visible(False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in seg]
    a1.legend(hs, [f"{l} ({v:.3f})" for l, v, _ in seg], loc="upper center",
              bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=False, fontsize=6.4,
              handlelength=1.1, handleheight=0.9, labelspacing=0.35)

    pc = sorted(d["per_class"], key=lambda x: -x["n"])
    yy = np.arange(len(pc))
    a2.barh(yy - 0.19, [p["teacher"] for p in pc], height=0.36,
            color=C["inertial"], label="4-sensor inertial teacher")
    a2.barh(yy + 0.19, [p["video"] for p in pc], height=0.36,
            color=C["video"], label="video-only model")
    a2.set_yticks(yy)
    a2.set_yticklabels([f"{CLS[p['cls']]} ({p['n']})" for p in pc], fontsize=6.4)
    a2.invert_yaxis()
    a2.set_xlim(0, 0.88)
    a2.set_xlabel("fraction of that class's wrong sets rescued")
    a2.set_title("by true class (set count in parentheses)", fontsize=7, pad=4)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.savefig(FIGD / "fig_residual.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 0 (setting)

def fig_setting(a):
    """Show the structure of the test data with real label sequences.

    (a) what was actually recorded: all 4 sensors + contiguous activity segments
    (b) what the test contains: only one sensor per second
    (c) what the model sees: the order is lost too
    """
    import pandas as pd
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    rec = a.setting_rec
    g = meta[(meta["rec"] == rec) & (meta["start"] % 50 == 0)].sort_values("start")
    g = g.iloc[a.setting_from: a.setting_from + a.setting_secs]
    y = g["label"].to_numpy()
    n = len(y)

    # colour the most frequent ones in a fixed order, fold the rest into Other
    vc = pd.Series(y).value_counts()
    top = list(vc.index[:len(CAT)])
    col = {c: CAT[i] for i, c in enumerate(top)}
    cols = np.array([col.get(v, OTHER) for v in y])

    rng = np.random.RandomState(0)
    assign = rng.randint(0, 4, n)
    SENS = ["right arm", "right leg", "left leg", "left arm"]

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 2.55),
                             gridspec_kw={"hspace": 0.75})
    for ax, mode, ttl in zip(
            axes, ["all", "one"],
            ["(a) what was recorded: four sensors, continuous exercise sets",
             "(b) what the test set keeps: one random sensor per second, shuffled and unlabelled"]):
        order = rng.permutation(n) if mode == "shuffled" else np.arange(n)
        for s in range(4):
            for j, i in enumerate(order):
                if mode != "all" and assign[i] != s:
                    continue
                ax.add_patch(plt.Rectangle((j, 3 - s + 0.12), 1.0, 0.76,
                                           fc="#a3a29d" if mode == "shuffled" else cols[i], ec="none"))
        ax.set_xlim(0, n); ax.set_ylim(0, 4)
        ax.set_yticks([3.5, 2.5, 1.5, 0.5]); ax.set_yticklabels(SENS, fontsize=6)
        ax.set_title(ttl, fontsize=7.2, pad=3, loc="left", color=INK)
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        if mode == "shuffled":
            ax.set_xticks([0, n]); ax.set_xticklabels(["", ""])
            ax.set_xlabel("clips in arbitrary order", fontsize=6.5, color=INK2)
        else:
            tk = np.arange(0, n + 1, 60)
            ax.set_xticks(tk); ax.set_xticklabels([f"{v}" for v in tk], fontsize=6)
            ax.set_xlabel("time in the recording (s)", fontsize=6.5, color=INK2)

    hs = [plt.Rectangle((0, 0), 1, 1, color=col[c]) for c in top]
    lb = [CLS[c] for c in top]
    if len(vc) > len(CAT):
        hs.append(plt.Rectangle((0, 0), 1, 1, color=OTHER)); lb.append("other")
    axes[-1].legend(hs, lb, loc="upper center", bbox_to_anchor=(0.5, -0.55),
                    ncol=4, frameon=False, fontsize=6.2, handlelength=1.1,
                    title="colour: true activity, not given to the model", title_fontsize=6.2)
    fig.savefig(FIGD / "fig_setting.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 2 (visualizing propagation)

def fig_refine(a):
    """Show what graph refinement actually does with a 2D projection of real data."""
    import pandas as pd
    from postprocess import prior_correct, tune_tau
    from refine import drop_top_pcs, multistage_at

    P, y, sbj, F0 = load_eval_set(TAGS, W, seed=42)
    Vg = load_vgraph_oof(VGRAPH, TAGS[0], seed=42)
    src = []
    for tg in VSRC:
        d = load_vgraph_oof(tg, TAGS[0], seed=42)
        M = np.empty((len(y), 19), np.float32)
        for s in np.unique(sbj):
            M[sbj == s] = d[s]
        src.append(M)
    V = np.power(np.prod(src, 0), 1 / len(src)); V /= V.sum(1, keepdims=True)

    s = a.refine_subject if a.refine_subject in set(sbj) else np.unique(sbj)[0]
    m = sbj == s
    f = drop_top_pcs(F0[s], 30)
    fn = f - f.mean(0); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    G = np.concatenate([fn, GAMMA * np.sqrt(Vg[s])], 1)
    Q = multistage_at(P[m], G, V[m], STAGES, k=K, alpha=ALPHA, iters=ITERS,
                      temp=TEMP, g=0.2, where="every")
    ys = y[m]
    _, tau = tune_tau(Q, ys)
    pred0, pred1 = P[m].argmax(1), prior_correct(Q, tau).argmax(1)

    # the 2D projection uses principal components of the raw video features (for visualization only, separate from the pipeline)
    c = F0[s] - F0[s].mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ Vt[:2].T

    import pandas as pd
    vc = pd.Series(ys).value_counts()
    top = list(vc.index[:len(CAT)])
    col = {cc: CAT[i] for i, cc in enumerate(top)}
    def cmap(v):
        return np.array([col.get(z, OTHER) for z in v])

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.35),
                             gridspec_kw={"wspace": 0.04})
    for ax, (lab, ttl) in zip(axes, [
            (ys, "(a) true activity"),
            (pred0, f"(b) base classifier\nmacro-F1 {macro_f1(ys, pred0):.3f}"),
            (pred1, f"(c) after propagation\nmacro-F1 {macro_f1(ys, pred1):.3f}")]):
        ax.scatter(xy[:, 0], xy[:, 1], s=2.4, c=cmap(lab), lw=0, alpha=0.85)
        ax.set_title(ttl, fontsize=7.2, pad=4, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect("equal", adjustable="datalim")

    hs = [plt.Line2D([], [], marker="o", ls="", ms=4, color=col[cc]) for cc in top]
    lb = [CLS[cc] for cc in top]
    if len(vc) > len(CAT):
        hs.append(plt.Line2D([], [], marker="o", ls="", ms=4, color=OTHER))
        lb.append("other")
    axes[1].legend(hs, lb, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                   ncol=4, frameon=False, fontsize=6.2, handlelength=0.8,
                   columnspacing=1.1)
    fig.text(0.5, -0.26, f"subject {s}; 2-D projection of the video embeddings "
             f"(visualisation only)", ha="center", fontsize=6.2, color=INK2)
    fig.savefig(FIGD / "fig_refine.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 8/9 (WEAR style)

def fig_confusion(a):
    """Confusion matrices (corresponding to Fig.4 of the WEAR paper). Shows what was resolved by refinement."""
    from sklearn.metrics import confusion_matrix
    d = np.load(DATD / "preds.npz")
    y = d["y"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3),
                             gridspec_kw={"wspace": 0.30})
    for ax, key, ttl in [(axes[0], "base", "(a) inertial ensemble"),
                         (axes[1], "final", "(b) final system")]:
        cm = confusion_matrix(y, d[key], labels=list(range(19)), normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{ttl}\nmacro-F1 {macro_f1(y, d[key]):.3f}", fontsize=7.5, pad=4)
        ax.set_xticks(range(19)); ax.set_yticks(range(19))
        ax.set_xticklabels(CLS, rotation=90, fontsize=4.6)
        ax.set_yticklabels(CLS if ax is axes[0] else [""] * 19, fontsize=4.6)
        ax.set_xlabel("predicted", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("true", fontsize=7)
        # outline the null row/column (the most discussed off-diagonal in this field)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 19, 1, fill=False, ec="#c0504d", lw=0.8))
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 19, fill=False, ec="#c0504d", lw=0.8))
    cb = fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02)
    cb.ax.tick_params(labelsize=6); cb.set_label("row-normalised", fontsize=6.5)
    fig.savefig(FIGD / "fig_confusion.pdf")
    plt.close(fig)


def fig_timeline(a):
    """Lay out the ground truth and each stage's predictions on a time axis (corresponding to Fig.5 of the WEAR paper).

    ⚠ Temporal order is used **for visualization only**. The method never uses order.
    The subject is not chosen arbitrarily: take the subject whose final-stage vs refinement difference is closest to the OOF median.
    """
    import pandas as pd
    d = np.load(DATD / "preds.npz")
    y, sbj = d["y"], d["sbj"]
    meta = pd.read_parquet(PREP / "win_meta.parquet")
    rng = np.random.RandomState(42)
    tile = np.where(meta["start"].to_numpy() % 50 == 0)[0]
    valid = np.load(PREP / "valid_mask.npy")
    sens = rng.randint(0, 4, len(tile))
    ok = valid[tile, sens]
    tile2 = tile[ok]
    o = np.load(WORK / TAGS[0] / "oof.npy")[tile2 + sens[ok] * len(meta)]
    tile2 = tile2[o.sum(1) > 1e-6]
    m = meta.iloc[tile2].reset_index(drop=True)
    assert len(m) == len(y), (len(m), len(y))

    # neutral subject selection
    dl = {s: (d["final"][sbj == s] == y[sbj == s]).mean()
             - (d["refine"][sbj == s] == y[sbj == s]).mean() for s in np.unique(sbj)}
    med = np.median(list(dl.values()))
    s = min(dl, key=lambda k: abs(dl[k] - med))
    sel = np.where(sbj == s)[0]
    sel = sel[np.argsort(m["start"].to_numpy()[sel])]

    vc = pd.Series(y[sel]).value_counts()
    top = list(vc.index[:len(CAT)])
    col = {c: CAT[i] for i, c in enumerate(top)}

    rows = [("ground truth", y[sel]), ("inertial ensemble", d["base"][sel]),
            ("+ propagation", d["refine"][sel]), ("+ video pooling", d["inject"][sel]),
            ("final system", d["final"][sel])]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    n = len(sel)
    for r, (nm, v) in enumerate(rows):
        c = [col.get(z, OTHER) for z in v]
        # draw runs of the same colour together (per-window rectangles make the PDF heavy)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and c[j + 1] == c[i]:
                j += 1
            ax.add_patch(plt.Rectangle((i, len(rows) - 1 - r + 0.12), j - i + 1, 0.76,
                                       fc=c[i], ec="none"))
            i = j + 1
        if nm != "ground truth":
            ax.text(n + n * 0.012, len(rows) - 1 - r + 0.5,
                    f"{(v == y[sel]).mean():.2f}", fontsize=6.3, va="center", color=INK2)
    ax.text(n + n * 0.012, len(rows) - 0.5 + 0.12, "acc", fontsize=6.0,
            va="center", color=INK2)
    ax.set_xlim(0, n * 1.05); ax.set_ylim(0, len(rows))
    ax.set_yticks([len(rows) - 1 - r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([nm for nm, _ in rows], fontsize=6.4)
    ax.set_xlabel("time in the recording (s)", fontsize=6.8, color=INK2)
    step = 300 if n > 1200 else 120
    tk = np.arange(0, n + 1, step)
    ax.set_xticks(tk); ax.set_xticklabels([str(v) for v in tk], fontsize=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    hs = [plt.Rectangle((0, 0), 1, 1, color=col[c]) for c in top]
    lb = [CLS[c] for c in top]
    if len(vc) > len(CAT):
        hs.append(plt.Rectangle((0, 0), 1, 1, color=OTHER)); lb.append("other")
    ax.legend(hs, lb, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
              frameon=False, fontsize=6.2, handlelength=1.1)
    fig.savefig(FIGD / "fig_timeline.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 6 (PC-removal anomaly)

def fig_pcdrop(d):
    """The anomaly that discarding the most informative components improves the system.

    (a) the top PCs have the largest variance and activity discriminability = they are not nuisance.
    (b) yet removing them raises the score (interior optimum). At the same time neighbour purity drops monotonically.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5),
                                 gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.36})
    CV, CE = "#7f7f7f", "#2f6f4e"
    # neighbour purity is shown alongside on the (a) side rather than in a separate panel in the same green as eta^2 in (a)
    # (no dual axis)

    i = np.array(d["pc_index"], float)
    a1.plot(i, d["var_ratio"], "-o", ms=3, lw=1.2, color=CV, label="variance explained")
    a1.plot(i, d["act_eta2"], "-s", ms=3, lw=1.4, color=CE,
            label=r"activity $\eta^2$ (discriminability)")
    a1.plot(i, d["time_r2"], "-^", ms=3, lw=1.0, color="#b07d2b",
            label=r"$R^2$ with recording time")
    a1.axvspan(0.7, 30, color="#c0504d", alpha=0.12)
    a1.text(2.4, 0.28, "removed\n(top 30)", fontsize=6.3, color="#c0504d", ha="center")
    a1.set_xscale("log"); a1.set_xlabel("principal component index")
    a1.set_ylim(0, 0.80)
    a1.set_title("(a) the removed components are the\nmost activity-discriminative",
                 fontsize=7.5, pad=5)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=6.2, ncol=1)

    m = np.array(d["prod"]["m"], float)
    f1 = np.array(d["prod"]["f1"]); sd = np.array(d["prod"]["std"])
    a2.errorbar(m, f1, yerr=sd, marker="o", ms=3.5, lw=1.4, capsize=2,
                color="#1f4e79", label="pipeline macro-F1 (OOF)")
    a2.axhline(f1[0], color="#999999", ls=":", lw=1)
    a2.set_xlabel("number of top components removed")
    a2.set_ylabel("pipeline macro-F1")
    k = int(np.argmax(f1))
    a2.annotate(f"best (m={int(m[k])})\n${f1[k] - f1[0]:+.4f}$", xy=(m[k], f1[k]), xytext=(m[k] + 22, f1[k] - 0.0012),
                fontsize=6.4, color="#1f4e79", arrowprops=dict(arrowstyle="-", color="#1f4e79", lw=0.6))

    h1, l1 = a2.get_legend_handles_labels()
    a2.legend(h1, l1, loc="lower left", frameon=False, fontsize=6.2)
    a2.set_title("(b) removing them improves the system\nwhile degrading the graph",
                 fontsize=7.5, pad=5)
    fig.savefig(FIGD / "fig_pcdrop.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 5 (2x2 control)

def fig_2x2(d):
    """Controlled experiment on the mechanism. Left = what the graph groups by, right = the 2x2 effects.

    Colour convention: (a) is coloured by "metric" (green = activity label / amber = sensor location),
    (b) by "modality" (blue = video / red = inertial). Colour meanings are not mixed across panels.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.7),
                                 gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.34})
    G = ["video", "inertial"]
    key = {"video": "video", "inertial": "inertial"}
    CL, CS = "#2f6f4e", "#b07d2b"

    # --- (a) what does the graph group by
    x = np.arange(2)
    lab = [d["purity"][key[g]] for g in G]
    sen = [d["sensor"][key[g]] for g in G]
    a1.bar(x - 0.19, lab, 0.36, color=CL, label="share true activity label")
    a1.bar(x + 0.19, sen, 0.36, color=CS, label="share sensor location")
    a1.axhline(0.25, color="#555555", ls=":", lw=1)
    a1.text(1.52, 0.255, "chance (0.25)", fontsize=6.2,
            color="#555555", ha="right", va="bottom")
    for xi, v in zip(x - 0.19, lab):
        a1.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.6)
    for xi, v in zip(x + 0.19, sen):
        a1.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.6)
    a1.set_xticks(x); a1.set_xticklabels([f"{g}\ngraph" for g in G])
    a1.set_xlim(-0.55, 1.55)
    a1.set_ylim(0, 0.92); a1.set_ylabel(f"fraction of $k$={d['k']} neighbours")
    a1.set_title("(a) what does the graph group by?", fontsize=7.5, pad=5)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=1,
              frameon=False, fontsize=6.5, handlelength=1.1)

    # --- (b) 2x2 effects
    B = ["inertial", "video"]
    col = {"inertial": C["inertial"], "video": C["video"]}
    for i, g in enumerate(G):
        for j, b in enumerate(B):
            base = d["cells"][f"{key[b]}|none"]["mean"]
            v = d["cells"][f"{key[b]}|{key[g]}"]["mean"] - base
            xp = i + (j - 0.5) * 0.38
            a2.bar(xp, v, 0.34, color=col[b],
                   label=f"base: {b}" if i == 0 else None)
            a2.text(xp, v + (0.010 if v > 0 else -0.010),
                    f"{v:+.3f}", ha="center", fontsize=6.6,
                    va="bottom" if v > 0 else "top")
            a2.text(xp, v + (0.037 if v > 0 else -0.032),
                    "same" if b == g else "cross", ha="center", fontsize=6.0,
                    color="#666666", va="bottom" if v > 0 else "top")
    a2.axhline(0, color="#333333", lw=0.9)
    a2.set_xticks(np.arange(2)); a2.set_xticklabels([f"{g}\ngraph" for g in G])
    a2.set_xlim(-0.55, 1.55)
    a2.set_ylim(-0.115, 0.255)
    a2.set_ylabel(r"$\Delta$ macro-F1 vs. no propagation")
    a2.set_title("(b) the graph source sets the sign;\nbase--graph overlap only scales it",
                 fontsize=7.5, pad=5)
    a2.legend(loc="lower left", frameon=False, fontsize=6.6)
    fig.savefig(FIGD / "fig_2x2.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ fig 4 (conceptual figure)

def toy_states():
    """For the schematic: the state at each stage after applying the actual update rules to 12 windows."""
    CA, CB = CAT[1], CAT[2]
    pos = np.array([[0.12, 0.68], [0.28, 0.82], [0.18, 0.40], [0.33, 0.56], [0.27, 0.22], [0.45, 0.74],
                    [0.58, 0.60], [0.73, 0.26], [0.88, 0.50], [0.70, 0.52], [0.82, 0.78], [0.57, 0.33]])
    y = np.array([0] * 6 + [1] * 6)
    limb = np.array([0, 2, 1, 3, 2, 0, 1, 3, 0, 2, 1, 3])
    p0 = np.where(y == 0, 0.85, 0.15).astype(float)
    for i, v in [(3, 0.30), (2, 0.52), (7, 0.66), (8, 0.46), (5, 0.55)]:
        p0[i] = v
    P = np.stack([p0, 1 - p0], 1)
    V = np.stack([np.where(y == 0, 0.65, 0.35), np.where(y == 0, 0.35, 0.65)], 1)
    n = len(y)
    def knn(F, k=3):
        D = ((F[:, None] - F[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, 1)[:, :k]
        r = np.zeros((n, n), bool); r[np.arange(n)[:, None], nn] = True
        return r & r.T
    def prop(Q0, A, alpha=0.75, it=5):
        deg = A.sum(1, keepdims=True); Wm = A / np.maximum(deg, 1)
        Q = Q0.copy()
        for _ in range(it):
            Q = np.where(deg > 0, (1 - alpha) * Q0 + alpha * (Wm @ Q), Q0)
        return Q
    def step(Q, A):
        R = prop(Q, A) ** 1.4; R /= R.sum(1, keepdims=True)
        g = 0.2 / 3; R = R ** (1 - g) * V ** g; return R / R.sum(1, keepdims=True)
    A1 = knn(pos); Q1 = step(P, A1)
    A2 = knn(np.hstack([pos, 0.35 * np.sqrt(Q1)])); Q2 = step(Q1, A2)
    A3 = knn(np.hstack([pos, 0.35 * np.sqrt(Q2)])); Q3 = step(Q2, A3)
    to = plt.matplotlib.colors.to_rgb
    def colour(Q, hard=False):
        base = np.where(Q[:, :1] >= 0.5, np.array(to(CA)), np.array(to(CB)))
        if hard:
            return base
        conf = np.abs(Q[:, 0] - 0.5) * 2
        grey = np.array(to("#d9d9d9"))
        return grey + (base - grey) * np.clip(conf, 0.12, 1)[:, None]
    return dict(pos=pos, limb=limb, P=P, A=[A1, A2, A3], Q=[Q1, Q2, Q3], colour=colour, CA=CA, CB=CB)


def draw_toy(ax, st, A=None, Q=None, hard=False, grey=False, s=14, lw=0.45):
    pos, limb = st["pos"], st["limb"]
    mk = ["o", "s", "^", "D"]
    ax.set_xlim(0.02, 0.98); ax.set_ylim(0.12, 0.90); ax.axis("off")
    if A is not None:
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if A[i, j]:
                    ax.plot(*pos[[i, j]].T, color="#6f6f6f", lw=lw, zorder=1)
    cols = np.tile(np.array([0.85, 0.85, 0.85]), (len(pos), 1)) if grey else st["colour"](Q, hard)
    for i in range(len(pos)):
        ax.scatter(*pos[i], s=s, marker=mk[limb[i]], c=[cols[i]], edgecolors=INK, linewidths=0.35, zorder=3)



def fig_pipeline():
    """Data flow from input to final labels. A schematic of each stage's contents is embedded inside the per-subject box."""
    st = toy_states()
    fig = _orig_figure(figsize=(7.0, 4.25)); ax = fig.add_subplot(111)   # a schematic laid out in coordinates, so do not shrink
    ax.axis("off"); ax.set_xlim(0, 106); ax.set_ylim(0, 66)
    BLUE, GREY = "#1f4e79", "#7f7f7f"
    F_IN, F_MOD, F_TR = "#f4f4f4", "#eaf0f6", "#dce8f2"

    def box(x, y, w, h, title, sub="", fc=F_MOD, ec="#333333", ls="-", tc=INK, fs=6.6, ty=0.70, sy=0.32):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8, ls=ls))
        if sub:
            ax.text(x + w / 2, y + h * ty, title, ha="center", va="center", fontsize=fs, color=tc)
            ax.text(x + w / 2, y + h * sy, sub, ha="center", va="center", fontsize=fs - 1.1,
                    color=INK2 if tc == INK else tc, linespacing=1.15)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, color=tc)

    def arr(x0, y0, x1, y1, col="#333333", lw=0.9, ls="-", cs="arc3"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=col, ls=ls, shrinkA=0, shrinkB=0,
                                    connectionstyle=cs, mutation_scale=7))

    def lab(x, y, t, col=INK2, fs=5.8, ha="left"):
        ax.text(x, y, t, fontsize=fs, color=col, ha=ha, va="center")

    def mini(x, y, w, h, **kw):
        a = ax.inset_axes([x, y, w, h], transform=ax.transData)
        draw_toy(a, st, **kw)
        return a

    for x, t in [(7.5, "input:\none test clip $i$"), (30.5, "per-clip models\n(trained, with labels)"),
                 (67.0, "per test participant:\nall of its clips, no labels"), (99.5, "output")]:
        ax.text(x, 64.3, t, ha="center", va="center", fontsize=6.4, color=INK, linespacing=1.1)

    # inputs and per-window models
    box(0.5, 45, 14, 11, "accelerometer", "one limb, 50$\\times$3\n+ placement", fc=F_IN, fs=5.9)
    box(0.5, 9, 14, 11, "video features", "VideoMAEv2\n15$\\times$768", fc=F_IN, fs=5.9)
    rows = [(49, "inertial ensemble", "LightGBM + 6$\\times$1D-CNN", F_MOD, "$P_i$"),
            (35, "inertial + video network", "both inputs", F_MOD, "$U_i$"),
            (21, "mean over 15 rows", "no training", F_IN, "$f_i$"),
            (7, "video-only models", "2 windowings,\ngeometric mean", "#d6e3f0", "$V_i$")]
    for y, t, sb, fc, nm in rows:
        box(19, y, 23, 9, t, sb, fc=fc, fs=6.0, ty=0.72, sy=0.30)
        lab(42.6, y + 6.3, nm, col=INK, fs=6.4)
    arr(14.5, 50.5, 19, 53.5); arr(14.5, 50.5, 19, 39.5)
    arr(14.5, 14.5, 19, 39.5); arr(14.5, 14.5, 19, 25.5); arr(14.5, 14.5, 19, 11.5)

    # per-subject box
    ax.add_patch(plt.Rectangle((45.5, 0.8), 44.5, 60.5, fc="none", ec=GREY, lw=0.8, ls=(0, (3, 2))))
    # schematic of P (colour = inertial prediction)
    ax.add_patch(plt.Rectangle((47, 47), 15, 12, fc="white", ec="#b8b7b2", lw=0.6))
    mini(47.3, 47.3, 14.4, 9.6, Q=st["P"])
    lab(54.5, 60.2, "$P$: one limb, some wrong", col=INK2, fs=4.8, ha="center")
    arr(42, 53.5, 47, 53.0)
    # graph features G (grey = position is from video)
    ax.add_patch(plt.Rectangle((47, 14), 15, 26, fc=F_TR, ec="#333333", lw=0.8))
    ax.text(54.5, 37.6, "graph features $G$", ha="center", va="center", fontsize=5.9)
    ax.text(54.5, 32.6, "stack $f_i$ into $F$,\nremove top-30 PCs,\ncentre, $\\ell_2$-normalise\n$G=[\\,F\\,\\|\\,0.5\\sqrt{U}\\,]$",
            ha="center", va="center", fontsize=4.9, color=INK2, linespacing=1.2)
    ax.add_patch(plt.Rectangle((47.6, 14.6), 13.8, 13.2, fc="white", ec="#c8c8c8", lw=0.5))
    mini(47.8, 15.0, 13.4, 11.4, grey=True)
    lab(54.5, 27.0, "position = video", col=INK2, fs=4.8, ha="center")
    arr(42, 25.5, 47, 25.5)
    arr(42, 39.5, 50.5, 40.0, cs="arc3,rad=-0.2")

    # stages (left: description, right: schematic = that stage's graph and the predictions after the stage)
    sy = [(44, "stage 1", "link by video\n($k$NN on $G$)\naverage $P$\nsharpen, $\\times V$"),
          (26.5, "stage 2", "relink by\n$[G\\|4\\sqrt{Q}]$\naverage\nsharpen, $\\times V$"),
          (9, "stage 3", "relink by\n$[G\\|4\\sqrt{Q}]$\naverage\nsharpen, $\\times V$")]
    for k_, (y, t, sb) in enumerate(sy):
        ax.add_patch(plt.Rectangle((65, y), 22.5, 15, fc=F_TR, ec="#333333", lw=0.8))
        ax.text(70.5, y + 12.6, t, ha="center", va="center", fontsize=6.4)
        ax.text(70.5, y + 6.0, sb, ha="center", va="center", fontsize=4.8, color=INK2, linespacing=1.35)
        ax.add_patch(plt.Rectangle((76.0, y + 0.8), 11.0, 13.4, fc="white", ec="#c8c8c8", lw=0.5))
        mini(76.2, y + 1.2, 10.6, 12.6, A=st["A"][k_], Q=st["Q"][k_], s=10, lw=0.4)
    arr(62, 53.0, 65, 53.0)
    for y, _, _ in sy:
        arr(62, 27.0, 65, y + 7.5)
    arr(70.5, 44, 70.5, 41.5); lab(71.2, 42.8, "$Q$", col=INK, fs=6.0)
    arr(70.5, 26.5, 70.5, 24.0); lab(71.2, 25.3, "$Q$", col=INK, fs=6.0)
    # V: from the right-hand bus to each stage
    bx = 88.9
    ax.plot([42, 44.2, 44.2, bx, bx], [11.5, 11.5, 2.3, 2.3, 55.0], color=BLUE, lw=0.8)
    for y, _, _ in sy:
        arr(bx, y + 11.0, 87.5, y + 11.0, col=BLUE, lw=0.8)
    lab(47, 3.9, "pool $Q \\propto Q^{1-g/3}\\,V^{g/3}$ after every stage ($g$ = 0.2)", col=BLUE)

    # output
    box(92.5, 30, 13, 12, "prior\ncorrection", "$\\arg\\max_c\\, Q_c/\\pi_c^{\\tau}$", fc=F_MOD, fs=6.4)
    ax.add_patch(plt.Rectangle((92.5, 6.5), 13, 16, fc="#f4f4f4", ec="#333333", lw=0.8))
    ax.text(99.0, 20.3, "label $\\hat{y}_i$", ha="center", va="center", fontsize=6.4)
    mini(93.0, 7.0, 12.0, 11.8, Q=st["Q"][2], hard=True)
    arr(87.5, 16.5, 92.5, 36.0, cs="arc3,rad=0.3")
    arr(99.0, 30, 99.0, 22.5)
    box(92.5, 47, 13, 9, "count band", "Discussion only", fc="white", ec=GREY, ls=(0, (2, 1.5)), tc=GREY, fs=6.0)
    arr(99.0, 47, 99.0, 42, col=GREY, ls=(0, (2, 1.5)))
    # schematic legend
    mkh = [plt.Line2D([], [], marker="o", ls="", ms=3.6, color=st["CA"]),
           plt.Line2D([], [], marker="o", ls="", ms=3.6, color=st["CB"]),
           plt.Line2D([], [], marker="o", ls="", ms=3.6, color="#d9d9d9")]
    mkh += [plt.Line2D([], [], marker=m, ls="", ms=3.2, mfc="white", mec=INK, mew=0.6) for m in ["o", "s", "^", "D"]]
    fig.legend(mkh, ["activity A", "activity B", "unsure", "R wrist", "R ankle", "L ankle", "L wrist"],
               loc="upper center", bbox_to_anchor=(0.55, 0.125), ncol=7, frameon=False, fontsize=5.4,
               handlelength=0.9, columnspacing=1.0)
    fig.text(0.55, 0.085, "white panels: a schematic of 12 clips with the actual update rules applied;  "
             "position = video embedding,  fill = activity prediction,  lines = mutual $k$-NN edges,  "
             "shape = limb of the accelerometer",
             ha="center", va="top", fontsize=5.2, color=INK2)
    fig.savefig(FIGD / "fig_pipeline.pdf")
    plt.close(fig)

def data_headtilt():
    """Ego-Exo4D, 6 participants: linearly decode head tilt (the within-participant max-variance component of the gravity direction, z-scored) from the
    top-30 subspace of the VideoMAEv2 embedding and its complement (take-level GroupKFold, OOF). Returns all windows pooled."""
    import collections, glob
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(DATA_DIR + "/egoexo4d/features/*.npz")):
        z = np.load(f); ok = z["ok"]
        by[int(z["participant"])].append((z["F"].mean(1)[ok], z["pose"][ok]))
    out = {"true": [], "top": [], "rest": []}
    for p, v in by.items():
        F = np.concatenate([a for a, _ in v]); G = np.concatenate([b for _, b in v])[:, :3]
        grp = np.concatenate([[i] * len(a) for i, (a, _) in enumerate(v)])
        F = F - F.mean(0)
        _, _, Vt = np.linalg.svd(F, full_matrices=False)
        j = int(np.argmax(G.std(0))); t = (G[:, j] - G[:, j].mean()) / G[:, j].std()
        # no need to align the sign with the top-30 prediction (it is a regression, so the direction is automatic)
        Xs = {"top": F @ Vt[:30].T, "rest": F - (F @ Vt[:30].T) @ Vt[:30]}
        cv = GroupKFold(min(5, len(v)))
        out["true"] += t.tolist()
        for k, X in Xs.items():
            pr = np.zeros_like(t)
            for tr, te in cv.split(X, t, grp):
                pr[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], t[tr]).predict(X[te])
            out[k] += pr.tolist()
    return out


def fig_teaser_v1(a):
    """Teaser: video selects neighbours, inertial data predicts each window, and predictions propagate along neighbours.

    All real data. In (b), pick one "window the base got wrong and refinement got right"
    and connect its actual k neighbours with lines. Marker shapes show that the neighbours' sensor locations are spread across the four limbs.
    The 2D coordinates are principal components of the raw video features, for visualization only (the pipeline works in 768 dimensions).
    """
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from postprocess import prior_correct, tune_tau
    from probe_invariance import load_all
    from refine import drop_top_pcs, knn_graph, multistage_at

    P, Vg, y, sbj, sens, start, Fv, _, tile = load_all(TAGS, W, VGRAPH, seed=42)
    src = []
    for tg in VSRC:
        d = load_vgraph_oof(tg, TAGS[0], seed=42)
        M = np.empty((len(y), 19), np.float32)
        for s_ in np.unique(sbj):
            M[sbj == s_] = d[s_]
        src.append(M)
    V = np.power(np.prod(src, 0), 1 / len(src)); V /= V.sum(1, keepdims=True)

    s = a.refine_subject if a.refine_subject in set(sbj) else np.unique(sbj)[0]
    m = sbj == s
    f = drop_top_pcs(Fv[s], 30)
    fn = f - f.mean(0); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    G = np.concatenate([fn, GAMMA * np.sqrt(Vg[m])], 1)
    Q = multistage_at(P[m], G, V[m], STAGES, k=K, alpha=ALPHA, iters=ITERS,
                      temp=TEMP, g=0.2, where="every")
    ys, ss = y[m], sens[m]
    _, tau = tune_tau(Q, ys)
    pred0, pred1 = P[m].argmax(1), prior_correct(Q, tau).argmax(1)

    # first-stage graph (same features and same k as the pipeline)
    Gn = G - G.mean(0); Gn /= np.linalg.norm(Gn, axis=1, keepdims=True) + 1e-8
    nn, _ = knn_graph(Gn, K, center=False)

    c = Fv[s] - Fv[s].mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ Vt[:2].T

    vc = pd.Series(ys).value_counts()
    top = list(vc.index[:len(CAT)])
    col = {cc: CAT[i] for i, cc in enumerate(top)}
    cmap = lambda v: np.array([col.get(z, OTHER) for z in v])
    limb_mk = ["o", "s", "^", "D"]                       # in SENSORS order
    limb_nm = ["R wrist", "R ankle", "L ankle", "L wrist"]

    # window to show: base wrong, final correct. True class is a coloured class (not null).
    # neighbours on diverse limbs, majority of neighbours correct, and neighbours reasonably spread in the 2D projection.
    cand = np.where((pred0 != ys) & (pred1 == ys) & (ys != 0) & np.isin(ys, top))[0]
    def score(q):
        nb = nn[q]
        spread = np.ptp(xy[nb], 0).mean()
        return (len(np.unique(ss[nb])), round((pred0[nb] == ys[q]).mean(), 1), spread)
    q = max(cand, key=score)
    nb = nn[q]
    pts = np.vstack([xy[nb], xy[q][None]])
    cx, cy = (pts.max(0) + pts.min(0)) / 2
    half = np.ptp(pts, 0).max() * 0.62 + 1e-6
    zx, zy = (cx - half, cx + half), (cy - half, cy + half)
    # (c)(d) fit the whole cloud into squares of equal size (same box shape so the titles align)
    gx, gy = (xy.max(0) + xy.min(0)) / 2
    R = np.ptp(xy, 0).max() / 2 * 1.03
    fx, fy = (gx - R, gx + R), (gy - R, gy + R)

    # for single-column layout (text width 5.48 in): the two (a) panels on top, (b)(c)(d) below
    fig = plt.figure(figsize=(5.48, 3.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 1.45], hspace=1.25, top=0.87, bottom=0.17)
    gt = gs[0].subgridspec(1, 3, wspace=0.62, width_ratios=[1, 1, 1.25])
    axH, axL, ax0b = fig.add_subplot(gt[0]), fig.add_subplot(gt[1]), fig.add_subplot(gt[2])
    gb = gs[1].subgridspec(1, 3, wspace=0.22)
    ax1, ax2, ax3 = [fig.add_subplot(gb[0, i]) for i in range(3)]
    TS = 6.3
    STAR = r"$\star$"

    # (a) left: measured head orientation explains the top components (egocentric only). middle: image layout explains them (same for WEAR). right: what helps when subtracted
    eg = json.loads((DATD / "encgain.json").read_text())
    xb = np.arange(3); ENC = ["VMAE", "CLIP", "DINOv2"]

    def small_bars(ax, series, ylim, ylabel=None):
        w = 0.8 / len(series)
        for k, (vals, col, lab) in enumerate(series):
            ax.bar(xb + (k - (len(series) - 1) / 2) * w, vals, w, color=col, label=lab)
        ax.set_xticks(xb); ax.set_xticklabels(ENC, fontsize=5.0)
        ax.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1); ax.tick_params(axis="x", length=0, pad=1)
        ax.set_ylim(*ylim); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=5.2, labelpad=1)
        ax.legend(loc="upper left", fontsize=4.8, frameon=False, handlelength=0.8, borderaxespad=0.1, labelspacing=0.2)

    small_bars(axH, [(eg["ego_orient_pc12"], "#1f4e79", "egocentric"), (eg["exo_orient_pc12"], OTHER, "exocentric")],
               (0, 1.15), "$R^2$ of PC1-2\n(within recording)")
    axH.set_title("head orientation\n$\\rightarrow$ leading PCs", fontsize=5.8, pad=2)
    axH.text(0.5, -0.30, "measured tilt, height and heading\nof the head (Ego-Exo4D glasses)",
             transform=axH.transAxes, fontsize=4.6, color=INK2, ha="center", va="top")
    small_bars(axL, [(eg["wear_ego_layout_pc12"], "#2a78d6", "WEAR"), (eg["egoexo_ego_layout_pc12"], "#7fb0e6", "Ego-Exo4D")],
               (0, 1.15))
    axL.set_title("view layout\n$\\rightarrow$ leading PCs (egocentric)", fontsize=5.8, pad=2)
    axL.text(0.5, -0.30, "where sky, ground, buildings and\nown hands appear (segmentation)",
             transform=axL.transAxes, fontsize=4.6, color=INK2, ha="center", va="top")
    fig.text(0.02, 1.005, "(a) WHAT EGOCENTRIC VIDEO EMBEDDINGS ENCODE", fontsize=TS, ha="left", va="bottom", weight="bold")
    fig.text(0.02, 0.975, "where the camera looks (left, centre); the activity's appearance, not the view, "
             "harms the graph (right)", fontsize=TS - 0.5, ha="left", va="bottom")
    w3 = 0.26
    ax0b.bar(xb - w3, eg["sub_view"], w3, color=OTHER, label="the view (layout)")
    ax0b.bar(xb, eg["sub_appearance"], w3, color="#1baf7a", label="activity appearance")
    ax0b.bar(xb + w3, eg["sub_top30"], w3, color="#1f4e79", label="top 30 PCs")
    ax0b.axhline(0, lw=0.5, color=INK)
    ax0b.set_xticks(xb); ax0b.set_xticklabels(ENC, fontsize=5.0)
    ax0b.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1); ax0b.tick_params(axis="x", length=0, pad=1)
    for sp in ("top", "right"):
        ax0b.spines[sp].set_visible(False)
    ax0b.set_ylabel("WEAR macro-F1 change", fontsize=5.2, labelpad=1)
    ax0b.set_title("subtracting a component\nfrom the video features", fontsize=5.8, pad=2)
    ax0b.legend(loc="upper left", fontsize=4.8, frameon=False, handlelength=0.8, borderaxespad=0.1, labelspacing=0.2)
    ax0b.set_ylim(-0.016, 0.042)
    ax0b.text(0.5, -0.30, "activity appearance = the part the\nfour limbs predict (unavailable at test)",
              transform=ax0b.transAxes, fontsize=4.6, color=INK2, ha="center", va="top")

    def vaxes(ax):
        ax.set_xlabel("video PC1", fontsize=5.0, color="#1f4e79", labelpad=1)
        ax.set_ylabel("video PC2", fontsize=5.0, color="#1f4e79", labelpad=1)

    def base(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlim(*fx); ax.set_ylim(*fy)
        ax.set_aspect("equal", adjustable="box")
        vaxes(ax)

    # (b)(c)(d) all share the same windows, positions and lines. Only the colours change.
    # positions are video principal components recomputed from ★ and its neighbours (plus nearby windows) only (a local 2D layout)
    dist = ((fn - fn[q]) ** 2).sum(1)
    local = np.argsort(dist)[:160]
    lc = fn[local] - fn[local].mean(0)
    _, _, Vl = np.linalg.svd(lc, full_matrices=False)
    xy = xy.copy()
    xy_loc = np.full_like(xy, np.nan); xy_loc[local] = lc @ Vl[:2].T
    others = local[~np.isin(local, np.append(nb, q))]
    pts = xy_loc[np.append(nb, q)]
    cx, cy = (pts.max(0) + pts.min(0)) / 2
    half = np.ptp(pts, 0).max() * 0.60 + 1e-6
    zx, zy = (cx - half, cx + half), (cy - half, cy + half)
    xy_full = xy
    xy = xy_loc

    def zoom_panel(ax, colours, star_col, title, tcol, note):
        ax.scatter(xy[others, 0], xy[others, 1], s=3, c="#dcdbd6", lw=0, zorder=1)
        for j in nb:
            ax.plot([xy[q, 0], xy[j, 0]], [xy[q, 1], xy[j, 1]], lw=0.45, color=INK2, alpha=0.55, zorder=2)
        for li in range(4):
            jj = nb[ss[nb] == li]
            ax.scatter(xy[jj, 0], xy[jj, 1], s=24, marker=limb_mk[li],
                       facecolors=colours[jj] if colours is not None else "white",
                       edgecolors=INK, linewidths=0.6, zorder=4)
        ax.scatter(xy[q, 0], xy[q, 1], s=150, marker="*", c=[star_col], edgecolors=INK,
                   linewidths=0.6, zorder=5)
        ax.set_xlim(*zx); ax.set_ylim(*zy)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="box")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#b8b7b2"); sp.set_linewidth(0.6)
        vaxes(ax)
        ax.set_title(title, fontsize=TS, pad=3, color=tcol)
        ax.text(0.5, -0.11, note, transform=ax.transAxes, fontsize=5.1, color=INK2, va="top", ha="center")

    cnt = {limb_nm[li]: int((ss[nb] == li).sum()) for li in range(4)}
    agree = (pred0[nb] == ys[q]).mean()
    zoom_panel(ax1, None, "white",
               "(b) VIDEO $\\rightarrow$ graph $G$\nwho is $\\star$ linked to?", "#1f4e79",
               f"30 video neighbours of {STAR}\n(its limb: {limb_nm[ss[q]]}); theirs:\n"
               + ", ".join(f"{''.join(w[0] for w in k.split())} {v}" for k, v in cnt.items()))
    zoom_panel(ax2, cmap(pred0), cmap([pred0[q]])[0],
               f"(c) INERTIA $\\rightarrow$ $P$\nsame clips; F1 {macro_f1(ys, pred0):.3f}",
               "#c0504d",
               f"{STAR} alone is wrong, but\n{agree:.0%} of its neighbours\npredict its class ({CLS[ys[q]]})")
    zoom_panel(ax3, cmap(pred1), cmap([pred1[q]])[0],
               f"(d) average along links $\\rightarrow$ $Q$\nsame clips; F1 {macro_f1(ys, pred1):.3f}",
               INK,
               f"{STAR} takes its neighbours'\nconsensus and is\nnow correct")
    y_b = ax1.get_position().y1
    fig.text(0.02, y_b + 0.115, "(b)-(d) THE PRINCIPLE", fontsize=TS, ha="left", va="bottom", weight="bold")
    fig.text(0.02, y_b + 0.085, "the video decides which clips belong together; the inertial signal decides "
             "what they are doing", fontsize=TS - 0.5, ha="left", va="bottom")
    # overview in the corner of (b): which region is zoomed
    ins = ax1.inset_axes([0.0, 0.70, 0.30, 0.30])
    ins.scatter(xy_full[:, 0], xy_full[:, 1], s=0.3, c="#b8b7b2", lw=0)
    ins.scatter(xy_full[np.append(nb, q), 0], xy_full[np.append(nb, q), 1], s=1.2, c=INK, lw=0)
    ins.text(0.5, 0.97, "whole participant", transform=ins.transAxes, fontsize=4.0, ha="center", va="top",
             color=INK2)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_linewidth(0.4); sp.set_color("#8a8a8a")

    # legend (activity colour / limb shape)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=3.6, color=col[cc]) for cc in top]
    lb = [CLS[cc] for cc in top]
    if len(vc) > len(CAT):
        hs.append(plt.Line2D([], [], marker="o", ls="", ms=3.6, color=OTHER)); lb.append("other")
    hl = [plt.Line2D([], [], marker=mk, ls="", ms=3.4, mfc="white", mec=INK, mew=0.6)
          for mk in limb_mk]
    fig.legend(hs + hl, lb + limb_nm, loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=4, frameon=False, fontsize=5.6, handlelength=0.9, columnspacing=1.0)
    fig.savefig(FIGD / "fig_teaser.pdf")
    plt.close(fig)
    return dict(subject=int(s), query_limb=limb_nm[ss[q]], neighbour_limbs=cnt,
                agree=float(agree), f1_base=float(macro_f1(ys, pred0)),
                f1_refined=float(macro_f1(ys, pred1)))



def head_icon(fig, x, y, h, pitch, label, left=False):
    """Draw a head seen from the side (facing right) tilted by pitch degrees (up is positive), with the line of sight as an arrow. (x, y) is the lower-left corner, h the height (fraction of the figure)."""
    from matplotlib.patches import Circle, Polygon
    W_, H_ = fig.get_size_inches()
    ax = fig.add_axes([x, y, h * H_ / W_, h]); ax.set_xlim(-0.85, 2.05); ax.set_ylim(-1.45, 1.45); ax.axis("off")
    th = np.radians(pitch)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if left:                                                             # left-facing head (mirrored)
        R = np.diag([-1, 1]) @ R; ax.set_xlim(-2.05, 0.85)
    ax.add_patch(Circle((0, 0), 0.75, fc="white", ec=INK, lw=0.7, zorder=2))
    nose = np.array([[0.68, 0.36], [1.15, 0.05], [0.68, -0.16]]) @ R.T
    ax.add_patch(Polygon(nose, closed=True, fc="white", ec=INK, lw=0.7, zorder=1))
    ax.plot(*(np.array([0.38, 0.28]) @ R.T), "o", ms=1.6, c=INK, zorder=3)
    g0, g1 = np.array([1.12, 0.1]) @ R.T, np.array([2.0, 0.1]) @ R.T
    ax.annotate("", xy=g1, xytext=g0, arrowprops=dict(arrowstyle="-|>", color="#c0504d", lw=0.8, mutation_scale=5), annotation_clip=False)
    fig.text(x + h * H_ / W_ + 0.004, y + h / 2, label, fontsize=4.8, color="#c0504d", va="center")


def fig_teaser(a):
    """Teaser (a single storyline): (a) the egocentric video embedding encodes what the camera is looking at (scatter plot and frames at the axis ends)
    -> (b) so video neighbours are windows of the same set, spread over the four limbs (time axis) -> (c) averaging the neighbours' inertial predictions gets it right.
    All from sbj_0 (recording sbj_0; same as sbj_0 of the public videos), evaluation windows of seed 42. The system uses the paper's settings.
    """
    import pandas as pd
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import Rectangle
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from postprocess import prior_correct, tune_tau
    from probe_invariance import load_all
    from refine import drop_top_pcs, knn_graph, multistage_at

    SUBJ, REC = 0, "sbj_0"
    FR = Path(DATA_DIR + "/wear_frames") / REC
    P, Vg, y, sbj, sens, start, Fv, _, tile = load_all(TAGS, W, VGRAPH, seed=42)
    meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
    rec_all = meta["rec"].to_numpy()[tile]
    src = []
    for tg in VSRC:
        d = load_vgraph_oof(tg, TAGS[0], seed=42)
        M = np.empty((len(y), 19), np.float32)
        for s_ in np.unique(sbj):
            M[sbj == s_] = d[s_]
        src.append(M)
    V = np.power(np.prod(src, 0), 1 / len(src)); V /= V.sum(1, keepdims=True)
    m = sbj == SUBJ
    f = drop_top_pcs(Fv[SUBJ], 30)
    fn = f - f.mean(0); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
    G = np.concatenate([fn, GAMMA * np.sqrt(Vg[m])], 1)
    Q = multistage_at(P[m], G, V[m], STAGES, k=K, alpha=ALPHA, iters=ITERS, temp=TEMP, g=0.2, where="every")
    ys, ss, st, rc = y[m], sens[m], start[m] // 50, rec_all[m]
    _, tau = tune_tau(Q, ys)
    Pm = P[m]
    pred0, pred1 = Pm.argmax(1), prior_correct(Q, tau).argmax(1)
    Gn = G - G.mean(0); Gn /= np.linalg.norm(Gn, axis=1, keepdims=True) + 1e-8
    nn, w = knn_graph(Gn, K, center=False)
    # sets (runs of the same label within a recording)
    setid = np.zeros(len(ys), int); nid = 0
    for r in np.unique(rc):
        ii = np.where(rc == r)[0]; ii = ii[np.argsort(st[ii])]
        for k_, i in enumerate(ii):
            if k_ == 0 or ys[i] != ys[ii[k_ - 1]] or st[i] - st[ii[k_ - 1]] > 5:
                nid += 1
            setid[i] = nid
    limb_mk = ["o", "s", "^", "D"]; limb_nm = ["R wrist", "R ankle", "L ankle", "L wrist"]
    cand = np.where((pred0 != ys) & (pred1 == ys) & (ys != 0) & (rc == REC))[0]
    def score(q):
        nb = nn[q]
        return (len(np.unique(ss[nb])), round((setid[nb] == setid[q]).mean(), 1), round((pred0[nb] == ys[q]).mean(), 1))
    q = max(cand, key=score); nb = nn[q]

    # contrast: ★ under the same propagation on a graph built from inertial features (shown only when it is not correct and clearly worse than video)
    _, _, _, _, _, _, _, Fi_raw, _ = load_all(TAGS, W, VGRAPH, seed=42)
    Fi = np.nan_to_num((Fi_raw[SUBJ] - Fi_raw[SUBJ].mean(0)) / (Fi_raw[SUBJ].std(0) + 1e-6))
    Qin = multistage_at(P[m], Fi, V[m], STAGES, k=K, alpha=ALPHA, iters=ITERS, temp=TEMP, g=0.2, where="every")
    Qi = Qin[q] / Qin[q].sum()
    if int(np.argmax(prior_correct(Qin, tau)[q])) == ys[q] or Qi[ys[q]] > 0.5 * (Q[q][ys[q]] / Q[q].sum()):
        Qi = None
    fig = plt.figure(figsize=(5.48, 3.55))
    TS = 6.3
    GREEN, RED = "#008300", "#c0504d"

    # ---------- (a) scatter plot and frames at the axis ends (windows of recording sbj_0, PCA of the raw video features)
    ia = np.where(rc == REC)[0]
    c = Fv[SUBJ][ia] - Fv[SUBJ][ia].mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    xy = c @ Vt[:2].T
    xy[:, 1] *= -1                                  # PC signs are arbitrary. Align so up = sky (looking up), down = limbs (looking down)
    axa = fig.add_axes([0.175, 0.30, 0.27, 0.40])
    axa.scatter(xy[:, 0], xy[:, 1], s=0.8, c="#b8b7b2", lw=0, rasterized=True)
    axa.set_xticks([]); axa.set_yticks([])
    axa.text(0.99, 0.01, "PC1", transform=axa.transAxes, fontsize=5.4, color="#1f4e79", ha="right", va="bottom")
    axa.text(0.5, -0.035, "PCA of VideoMAEv2 video embeddings (1 s clips, one recording)", transform=axa.transAxes,
             fontsize=4.8, color=INK2, ha="center", va="top")
    axa.text(0.01, 0.99, "PC2", transform=axa.transAxes, fontsize=5.4, color="#1f4e79", ha="left", va="top")
    secs = st[ia]
    # show the linear gradient over PC1-2 of the camera pitch estimated by GeoCalib (up is positive) as an arrow. Length proportional to sqrt(R^2)
    gc = np.load(f"{DATA_DIR}/geocalib/wear/{REC}.npz")
    okg = secs < len(gc["rp"])
    pit = np.degrees(gc["rp"][secs[okg], 1])
    Xg = np.column_stack([xy[okg], np.ones(okg.sum())]); bg, *_ = np.linalg.lstsq(Xg, pit, rcond=None)
    r2g = 1 - ((pit - Xg @ bg) ** 2).sum() / ((pit - pit.mean()) ** 2).sum()
    gdir = bg[:2] / np.linalg.norm(bg[:2])
    span = np.percentile(xy @ gdir, [3, 97]) * np.sqrt(max(r2g, 0))
    ctr = xy.mean(0)
    axa.annotate("", xy=ctr + gdir * span[1], xytext=ctr + gdir * span[0],
                 arrowprops=dict(arrowstyle="->", color="#c0504d", lw=1.1), zorder=4)
    tip = ctr + gdir * span[1]
    axa.text(ctr[0] + gdir[0] * span[1] * 0.3 - 0.03 * np.ptp(xy[:, 0]), ctr[1] + gdir[1] * span[1] * 0.3, f"head tilts up\n(GeoCalib, $R^2$ = {r2g:.2f})", fontsize=4.8,
             color="#c0504d", ha="right", va="center", zorder=4, clip_on=False,
             bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    # head icons on the extension of the arrow: looking down at the tail end, looking up at the tip end
    axa.set_xlim(axa.get_xlim()); axa.set_ylim(axa.get_ylim())
    to_fig = lambda p_: fig.transFigure.inverted().transform(axa.transData.transform(p_))
    IH = 0.08; IW = IH * 3.55 / 5.48
    for end, ext, pitch_, lab, lf in [(span[0], 1.3, -50, "looking\ndown", False), (span[1], 1.45, 55, "looking\nup", True)]:
        a0, a1 = ctr + gdir * end, ctr + gdir * end * ext
        axa.plot([a0[0], a1[0]], [a0[1], a1[1]], ls=(0, (2, 1.5)), c=INK2, lw=0.5, zorder=3, clip_on=False)
        cx, cy = to_fig(ctr + gdir * end * (ext + 0.32))
        head_icon(fig, cx - IW / 2, cy - IH / 2, IH, pitch_, lab, left=lf)
    def ends(k, low):
        o = np.argsort(xy[:, k]); o = o if low else o[::-1]
        pick = []
        for i in o:
            if all(abs(secs[i] - secs[j]) > 60 for j in pick):
                pick.append(i)
            if len(pick) == 2:
                break
        return pick
    TW, TH = 0.128, 0.128 * 5.48 / 3.55 * 9 / 16
    spots = [(ends(0, True), "ground, close", [(0.012, 0.50), (0.012, 0.50 - TH - 0.03)]),
             (ends(0, False), "horizon ahead", [(0.462, 0.50), (0.462, 0.50 - TH - 0.03)]),
             (ends(1, False), "sky", [(0.18, 0.735), (0.18 + TW + 0.012, 0.735)]),
             (ends(1, True), "own limbs", [(0.18, 0.045), (0.18 + TW + 0.012, 0.045)])]
    # in each quadrant, match the positions of the two frames to the order of the points so leader lines do not cross
    # (left/right quadrants: upper box for the upper point; top/bottom quadrants: left box for the left point)
    spots = [(sorted(idx, key=lambda i: -xy[i, 1]) if nm in ("ground, close", "horizon ahead")
              else sorted(idx, key=lambda i: xy[i, 0]), nm, poss) for idx, nm, poss in spots]
    (DATD / "teaser_frames.json").write_text(json.dumps({nm: [int(secs[i]) for i in idx] for idx, nm, _ in spots}))
    for idx, nm, poss in spots:
        axa.scatter(xy[idx, 0], xy[idx, 1], s=9, c=INK, marker="x", lw=0.7, zorder=3)
        (x0, y0) = poss[0]
        if nm in ("own limbs", "sky"):
            fig.text(poss[1][0] + TW + 0.02, y0 + TH / 2, nm, fontsize=5.4, weight="bold", color="#1f4e79", va="center")
        else:
            fig.text(x0 + TW / 2, y0 + TH + 0.022, nm, fontsize=5.4, weight="bold", color="#1f4e79", ha="center", va="bottom")
        for i, (x0, y0) in zip(idx, poss):
            # stack the frames of neighbouring seconds offset to the upper right to show it is video (fainter further back)
            for k_, dd in [(2, 0.010), (1, 0.005)]:
                fb = FR / f"{secs[i] + 2 * k_:05d}.jpg"
                ab = fig.add_axes([x0 + dd, y0 + dd * 1.4, TW, TH])
                ab.imshow(Image.open(fb if fb.exists() else FR / f"{secs[i]:05d}.jpg"), alpha=1 - 0.25 * k_)
                ab.add_patch(Rectangle((0, 0), 1, 1, transform=ab.transAxes, fill=False, ec="white", lw=0.8, clip_on=False))
                ab.axis("off")
            ai = fig.add_axes([x0, y0, TW, TH]); ai.imshow(Image.open(FR / f"{secs[i]:05d}.jpg")); ai.axis("off")
            ai.add_patch(Rectangle((0, 0), 1, 1, transform=ai.transAxes, fill=False, ec="white", lw=0.8, clip_on=False))
            axa.annotate("", xy=(xy[i, 0], xy[i, 1]), xycoords="data", xytext=(x0 + TW / 2, y0 + TH / 2),
                         textcoords="figure fraction", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.35, alpha=0.6))
    fig.text(0.012, 0.965, "(a) THE VIDEO SEES WHERE", fontsize=TS, weight="bold", va="bottom")
    fig.text(0.012, 0.93, "frames at the ends of the leading PCs of\nthe video embedding (one recording)",
             fontsize=5.2, color=INK2, va="top")


    # ---------- (b) three vertical panels: ① link by video → ② predict with each clip's sensor → ③ average along the links
    in_set = (setid[nb] == setid[q]).mean()
    same_rec = rc[nb] == REC
    wrong = pred0[q]
    LIMB_S = {0: "R wrist", 1: "R ankle", 2: "L ankle", 3: "L wrist"}
    LIMB_L = {0: "right wrist", 1: "right ankle", 2: "left ankle", 3: "left wrist"}
    MK = {0: "o", 1: "s", 2: "^", 3: "D"}
    GREY_N = "#c9c8c3"
    def pcol(c):
        return GREEN if c == ys[q] else (RED if c == wrong else "#8c8b86")
    def step(x, y, n):
        fig.text(x, y, n, fontsize=5.6, weight="bold", color="white", ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.18", fc="#1f4e79", ec="none"))
    def down(x, y0, y1):
        fig.add_artist(plt.annotate("", xy=(x, y1), xytext=(x, y0), xycoords="figure fraction",
                                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7)))
    # 2D layout of ★ and its 30 neighbours in the graph features (video with top components removed)
    ids = [q] + list(nb)
    Z = Gn[ids] - Gn[ids].mean(0)
    _, _, Vz = np.linalg.svd(Z, full_matrices=False)
    P2 = Z @ Vz[:2].T
    P2 = (P2 - P2[0]) / np.abs(P2 - P2[0]).max()          # centre on ★
    ACC = np.load(WORK / "prep" / "inertial.npy", mmap_mode="r")
    rows_m = tile[m]
    def accplot(rect, i_, lw=0.5):
        ax_ = fig.add_axes(rect)
        a_ = np.asarray(ACC[rows_m[i_], ss[i_]], np.float32)
        a_ = (a_ - a_.mean(0)) / (np.abs(a_ - a_.mean(0)).max() + 1e-6)
        for d_, col_ in enumerate(["#1f4e79", "#c47a2c", "#5b8c5a"]):
            ax_.plot(a_[:, d_] + (1 - d_) * 0.9, color=col_, lw=lw)
        ax_.set_xlim(0, 49); ax_.set_ylim(-2.1, 2.1); ax_.set_xticks([]); ax_.set_yticks([])
        for sp_ in ax_.spines.values():
            sp_.set_color("#b8b7b2"); sp_.set_linewidth(0.5)
        return ax_
    def graph(rect, colors, star_col, show_shape):
        ax_ = fig.add_axes(rect)
        for k_ in range(1, len(ids)):
            ax_.plot([P2[0, 0], P2[k_, 0]], [P2[0, 1], P2[k_, 1]], color="#9a9994", lw=0.3, zorder=1)
        for k_, i_ in enumerate(ids[1:], 1):
            ax_.scatter(P2[k_, 0], P2[k_, 1], s=9, marker=MK[ss[i_]] if show_shape else "o", c=colors[k_ - 1],
                        edgecolors="white", linewidths=0.3, zorder=2)
        ax_.scatter(P2[0, 0], P2[0, 1], s=70, marker="*", c=star_col, edgecolors=INK, linewidths=0.5, zorder=3)
        ax_.set_xlim(-1.15, 1.15); ax_.set_ylim(-1.15, 1.15); ax_.axis("off")
        return ax_
    GX, GW, GH = 0.635, 0.15, 0.215                          # graph left edge, width, height
    TXX = GX + GW + 0.02                                     # left edge of the description on the right
    fig.text(0.61, 0.965, "(b) VIDEO LINKS THE CLIPS, SENSORS LABEL THEM", fontsize=TS, weight="bold", va="bottom")
    # ① link by video (grey = no label yet)
    y1 = 0.665
    step(0.618, y1 + GH + 0.015, "1")
    fig.text(GX + 0.012, y1 + GH + 0.015, "link $\\star$ to its 30 nearest clips in the video", fontsize=5.2, weight="bold",
             va="center")
    ga = graph([GX, y1, GW, GH], [GREY_N] * len(nb), GREY_N, False)
    # one from each of the four limbs, with photo and leader line
    cand = [i for i in nb if rc[i] == REC]
    ph = []
    for l_ in [3, 0, 2, 1]:
        c_ = [i for i in cand if ss[i] == l_]
        if c_:
            ph.append(min(c_, key=lambda i: abs(st[i] - st[q])))
    PW = 0.06; PH = PW * 5.48 / 3.55 * 9 / 16
    for k_, i in enumerate(ph):
        px = TXX + (k_ % 2) * (PW + 0.012); py = y1 + 0.135 - (k_ // 2) * (PH + 0.03)
        an = fig.add_axes([px, py, PW, PH]); an.imshow(Image.open(FR / f"{st[i]:05d}.jpg")); an.axis("off")
        fig.text(px + PW / 2, py + PH + 0.006, f"{LIMB_S[ss[i]]} clip", fontsize=4.2, ha="center", va="bottom", color=INK)
        kk = ids.index(i)
        ga.annotate("", xy=(P2[kk, 0], P2[kk, 1]), xycoords="data", xytext=(px, py + PH / 2), textcoords="figure fraction",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.35, alpha=0.7))
    fig.text(TXX, y1 + 0.012, f"edges: nearest clips in the video;\n{in_set:.0%} from $\\star$'s own set, on all four limbs",
             fontsize=4.5, color=INK2, va="top", linespacing=1.15)
    down(GX + GW / 2, y1 - 0.005, y1 - 0.045)
    # ② predict from each clip's own acceleration (colour = that prediction, shape = limb)
    y2 = 0.375
    step(0.618, y2 + GH + 0.015, "2")
    fig.text(GX + 0.012, y2 + GH + 0.015, "predict each clip from its own accelerometer", fontsize=5.2, weight="bold",
             va="center")
    graph([GX, y2, GW, GH], [pcol(pred0[i]) for i in nb], RED, True)
    fig.text(TXX, y2 + GH - 0.02, f"$\\star$'s accelerometer ({LIMB_L[ss[q]]} only)", fontsize=4.5, color=INK2, va="bottom")
    accplot([TXX, y2 + 0.11, 0.13, 0.065], q)
    fig.text(TXX, y2 + 0.085, f"$\\star$'s own prediction: {CLS[wrong]} (wrong)", fontsize=4.8, color=RED, va="center")
    for k_, l_ in enumerate([0, 1, 2, 3]):
        lx = TXX + (k_ % 2) * 0.075; ly = y2 + 0.04 - (k_ // 2) * 0.03
        fig.add_artist(plt.Line2D([lx], [ly], marker=MK[l_], color="#555555", ms=3.0, ls="", transform=fig.transFigure))
        fig.text(lx + 0.008, ly, LIMB_L[l_], fontsize=4.3, color=INK2, va="center")
    down(GX + GW / 2, y2 - 0.005, y2 - 0.045)
    # ③ average along the links (★ takes its neighbours' predictions)
    y3 = 0.085
    step(0.618, y3 + GH + 0.015, "3")
    fig.text(GX + 0.012, y3 + GH + 0.015, "average the predictions over the links", fontsize=5.2, weight="bold", va="center")
    graph([GX, y3, GW, GH], [pcol(pred0[i]) for i in nb], GREEN, True)
    fig.text(TXX, y3 + GH * 0.62, f"$\\star$ after averaging:\n{CLS[ys[q]]} (correct)", fontsize=5.0, color=GREEN,
             weight="bold", va="center", linespacing=1.15)
    fig.text(TXX, y3 + GH * 0.28, "colour: each clip's prediction\n(green: true class, red: null)", fontsize=4.5,
             color=INK2, va="center", linespacing=1.15)
    fig.savefig(FIGD / "fig_teaser.pdf")
    plt.close(fig)
    return dict(query_time=int(st[q]), true=CLS[ys[q]], own=CLS[wrong], in_set=float(in_set),
                limbs=len(np.unique(ss[nb])), same_rec=int(same_rec.sum()), agree=float((pred0[nb] == ys[q]).mean()))


def fig_pcends(a):
    """Appendix: frames at both ends of the first and second principal components of the video embedding (VideoMAEv2, centered within recording) of 3 participants (recordings sbj_0 / sbj_3 / sbj_10)
    (2 per end, windows at least 60 s apart). To show that teaser (a) is one participant picked as an example."""
    import pandas as pd
    from PIL import Image
    meta = pd.read_parquet(WORK / "prep" / "win_meta.parquet")
    video = np.load(WORK / "prep" / "video_raw_head.npy", mmap_mode="r")
    recs = ["sbj_0", "sbj_3", "sbj_10"]
    cols = [(0, True, "PC1 low"), (0, False, "PC1 high"), (1, True, "PC2 low"), (1, False, "PC2 high")]
    fig = plt.figure(figsize=(5.48, 2.35))
    TW = 0.117; GAP = 0.005; TH = TW * 5.48 / 2.35 * 9 / 16
    for r_, rec in enumerate(recs):
        t = meta[(meta.start % 50 == 0) & (meta.rec == rec)]
        F = np.asarray(video[t.index.to_numpy()], np.float32).mean(1); F -= F.mean(0)
        _, _, Vt = np.linalg.svd(F, full_matrices=False); Z = F @ Vt[:2].T
        sec = t.start.to_numpy() // 50
        y0 = 0.64 - r_ * (TH + 0.04)
        fig.text(0.0, y0 + TH / 2, f"participant {rec.split('_')[1]}", fontsize=5.6, va="center")
        for c_, (k, low, nm) in enumerate(cols):
            o = np.argsort(Z[:, k]); o = o if low else o[::-1]
            pick = []
            for i in o:
                if all(abs(sec[i] - sec[j]) > 60 for j in pick):
                    pick.append(i)
                if len(pick) == 2:
                    break
            for j, i in enumerate(pick):
                x0 = 0.1 + c_ * (2 * TW + 3 * GAP) + j * (TW + GAP)
                ai = fig.add_axes([x0, y0, TW, TH])
                ai.imshow(Image.open(Path(DATA_DIR + "/wear_frames") / rec / f"{sec[i]:05d}.jpg")); ai.axis("off")
            if r_ == 0:
                fig.text(0.1 + c_ * (2 * TW + 3 * GAP) + TW, y0 + TH + 0.02, nm, fontsize=5.8, ha="center", weight="bold")
    fig.savefig(FIGD / "fig_pcends.pdf")
    plt.close(fig)


def fig_egoexo_view(a):
    """Ego-Exo4D (built like Fig. 1a): the top components of the egocentric embedding follow the measured head tilt; the exocentric ones of the same take do not.
    (a) egocentric view of participant 908 (soccer): VideoMAEv2 PC1/PC2 centered within the take, coloured by measured head tilt, with the tilt gradient
        as an arrow, head icons on its extension, and frames of the windows at both ends (with the frames 2 s and 4 s later stacked behind)
    (b) the exocentric camera embedding of the same take coloured by the same tilt  (c) R^2 for all 6 participants (head tilt ~ PC1+PC2, within take)
    PC signs are arbitrary, so they are aligned so that tilt increases toward the upper right.
    """
    import json, subprocess
    from matplotlib.patches import Rectangle
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from egoexo_tilt_view import load, within
    ROOT = Path(DATA_DIR + "/egoexo4d")
    takes = {t["take_name"]: t for t in json.load(open(ROOT / "takes.json"))}
    ego, exo = load("features"), load("features_exo_videomae")
    P0 = 908
    def fit(Z, tilt):
        X = np.column_stack([Z, np.ones(len(Z))]); b, *_ = np.linalg.lstsq(X, tilt, rcond=None)
        return b, 1 - ((tilt - X @ b) ** 2).sum() / ((tilt - tilt.mean()) ** 2).sum()
    r2 = {p: [fit(*within(by, p)[:2])[1] for by in (ego, exo)] for p in ego}

    def grab(tk, n):
        t = takes[tk]
        vid = sorted((ROOT / t["root_dir"] / "frame_aligned_videos/downscaled/448").glob("aria*_214-1.mp4"))[0]
        buf = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(vid), "-vf",
                              f"select='eq(n\\,{n})',scale=224:224", "-vsync", "0", "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE).stdout
        return np.frombuffer(buf, np.uint8).reshape(224, 224, 3) if len(buf) == 224 * 224 * 3 else None

    fig = plt.figure(figsize=(5.48, 2.6))
    TS = 6.3
    cm = plt.get_cmap("coolwarm")
    Ze, te, take, win = within(ego, P0)
    Zx, tx, _, _ = within(exo, P0)
    lim = np.percentile(np.abs(te), 97)
    FW = 0.085; FH = FW * 5.48 / 2.6
    for k_, (Z, tl, x0, w0) in enumerate([(Ze, te, 0.115, 0.22), (Zx, tx, 0.49, 0.17)]):
        Z = Z.copy()
        b, r2p = fit(Z, tl)
        Z[:, 0] *= np.sign(b[0]) or 1; Z[:, 1] *= np.sign(b[1]) or 1       # tilt increases toward the upper right
        b, r2p = fit(Z, tl)
        ax = fig.add_axes([x0, 0.17, w0, 0.58])
        o = np.argsort(np.abs(tl))
        sc = ax.scatter(Z[o, 0], Z[o, 1], c=tl[o], cmap=cm, vmin=-lim, vmax=lim, s=1.4, lw=0, rasterized=True)
        g = b[:2] / (np.linalg.norm(b[:2]) + 1e-9)
        span = np.percentile(Z @ g, [3, 97]) * np.sqrt(max(r2p, 0))     # arrow length proportional to sqrt(R^2)
        ctr = Z.mean(0)
        ax.annotate("", xy=ctr + g * span[1], xytext=ctr + g * span[0],
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1), zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.text(0.99, 0.01, "PC1", transform=ax.transAxes, fontsize=5.4, color="#1f4e79", ha="right", va="bottom")
        ax.text(0.01, 0.99, "PC2", transform=ax.transAxes, fontsize=5.4, color="#1f4e79", ha="left", va="top")
        ax.text(0.5, -0.04, "PCA of VideoMAEv2 embeddings\n(" + ("egocentric" if k_ == 0 else "exocentric")
                + " camera, participant 908's takes)", transform=ax.transAxes, fontsize=4.6, color=INK2,
                ha="center", va="top")
        ax.text(0.03, 0.06, f"head tilt explains\nPC1-2: $R^2$ = {r2p:.2f}", transform=ax.transAxes, fontsize=5.0,
                color=INK, ha="left", va="bottom")
        if k_ == 1:
            continue
        # head icons on the extension, and frames of the windows at both ends (publication in papers is permitted under Purpose (1) of the Ego-Exo4D licence)
        ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())
        to_fig = lambda p_: fig.transFigure.inverted().transform(ax.transData.transform(p_))
        IH = 0.10; IW = IH * 2.6 / 5.48
        proj = Z @ g
        for end, pitch_, lab, lf, i, fx in [(span[0], -50, "looking\ndown", False, int(np.argmin(proj)), 0.012),
                                            (span[1], 55, "looking\nup", True, int(np.argmax(proj)), x0 + w0 + 0.012)]:
            ax.scatter(Z[i, 0], Z[i, 1], s=12, marker="x", c=INK, lw=0.8, zorder=5)
            n = int(win[i]) * 30 + 15
            fy = to_fig(ctr)[1] - FH / 2
            head_icon(fig, fx + FW / 2 - IW / 2, fy + FH + 0.03, IH, pitch_, "", left=lf)
            for kk, dd in [(2, 0.010), (1, 0.005), (0, 0.0)]:
                img = grab(take[i], n + 60 * kk) if kk else grab(take[i], n)
                if img is None:
                    img = grab(take[i], n)
                ai = fig.add_axes([fx + dd, fy + dd * 1.4, FW, FH])
                ai.imshow(img, alpha=1 - 0.25 * kk); ai.axis("off")
                ai.add_patch(Rectangle((0, 0), 1, 1, transform=ai.transAxes, fill=False, ec="white", lw=0.8,
                                       clip_on=False))
            fig.text(fx + FW / 2, fy - 0.02, lab.replace("\n", " "), fontsize=5.2, weight="bold", color="#1f4e79",
                     ha="center", va="top")
            px, py = to_fig(Z[i])
            fig.add_artist(plt.Line2D([px, fx + FW / 2], [py, fy + FH / 2], color=INK2, lw=0.35, alpha=0.6))
    cax = fig.add_axes([0.675, 0.17, 0.008, 0.58])
    cb = fig.colorbar(sc, cax=cax); cb.ax.tick_params(labelsize=4.8, length=1.5)
    cb.set_label("measured head tilt (deg, up +)", fontsize=4.8, labelpad=0.5)
    axc = fig.add_axes([0.815, 0.17, 0.175, 0.58])
    ps = sorted(r2, key=lambda p: -r2[p][0]); xb = np.arange(len(ps))
    axc.bar(xb - 0.2, [r2[p][0] for p in ps], 0.4, color="#1f4e79", label="egocentric")
    axc.bar(xb + 0.2, [max(r2[p][1], 0) for p in ps], 0.4, color=OTHER, label="exocentric")
    dom = {356: "basket", 383: "basket", 908: "soccer", 909: "soccer", 519: "dance", 520: "dance"}
    axc.set_xticks(xb); axc.set_xticklabels([f"{p}\n{dom[p][:3]}" for p in ps], fontsize=4.2)
    axc.tick_params(axis="y", labelsize=4.8, length=1.5, pad=1); axc.tick_params(axis="x", length=0, pad=1)
    axc.set_ylim(0, 1.05)
    for sp_ in ("top", "right"):
        axc.spines[sp_].set_visible(False)
    axc.set_ylabel("$R^2$ (head tilt ~ PC1-2)", fontsize=5.2, labelpad=1)
    axc.legend(fontsize=4.8, frameon=False, loc="upper right", handlelength=0.8)
    fig.text(0.012, 0.975, "(a) EGOCENTRIC: THE HEAD SETS PC1-2", fontsize=TS, weight="bold", va="top")
    fig.text(0.012, 0.915, "one participant (soccer); colour = measured head tilt", fontsize=5.2, color=INK2, va="top")
    fig.text(0.47, 0.975, "(b) EXOCENTRIC", fontsize=TS, weight="bold", va="top")
    fig.text(0.47, 0.915, "same takes, a static camera:\nhead tilt does not set PC1-2", fontsize=5.2, color=INK2, va="top")
    fig.text(0.765, 0.975, "(c) ALL SIX", fontsize=TS, weight="bold", va="top")
    fig.text(0.765, 0.915, "participants", fontsize=5.2, color=INK2, va="top")
    fig.savefig(FIGD / "fig_egoexo_view.pdf")
    plt.close(fig)
    return {p: [round(v, 3) for v in r2[p]] for p in r2}

def fig_causal(a):
    """For the introduction: how the data arise (left) and how our method traces it backwards (right).
    Observable nodes contain real data: limb motion = 1 s of WEAR acceleration (only one of the four limbs coloured),
    video "appearance" and "scenery" = the same frames as Fig. 1 (a) (own limbs / horizon ahead). Unobserved nodes are text only."""
    import pandas as pd
    from matplotlib.patches import FancyBboxPatch
    from PIL import Image
    ROOTD = Path(__file__).resolve().parents[1]
    fr = json.loads((DATD / "teaser_frames.json").read_text())
    FRD = Path(DATA_DIR + "/wear_frames/sbj_0")
    img_app = Image.open(FRD / f"{fr['own limbs'][0]:05d}.jpg")
    img_view = Image.open(FRD / f"{fr['horizon ahead'][0]:05d}.jpg")
    acc = pd.read_csv(ROOTD / "input/train/inertial_feat/sbj_0.csv", nrows=50 * 40)
    t0 = 50 * 20                                                      # 1 s of jogging
    limbs = [("L wrist", "left_arm"), ("L ankle", "left_leg"), ("R ankle", "right_leg"), ("R wrist", "right_arm")]
    sig = {nm: acc[[f"{c}_acc_{k}" for k in "xyz"]].to_numpy()[t0:t0 + 50] for nm, c in limbs}

    fig = plt.figure(figsize=(5.48, 2.45))
    BLUE, GREEN, RED, GREY = "#1f4e79", "#008300", "#c0504d", "#e9e8e4"

    def box(ax, x, y, w, h, title, sub="", fc="white", ec=INK, tc=INK, lw=0.7, ls="-", top=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=fc, ec=ec, lw=lw, ls=ls))
        ty = (y + h / 2 - 0.035) if (sub or top) else y
        ax.text(x, ty, title, fontsize=5.8, weight="bold", ha="center", va="center", color=tc)
        if sub:
            ax.text(x, ty - 0.034, sub, fontsize=4.5, ha="center", va="top", color=INK2, linespacing=1.1)

    def arr(ax, p0, p1, c=INK, lw=0.8):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, mutation_scale=6,
                                                           shrinkA=0, shrinkB=0))

    def trace(ax, x0, y0, w, h):
        """1 s of acceleration for the four limbs. Only the worn limb is coloured, the others faint (not given)."""
        ia = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
        for k, (nm, _) in enumerate(limbs):
            v = np.linalg.norm(sig[nm], axis=1); v = (v - v.mean()) / (np.ptp(v) + 1e-6)
            on = nm == "R ankle"
            ia.plot(np.arange(50), 3 - k + 0.8 * v, color=RED if on else "#c8c7c2", lw=0.7 if on else 0.5)
            ia.text(-2, 3 - k, nm, fontsize=3.8, ha="right", va="center", color=INK if on else "#a09f9a")
        ia.set_xlim(0, 49); ia.set_ylim(-0.7, 3.7); ia.axis("off")

    def image(ax, im, x0, y0, w, h, faded=False):
        ia = ax.inset_axes([x0, y0, w, h], transform=ax.transData)
        ia.imshow(im, alpha=0.35 if faded else 1.0, aspect="auto"); ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_linewidth(0.4); sp.set_color("#888888")
        if faded:
            ia.plot([0, 1], [0, 1], transform=ia.transAxes, color=RED, lw=0.9)
            ia.plot([0, 1], [1, 0], transform=ia.transAxes, color=RED, lw=0.9)

    def video_block(ax, faded):
        ax.add_patch(FancyBboxPatch((0.46, 0.035), 0.525, 0.46, boxstyle="round,pad=0.008,rounding_size=0.015",
                                    fc=GREY, ec=INK, lw=0.7))
        ax.text(0.7225, 0.46, "egocentric video  $v$", fontsize=5.8, weight="bold", ha="center", va="center")
        box(ax, 0.595, 0.215, 0.245, 0.33, "appearance", tc=RED, top=True,
            ls=(0, (2, 1.5)) if faded else "-", ec=RED if faded else INK)
        box(ax, 0.855, 0.215, 0.245, 0.33, "view", tc=BLUE, top=True)
        image(ax, img_app, 0.485, 0.085, 0.22, 0.2, faded=faded)
        image(ax, img_view, 0.745, 0.085, 0.22, 0.2)

    def limb_block(ax):
        box(ax, 0.215, 0.265, 0.40, 0.46, "limb motion  $x$", fc=GREY, top=True)
        trace(ax, 0.10, 0.07, 0.28, 0.30)

    # ---------- left: how the data arise
    ax = fig.add_axes([0.0, 0.0, 0.5, 0.9]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.01, 0.975, "(a) HOW THE DATA ARISE", fontsize=6.3, weight="bold", va="top")
    box(ax, 0.50, 0.92, 0.62, 0.13, "exercise set  $s$", "one exercise at one place for tens of seconds")
    box(ax, 0.215, 0.69, 0.34, 0.13, "activity  $y$", "the exercise performed")
    box(ax, 0.78, 0.69, 0.40, 0.13, "circumstances  $c$", "of the set: place, posture, facing")
    limb_block(ax); video_block(ax, faded=False)
    arr(ax, (0.40, 0.855), (0.27, 0.755)); arr(ax, (0.60, 0.855), (0.73, 0.755))
    arr(ax, (0.215, 0.625), (0.215, 0.495)); arr(ax, (0.30, 0.625), (0.55, 0.38), c=RED)
    arr(ax, (0.90, 0.625), (0.90, 0.38), c=BLUE)
    ax.text(0.02, -0.035, "grey = observed per test clip; $s$, $y$, $c$ hidden", fontsize=4.6, color=INK2)

    # ---------- right: our method traces it backwards
    ax = fig.add_axes([0.5, 0.0, 0.5, 0.9]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.51, 0.975, "(b) HOW THE METHOD INVERTS IT", fontsize=6.3, weight="bold", va="top")
    limb_block(ax); video_block(ax, faded=True)
    box(ax, 0.215, 0.69, 0.36, 0.13, "what: $P$", "inertial classifier", ec=GREEN, tc=GREEN)
    box(ax, 0.78, 0.69, 0.40, 0.13, "same set", "video neighbours ($k$-NN graph)", ec=BLUE, tc=BLUE)
    box(ax, 0.50, 0.915, 0.62, 0.15, "label per set", "propagate $P$ over the graph:\nthe set's four limbs are pooled")
    arr(ax, (0.215, 0.495), (0.215, 0.625), c=GREEN); arr(ax, (0.90, 0.38), (0.90, 0.625), c=BLUE)
    arr(ax, (0.27, 0.755), (0.38, 0.84), c=GREEN); arr(ax, (0.73, 0.755), (0.62, 0.84), c=BLUE)
    ax.text(0.595, 0.303, "removed: top PCs", fontsize=4.4, color=RED, ha="center", va="center")
    ax.text(0.02, -0.035, "the video links clips of the same scene; the inertial signal says what they do",
            fontsize=4.6, color=INK2)
    fig.savefig(FIGD / "fig_causal.pdf")
    plt.close(fig)

def main(a):
    FIGD.mkdir(parents=True, exist_ok=True)
    print("fig_pipeline (concept)")
    fig_pipeline()
    if (DATD / "preds.npz").exists():
        for nm, fn2 in [("fig_confusion", fig_confusion), ("fig_timeline", fig_timeline)]:
            if a.only and nm.replace("fig_", "") not in a.only:
                continue
            print(nm); fn2(a)
    if not a.only or "teaser" in a.only:
        print("fig_teaser"); print("  ", fig_teaser(a))
    if not a.only or "pcends" in a.only:
        print("fig_pcends"); fig_pcends(a)
    if not a.only or "causal" in a.only:
        print("fig_causal"); fig_causal(a)
    if not a.only or "egoexo" in a.only:
        print("fig_egoexo_view"); print("  ", fig_egoexo_view(a))
    for nm, fn2 in [("fig_setting", fig_setting), ("fig_refine", fig_refine)]:
        if a.only and nm.replace("fig_", "") not in a.only:
            continue
        print(nm); fn2(a)
    pc = DATD / "pcdrop.json"
    if pc.exists():
        print("fig_pcdrop")
        fig_pcdrop(json.loads(pc.read_text()))
    tx = DATD / "twoxtwo.json"
    if tx.exists():
        print("fig_2x2")
        fig_2x2(json.loads(tx.read_text()))
    else:
        print("fig_2x2: figdata/twoxtwo.json not found. Run src/probe_2x2.py first")
    for nm, dfn, pfn in [("position", data_position, fig_position),
                         ("seat", data_seat, fig_seat),
                         ("residual", data_residual, fig_residual)]:
        if a.only and nm not in a.only:
            continue
        print(f"fig_{nm}")
        pfn(cached(nm, dfn, a.recompute))
    print("\nwrote:", sorted(p.name for p in FIGD.glob("*.pdf")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--setting-rec", default="sbj_0")
    p.add_argument("--setting-from", type=int, default=780)
    p.add_argument("--setting-secs", type=int, default=240)
    p.add_argument("--refine-subject", type=int, default=3)
    p.add_argument("--timeline-subject", type=int, default=0)
    p.add_argument("--timeline-from", type=int, default=780)
    p.add_argument("--timeline-secs", type=int, default=240)
    main(p.parse_args())


def fig_stages():
    """Schematic of one stage's contents (colours are set by actually computing propagation on 12 windows).

    position = video, fill = activity prediction (inertial P / post-propagation Q, grey = uncertain), shape = accelerometer limb,
    small blue-outlined square = vote of the video-only model V.
    """
    CA, CB = CAT[1], CAT[2]                       # jog (orange) / sit-ups (green)
    pos = np.array([[0.12, 0.68], [0.28, 0.82], [0.18, 0.40], [0.33, 0.56], [0.27, 0.22], [0.45, 0.74],
                    [0.58, 0.60], [0.73, 0.26], [0.88, 0.50], [0.70, 0.52], [0.82, 0.78], [0.57, 0.33]])
    y = np.array([0] * 6 + [1] * 6)
    limb = np.array([0, 2, 1, 3, 2, 0, 1, 3, 0, 2, 1, 3])
    p0 = np.where(y == 0, 0.85, 0.15).astype(float)
    for i, v in [(3, 0.30), (2, 0.52), (7, 0.66), (8, 0.46), (5, 0.55)]:
        p0[i] = v                                  # only one limb, so some windows are wrong or uncertain
    P = np.stack([p0, 1 - p0], 1)
    V = np.stack([np.where(y == 0, 0.65, 0.35), np.where(y == 0, 0.35, 0.65)], 1)
    n = len(y)

    def knn(F, k=3):
        D = ((F[:, None] - F[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, 1)[:, :k]
        r = np.zeros((n, n), bool); r[np.arange(n)[:, None], nn] = True
        return r & r.T                             # mutual kNN
    def prop(Q0, A, alpha=0.75, it=5):
        deg = A.sum(1, keepdims=True); Wm = A / np.maximum(deg, 1)
        Q = Q0.copy()
        for _ in range(it):
            Q = np.where(deg > 0, (1 - alpha) * Q0 + alpha * (Wm @ Q), Q0)
        return Q
    def sharpen(Q, t=1.4):
        R = Q ** t; return R / R.sum(1, keepdims=True)
    def pool(Q, g=0.2 / 3):
        R = Q ** (1 - g) * V ** g; return R / R.sum(1, keepdims=True)

    A1 = knn(pos)
    Qb = prop(P, A1); Q1 = pool(sharpen(Qb))
    A2 = knn(np.hstack([pos, 0.35 * np.sqrt(Q1)]))
    Q2 = pool(sharpen(prop(Q1, A2)))

    to = plt.matplotlib.colors.to_rgb
    def colour(Q):
        conf = np.abs(Q[:, 0] - 0.5) * 2
        base = np.where(Q[:, :1] >= 0.5, np.array(to(CA)), np.array(to(CB)))
        grey = np.array(to("#d9d9d9"))
        return grey + (base - grey) * np.clip(conf, 0.12, 1)[:, None]

    mk = ["o", "s", "^", "D"]
    fig, axs = plt.subplots(1, 5, figsize=(7.0, 1.9))
    fig.subplots_adjust(wspace=0.18, left=0.01, right=0.99, top=0.80, bottom=0.12)
    titles = ["(a) input\n$P$ from the inertial signal", "(b) stage 1\nlink by video, average",
              "(c) sharpen, then\ntimes video vote $V$", "(d) stages 2–3\nrelink by [video $\\|$ $Q$]",
              "(e) output\nlabel"]
    notes = ["one limb: some wrong\nor unsure", "$Q_i \\leftarrow 0.25P_i+0.75\\,\\mathrm{mean}_{\\mathrm{nbrs}}Q$",
             "$Q \\propto (Q^{1.4})^{1-g}\\,V^{g}$", "same-$Q$ clips\nnow link", "$\\arg\\max_c Q_c/\\pi_c^{\\tau}$"]
    states = [(P, None, False), (Qb, A1, False), (Q1, A1, True), (Q2, A2, False), (Q2, None, False)]
    for k_, (ax, t, nt, (Q, A, showv)) in enumerate(zip(axs, titles, notes, states)):
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.08, 0.95); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(plt.Rectangle((0.0, 0.08), 1.0, 0.87, fc="#fafafa", ec="#d0d0d0", lw=0.5, zorder=0))
        if A is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    if A[i, j]:
                        ax.plot(*pos[[i, j]].T, color="#6f6f6f", lw=0.8, zorder=1)
        cols = colour(Q) if k_ < 4 else np.where(Q[:, :1] >= 0.5, np.array(to(CA)), np.array(to(CB)))
        for i in range(n):
            ax.scatter(*pos[i], s=62, marker=mk[limb[i]], c=[cols[i]], edgecolors=INK, linewidths=0.55, zorder=3)
            if showv:
                ax.scatter(pos[i, 0] + 0.06, pos[i, 1] + 0.065, s=10, marker="s",
                           c=[CA if V[i, 0] > 0.5 else CB], edgecolors="#1f4e79", linewidths=0.7, zorder=4)
        ax.set_title(t, fontsize=6.3, pad=2, linespacing=1.1)
        ax.text(0.5, 0.06, nt, transform=ax.transAxes, ha="center", va="top", fontsize=5.5, color=INK2,
                linespacing=1.1)
    for i in range(4):
        x0 = axs[i].get_position().x1; x1 = axs[i + 1].get_position().x0
        fig.text((x0 + x1) / 2, 0.47, "$\\rightarrow$", ha="center", va="center", fontsize=9, color=INK2)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=4, color=CA), plt.Line2D([], [], marker="o", ls="", ms=4, color=CB),
          plt.Line2D([], [], marker="o", ls="", ms=4, color="#d9d9d9"),
          plt.Line2D([], [], marker="s", ls="", ms=3.0, color=CB, mec="#1f4e79", mew=0.8)]
    hs += [plt.Line2D([], [], marker=m, ls="", ms=3.6, mfc="white", mec=INK, mew=0.6) for m in mk]
    lb = ["jog", "sit-ups", "unsure", "video-only vote $V$", "R wrist", "R ankle", "L ankle", "L wrist"]
    fig.legend(hs, lb, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=8, frameon=False, fontsize=5.6,
               handlelength=0.9, columnspacing=1.0)
    fig.text(0.5, -0.13, "schematic of one test participant: position = video embedding;  lines = mutual $k$-NN edges;  "
             "fill = activity prediction;  shape = limb of the clip's accelerometer",
             ha="center", va="top", fontsize=5.4, color=INK2)
    fig.savefig(FIGD / "fig_stages.pdf")
    plt.close(fig)
