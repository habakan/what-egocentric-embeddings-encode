"""Statistical/frequency features from 1 s windows (50, 3). Computed vectorized over all windows at once."""
import os as _os, sys as _sys, pathlib as _pl  # noqa: E401  add folders under src/ to the import search path
_sys.path[:0] = [str(_p) for _p in sorted(_pl.Path(__file__).resolve().parents[1].iterdir()) if _p.is_dir()]
DATA_DIR = _os.environ.get("WEAR_DATA", str(_pl.Path(__file__).resolve().parents[2] / "data"))
import numpy as np


def window_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """x: (n, 50, 3) float32 -> (n, d) float32"""
    x = np.asarray(x, np.float32)
    n, t, c = x.shape
    feats, names = [], []

    def add(arr, nm):
        arr = np.asarray(arr, np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        feats.append(arr)
        names.extend(nm if isinstance(nm, list) else [nm])

    ax = ["x", "y", "z"]
    d1 = np.diff(x, axis=1)

    for f, nm in [(x.mean(1), "mean"), (x.std(1), "std"), (x.min(1), "min"),
                  (x.max(1), "max"), (np.median(x, 1), "med"),
                  (np.percentile(x, 25, axis=1), "q25"),
                  (np.percentile(x, 75, axis=1), "q75"),
                  (np.abs(d1).mean(1), "absdiff"), (d1.std(1), "diffstd"),
                  (np.sqrt((x ** 2).mean(1)), "rms")]:
        add(f, [f"{nm}_{a}" for a in ax])
    add(x.max(1) - x.min(1), [f"ptp_{a}" for a in ax])
    # 3rd/4th moments after subtracting the mean (skewness, kurtosis)
    xc = x - x.mean(1, keepdims=True)
    sd = x.std(1) + 1e-6
    add((xc ** 3).mean(1) / sd ** 3, [f"skew_{a}" for a in ax])
    add((xc ** 4).mean(1) / sd ** 4, [f"kurt_{a}" for a in ax])
    # zero-crossing rate
    add((np.diff(np.sign(xc), axis=1) != 0).mean(1), [f"zcr_{a}" for a in ax])

    # magnitude vector (orientation-invariant)
    m = np.linalg.norm(x, axis=2)
    add(m.mean(1), "mag_mean"); add(m.std(1), "mag_std")
    add(m.min(1), "mag_min"); add(m.max(1), "mag_max")
    add(m.max(1) - m.min(1), "mag_ptp")
    add(np.abs(np.diff(m, axis=1)).mean(1), "mag_absdiff")
    add((m ** 2).mean(1), "mag_energy")

    # inter-axis correlation
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a, b = xc[:, :, i], xc[:, :, j]
        add((a * b).mean(1) / (a.std(1) * b.std(1) + 1e-6), f"corr_{ax[i]}{ax[j]}")

    # frequency (50Hz, 50 points -> resolution 1Hz, usable 1..25Hz)
    sp = np.abs(np.fft.rfft(xc, axis=1))[:, 1:, :]              # (n, 25, 3)
    p = sp ** 2
    tot = p.sum(1) + 1e-8
    add(np.argmax(p, axis=1).astype(np.float32) + 1, [f"domfreq_{a}" for a in ax])
    add(p.max(1) / tot, [f"dompow_{a}" for a in ax])
    freqs = np.arange(1, sp.shape[1] + 1, dtype=np.float32)[None, :, None]
    add((p * freqs).sum(1) / tot, [f"centroid_{a}" for a in ax])
    pn = p / tot[:, None, :]
    add(-(pn * np.log(pn + 1e-12)).sum(1), [f"specent_{a}" for a in ax])
    for lo, hi in [(0, 3), (3, 6), (6, 12), (12, 25)]:
        add(p[:, lo:hi, :].sum(1) / tot, [f"band{lo}_{hi}_{a}" for a in ax])

    spm = np.abs(np.fft.rfft(m - m.mean(1, keepdims=True), axis=1))[:, 1:] ** 2
    tm = spm.sum(1) + 1e-8
    add(np.argmax(spm, 1).astype(np.float32) + 1, "mag_domfreq")
    add(spm.max(1) / tm, "mag_dompow")

    out = np.concatenate(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out), names
