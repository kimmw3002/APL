import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Per-row degree-1 leveling heatmaps from a CSV.")
parser.add_argument("csv", help="Path to the input CSV (semicolon-delimited).")
args = parser.parse_args()

SRC = args.csv
base = os.path.splitext(os.path.basename(SRC))[0]
OUT = os.path.join(os.path.dirname(os.path.abspath(SRC)), base + "_leveled.png")

data = np.loadtxt(SRC, delimiter=";")
N, M = data.shape
x = np.arange(M)


def masked_polyfit(arr, k=2.5):
    out = np.empty_like(arr)
    for i, row in enumerate(arr):
        a, b = np.polyfit(x, row, 1)
        resid = row - (a * x + b)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-30
        mask = np.abs(resid) < k * 1.4826 * mad
        if mask.sum() < 8:
            fit = a * x + b
        else:
            a2, b2 = np.polyfit(x[mask], row[mask], 1)
            fit = a2 * x + b2
        out[i] = row - fit
    return out


def robust_siegel(arr):
    from scipy.stats import siegelslopes
    out = np.empty_like(arr)
    for i, row in enumerate(arr):
        slope, intercept = siegelslopes(row, x)
        out[i] = row - (slope * x + intercept)
    return out


def median_of_differences(arr):
    diffs = np.median(np.diff(arr, axis=0), axis=1)
    offsets = np.concatenate([[0.0], np.cumsum(diffs)])
    aligned = arr - offsets[:, None]
    out = np.empty_like(aligned)
    for i, row in enumerate(aligned):
        a, b = np.polyfit(x, row, 1)
        out[i] = row - (a * x + b)
    return out


corr = {
    "Masked polyfit": masked_polyfit(data),
    "Robust (Siegel)": robust_siegel(data),
    "Median-of-differences": median_of_differences(data),
}

for name, c in corr.items():
    print(f"{name:25s}  min={c.min(): .3e}  max={c.max(): .3e}  ptp={np.ptp(c): .3e}")

stacked = np.stack(list(corr.values()))
vmin, vmax = np.percentile(stacked, [1, 99])

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
for ax, (name, c) in zip(axes, corr.items()):
    im = ax.imshow(c, cmap="viridis", aspect="equal", origin="lower",
                   vmin=vmin, vmax=vmax)
    ax.set_title(name)
    ax.set_xlabel("column")
axes[0].set_ylabel("row")
fig.colorbar(im, ax=axes, label="height (m, leveled)", shrink=0.85)
fig.suptitle(f"{os.path.basename(SRC)} — per-row degree-1 leveling (three methods)")
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
