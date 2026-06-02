import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Differential-mode heatmaps from a CSV (derivative of height).")
parser.add_argument("csv", help="Path to the input CSV (semicolon-delimited).")
args = parser.parse_args()

SRC = args.csv
base = os.path.splitext(os.path.basename(SRC))[0]
OUT = os.path.join(os.path.dirname(os.path.abspath(SRC)), base + "_differential.png")

data = np.loadtxt(SRC, delimiter=";")

# Differential mode: spatial derivative of the height map. Differentiating along
# the fast-scan (x) axis is the classic AFM "differential" view -- it removes the
# slowly-varying background/tilt and emphasises edges and slopes.
dx = np.gradient(data, axis=1)   # d/dx  (fast scan)
dy = np.gradient(data, axis=0)   # d/dy  (slow scan)
grad = np.hypot(dx, dy)          # |grad|

diff = {
    "d/dx (fast scan)": dx,
    "d/dy (slow scan)": dy,
    "|gradient|": grad,
}

for name, c in diff.items():
    print(f"{name:18s}  min={c.min(): .3e}  max={c.max(): .3e}  ptp={np.ptp(c): .3e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
for ax, (name, c) in zip(axes, diff.items()):
    # symmetric range for signed derivatives, 0-based for the magnitude
    if name.startswith("|"):
        vmin, vmax, cmap = 0.0, np.percentile(c, 99), "inferno"
    else:
        lim = np.percentile(np.abs(c), 99)
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    im = ax.imshow(c, cmap=cmap, aspect="equal", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(name)
    ax.set_xlabel("column")
    fig.colorbar(im, ax=ax, shrink=0.7)
axes[0].set_ylabel("row")
fig.suptitle(f"{os.path.basename(SRC)} — differential mode (height derivatives)")
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
