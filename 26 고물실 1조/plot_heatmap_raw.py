import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plain heatmap of a raw AFM CSV (no leveling).")
parser.add_argument("csv", help="Path to the input CSV (semicolon-delimited).")
args = parser.parse_args()

SRC = args.csv
base = os.path.splitext(os.path.basename(SRC))[0]
OUT = os.path.join(os.path.dirname(os.path.abspath(SRC)), base + "_heatmap.png")

data = np.loadtxt(SRC, delimiter=";")
vmin, vmax = np.percentile(data, [1, 99])

fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
im = ax.imshow(data, cmap="viridis", aspect="equal", origin="lower",
               vmin=vmin, vmax=vmax)
ax.set_xlabel("column")
ax.set_ylabel("row")
ax.set_title(os.path.basename(SRC))
fig.colorbar(im, ax=ax, label="height (m)", shrink=0.85)
fig.savefig(OUT, dpi=150)
print("saved:", OUT)
