"""Generate the report figures (vector PDF) for reports/sample/sample.tex.

Reproduces the leveled / raw / d-by-dx panels of measure_nid.py for the Contact
scan JJ_AFM_4.nid, rendered as publication-style heatmaps with um axes and a
colorbar. Output is PDF (vector) so it embeds crisply in LaTeX.

Convention (see repo AGENTS.md): each report's images/ folder holds BOTH this
script and the .pdf figures it produces. The script reads the source .nid and
reuses the repo's processing modules rather than reimplementing the leveling.

Run from anywhere:  python reports/sample/images/make_figures.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- paths: this file lives in <repo>/reports/sample/images/ -----------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA_ROOT = os.path.join(REPO, "26 고물실 1조")        # source data + processing modules
sys.path.insert(0, DATA_ROOT)

from NSFopen.read import read                  # noqa: E402
from export_leveled import masked_polyfit      # noqa: E402  (per-row masked polyfit leveling)
from measure_nid import pixel_nm               # noqa: E402  (nm-per-pixel from metadata)

# --- source scan + leveling parameters (measure_nid.py interactive defaults) --
NID = os.path.join(DATA_ROOT, "New_data_only", "JJ_data_0603", "JJ_AFM_4.nid")  # Contact, 20um
CHANNEL = "Forward"
DEG, K, REX = 2, 2.5, 0.0

# --- typography (finalized, see AGENTS.md) -----------------------------------
LABEL_FS = 15        # axis + colorbar label fontsize (~1.5x matplotlib default)
TICK_FS = 13         # axis + colorbar tick fontsize
CBAR_FRACTION = 0.07  # colorbar thickness (~1.5x the usual 0.046)
FIGSIZE = (6.0, 5.2)


def render(grid, extent, out, vmin, vmax, cmap="viridis", clabel="Height (nm)"):
    """Save grid as a heatmap (origin lower) with the given color scale/colormap.

    No title (by convention); units always present on the axis/colorbar labels.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, aspect="equal")
    ax.set_xlabel("x (µm)", fontsize=LABEL_FS)
    ax.set_ylabel("y (µm)", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    cbar = fig.colorbar(im, ax=ax, fraction=CBAR_FRACTION, pad=0.04)
    cbar.set_label(clabel, fontsize=LABEL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")   # PDF is vector; no dpi needed
    plt.close(fig)
    print(f"saved: {out}  vmin/vmax=[{vmin:.3f}, {vmax:.3f}]")


def main():
    afm = read(NID)
    z = np.asarray(afm.data[("Image", CHANNEL, "Z-Axis")], dtype=float)
    rows, cols = z.shape

    raw_nm = z * 1e9                                                    # m -> nm
    lev_nm = masked_polyfit(z, k=K, deg=DEG, right_exclude=REX) * 1e9   # m -> nm
    # d/dx along the fast-scan axis (columns) of RAW height, nm per pixel,
    # like np.gradient(z, axis=1) -> the JS gradX in measure_nid.py
    dx_nm = np.gradient(raw_nm, axis=1)

    px_x = pixel_nm(afm, "X", cols)        # nm per pixel
    px_y = pixel_nm(afm, "Y", rows)
    extent = [0, cols * px_x / 1000.0, 0, rows * px_y / 1000.0]         # um

    print(f"C ({os.path.relpath(NID, REPO)}): shape={rows}x{cols}  "
          f"px=({px_x:.2f}, {px_y:.2f}) nm")

    lev_lo, lev_hi = np.percentile(lev_nm, [1, 99])        # measure_nid colorScale
    raw_lo, raw_hi = np.percentile(raw_nm, [1, 99])
    dx_lim = np.percentile(np.abs(dx_nm), 99) or 1e-9      # symmetric range

    render(lev_nm, extent, os.path.join(HERE, "C_short_leveled.pdf"), lev_lo, lev_hi)
    render(raw_nm, extent, os.path.join(HERE, "C_short_raw.pdf"), raw_lo, raw_hi)
    render(dx_nm, extent, os.path.join(HERE, "C_short_dx.pdf"), -dx_lim, dx_lim,
           cmap="RdBu_r", clabel="dz/dx (nm/px)")


if __name__ == "__main__":
    main()
