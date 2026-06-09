"""Reproduce the "leveled" / "raw" / "d/dx" panels of measure_nid.py as standalone
figures, for the Contact (JJ_AFM_4) and Non-contact (JJ_AFM_2) scans.

Each scan's Forward Z-Axis is leveled with the interactive tool's masked degree-2
polyfit (k=2.5, right_exclude=0) and rendered with viridis (leveled/raw) or RdBu_r
(d/dx), origin lower, color-scaled to the 1st/99th percentile (symmetric for d/dx),
exactly as measure_nid.py does. Output: a publication-style figure with nm/um axes
and a colorbar, three per scan.

Note: tall and short objects live in the SAME scan, so C_short == C_tall and
NC_short == NC_tall for the full-image panels; outputs are named by tip mode (C/NC).

Run from the repo root:  python New_data_only/JJ_data_0603/plot_C_short_leveled.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root = two levels up (New_data_only/JJ_data_0603/..) -> for repo-root imports
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from NSFopen.read import read                  # noqa: E402
from export_leveled import masked_polyfit      # noqa: E402
from measure_nid import pixel_nm               # noqa: E402

CHANNEL = "Forward"
DEG, K, REX = 2, 2.5, 0.0                       # measure_nid.py interactive defaults

JJ_DATA = os.path.join(REPO, "JJ_data")

# (output dir, prefix, nid path relative to REPO).  tall/short share a scan, so the
# full-image panels are named by tip mode (C/NC). Outputs land next to their nid.
CASES = [
    (HERE, "C", "New_data_only/JJ_data_0603/JJ_AFM_4.nid"),    # Contact, 20um (new)
    (HERE, "NC", "New_data_only/JJ_data_0603/JJ_AFM_2.nid"),   # Non-contact, 20um (new)
    (JJ_DATA, "C", "JJ_data/JJ_AFM_5.nid"),                    # Contact, 20um (old)
    (JJ_DATA, "NC", "JJ_data/JJ_AFM_7.nid"),                   # Non-contact, 20um (old)
]


def render(grid, extent, out, vmin, vmax, cmap="viridis", clabel="height (nm)"):
    """Save grid as a heatmap (origin lower) with the given color scale/colormap."""
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, aspect="equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(clabel)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}  vmin/vmax=[{vmin:.3f}, {vmax:.3f}]")


def process(out_dir, prefix, nid_rel):
    afm = read(os.path.join(REPO, nid_rel))
    z = np.asarray(afm.data[("Image", CHANNEL, "Z-Axis")], dtype=float)
    rows, cols = z.shape

    raw_nm = z * 1e9                                                    # m -> nm
    lev_nm = masked_polyfit(z, k=K, deg=DEG, right_exclude=REX) * 1e9   # m -> nm
    # d/dx along the fast-scan axis (columns) of RAW height, nm per pixel,
    # like np.gradient(z, axis=1) -> the JS gradX in measure_nid.py
    dx_nm = np.gradient(raw_nm, axis=1)

    px_x = pixel_nm(afm, "X", cols)        # nm per pixel
    px_y = pixel_nm(afm, "Y", rows)
    # extent in micrometers for readable real-space axes
    extent = [0, cols * px_x / 1000.0, 0, rows * px_y / 1000.0]

    print(f"{prefix} ({nid_rel}): shape={rows}x{cols}  px=({px_x:.2f}, {px_y:.2f}) nm")
    lev_lo, lev_hi = np.percentile(lev_nm, [1, 99])           # JS colorScale
    raw_lo, raw_hi = np.percentile(raw_nm, [1, 99])
    dx_lim = np.percentile(np.abs(dx_nm), 99) or 1e-9         # symmetric range
    p = os.path.join(out_dir, prefix)
    render(lev_nm, extent, f"{p}_short_leveled.png", lev_lo, lev_hi)
    render(raw_nm, extent, f"{p}_short_raw.png", raw_lo, raw_hi)
    render(dx_nm, extent, f"{p}_short_dx.png", -dx_lim, dx_lim,
           cmap="RdBu_r", clabel="d/dx (nm/px)")


def main():
    for out_dir, prefix, nid_rel in CASES:
        process(out_dir, prefix, nid_rel)


if __name__ == "__main__":
    main()
