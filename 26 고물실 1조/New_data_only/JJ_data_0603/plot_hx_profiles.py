"""Reproduce the measure_nid.py swath-averaged profile and plot h(x) for the
8 cases (tall/short x C/NC x new/old) recorded in nid_measurements.csv.

For each case this reads the .nid Z-Axis, levels it with the exact same masked
polyfit (deg=2, k=2.5, right_exclude=0 -- the interactive tool's defaults), then
samples a 21-px ALIGNED swath profile (faithful port of the JS sampleProfile /
shiftToAlign / bilinearG in measure_nid.py) and plots only h(x).

Each h(x) point's marker is colored by how many of the 21 parallel cross-sections
actually contributed to that point's average (cnt[i]) -- interior points use all
21, points near the ends lose lines whose alignment shift falls out of range.

Run from the repo root:  python New_data_only/JJ_data_0603/plot_hx_profiles.py
"""
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# repo root = two levels up from this file (New_data_only/JJ_data_0603/..)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from NSFopen.read import read                  # noqa: E402
from export_leveled import masked_polyfit       # noqa: E402
from measure_nid import pixel_nm                # noqa: E402

# fraction of the x-axis the peak feature (~2*FWHM bump) should occupy
PEAK_AXIS_FRAC = 0.70

OUT_DIR = os.path.join(HERE, "hx_plots")

# label, nid (repo-relative), channel, x0, y0, x1, y1   (coords as exported to CSV)
CASES = [
    ("short_C_new",  "New_data_only/JJ_data_0603/JJ_AFM_4.nid", "Forward", 482, 430, 400, 342),
    ("tall_C_new",   "New_data_only/JJ_data_0603/JJ_AFM_4.nid", "Forward", 512, 514, 406, 607),
    ("short_NC_new", "New_data_only/JJ_data_0603/JJ_AFM_2.nid", "Forward", 570, 402, 493, 319),
    ("tall_NC_new",  "New_data_only/JJ_data_0603/JJ_AFM_2.nid", "Forward", 586, 473, 422, 623),
    ("short_C_old",  "JJ_data/JJ_AFM_5.nid",                    "Forward",  96, 120, 130, 148),
    ("tall_C_old",   "JJ_data/JJ_AFM_5.nid",                    "Forward",  92, 109, 126,  80),
    ("short_NC_old", "JJ_data/JJ_AFM_7.nid",                    "Forward", 120,  93, 146, 121),
    ("tall_NC_old",  "JJ_data/JJ_AFM_7.nid",                    "Forward", 115,  78, 144,  48),
]

SWATH = 21          # swath_px from the CSV (all rows)
DEG, K, REX = 2, 2.5, 0.0   # interactive tool defaults (CSV stores no leveling params)


def nice_title(label):
    """'short_C_new' -> 'Short Electrode, Contact Tip' (size + tip mode only)."""
    parts = label.split("_")
    size = "Tall" if parts[0] == "tall" else "Short"
    tip = "Contact" if parts[1] == "C" else "Non-contact"
    return f"{size} Electrode, {tip} Tip"


def bilinear(grid, c, r):
    """Bilinear sample of grid[row][col] at (c=col, r=row), clamped to bounds.
    Mirrors bilinearG in measure_nid.py (display flip does NOT affect sampling)."""
    rows, cols = grid.shape
    c = min(max(c, 0.0), cols - 1.0)
    r = min(max(r, 0.0), rows - 1.0)
    c0, r0 = int(math.floor(c)), int(math.floor(r))
    c1, r1 = min(c0 + 1, cols - 1), min(r0 + 1, rows - 1)
    fc, fr = c - c0, r - r0
    return (grid[r0][c0] * (1 - fc) * (1 - fr) + grid[r0][c1] * fc * (1 - fr)
            + grid[r1][c0] * (1 - fc) * fr + grid[r1][c1] * fc * fr)


def shift_to_align(row, ref, max_lag):
    """Integer lag in [-max_lag, max_lag] maximizing mean-centered cross-covariance
    of `row` with `ref`; return aligned copy (out-of-range entries -> NaN).
    Faithful port of shiftToAlign in measure_nid.py."""
    n = len(row)
    rm, om = ref.mean(), row.mean()
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        s, c = 0.0, 0
        for i in range(n):
            j = i + lag
            if j < 0 or j >= n:
                continue
            s += (ref[i] - rm) * (row[j] - om)
            c += 1
        if c > 0:
            sc = s / c
            if sc > best:
                best, best_lag = sc, lag
    out = np.full(n, np.nan)
    for i in range(n):
        j = i + best_lag
        if 0 <= j < n:
            out[i] = row[j]
    return out


def sample_profile(grid, px_x, px_y, x0, y0, x1, y1, width):
    """Port of sampleProfile (measure_nid.py): aligned 21-px swath average.

    Returns (dist_nm, prof_nm, cnt) where cnt[i] = number of cross-sections
    contributing to h(x) point i after alignment."""
    dx, dy = x1 - x0, y1 - y0
    lenpx = math.hypot(dx, dy) or 1e-9
    dxn, dyn = dx * px_x, dy * px_y
    lennm = math.hypot(dxn, dyn) or 1.0
    perp_x, perp_y = -dy / lenpx, dx / lenpx           # perpendicular (pixel space)
    half = (width - 1) / 2
    steps = max(8, round(lenpx))
    n = steps + 1
    dist = np.array([i / steps * lennm for i in range(n)])

    # collect the `width` parallel cross-sections
    lines = []
    for w in range(-int(half), int(half) + 1):
        row = np.empty(n)
        for i in range(n):
            t = i / steps
            row[i] = bilinear(grid, x0 + dx * t + w * perp_x, y0 + dy * t + w * perp_y)
        lines.append(row)
    lines = np.array(lines)

    if width == 1:
        return dist, lines[0], np.ones(n, dtype=int)

    # align every cross-section to the center one, then average
    ref = lines[len(lines) // 2]
    max_lag = min(n - 2, int(math.ceil(half)) + 3)
    acc = np.zeros(n)
    cnt = np.zeros(n, dtype=int)
    for row in lines:
        a = shift_to_align(row, ref, max_lag)
        good = ~np.isnan(a)
        acc[good] += a[good]
        cnt[good] += 1
    prof = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1), ref)
    return dist, prof, cnt


def analyze_profile(dist, prof, peak_avg_n, fwhm_band_sigma=1.0, basefrac=0.25):
    """Faithful port of measure_nid.analyze() (auto baseline path): returns
    height + FWHM and their uncertainties exactly as the interactive tool.

    height_err = baseline-residual noise (sigma).
    fwhm_err   = half the span between the half-max crossings evaluated at
                 (50% +/- fwhm_band_sigma*noise)  -- the tool's FWHM band.
    """
    n = len(prof)
    k = max(1, round(n * basefrac))
    base_idx = [i for i in range(n) if i < k or i >= n - k]
    rod_idx = [i for i in range(n) if k <= i < n - k]
    base_safe = base_idx if len(base_idx) >= 2 else list(range(n))

    # sloped baseline: linear fit to baseline points; noise = residual std
    A = np.vstack([np.ones(len(base_safe)), dist[base_safe]]).T
    (bl_base, bl_slope), *_ = np.linalg.lstsq(A, prof[base_safe], rcond=None)
    resid = prof[base_safe] - (bl_base + bl_slope * dist[base_safe])
    noise = float(resid.std()) or 1e-9          # population std, like JS std()

    det = prof - (bl_base + bl_slope * dist)
    search = rod_idx if rod_idx else list(range(n))
    pi = search[0]
    for i in search:
        if det[i] > det[pi]:
            pi = i
    search_set = set(search)
    avg_idx = [i for i in range(max(0, pi - peak_avg_n), min(n - 1, pi + peak_avg_n) + 1)
               if i in search_set] or [pi]
    height = float(np.mean(det[avg_idx]))
    half_det = height / 2

    def cross_left(level):
        for i in range(pi, 0, -1):
            if det[i] >= level and det[i - 1] < level:
                f = (level - det[i - 1]) / (det[i] - det[i - 1])
                return dist[i - 1] + (dist[i] - dist[i - 1]) * f
        return None

    def cross_right(level):
        for i in range(pi, n - 1):
            if det[i] >= level and det[i + 1] < level:
                f = (level - det[i]) / (det[i + 1] - det[i])
                return dist[i] + (dist[i + 1] - dist[i]) * f
        return None

    xL, xR = cross_left(half_det), cross_right(half_det)
    fwhm = abs(xR - xL) if (xL is not None and xR is not None) else float("nan")

    tol = fwhm_band_sigma * noise
    lo, hi = max(0.0, half_det - tol), half_det + tol
    xLlo, xLhi = cross_left(lo), cross_left(hi)
    xRlo, xRhi = cross_right(lo), cross_right(hi)
    fwhm_err = float("nan")
    if None not in (xLlo, xLhi, xRlo, xRhi):
        left_inner, left_outer = max(xLlo, xLhi), min(xLlo, xLhi)
        right_inner, right_outer = min(xRlo, xRhi), max(xRlo, xRhi)
        fwhm_min = max(0.0, right_inner - left_inner)
        fwhm_max = max(0.0, right_outer - left_outer)
        fwhm_err = 0.5 * abs(fwhm_max - fwhm_min)

    return dict(height=height, height_err=noise, fwhm=fwhm, fwhm_err=fwhm_err,
                center=float(dist[pi]), xL=xL, xR=xR, pi=pi,
                peak_avg_count=len(avg_idx))


def peak_xlim(dist, center, fwhm):
    """x-limits centered on `center` so the ~2*FWHM peak fills PEAK_AXIS_FRAC
    of the axis. Falls back to the full data range when FWHM is unavailable,
    and clips to the data extent so we never pad past the measured profile."""
    lo_data, hi_data = float(dist[0]), float(dist[-1])
    if fwhm is None or (isinstance(fwhm, float) and math.isnan(fwhm)) or fwhm <= 0:
        return lo_data, hi_data
    half = fwhm / PEAK_AXIS_FRAC          # half-window: 2*FWHM bump -> ~70%
    return max(lo_data, center - half), min(hi_data, center + half)


def estimate_height(dist, prof, frac=0.25):
    """Sloped-baseline peak-minus-baseline height, for a CSV sanity check only
    (not plotted). Mirrors the auto-baseline path in measure_nid.analyze()."""
    n = len(prof)
    k = max(1, round(n * frac))
    base_idx = list(range(0, k)) + list(range(n - k, n))
    A = np.vstack([np.ones(len(base_idx)), dist[base_idx]]).T
    (b, s), *_ = np.linalg.lstsq(A, prof[base_idx], rcond=None)
    det = prof - (b + s * dist)
    rod = det[k:n - k] if n - 2 * k > 0 else det
    return float(rod.max())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'case':14s} {'nid':40s} {'px(nm)':>7s} {'n':>4s} "
          f"{'cnt min/med/max':>16s} {'height(nm)':>10s}")
    print("-" * 100)

    for idx, (label, nid_rel, ch, x0, y0, x1, y1) in enumerate(CASES, 1):
        afm = read(os.path.join(REPO, nid_rel))
        z = np.asarray(afm.data[("Image", ch, "Z-Axis")], dtype=float)
        rows, cols = z.shape
        lev = masked_polyfit(z, k=K, deg=DEG, right_exclude=REX) * 1e9   # m -> nm
        px_x = pixel_nm(afm, "X", cols)
        px_y = pixel_nm(afm, "Y", rows)

        dist, prof, cnt = sample_profile(lev, px_x, px_y, x0, y0, x1, y1, SWATH)
        height = estimate_height(dist, prof)
        A = analyze_profile(dist, prof, peak_avg_n=2)

        print(f"{label:14s} {os.path.basename(nid_rel):40s} {px_x:7.2f} {len(prof):4d} "
              f"{cnt.min():2d}/{int(np.median(cnt)):2d}/{cnt.max():2d}{'':>8s} {height:10.2f}")

        # ---- plot: h(x) only (line + small markers, single color) ----
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        ax.plot(dist, prof, "-o", color="#3366cc", lw=1.0, ms=2.5, zorder=2)
        ax.set_xlim(*peak_xlim(dist, A["center"], A["fwhm"]))
        ax.set_xlabel("distance (nm)")
        ax.set_ylabel("height (nm)")
        ax.set_title(nice_title(label))
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        out = os.path.join(OUT_DIR, f"hx_{idx}_{label}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)

    print(f"\nsaved 8 PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
