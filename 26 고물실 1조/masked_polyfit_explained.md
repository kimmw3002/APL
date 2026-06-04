# How `masked_polyfit` works

`masked_polyfit` does **per-row line removal with outlier rejection**. The goal is to flatten each row of the AFM image by subtracting a best-fit line — but it ignores real surface features (bumps, particles) when computing that line, so they don't bias the leveling.

Here's the flow, row by row (`plot_heatmap_leveled.py:20-33`):

## 1. First pass — fit a line to the whole row

```python
a, b = np.polyfit(x, row, 1)        # degree-1 fit: row ≈ a*x + b
resid = row - (a * x + b)           # what's left after subtracting the line
```

This is a naive line fit. The problem: if the row has a tall feature, that feature pulls the line up and tilts it.

## 2. Estimate the noise scale robustly (MAD)

```python
mad = np.median(np.abs(resid - np.median(resid))) + 1e-30
```

`mad` = **Median Absolute Deviation**, a robust alternative to standard deviation. Unlike `std`, the median isn't thrown off by a few large outliers. The `1e-30` just prevents division-by-zero on a perfectly flat row.

## 3. Build a mask of "inliers"

```python
mask = np.abs(resid) < k * 1.4826 * mad
```

- `1.4826` is the constant that converts MAD into a standard-deviation-equivalent (for Gaussian noise, `σ ≈ 1.4826 · MAD`).
- `k = 2.5` means "keep points within 2.5σ of the fit line."
- Points further away — i.e. real surface features sitting above/below the baseline — get masked **out**.

## 4. Second pass — refit using only inliers

```python
if mask.sum() < 8:
    fit = a * x + b                 # too few inliers → fall back to first fit
else:
    a2, b2 = np.polyfit(x[mask], row[mask], 1)   # refit on background only
    fit = a2 * x + b2
out[i] = row - fit                  # subtract the clean baseline
```

The refit uses only the background pixels, so the line represents the true tilt/offset of the substrate rather than being skewed by features. Then it subtracts that line, leaving a flattened row. The `< 8` guard avoids fitting a line through too few points (unreliable), reverting to the naive fit in that case.

## Net effect

Each row is leveled so its background sits at ~0, while genuine topography is preserved instead of being partially subtracted away. This is the standard "iterate-once with sigma clipping" trick — the same idea as the other two methods in the file (Siegel slopes and median-of-differences), which are robust to outliers in different ways.

> **Note:** it's a **single** refinement pass (one mask, one refit), not iterated to convergence. For most AFM rows that's plenty, but if a row had extreme features you could loop steps 1–3 until the mask stabilizes.
