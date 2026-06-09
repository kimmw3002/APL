# Forward vs Backward h(x) analysis (8 cases)

Center-aligned overlay of the Forward and Backward scans, with height &
FWHM reproduced by the exact `measure_nid.py` method and tested for
Forward = Backward agreement. Plots: `hx_plots/hx_<n>_<label>_fwbw_centered.png`.

## Method

- Processing identical to `measure_nid.py` for both channels: leveling
  (deg=2, k=2.5, right_exclude=0) -> 21px **aligned** swath -> sloped baseline.
- **height** = peak-minus-baseline, averaged over the peak +/- `peak_avg_n` pts;
  **height_err** = baseline-residual noise sigma.
- **FWHM** from the 50% crossings; **fwhm_err** = the tool's half-max band
  (50% +/- 1 sigma_noise).
- **Pixel (lateral) uncertainty** on FWHM: sigma_px = px/sqrt(6)
  (two independent half-max edge positions on the pixel grid).
  Combined in quadrature: sigma_tot = sqrt(fwhm_err^2 + sigma_px^2).
  Height is a z-axis quantity -> no pixel term.
- **Significance**: z = |F - B| / sqrt(sigma_F^2 + sigma_B^2); z < 2 => not significant.

## 1. Measured values (nm)

| # | case | px | ch | height +/- err | FWHM +/- fit | sigma_px | FWHM +/- tot |
|---|------|----|----|----------------|--------------|----------|--------------|
| 1 | short_C_new | 19.53 | Forward | 23.50 +/- 3.12 | 438.4 +/- 52.1 | 8.0 | 438.4 +/- 52.7 |
| 1 | short_C_new | 19.53 | Backward | 21.24 +/- 2.84 | 480.5 +/- 46.0 | 8.0 | 480.5 +/- 46.7 |
| 2 | tall_C_new | 19.53 | Forward | 95.74 +/- 1.88 | 400.6 +/- 9.9 | 8.0 | 400.6 +/- 12.7 |
| 2 | tall_C_new | 19.53 | Backward | 94.09 +/- 4.84 | 400.1 +/- 21.4 | 8.0 | 400.1 +/- 22.9 |
| 3 | short_NC_new | 19.53 | Forward | 22.11 +/- 2.69 | 763.7 +/- 189.4 | 8.0 | 763.7 +/- 189.5 |
| 3 | short_NC_new | 19.53 | Backward | 20.04 +/- 5.73 | 552.3 +/- 291.1 | 8.0 | 552.3 +/- 291.2 |
| 4 | tall_NC_new | 19.53 | Forward | 95.11 +/- 2.73 | 929.3 +/- 20.0 | 8.0 | 929.3 +/- 21.5 |
| 4 | tall_NC_new | 19.53 | Backward | 93.46 +/- 6.67 | 976.2 +/- 65.8 | 8.0 | 976.2 +/- 66.2 |
| 5 | short_C_old | 78.12 | Forward | 21.65 +/- 0.42 | 595.8 +/- 9.3 | 31.9 | 595.8 +/- 33.2 |
| 5 | short_C_old | 78.12 | Backward | 21.10 +/- 1.57 | 528.4 +/- 39.8 | 31.9 | 528.4 +/- 51.0 |
| 6 | tall_C_old | 78.12 | Forward | 94.55 +/- 0.50 | 463.7 +/- 2.8 | 31.9 | 463.7 +/- 32.0 |
| 6 | tall_C_old | 78.12 | Backward | 93.15 +/- 3.02 | 492.3 +/- 17.5 | 31.9 | 492.3 +/- 36.4 |
| 7 | short_NC_old | 78.12 | Forward | 21.28 +/- 0.75 | 737.0 +/- 15.2 | 31.9 | 737.0 +/- 35.3 |
| 7 | short_NC_old | 78.12 | Backward | 20.28 +/- 2.50 | 708.3 +/- 50.1 | 31.9 | 708.3 +/- 59.4 |
| 8 | tall_NC_old | 78.12 | Forward | 101.30 +/- 0.78 | 815.2 +/- 4.6 | 31.9 | 815.2 +/- 32.2 |
| 8 | tall_NC_old | 78.12 | Backward | 95.85 +/- 16.60 | 801.2 +/- 104.2 | 31.9 | 801.2 +/- 109.0 |

## 2. Forward vs Backward significance

### Height (sigma = noise)

| case | Forward | Backward | d | sigma_d | z | agree? |
|------|---------|----------|---|---------|---|--------|
| short_C_new | 23.50 | 21.24 | 2.26 | 4.21 | 0.54 | yes |
| tall_C_new | 95.74 | 94.09 | 1.65 | 5.19 | 0.32 | yes |
| short_NC_new | 22.11 | 20.04 | 2.07 | 6.33 | 0.33 | yes |
| tall_NC_new | 95.11 | 93.46 | 1.66 | 7.21 | 0.23 | yes |
| short_C_old | 21.65 | 21.10 | 0.55 | 1.62 | 0.34 | yes |
| tall_C_old | 94.55 | 93.15 | 1.40 | 3.06 | 0.46 | yes |
| short_NC_old | 21.28 | 20.28 | 1.00 | 2.61 | 0.38 | yes |
| tall_NC_old | 101.30 | 95.85 | 5.45 | 16.62 | 0.33 | yes |

### FWHM (sigma = sigma_tot)

| case | Forward | Backward | d | sigma_d | z | agree? |
|------|---------|----------|---|---------|---|--------|
| short_C_new | 438.4 | 480.5 | -42.1 | 70.4 | 0.60 | yes |
| tall_C_new | 400.6 | 400.1 | 0.5 | 26.2 | 0.02 | yes |
| short_NC_new | 763.7 | 552.3 | 211.4 | 347.5 | 0.61 | yes |
| tall_NC_new | 929.3 | 976.2 | -46.9 | 69.6 | 0.67 | yes |
| short_C_old | 595.8 | 528.4 | 67.4 | 60.9 | 1.11 | yes |
| tall_C_old | 463.7 | 492.3 | -28.6 | 48.5 | 0.59 | yes |
| short_NC_old | 737.0 | 708.3 | 28.7 | 69.1 | 0.42 | yes |
| tall_NC_old | 815.2 | 801.2 | 13.9 | 113.6 | 0.12 | yes |

## 3. Conclusions

- **Height:** Forward = Backward in all 8 cases (every z < 2; largest z = 0.54 at short_C_new). Trace/retrace heights are
  statistically indistinguishable.
- **FWHM:** Forward = Backward in all 8 cases (every z < 2; largest z = 1.11 at short_C_old). With the half-max band and
  pixel uncertainty included, no trace/retrace FWHM difference is significant.
- The visible **x-offset between the raw Forward/Backward peaks is scanner
  hysteresis** (trace vs retrace), removed here by center-aligning; it does
  not affect height or FWHM, which is consistent with the agreement above.
- Net: the measurement is **direction-independent** within uncertainty, so
  using the Forward channel (as in the main analysis) is justified.

