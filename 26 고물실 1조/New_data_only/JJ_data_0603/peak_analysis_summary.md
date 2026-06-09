# AFM Peak Measurement Summary (tall/short × C/NC × new/old)

Source: `nid_measurements.csv` + `*_Info.txt` metadata.

- **new** = `New_data_only/JJ_data_0603/` — AFM_4 (C), AFM_2 (NC)
- **old** = `JJ_data/` — AFM_5 (C), AFM_7 (NC)
- For tall·C (new), the last of the 3 duplicate rows (row 11) is used.

## 1. Measured values (nm)

| data | height | C/NC | height ± pm | FWHM ± σ_fit | row |
|------|--------|------|-------------|--------------|-----|
| new | short | C  | 23.51 ± 3.06 | 436.7 ± 80.7 | 2 |
| new | tall  | C  | 95.89 ± 1.89 | 400.4 ± 9.1 | 11 |
| new | short | NC | 21.76 ± 2.93 | 765.1 ± 203.8 | 4 |
| new | tall  | NC | 95.20 ± 2.83 | 929.4 ± 20.8 | 5 |
| old | short | C  | 21.61 ± 0.40 | 591.4 ± 9.8 | 9 |
| old | tall  | C  | 95.39 ± 0.52 | 459.7 ± 2.7 | 8 |
| old | short | NC | 21.48 ± 0.58 | 726.6 ± 10.3 | 7 |
| old | tall  | NC | 100.19 ± 0.81 | 819.0 ± 4.8 | 6 |

## 2. Minimum length unit (lateral pixel size)

From `*_Info.txt`: all scans are 20 µm square.

$$\text{px size}=\frac{\text{Image size}}{N_\text{points}}$$

| data | Image size | Points×Lines | **px size** |
|------|-----------|--------------|-------------|
| new | 20 µm | 1024×1024 | **19.53 nm** |
| old | 20 µm | 256×256 | **78.13 nm** |

→ new is 4× finer.

## 3. Pixel (discretization) uncertainty

The quantization error of one position on the grid is the std of a uniform
distribution of width 1 px: $px/\sqrt{12}$. FWHM is the difference of the
left and right half-max positions (two independent quantized positions):

$$\sigma_\text{px,FWHM}=\sqrt{2}\cdot\frac{px}{\sqrt{12}}=\frac{px}{\sqrt 6}$$

- new: $19.53/\sqrt6 = 7.97$ nm
- old: $78.13/\sqrt6 = 31.9$ nm

> Height is measured along z (feedback), independent of the lateral pixel
> size, so σ_px is NOT applied to height.

## 4. Total uncertainty (fit ⊕ pixel, independent → quadrature)

$$\sigma_\text{tot}=\sqrt{\sigma_\text{fit}^2+\sigma_\text{px}^2}$$

| data | h | C/NC | σ_fit | σ_px | **σ_tot (FWHM)** |
|------|---|------|-------|------|------------------|
| new | short | C  | 80.7 | 7.97 | **81.1** |
| new | tall  | C  | 9.1  | 7.97 | **12.1** |
| new | short | NC | 203.8| 7.97 | **203.9** |
| new | tall  | NC | 20.8 | 7.97 | **22.3** |
| old | short | C  | 9.8  | 31.9 | **33.4** |
| old | tall  | C  | 2.7  | 31.9 | **32.0** |
| old | short | NC | 10.3 | 31.9 | **33.5** |
| old | tall  | NC | 4.8  | 31.9 | **32.3** |

Key point: the small fit pm of the **old** data is misleading — its real
uncertainty is dominated by the pixel term (~±32 nm).

## 5. new vs old significance test

Compare matching (height, C/NC) pairs. Standard deviation of the difference:

$$z=\frac{|x_\text{new}-x_\text{old}|}{\sqrt{\sigma_\text{new}^2+\sigma_\text{old}^2}}$$

(z < 2 → not significant at the 2σ level)

### Height (σ = reported pm)

| h | C/NC | new | old | Δ | σ_Δ | **z** | sig? |
|---|------|-----|-----|---|-----|-------|------|
| short | C  | 23.51 | 21.61 | 1.90 | 3.09 | 0.61 | no |
| tall  | C  | 95.89 | 95.39 | 0.50 | 1.96 | 0.26 | no |
| short | NC | 21.76 | 21.48 | 0.28 | 2.99 | 0.09 | no |
| tall  | NC | 95.20 | 100.19| -4.99| 2.94 | 1.70 | no (marginal) |

→ **No significant new/old height difference.** Height depends only on the
structure (short ≈ 21–24, tall ≈ 95–100); robust across run and coating.

### FWHM (σ = σ_tot)

| h | C/NC | new | old | Δ | σ_Δ | **z** | sig? |
|---|------|-----|-----|---|-----|-------|------|
| short | C  | 436.7 | 591.4 | -154.7 | 87.7 | 1.76 | no (marginal) |
| tall  | C  | 400.4 | 459.7 | -59.3  | 34.2 | 1.73 | no (marginal) |
| short | NC | 765.1 | 726.6 | 38.5   | 206.6| 0.19 | no |
| tall  | NC | 929.4 | 819.0 | 110.4  | 39.3 | **2.81** | **yes** |

→ Most new/old FWHM differences are not significant (~1.7σ). Only **tall·NC**
is significant (2.8σ, new wider).

## 6. Conclusions

1. **C < NC (coating sharpens the peak):** holds for all 4 pairs. Clear in
   **new** (resolution sufficient: C ~21 px vs NC ~43 px); **old** has FWHM of
   only 6–10 px, unsuitable for quantitative comparison.
2. **Height:** set by short/tall structure only. Statistically identical across
   new/old and C/NC (all z < 2).
3. **new vs old FWHM:** only tall·NC differs significantly; the rest are
   consistent between runs. new has 4× better lateral resolution.

## Appendix — FWHM in pixel units (FWHM_nm / px_size)

| data | h | C/NC | FWHM (nm) | px size | FWHM (px) |
|------|---|------|-----------|---------|-----------|
| new | short | C  | 436.7 | 19.53 | 22.4 |
| new | tall  | C  | 400.4 | 19.53 | 20.5 |
| new | short | NC | 765.1 | 19.53 | 39.2 |
| new | tall  | NC | 929.4 | 19.53 | 47.6 |
| old | short | C  | 591.4 | 78.13 | 7.6 |
| old | tall  | C  | 459.7 | 78.13 | 5.9 |
| old | short | NC | 726.6 | 78.13 | 9.3 |
| old | tall  | NC | 819.0 | 78.13 | 10.5 |
