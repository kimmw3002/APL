# Measurement metrics in `measure_nid.py`

When you draw a line across a feature, the tool samples a **profile** and reports
**rod height**, **FWHM**, and a **right−left height difference**, all on a single
**peak − baseline** basis. This note explains how each value and its uncertainty
are computed, including the swath averaging (with cross-section alignment) and the
auto/manual baseline options.

Everything is computed in `analyze()` in the embedded page of `measure_nid.py`.

## The profile (swath averaging + alignment)

The line is sampled at uniform steps along its length. At each step the tool
averages **W parallel cross-sections** offset perpendicular to the line — the
"swath" (*Swath width* slider, in pixels). A rod is extended along its axis, so
averaging W cross-sections cuts random noise by ~√W. Off-grid samples use
bilinear interpolation. `W = 1` is a single 1-pixel line.

**Alignment (registration), the "align swath" checkbox — default on.** The swath
offsets are perpendicular to the *drawn line*. If that direction is not exactly
the rod's long axis (line slightly off-perpendicular, or a curved/tilted rod),
each offset cross-section cuts the rod at a slightly shifted lateral position;
plain averaging then smears the edges and **a rectangular rod's flat top
(plateau) rounds into a bump**. With alignment on, each cross-section is shifted
to the integer lag that maximizes its cross-covariance with the center
cross-section *before* averaging, so sharp edges and the plateau survive. (This
matters for Josephson-junction rods, which are rectangular and must keep a flat
top.) Turn it off to see the raw, un-registered average.

- `prof[i]` = swath-averaged (optionally aligned) value at distance `dist[i]` (nm).
- The profile comes from the **leveled** or **raw** map ("Measure on").

Throughout, `std(·)` is the population standard deviation, `median(·)` the sample
median.

## Baseline: which points are "background" vs "rod"

Two modes (the *Baseline* selector):

- **auto** — the *outer `k%`* of each end is baseline (`k = round(N·frac)`, the
  "Baseline window" slider); the inner part is the rod region.
- **manual** — drag the **rod start / rod end** handles on the profile; every
  sample *outside* `[ba, bb]` is baseline, inside is rod. Use this for low-SNR
  rods whose tails would otherwise leak into an auto window.

A **baseline line** `base(x) = blBase + blSlope·x` is fit by linear least squares
to the baseline points (so a residual-tilted background is handled). The shared
noise estimate is the scatter of the baseline points about that line:

```
noise = std( prof[baseline] − base(dist[baseline]) )
```

---

## 1. Rod height = peak − baseline

```
det(x) = prof(x) − base(x)                 # baseline-subtracted profile
peak   = max( det over the rod region )    # tallest point inside the rod
height = peak
```

**Error:** `σ_height = noise` (the baseline scatter). The peak is taken as the
tallest sample; swath averaging (and alignment) is what makes that sample
trustworthy at low SNR. For a flat-topped (rectangular) rod the peak sits on the
plateau, so the plateau level minus baseline is what's measured.

---

## 2. FWHM (full width at half maximum)

```
half = height/2                            # in baseline-subtracted units
```

From the peak, walk outward to the first sample on each side that drops below
`half`, and linearly interpolate the crossing position between the two bracketing
samples; `FWHM = |xR − xL|`. (For a trapezoidal/rectangular rod the crossings
land on the sloped sides, giving the correct full width at half height.) Reported
as "—" if a side never reaches half-max (line too short).

**Error — baseline noise propagated through the crossing slope**
```
slopeL = (prof[i]−prof[i−1])/(dist[i]−dist[i−1])    # left crossing
slopeR = (prof[i+1]−prof[i])/(dist[i+1]−dist[i])    # right crossing
σ_FWHM = √( (noise/|slopeL|)² + (noise/|slopeR|)² )
```
Steep edges → tight FWHM error; shallow edges → large error; a flat crossing
gives an effectively infinite per-side term.

---

## 3. Height difference = right − left

A step-height metric: the difference between the two baseline plateaus.

```
left  = median(baseline points left of center)
right = median(baseline points right of center)
Δh    = right − left
σ_Δh  = √( std(left)² + std(right)² )      # difference of two independent means
```

Best for step samples (flat-left vs flat-right plateaus). On the profile the two
end levels are drawn as **purple** segments.

---

## What you see (so the analysis is legible)

- **On the heatmaps**: the swath band (the averaged area), the line colored
  **orange over the rod range** and **gray-dashed over the baseline**, with ticks
  at the rod boundaries and the peak position — so the rod range looks sensical on
  the real image.
- **In the profile panel** (kept deliberately clean — no text labels): baseline
  (gray) and rod (orange) shaded regions; points colored by role (rod blue,
  baseline gray); the dashed baseline and half-max lines; a vertical **height**
  guide and a horizontal **FWHM** guide with crossing markers; the purple
  left/right end levels; and draggable rod-start/end handles in manual mode. The
  numeric values live in the readout below the plot and in the results table.

---

## Shared caveats

Deliberately simple, conservative error bars — they describe how much the
**background scatters**, not formal confidence intervals:

- They use the raw `noise` (baseline std), not the standard error of the mean
  (`σ/√n`). Swath averaging reduces `noise` (and thus the error bars) by ~√W.
- They assume the baseline points are pure background. In auto mode a feature
  tail leaking into the outer `k%` inflates `noise` and biases `base`; use manual
  mode to exclude it.
- The peak is treated as one sample (no peak-pixel noise, no tip convolution).
- Alignment uses an integer-pixel lag from cross-covariance; a rod with very low
  contrast or a swath wider than the rod's straight section can still mis-register
  — widen the swath only over the straight part of the rod.
- Pixels are treated as independent; real AFM noise is line-to-line correlated,
  so true uncertainties can be larger than the quadrature suggests.

See also [`masked_polyfit_explained.md`](masked_polyfit_explained.md) for how the
leveled map (the default measurement surface) is produced, and
[`AFM_line_drift_notes.md`](AFM_line_drift_notes.md) for the per-row drift that
motivates leveling.
