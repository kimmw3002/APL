# AFM line-to-line drift and the d/dy stripe artifact

## Symptom

In differential-mode plots, the derivative along the **slow-scan (y)** axis
(`d/dy`) and the gradient magnitude are full of horizontal stripes, while the
derivative along the **fast-scan (x)** axis (`d/dx`) is clean.

Measured on `JJ_data/Data_1_Raw.csv`:

| quantity | value |
|---|---|
| fraction of `d/dy` variance from a per-row constant offset | **~80%** |
| same fraction for `d/dx` | **~0.01%** |

So the stripes are **line-to-line baseline offsets**, not real topography.

## Why the drift happens (per-row drift)

Raster scanning separates two timescales. With `Time/Line = 1 s` and 512 lines,
one line is swept in ~1 s but the **whole frame takes ~8.5 minutes**. The slow
(y) axis advances one line at a time across those minutes. Every drift source
acts on the seconds-to-minutes timescale, so it barely moves during a single
line but accumulates strongly between lines:

- **Thermal drift** — small temperature changes expand/contract the scanner,
  sample, and cantilever holder, shifting the cantilever rest deflection and the
  tip–sample distance by nm over tens of seconds. Usually the dominant source.
- **Piezo creep & hysteresis** — after the y-piezo steps, it keeps relaxing
  toward its commanded position, nudging the absolute Z baseline line by line.
  Strongest near the start of a scan.
- **Feedback baseline / setpoint wander** — low integral gain, a drifting
  deflection/amplitude setpoint, or laser-spot/photodiode drift slowly change the
  height the loop settles to.
- **1/f (low-frequency) noise** — electronic and mechanical noise concentrated
  at long timescales, i.e. between lines rather than within a line.

All are **slow (≫ 1 s)**, so to a single 1-second line they look like a constant
offset — and that offset changes from one line to the next.

## Why each row is "safe"

A row is safe in **relative** height, not absolute:

1. **Acquired fast and together** — all points of a line are taken within ~1 s,
   so slow drift moves a negligible amount across the line. Every point shares
   essentially the same instantaneous baseline.
2. **Common-mode cancellation** — because they share that baseline, point-to-point
   differences within a row (`d/dx`, relative topography) cancel the drift; it is
   a common offset that subtracts away.
3. **Per-row leveling mops up the rest** — any small residual tilt accumulated
   during the 1 s is removed by the per-line degree-1 fit.

## Consequence

- **Within a row** the fast scan outruns the drift → relative heights are
  trustworthy → `d/dx` is clean.
- **Between rows** the drift has had ~1 s (×512) to accumulate → absolute offsets
  are corrupted → `d/dy` is striped.

This is the same artifact as the horizontal seams in the leveled plots, and it is
exactly why **line-by-line leveling** exists: it discards the one quantity (each
line's absolute offset) that the drift actually corrupts.

## Fix

Level the rows (subtract each line's offset/tilt) **before** differentiating.
Then `d/dy` reflects real topography and the stripes disappear.
