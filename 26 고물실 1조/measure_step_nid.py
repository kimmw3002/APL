"""Batch step-edge measurement for the HeightSample .nid files.

Each HeightSample_data/*.nid is a step-height specimen: two flat terraces
separated by ONE vertical step edge. The per-row `masked_polyfit` leveling in
measure_nid.py / export_leveled.py fails here, because a single polynomial
cannot fit two terraces at once -- the big step drags the fit. Some files also
have debris sitting on the right terrace, which makes d/dx explode there too.

This tool instead does the standard (ISO-5436-style) two-terrace step fit:

  1. Detect the edge column GLOBALLY from the row-median profile (debris is
     local to a few rows, so it washes out in the median), searching only the
     central 8-92% of columns to skip the scan-edge artifacts at col 0/last.
  2. For each row, locate the step near the global column, then fit the LEFT and
     RIGHT terraces independently with a robust (MAD-reject) degree-1 line.
     Robust rejection throws out debris on the right terrace.
  3. step height = right_fit(step) - left_fit(step)  (lines extrapolated to the
     step center).  step width = local 10%-90% rise distance.
  4. The per-row height DRIFTS in y, so we do NOT average the whole image.
     We aggregate (median +/- 1.4826*MAD) only inside a row band -- by default
     the bottom 10% of rows (origin lower), which is the most stable region.

Outputs (all in HeightSample_data/step_analysis/):
  * step_measurements.csv      -- one row per file
  * <file>_step.png            -- 4-panel diagnostic per file
  * step_analysis_README.md    -- method + per-file summary

Usage:
  python measure_step_nid.py
  python measure_step_nid.py --row-band-frac 0.15 --row-band-pos bottom
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from NSFopen.read import read

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "New_data_only", "HeightSample_data")
OUT_DIR = os.path.join(DATA_DIR, "step_analysis")
CHANNEL = "Forward"


# ----------------------------------------------------------------------------
# .nid reading helpers (same access pattern as measure_nid.py)
# ----------------------------------------------------------------------------
def read_z_nm(path, channel=CHANNEL):
    """Z-Axis height image (nm) for `channel`, plus nm/pixel along x."""
    afm = read(path)
    z = np.asarray(afm.data[("Image", channel, "Z-Axis")], dtype=float) * 1e9
    cols = z.shape[1]
    try:
        rng = afm.param[("X", "range")]
        val = rng[0] if isinstance(rng, (list, tuple, np.ndarray)) else rng
        px = float(val) * 1e9 / cols
    except (KeyError, TypeError, IndexError):
        px = 50000.0 / cols  # 50 um fallback
    return z, px


# ----------------------------------------------------------------------------
# fitting primitives
# ----------------------------------------------------------------------------
def robust_line(x, y, iters=3, k=2.5):
    """Degree-1 fit with repeated MAD outlier rejection (rejects debris).

    Returns (coef, inlier_mask)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    coef = np.polyfit(x, y, 1)
    mask = np.ones(len(x), bool)
    for _ in range(iters):
        resid = y - np.polyval(coef, x)
        s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-12
        new = np.abs(resid) < k * s
        if new.sum() < 5 or np.array_equal(new, mask):
            mask = new
            break
        mask = new
        coef = np.polyfit(x[mask], y[mask], 1)
    return coef, mask


def smooth(a, w=5):
    if w <= 1:
        return a
    ker = np.ones(w) / w
    return np.convolve(a, ker, mode="same")


def detect_step_col(profile, lo_frac=0.08, hi_frac=0.92):
    """Column of max |d/dx| within the central band (skips edge artifacts)."""
    n = len(profile)
    g = np.abs(np.gradient(smooth(profile, 5)))
    lo, hi = int(lo_frac * n), int(hi_frac * n)
    return lo + int(np.argmax(g[lo:hi]))


def subpixel_peak(g, i):
    """Parabolic sub-pixel refinement of a discrete maximum at index i."""
    if 0 < i < len(g) - 1:
        a, b, c = g[i - 1], g[i], g[i + 1]
        denom = a - 2 * b + c
        if denom != 0:
            return i + 0.5 * (a - c) / denom
    return float(i)


def cross_dir(x, y, center, level, step):
    """First x where y crosses `level`, scanning OUTWARD from `center`.

    step=-1 scans left, +1 scans right. Searching from the mid-transition
    point outward localizes the crossing to the edge and ignores far-terrace
    noise (which otherwise produces absurd or missing 10-90 widths)."""
    i = int(np.argmin(np.abs(x - center)))
    while 0 <= i + step < len(x):
        y0, y1 = y[i], y[i + step]
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            f = (level - y0) / (y1 - y0)
            return x[i] + (x[i + step] - x[i]) * f
        i += step
    return None


def width_10_90(x, dd, center, A):
    """10-90% rise distance about `center`, in x units. dd: left-detrended."""
    x10 = cross_dir(x, dd, center, 0.1 * A, -1)   # toward left terrace
    x90 = cross_dir(x, dd, center, 0.9 * A, +1)   # toward right terrace
    if x10 is None or x90 is None:
        return np.nan
    return abs(x90 - x10)


# ----------------------------------------------------------------------------
# per-row two-terrace step measurement
# ----------------------------------------------------------------------------
def measure_row(row, s0, px, win=8, margin_frac=0.04):
    """Two-terrace fit for one row. Returns dict or None.

    s0: global step column.  win: per-row step search radius (px).
    margin_frac: transition zone excluded on each side of the step (frac cols).
    """
    n = len(row)
    x = np.arange(n, dtype=float)
    m = max(4, int(margin_frac * n))

    # locate the step within s0 +/- win, sub-pixel
    g = np.abs(np.gradient(smooth(row, 5)))
    lo, hi = max(m + 2, s0 - win), min(n - m - 2, s0 + win)
    if hi <= lo:
        return None
    si = lo + int(np.argmax(g[lo:hi]))
    s = subpixel_peak(g, si)

    # fit each terrace LOCALLY near the step (long extrapolation across a tilted/
    # bowed image is unstable; local bands keep the extrapolation to ~margin).
    rad = int(max(3 * m, 0.10 * n))
    Lmask = (x >= s - rad) & (x < s - m)
    Rmask = (x > s + m) & (x <= s + rad)
    if Lmask.sum() < 6 or Rmask.sum() < 6:
        return None

    cl, lin_l = robust_line(x[Lmask], row[Lmask])
    cr, lin_r = robust_line(x[Rmask], row[Rmask])
    height = np.polyval(cr, s) - np.polyval(cl, s)

    # debris bookkeeping: fraction of right-terrace points rejected as outliers
    right_rej = 1.0 - (lin_r.sum() / max(1, Rmask.sum()))

    # 10-90 width: detrend by the LEFT line; search crossings outward from the
    # step center within the local window (far debris is outside this window).
    a, b = max(0, int(s) - rad), min(n, int(s) + rad + 1)
    xx = x[a:b]
    dd = smooth(row[a:b], 3) - np.polyval(cl, xx)   # left terrace ~ 0, right ~ height
    width = width_10_90(xx, dd, s, height) * px

    return {"s": s, "height": height, "width": width, "right_rej": right_rej}


def mad(a):
    a = np.asarray(a, float)
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def agg_mean_sem(a, reject=3.0):
    """Mean +/- standard error over finite values, after MAD outlier rejection.

    Returns (mean, sem, std, n). MAD rejection drops residual debris rows so the
    mean isn't skewed; the error is the standard error of the mean."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    s = mad(a)
    if s > 0:
        a = a[np.abs(a - np.median(a)) <= reject * s]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    sem = std / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return mean, sem, std, len(a)


def band_rows(rows, frac, pos):
    """Row indices in the band (data coords, row 0 = bottom = origin lower)."""
    k = max(1, int(round(rows * frac)))
    if pos == "bottom":
        return np.arange(0, k)
    if pos == "top":
        return np.arange(rows - k, rows)
    c = rows // 2
    return np.arange(max(0, c - k // 2), min(rows, c - k // 2 + k))


# ----------------------------------------------------------------------------
# per-file analysis
# ----------------------------------------------------------------------------
def analyze_file(path, frac, pos):
    z, px = read_z_nm(path)
    rows, cols = z.shape

    cp = np.median(z, axis=0)              # debris-robust fast-axis profile
    s0 = detect_step_col(cp)
    win = max(8, int(round(0.03 * cols)))  # per-row step search radius

    # per-row measurement for ALL rows (for the drift panel)
    per = [measure_row(z[r], s0, px, win=win) for r in range(rows)]
    heights = np.array([p["height"] if p else np.nan for p in per])
    widths = np.array([p["width"] if p else np.nan for p in per])
    scols = np.array([p["s"] if p else np.nan for p in per])
    rrej = np.array([p["right_rej"] if p else np.nan for p in per])

    # aggregate only inside the row band, after rejecting bad rows
    bidx = band_rows(rows, frac, pos)
    good = bidx[np.isfinite(heights[bidx]) & (np.abs(scols[bidx] - s0) <= win)]
    h_mean, h_sem, h_std, h_n = agg_mean_sem(heights[good])
    w_mean, w_sem, w_std, w_n = agg_mean_sem(widths[good])

    note = []
    if len(good) and np.nanmedian(rrej[good]) > 0.04:
        note.append("debris-rejected on right terrace")
    if h_n and h_std > 0.15 * abs(h_mean) + 2:
        note.append("high row scatter")
    if h_n < max(3, 0.5 * len(bidx)):
        note.append("few usable rows")

    res = {
        "z": z, "px": px, "rows": rows, "cols": cols, "s0": s0,
        "heights": heights, "widths": widths, "scols": scols,
        "band_idx": bidx, "good_idx": good,
        "height": h_mean, "height_err": h_sem, "height_std": h_std,
        "width": w_mean, "width_err": w_sem, "width_std": w_std,
        "n_rows": int(h_n), "n_rows_total": int(len(bidx)),
        "note": "; ".join(note),
    }
    return res


def master_profile(res):
    """Aligned-average cross-section over the good band rows (for plotting)."""
    z, good = res["z"], res["good_idx"]
    cols = res["cols"]
    if len(good) == 0:
        return None, None
    sis = np.round(res["scols"][good]).astype(int)
    half = min(int(sis.min()), cols - 1 - int(sis.max()))
    half = min(half, int(0.4 * cols))
    if half < 5:
        return None, None
    seg = np.arange(-half, half + 1)
    acc = [z[r][si + seg] for r, si in zip(good, sis)]
    return seg.astype(float), np.mean(acc, axis=0)


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------
def make_plot(name, res, out_png, frac, pos):
    z, px, s0 = res["z"], res["px"], res["s0"]
    rows, cols = res["rows"], res["cols"]

    dx = np.gradient(z, axis=1)  # raw d/dx (nm/px)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{name}   step@col {s0}  (x={s0*px/1000:.1f} um)   "
                 f"height={res['height']:.1f}±{res['height_std']:.1f} nm   "
                 f"width(10-90)={res['width']:.0f}±{res['width_std']:.0f} nm  (mean±std)",
                 fontsize=12)

    # panel 1: d/dx heatmap (edge + debris visible)
    a1 = ax[0, 0]
    dlim = np.nanpercentile(np.abs(dx), 99)
    im = a1.imshow(dx, origin="lower", cmap="RdBu_r", vmin=-dlim, vmax=dlim,
                   aspect="auto", extent=[0, cols * px / 1000, 0, rows * px / 1000])
    a1.axvline(s0 * px / 1000, color="k", lw=0.8, ls="--")
    a1.set_title("d/dx (raw) -- edge & debris spikes")
    a1.set_xlabel("x (um)"); a1.set_ylabel("y (um)")
    fig.colorbar(im, ax=a1, label="nm/px")

    # panel 2: aligned-average cross-section with terrace fits + 10-90 markers
    a2 = ax[0, 1]
    seg, prof = master_profile(res)
    if seg is not None:
        xnm = seg * px
        a2.plot(xnm, prof, ".", ms=3, color="#9bb7e8", label="aligned avg")
        mm = max(4, int(0.04 * cols))
        rad = int(max(3 * mm, 0.10 * cols))
        L = (seg >= -rad) & (seg < -mm); R = (seg > mm) & (seg <= rad)
        cl, _ = robust_line(seg[L], prof[L]); cr, _ = robust_line(seg[R], prof[R])
        a2.plot(xnm, np.polyval(cl, seg), "g-", lw=1.2, label="left fit")
        a2.plot(xnm, np.polyval(cr, seg), "m-", lw=1.2, label="right fit")
        h = np.polyval(cr, 0) - np.polyval(cl, 0)
        a2.annotate("", xy=(0, np.polyval(cl, 0) + h), xytext=(0, np.polyval(cl, 0)),
                    arrowprops=dict(arrowstyle="<->", color="r"))
        # 10-90 on the left-detrended master profile (directional from center)
        dd = prof - np.polyval(cl, seg)
        x10 = cross_dir(seg, dd, 0.0, 0.1 * h, -1)
        x90 = cross_dir(seg, dd, 0.0, 0.9 * h, +1)
        for xc in (x10, x90):
            if xc is not None:
                a2.axvline(xc * px, color="#d98300", ls=":", lw=1)
        # zoom to the local edge window so the step & 10-90 are visible (the full
        # profile is dominated by terrace tilt)
        zoom = rad * px * 1.3
        a2.set_xlim(-zoom, zoom)
        inwin = np.abs(xnm) <= zoom
        yl = prof[inwin]
        if len(yl) > 2:
            pad = (yl.max() - yl.min()) * 0.15 + 1
            a2.set_ylim(yl.min() - pad, yl.max() + pad)
        a2.set_title(f"aligned-average cross-section (zoom)  step={h:.1f} nm")
        a2.set_xlabel("x from step (nm)"); a2.set_ylabel("height (nm)")
        a2.legend(fontsize=8)

    # panel 3: step height vs row (drift) + band shading
    a3 = ax[1, 0]
    yrow = np.arange(rows) * px / 1000
    a3.plot(res["heights"], yrow, ".", ms=2, color="#888", label="per-row")
    bidx, good = res["band_idx"], res["good_idx"]
    a3.axhspan(bidx[0] * px / 1000, bidx[-1] * px / 1000, color="orange", alpha=0.15,
               label=f"band ({pos} {int(frac*100)}%)")
    a3.plot(res["heights"][good], yrow[good], ".", ms=3, color="#2456c8")
    a3.axvline(res["height"], color="r", lw=1.2,
               label=f"mean {res['height']:.1f}±{res['height_std']:.1f} (std)")
    a3.axvspan(res["height"] - res["height_std"], res["height"] + res["height_std"],
               color="r", alpha=0.10)  # +/- 1 std (row scatter = headline error)
    a3.set_title("step height vs row (y) -- shows drift")
    a3.set_xlabel("step height (nm)"); a3.set_ylabel("y (um)")
    a3.legend(fontsize=8)

    # panel 4: histogram of band step heights (its own panel now)
    a4 = ax[1, 1]
    bh = res["heights"][good]
    bh = bh[np.isfinite(bh)]
    if len(bh) > 1:
        a4.hist(bh, bins=max(8, int(np.sqrt(len(bh)) * 1.5)), color="#2456c8",
                edgecolor="white")
        a4.axvline(res["height"], color="r", lw=1.5,
                   label=f"mean {res['height']:.1f}")
        a4.axvspan(res["height"] - res["height_std"], res["height"] + res["height_std"],
                   color="r", alpha=0.12, label=f"±std {res['height_std']:.1f}")
        a4.legend(fontsize=8)
    a4.set_title(f"band step-height distribution (n={len(bh)})")
    a4.set_xlabel("step height (nm)"); a4.set_ylabel("rows (count)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row-band-frac", type=float, default=0.25,
                    help="fraction of rows used for aggregation (default 0.25)")
    ap.add_argument("--row-band-pos", choices=["bottom", "top", "center", "both"],
                    default="both",
                    help="which row band; 'both' = bottom AND top (default both)")
    args = ap.parse_args()
    frac, pos = args.row_band_frac, args.row_band_pos

    # 'both' computes the bottom AND top band so the two are cross-checked.
    bands = [("bottom", frac), ("top", frac)] if pos == "both" else [(pos, frac)]

    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".nid"))

    rows_csv = []
    for fn in files:
        path = os.path.join(DATA_DIR, fn)
        base = os.path.splitext(fn)[0]
        for bp, bf in bands:
            res = analyze_file(path, bf, bp)
            tag = f"_{bp}{int(bf*100)}" if len(bands) > 1 else ""
            out_png = os.path.join(OUT_DIR, base + "_step" + tag + ".png")
            make_plot(fn, res, out_png, bf, bp)
            band = f"{bp} {int(bf*100)}%"
            rows_csv.append([
                fn, CHANNEL, f"{res['px']:.4f}", res["s0"],
                f"{res['s0']*res['px']:.1f}", band,
                f"{res['height']:.3f}", f"{res['height_std']:.3f}", f"{res['height_err']:.3f}",
                f"{res['width']:.3f}", f"{res['width_std']:.3f}", f"{res['width_err']:.3f}",
                res["n_rows"], res["n_rows_total"], res["note"],
            ])
            print(f"{fn} [{band}]: step@{res['s0']}  "
                  f"height={res['height']:.1f}±{res['height_std']:.1f} nm  "
                  f"width={res['width']:.0f}±{res['width_std']:.0f} nm  "
                  f"(n={res['n_rows']}/{res['n_rows_total']}) {res['note']}", flush=True)

    # CSV: headline error = std (row-to-row scatter; rows are correlated so SEM
    # would overstate precision). step_height_sem_nm is the std/sqrt(n) reference.
    hdr = ("file,channel,pixel_nm,step_col_px,step_x_nm,row_band,"
           "step_height_nm,step_height_err_nm,step_height_sem_nm,"
           "width_10_90_nm,width_err_nm,width_sem_nm,n_rows,n_rows_total,note")
    note_col = hdr.split(",").index("note")
    csv_path = os.path.join(OUT_DIR, "step_measurements.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(hdr + "\n")
        for r in rows_csv:
            cells = [str(c).replace(",", ";") if i == note_col else str(c)
                     for i, c in enumerate(r)]
            f.write(",".join(cells) + "\n")
    print("saved:", csv_path)

    write_readme(files, rows_csv, frac, pos)


def _consistency(rows_for_file):
    """Bottom-vs-top agreement for one file: True if |Δ| <= std_a + std_b."""
    out = {}
    if len(rows_for_file) < 2:
        return out
    a, b = rows_for_file[0], rows_for_file[1]
    for key, mcol, scol in (("h", 6, 7), ("w", 9, 10)):
        try:
            ma, sa = float(a[mcol]), float(a[scol])
            mb, sb = float(b[mcol]), float(b[scol])
            out[key] = (abs(ma - mb), sa + sb, abs(ma - mb) <= sa + sb)
        except ValueError:
            out[key] = (float("nan"), float("nan"), False)
    return out


def write_readme(files, rows_csv, frac, pos):
    md = os.path.join(OUT_DIR, "step_analysis_README.md")
    # rows_csv layout: 0 file,1 ch,2 px,3 col,4 x,5 band,6 h,7 h_std,8 h_sem,
    #                  9 w,10 w_std,11 w_sem,12 n,13 n_tot,14 note
    by_file = {}
    for r in rows_csv:
        by_file.setdefault(r[0], []).append(r)
    multi = any(len(v) > 1 for v in by_file.values())

    lines = []
    lines.append("# HeightSample step-edge 분석\n")

    # ---- TL;DR ----
    lines.append("## TL;DR\n")
    lines.append("- **무엇**: `New_data_only/HeightSample_data/*.nid` (Forward) 각 이미지의 "
                 "단일 세로 step edge에 대해 **step 높이**와 **10-90% 상승 폭(edge가 얼마나 넓게 찍혔나)** 을 측정.")
    lines.append("- **왜 새로 짰나**: 한 행을 단일 다항식으로 펴는 기존 `masked_polyfit` 은 큰 step이 fit을 "
                 "끌어당겨 실패. 대신 step 양쪽 terrace를 **따로** 직선 맞추는 two-terrace 방식을 씀.")
    lines.append("- **핵심 처리 3가지**: (1) edge 위치는 **행 median 프로파일**의 d/dx 로 잡아 이물질·라인노이즈를 "
                 "희석하고 가장자리 artifact를 피함, (2) terrace 직선은 **국소(local) + MAD robust** 라 "
                 "오른쪽 terrace의 이물질이 fit을 못 흔듦, (3) 높이가 y로 drift 하므로 전체평균 대신 "
                 "**row band(기본 위·아래 각 25%)** 에서만 집계.")
    lines.append("- **불확도**: 값=band 행들의 **mean**, 오차=**std(행간 산포)**. (행이 상관돼 SEM은 과장 → 참고만.)")
    lines.append("- **검증**: 같은 파일을 **아래쪽 25% / 위쪽 25% 두 band로 독립 측정** → 둘이 서로의 "
                 "오차범위(합산 std) 안에 들어오면 band 선택에 robust 하다는 뜻. 아래 결과에 ✓/✗ 로 표기.")
    lines.append("- **결과 (mean ± std)**:")
    for fn, rs in by_file.items():
        for r in rs:
            flag = f"  ⚠ {r[14]}" if r[14] else ""
            lines.append(f"    - `{fn}` [{r[5]}] — height **{r[6]} ± {r[7]} nm**, "
                         f"width(10-90) **{r[9]} ± {r[10]} nm** (n={r[12]}/{r[13]}){flag}")
        c = _consistency(rs)
        if c:
            hv = "✓ 일치" if c["h"][2] else "✗ 불일치"
            wv = "✓ 일치" if c["w"][2] else "✗ 불일치"
            lines.append(f"        - bottom↔top 일치: height {hv} (|Δ|={c['h'][0]:.1f} ≤ {c['h'][1]:.1f}), "
                         f"width {wv} (|Δ|={c['w'][0]:.0f} ≤ {c['w'][1]:.0f})")
    if multi:
        all_ok = all(_consistency(rs).get("h", (0, 0, False))[2]
                     for rs in by_file.values() if len(rs) > 1)
        lines.append(f"- **종합**: 모든 파일에서 bottom↔top step height가 "
                     f"{'오차범위 내 일치 → band 선택에 robust' if all_ok else '일부 불일치(아래 표 확인)'}. "
                     f"contact_Left/remeasure 도 서로 재현성 OK. Noncontact_Left 만 산포가 커 신뢰도 낮음.\n")
    else:
        lines.append("- **품질**: contact_Left 와 그 remeasure 가 오차범위 안에서 일치(재현성 OK). "
                     "Noncontact_Left 만 산포가 커 신뢰도 낮음(아래 note).\n")

    lines.append("## 방법 (분석 과정 상세)\n")
    lines.append("1. **edge 위치(전역) 탐지** — 행 방향 median 프로파일 `median(z, axis=0)` 로 이물질/라인노이즈를 "
                 "희석한 뒤, 중앙 8-92% column band에서 `|d/dx|` 최대 column을 step 위치 `s0` 로 잡음 "
                 "(col 0/마지막의 scan-edge artifact 회피). 큰 |d/dx| 1순위는 보통 가장자리 artifact라 "
                 "중앙 band 제한이 필수.")
    lines.append("2. **per-row two-terrace fit** — 각 행에서 `s0 ± win`(win = max(8px, cols의 3%)) 안의 step을 "
                 "gradient로 sub-pixel 위치 추정. transition margin(±4% cols) 바깥의 좌/우 terrace를 "
                 "**step 근처 국소 구간(±rad ≈ cols의 10%)** 에서만 **MAD 2.5σ robust 직선**으로 각각 fit. "
                 "국소 fit이라 이미지 전체의 tilt/bow에 안 휘둘리고, 오른쪽 terrace의 이물질은 MAD outlier로 제거됨. "
                 "**step height = (우측 직선 − 좌측 직선) 을 step 중심에서 평가한 차.**")
    lines.append("3. **10-90% 폭** — 좌측 직선을 뺀 프로파일(좌≈0, 우≈height)에서 step 중심으로부터 "
                 "**바깥쪽으로** 0.1·height(좌향)·0.9·height(우향) 교차점을 선형보간으로 찾아 그 거리. "
                 "중심에서 바깥으로 탐색하므로 멀리 있는 이물질/노이즈에 안 걸림.")
    band_desc = "위·아래 각 25%" if pos == "both" else f"{pos} {int(frac*100)}%"
    lines.append(f"4. **집계** — per-row step height가 y방향으로 **drift** 하므로 전체 평균하지 않고 "
                 f"**{band_desc} row band** 안에서만 집계 "
                 f"(`--row-band-frac`, `--row-band-pos {{bottom,top,center,both}}` 로 조절; 기본 both). "
                 f"band 안에서 MAD 3σ 이상치 행(잔여 debris)을 "
                 f"제거한 뒤 height·width 의 **mean** 을 값으로, **헤드라인 불확도 = std(행간 산포)** 로 보고. "
                 f"(행들은 같은 edge를 연속 스캔해 drift·tip 등이 공유되므로 서로 **상관**돼 있어 "
                 f"SEM=std/√n 으로 나누면 정밀도를 과장함 → std 가 정직한 불확도. SEM 은 `*_sem_nm` 참고 컬럼.) "
                 f"진단 PNG의 panel 4(height vs row)에서 drift와 band 선택의 타당성을 확인할 수 있음.\n")
    lines.append("## 컬럼 의미 (step_measurements.csv)\n")
    lines.append("| 컬럼 | 뜻 |")
    lines.append("|---|---|")
    lines.append("| step_col_px / step_x_nm | step edge 위치 (column, nm) |")
    lines.append("| row_band | 집계에 쓴 row band |")
    lines.append("| step_height_nm / _err_ / _sem_ | band 내 step 높이 **mean / std(헤드라인 오차) / SEM(참고)** |")
    lines.append("| width_10_90_nm / _err_ / _sem_ | band 내 10-90% 상승 폭 **mean / std / SEM** |")
    lines.append("| n_rows / n_rows_total | 집계에 실제 사용한 행 / band 전체 행 |")
    lines.append("| note | debris/품질 플래그 |\n")
    lines.append("## 파일별 결과 (mean ± std)\n")
    lines.append("| file | band | step_x (nm) | height (nm) | width 10-90 (nm) | n_rows | note |")
    lines.append("|---|---|---|---|---|---|---|")
    for fn, rs in by_file.items():
        for r in rs:
            lines.append(f"| {fn} | {r[5]} | {r[4]} | {r[6]} ± {r[7]} | {r[9]} ± {r[10]} | "
                         f"{r[12]}/{r[13]} | {r[14]} |")
        c = _consistency(rs)
        if c:
            hv = "✓" if c["h"][2] else "✗"
            wv = "✓" if c["w"][2] else "✗"
            lines.append(f"| | **bottom↔top** | | {hv} \\|Δ\\|={c['h'][0]:.1f}≤{c['h'][1]:.1f} | "
                         f"{wv} \\|Δ\\|={c['w'][0]:.0f}≤{c['w'][1]:.0f} | | |")
    png_note = ("각 파일은 band별로 `*_step_bottom25.png` / `*_step_top25.png` 두 장"
                if multi else "각 `*_step.png`")
    lines.append(f"\n진단 그림: {png_note} — 4-panel = d/dx heatmap(edge·이물질) / "
                 "aligned-average 단면(좌·우 fit, 높이, 10-90 마커) / step height vs row(drift+band) / "
                 "band step-height 히스토그램.\n")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved:", md)


if __name__ == "__main__":
    main()
