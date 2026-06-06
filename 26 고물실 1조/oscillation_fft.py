"""Extract the z-level oscillation seen in the left part of an AFM scan and
measure its frequency in Hz.

Context
-------
In New_data_only/JJ_data_0603/JJ_AFM_4.nid and JJ_AFM_2.nid the leveled
topography (deg-2 masked polyfit, k=2.5, right_exclude=0.0) shows a clean
sinusoidal z oscillation in the left ~30% of every scan line (the clean
substrate region, before the sample feature on the right).

Each scan line takes 500 ms, so the fast (column) axis is a time axis:
    dt = 0.500 s / N_cols   per pixel.
The oscillation is *free-running* (its phase drifts line to line, so it shows
as diagonal stripes and averaging lines together cancels it). We therefore
treat each clean substrate segment as its own time-series segment and average
the per-segment periodograms (Welch's method) to recover the frequency.

The sample feature sits in the MIDDLE of the line, so we use the left AND right
EDGE_FRAC (25%) of every line as clean segments -> 2 segments per line.

Outputs, per file, a 3-panel PNG and prints the peak frequency.
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from NSFopen.read import read

from export_leveled import masked_polyfit

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "New_data_only/JJ_data_0603/JJ_AFM_4.nid",
    "New_data_only/JJ_data_0603/JJ_AFM_2.nid",
]

LINE_TIME = 0.500     # s per scan line (given)
EDGE_FRAC = 0.25      # use the left AND right 25% of each line (clean substrate)
ROW_FRAC = 0.20       # drop the top AND bottom 20% of rows (start/end drift)
CHANNEL = "Forward"


def parabolic_peak(P, k):
    """Sub-bin peak location (in bins) by parabolic interpolation around k."""
    if k <= 0 or k >= len(P) - 1:
        return float(k)
    a, b, c = P[k - 1], P[k], P[k + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(k)
    return k + 0.5 * (a - c) / denom


def mode_label(afm):
    """contact / non-contact from the nid metadata (Op. mode)."""
    info = afm.param[("HeaderDump", "DataSet-Info")]
    mode = str(info.get("Op. mode", ""))
    if "Static" in mode:
        return "contact"
    if "Dynamic" in mode:
        return "non-contact"
    return "unknown"


def detrend_rows(block):
    """Remove a per-row linear trend (avoids spectral leakage from a ramp)."""
    out = np.empty_like(block)
    t = np.arange(block.shape[1])
    for i, row in enumerate(block):
        a, b = np.polyfit(t, row, 1)
        out[i] = row - (a * t + b)
    return out


def analyze(rel):
    afm = read(os.path.join(HERE, rel))
    label = mode_label(afm)            # contact / non-contact (for the filename)
    z = np.asarray(afm.data[("Image", CHANNEL, "Z-Axis")], float)
    lev = masked_polyfit(z, k=2.5, deg=2, right_exclude=0.0) * 1e9  # nm
    R, C = lev.shape
    nE = int(C * EDGE_FRAC)
    r0 = int(R * ROW_FRAC)
    r1 = R - r0                        # keep the middle rows only
    dt = LINE_TIME / C                 # s per column (within a line)

    # clean substrate segments: left and right edge of the kept (middle) lines
    # (the feature is in the middle columns; start/end rows have drift).
    core = lev[r0:r1]
    disp_left = core[:, :nE]           # leveled bands actually analyzed (for display)
    disp_right = core[:, -nE:]
    left = detrend_rows(disp_left)     # per-row linear detrend before FFT
    right = detrend_rows(disp_right)
    segs = np.vstack([left, right])

    # averaged periodogram (Welch), Hann window.
    # The oscillation phase drifts line to line, so we average POWER across
    # segments (incoherent average) and take sqrt at the end.
    win = np.hanning(nE)
    cg = np.sum(win)                  # coherent gain of the window (for amplitude)
    freq = np.fft.rfftfreq(nE, dt)
    P = np.zeros(len(freq))
    for row in segs:
        P += np.abs(np.fft.rfft(row * win)) ** 2
    P /= len(segs)

    # peak (skip DC bin), refined to sub-bin precision
    pk = np.argmax(P[1:]) + 1
    df = freq[1]
    f_peak = parabolic_peak(P, pk) * df
    # Coherent-gain amplitude calibration: for a pure tone z=A*cos(2*pi*f0*t)
    # the windowed rFFT peak is |X|=A/2*sum(win), so 2*|X|/sum(win) = A.
    # -> peak of amp_spectrum is the physical peak amplitude A (nm) of the tone.
    amp_spectrum = 2.0 * np.sqrt(P) / cg   # peak amplitude per bin (nm)

    return dict(rel=rel, label=label, lev=lev, disp_left=disp_left, disp_right=disp_right,
                nE=nE, r0=r0, r1=r1, dt=dt, R=R, C=C,
                nseg=len(segs), freq=freq, P=P, amp=amp_spectrum, pk=pk,
                df=df, f_peak=f_peak)


def plot(res):
    rel = res["rel"]
    base = res["label"]               # filename = contact / non-contact (from metadata)
    out = os.path.join(HERE, os.path.dirname(rel), base + ".png")
    lev, nE, r0, r1, dt = res["lev"], res["nE"], res["r0"], res["r1"], res["dt"]
    disp_left, disp_right = res["disp_left"], res["disp_right"]
    freq, amp, f_peak = res["freq"], res["amp"], res["f_peak"]

    line_ms = LINE_TIME * 1e3
    edge_ms = EDGE_FRAC * line_ms
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    lim = np.percentile(np.abs(np.concatenate([disp_left.ravel(), disp_right.ravel()])), 99)

    # (1) full leveled image; mark the column edges + row cut actually used
    im = ax[0].imshow(lev, cmap="RdBu_r", aspect="auto", origin="lower",
                      vmin=-lim, vmax=lim, extent=[0, line_ms, 0, res["R"]])
    for xv in (edge_ms, line_ms - edge_ms):
        ax[0].axvline(xv, color="k", ls="--", lw=1.0)
    for yv in (r0, r1):
        ax[0].axhline(yv, color="k", ls="--", lw=1.0)
    ax[0].set_title(f"{base}: full leveled z (nm) — analyzed box dashed")
    ax[0].set_xlabel("time within line (ms)")
    ax[0].set_ylabel("scan line #")
    fig.colorbar(im, ax=ax[0], shrink=0.8, label="z (nm)")

    # (2) leveled heatmap of ONLY the analyzed region: left band | right band
    band = np.hstack([disp_left, disp_right])
    im2 = ax[1].imshow(band, cmap="RdBu_r", aspect="auto", origin="lower",
                       vmin=-lim, vmax=lim, extent=[0, 2 * nE, r0, r1])
    ax[1].axvline(nE, color="k", lw=1.2)   # divider between left and right bands
    ax[1].set_title(f"analyzed region (leveled z, nm)\nleft {int(EDGE_FRAC*100)}% | right {int(EDGE_FRAC*100)}%, rows {r0}-{r1}")
    ax[1].set_xlabel("edge column (left band | right band)")
    ax[1].set_ylabel("scan line #")
    fig.colorbar(im2, ax=ax[1], shrink=0.8, label="z (nm)")

    # (3) averaged amplitude spectrum vs Hz
    ax[2].plot(freq, amp, lw=1.0)
    ax[2].axvline(f_peak, color="r", ls="--", lw=1.0)
    ax[2].set_title(f"averaged spectrum (Welch, {res['nseg']} segments)")
    ax[2].set_xlabel("frequency (Hz)")
    ax[2].set_ylabel("peak amplitude (nm)")
    ax[2].set_xlim(0, min(300, freq[-1]))
    ax[2].annotate(f"{f_peak:.1f} Hz",
                   xy=(f_peak, amp[res['pk']]),
                   xytext=(f_peak + 20, amp[res['pk']] * 0.9),
                   color="r", fontsize=10,
                   arrowprops=dict(arrowstyle="->", color="r"))

    fig.suptitle(f"{base}  -  z oscillation at line edges  "
                 f"(line={line_ms:.0f} ms, dt={dt*1e3:.4f} ms/col, "
                 f"fs={1/dt:.0f} Hz, res={res['df']:.2f} Hz)")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    for rel in FILES:
        res = analyze(rel)
        out = plot(res)
        print(f"{rel}  -> {res['label']}")
        print(f"   N_cols={res['C']}  edge={res['nE']} cols x2  "
              f"rows {res['r0']}-{res['r1']}  segments={res['nseg']}")
        print(f"   dt={res['dt']*1e3:.4f} ms/col  fs={1/res['dt']:.1f} Hz  "
              f"Nyquist={0.5/res['dt']:.1f} Hz  resolution={res['df']:.2f} Hz")
        print(f"   PEAK = {res['f_peak']:.2f} Hz  (period {1e3/res['f_peak']:.3f} ms)")
        print(f"   saved: {out}")


if __name__ == "__main__":
    main()
