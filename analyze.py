import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)


def discover_groups(pattern, regex):
    """Find files matching pattern, group by regex capture group 1."""
    files = glob.glob(os.path.join(BASE_DIR, pattern))
    groups = {}
    for fpath in files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        m = re.match(regex, fname)
        if not m:
            continue
        group, num = m.group(1), int(m.group(2))
        groups.setdefault(group, []).append((num, fname))
    for g in groups:
        groups[g].sort()
        groups[g] = [label for _, label in groups[g]]
    sorted_g = sorted(groups.keys(), key=lambda g: int(re.match(r"(\d+)", g).group(1)))
    return groups, sorted_g


def run_analysis(series_name, groups, sorted_groups):
    """Run full analysis pipeline for one series."""
    prefix = f"[{series_name}] "
    img_prefix = f"{series_name}_" if series_name else ""
    all_labels = [label for g in sorted_groups for label in groups[g]]

    print(f"\n{'='*60}")
    print(f" {series_name}: {len(all_labels)} files in {len(sorted_groups)} groups: {sorted_groups}")
    print(f"{'='*60}\n")

    # ── Load all data ──
    data = {}
    for label in all_labels:
        filepath = os.path.join(BASE_DIR, f"{label}.txt")
        with open(filepath, "r") as f:
            lines = f.readlines()
        V = np.array([float(x) for x in lines[0].strip().split("\t")])
        t = np.array([float(x) for x in lines[1].strip().split("\t")])
        data[label] = {"V": V, "t": t}

    # ── Per-file statistics and V(t) plots ──
    results = []
    for label in all_labels:
        V, t = data[label]["V"], data[label]["t"]
        mean_V = np.mean(V)
        std_V = np.std(V, ddof=1)
        results.append({"label": label, "mean": mean_V, "stdev": std_V})

        plt.figure(figsize=(10, 4))
        plt.plot(t * 1e3, V * 1e3, linewidth=0.5)
        plt.xlabel("t (ms)")
        plt.ylabel("V (mV)")
        plt.title(f"{label}")
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f"{label}.png"), dpi=150)
        plt.close()
        print(f"{label}: mean={mean_V:.6e} V, stdev={std_V:.6e} V")

    # ── PSD for each file ──
    psd_data = {}
    for label in all_labels:
        V, t = data[label]["V"], data[label]["t"]
        dt = t[1] - t[0]
        N = len(V)
        freq = np.fft.rfftfreq(N, d=dt)
        Vfft = np.fft.rfft(V)
        psd = (2.0 * dt / N) * np.abs(Vfft) ** 2
        psd_data[label] = {"freq": freq, "psd": psd}

        plt.figure(figsize=(8, 4))
        plt.loglog(freq[1:], psd[1:], linewidth=0.5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(r"PSD (V$^2$/Hz)")
        plt.title(f"PSD — {label}")
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f"{label}_psd.png"), dpi=150)
        plt.close()

    # ── Overlay V(t) plots per group ──
    for group in sorted_groups:
        plt.figure(figsize=(10, 4))
        for label in groups[group]:
            V, t = data[label]["V"], data[label]["t"]
            plt.plot(t * 1e3, V * 1e3, linewidth=0.4, alpha=0.7, label=label)
        plt.xlabel("t (ms)")
        plt.ylabel("V (mV)")
        plt.title(f"V(t) overlay — {group} [{series_name}]")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR, f"{img_prefix}{group}_overlay.png"), dpi=150)
        plt.close()

    # ── Average PSD comparison ──
    avg_psd = {}
    for group in sorted_groups:
        psds = [psd_data[label]["psd"] for label in groups[group]]
        avg_psd[group] = np.mean(psds, axis=0)

    freq_ref = psd_data[all_labels[0]]["freq"]

    plt.figure(figsize=(8, 5))
    for group in sorted_groups:
        plt.loglog(freq_ref[1:], avg_psd[group][1:], linewidth=1.0, label=f"{group} (avg)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"PSD (V$^2$/Hz)")
    plt.title(f"Average PSD comparison [{series_name}]")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"{img_prefix}psd_comparison.png"), dpi=150)
    plt.close()

    # ── Histogram comparison ──
    plt.figure(figsize=(8, 5))
    bins = np.linspace(-3e-3, 3e-3, 100)
    for group in sorted_groups:
        all_V = np.concatenate([data[label]["V"] for label in groups[group]])
        plt.hist(all_V * 1e3, bins=bins * 1e3, alpha=0.6, density=True, label=group)
    plt.xlabel("V (mV)")
    plt.ylabel("Probability density")
    plt.title(f"Voltage distribution [{series_name}]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"{img_prefix}histogram_comparison.png"), dpi=150)
    plt.close()

    # ── Group statistics ──
    print(f"\n--- {series_name} Group Summary ---")
    group_stdevs = {}
    for group in sorted_groups:
        stdevs = [r["stdev"] for r in results if r["label"] in groups[group]]
        avg_std = np.mean(stdevs)
        group_stdevs[group] = avg_std
        print(f"{group} avg stdev: {avg_std:.6e} V")

    # ── Linear fit: stdev vs cable length (per-file data) ──
    lengths_grp = np.array([int(re.match(r"(\d+)", g).group(1)) for g in sorted_groups])
    stdevs_grp = np.array([group_stdevs[g] for g in sorted_groups])

    # Use all individual files for proper uncertainty
    lengths_all = []
    stdevs_all = []
    for group in sorted_groups:
        L = int(re.match(r"(\d+)", group).group(1))
        for r in results:
            if r["label"] in groups[group]:
                lengths_all.append(L)
                stdevs_all.append(r["stdev"])
    lengths_all = np.array(lengths_all)
    stdevs_all = np.array(stdevs_all)

    coeffs, cov = np.polyfit(lengths_all, stdevs_all, 1, cov=True)
    slope, intercept = coeffs
    slope_err = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    ss_res = np.sum((stdevs_all - (slope * lengths_all + intercept)) ** 2)
    ss_tot = np.sum((stdevs_all - np.mean(stdevs_all)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\nLinear fit (N={len(stdevs_all)} points):")
    print(f"  slope     = ({slope:.6e} ± {slope_err:.6e}) V/cm")
    print(f"  intercept = ({intercept:.6e} ± {intercept_err:.6e}) V")
    print(f"  R² = {r_squared:.6f}")

    fit_x = np.linspace(lengths_grp.min() * 0.9, lengths_grp.max() * 1.1, 100)
    fit_y = slope * fit_x + intercept

    plt.figure(figsize=(8, 5))
    plt.scatter(lengths_all, stdevs_all * 1e3, alpha=0.5, s=20, label="Individual files")
    plt.scatter(lengths_grp, stdevs_grp * 1e3, zorder=5, s=60, marker='D', label="Group avg")
    plt.plot(fit_x, fit_y * 1e3, 'r--',
             label=f"Fit: ({slope*1e6:.2f}±{slope_err*1e6:.2f}) μV/cm (R²={r_squared:.4f})")
    plt.xlabel("Cable length (cm)")
    plt.ylabel("Stdev (mV)")
    plt.title(f"Noise (stdev) vs Cable Length [{series_name}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"{img_prefix}stdev_vs_length.png"), dpi=150)
    plt.close()

    return results, group_stdevs, slope, slope_err, intercept, intercept_err


# ── Run for base series (no prefix) ──
groups_base, sorted_base = discover_groups("*cm-*.txt", r"^(\d+cm)-(\d+)$")
results_base, gstd_base, slope_base, slope_base_err, int_base, int_base_err = \
    run_analysis("base", groups_base, sorted_base)

# ── Run for 1M_ series ──
groups_1m, sorted_1m = discover_groups("1M_*cm-*.txt", r"^1M_(\d+cm)-(\d+)$")
if sorted_1m:
    results_1m, gstd_1m, slope_1m, slope_1m_err, int_1m, int_1m_err = \
        run_analysis("1M", groups_1m, sorted_1m)

    # ── Z-test: base vs 1M ──
    print(f"\n{'='*60}")
    print(f" Z-test: base vs 1M")
    print(f"{'='*60}")
    for name, v1, e1, v2, e2 in [
        ("slope", slope_base, slope_base_err, slope_1m, slope_1m_err),
        ("intercept", int_base, int_base_err, int_1m, int_1m_err),
    ]:
        z = (v1 - v2) / np.sqrt(e1**2 + e2**2)
        print(f"\n  {name}:")
        print(f"    base = {v1:.6e} ± {e1:.6e}")
        print(f"    1M   = {v2:.6e} ± {e2:.6e}")
        print(f"    z = {z:.4f},  |z| = {abs(z):.4f}  {'(p<0.05, significant)' if abs(z) > 1.96 else '(not significant)'}")
else:
    results_1m, gstd_1m = [], {}

# ── Write combined statistics.csv ──
csv_path = os.path.join(BASE_DIR, "statistics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["series", "label", "mean", "stdev"])
    for r in results_base:
        writer.writerow(["base", r["label"], r["mean"], r["stdev"]])
    for r in results_1m:
        writer.writerow(["1M", r["label"], r["mean"], r["stdev"]])
    writer.writerow([])
    writer.writerow(["series", "group", "avg_stdev"])
    for g in sorted_base:
        writer.writerow(["base", g, gstd_base[g]])
    for g in sorted_1m:
        writer.writerow(["1M", g, gstd_1m[g]])

print(f"\nSaved plots to {IMG_DIR}")
print(f"Saved statistics to {csv_path}")
