import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

    # ── Fit: sigma = sqrt(a^2 + b^2 * L^2) ──
    lengths_grp = np.array([int(re.match(r"(\d+)", g).group(1)) for g in sorted_groups])
    stdevs_grp = np.array([group_stdevs[g] for g in sorted_groups])

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

    def model(L, a, b):
        return np.sqrt(a**2 + b**2 * L**2)

    p0 = [stdevs_all.min(), 1e-6]
    popt, pcov = curve_fit(model, lengths_all, stdevs_all, p0=p0)
    a, b = popt
    a_err, b_err = np.sqrt(np.diag(pcov))

    ss_res = np.sum((stdevs_all - model(lengths_all, a, b)) ** 2)
    ss_tot = np.sum((stdevs_all - np.mean(stdevs_all)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\nFit: sigma = sqrt(a² + b²L²)  (N={len(stdevs_all)} points)")
    print(f"  a = ({a:.6e} ± {a_err:.6e}) V")
    print(f"  b = ({b:.6e} ± {b_err:.6e}) V/cm")
    print(f"  R² = {r_squared:.6f}")

    fit_x = np.linspace(lengths_grp.min() * 0.9, lengths_grp.max() * 1.1, 100)
    fit_y = model(fit_x, a, b)

    plt.figure(figsize=(8, 5))
    plt.scatter(lengths_all, stdevs_all * 1e3, alpha=0.5, s=20, label="Individual files")
    plt.scatter(lengths_grp, stdevs_grp * 1e3, zorder=5, s=60, marker='D', label="Group avg")
    plt.plot(fit_x, fit_y * 1e3, 'r--',
             label=f"$\\sqrt{{a^2+b^2 L^2}}$: a={a*1e3:.3f}mV, b={b*1e6:.2f}μV/cm (R²={r_squared:.4f})")
    plt.xlabel("Cable length (cm)")
    plt.ylabel("Stdev (mV)")
    plt.title(f"Noise vs Cable Length [{series_name}]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"{img_prefix}stdev_vs_length.png"), dpi=150)
    plt.close()

    return results, group_stdevs, a, a_err, b, b_err


# ── Auto-detect all series by prefix ──
all_files = glob.glob(os.path.join(BASE_DIR, "*cm-*.txt"))
prefixes = set()
for fpath in all_files:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    m = re.match(r"^(.*?)(\d+cm)-(\d+)$", fname)
    if m:
        prefixes.add(m.group(1))

# Sort prefixes: "" first, then by numeric part (1M_, 10M_, ...)
def prefix_sort_key(p):
    if p == "":
        return (0, "")
    m = re.match(r"(\d+)", p)
    return (int(m.group(1)), p) if m else (999, p)

sorted_prefixes = sorted(prefixes, key=prefix_sort_key)

series_results = {}
for prefix in sorted_prefixes:
    series_name = prefix.rstrip("_") if prefix else "base"
    if prefix:
        pattern = f"{prefix}*cm-*.txt"
        regex = rf"^{re.escape(prefix)}(\d+cm)-(\d+)$"
    else:
        pattern = "*cm-*.txt"
        regex = r"^(\d+cm)-(\d+)$"
    groups, sorted_g = discover_groups(pattern, regex)
    if sorted_g:
        res = run_analysis(series_name, groups, sorted_g)
        series_results[series_name] = {
            "results": res[0], "gstd": res[1],
            "a": res[2], "a_err": res[3], "b": res[4], "b_err": res[5],
            "groups": groups, "sorted_groups": sorted_g,
        }

# ── Z-test: all pairs ──
series_names = list(series_results.keys())
if len(series_names) >= 2:
    print(f"\n{'='*60}")
    print(f" Z-tests [sigma = sqrt(a² + b²L²)]")
    print(f"{'='*60}")
    for i in range(len(series_names)):
        for j in range(i + 1, len(series_names)):
            s1, s2 = series_names[i], series_names[j]
            d1, d2 = series_results[s1], series_results[s2]
            print(f"\n  --- {s1} vs {s2} ---")
            for name in ["a", "b"]:
                v1, e1 = d1[name], d1[f"{name}_err"]
                v2, e2 = d2[name], d2[f"{name}_err"]
                z = (v1 - v2) / np.sqrt(e1**2 + e2**2)
                sig = "(p<0.05)" if abs(z) > 1.96 else "(not sig.)"
                print(f"    {name}: {v1:.4e}±{e1:.4e} vs {v2:.4e}±{e2:.4e}  z={z:.3f} {sig}")

# ── Write combined statistics.csv ──
csv_path = os.path.join(BASE_DIR, "statistics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["series", "label", "mean", "stdev"])
    for sname in series_names:
        for r in series_results[sname]["results"]:
            writer.writerow([sname, r["label"], r["mean"], r["stdev"]])
    writer.writerow([])
    writer.writerow(["series", "group", "avg_stdev"])
    for sname in series_names:
        for g in series_results[sname]["sorted_groups"]:
            writer.writerow([sname, g, series_results[sname]["gstd"][g]])

print(f"\nSaved plots to {IMG_DIR}")
print(f"Saved statistics to {csv_path}")
