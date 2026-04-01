import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import glob
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

# ── Auto-detect and group *cm-*.txt files ──
files = glob.glob(os.path.join(BASE_DIR, "*cm-*.txt"))
groups = {}  # e.g. {"109cm": ["109cm-1", "109cm-2", ...]}
for fpath in files:
    fname = os.path.splitext(os.path.basename(fpath))[0]
    m = re.match(r"^(\d+cm)-(\d+)$", fname)
    if not m:
        continue
    group, num = m.group(1), int(m.group(2))
    groups.setdefault(group, []).append((num, fname))

for g in groups:
    groups[g].sort()
    groups[g] = [label for _, label in groups[g]]

sorted_groups = sorted(groups.keys(), key=lambda g: int(re.match(r"(\d+)", g).group(1)))
all_labels = [label for g in sorted_groups for label in groups[g]]

print(f"Found {len(all_labels)} files in {len(sorted_groups)} groups: {sorted_groups}\n")

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
    psd = (2.0 * dt / N) * np.abs(Vfft) ** 2  # V^2/Hz
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
    plt.title(f"V(t) overlay — {group}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"{group}_overlay.png"), dpi=150)
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
plt.title("Average PSD comparison")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "psd_comparison.png"), dpi=150)
plt.close()

# ── Histogram comparison ──
plt.figure(figsize=(8, 5))
bins = np.linspace(-3e-3, 3e-3, 100)
for group in sorted_groups:
    all_V = np.concatenate([data[label]["V"] for label in groups[group]])
    plt.hist(all_V * 1e3, bins=bins * 1e3, alpha=0.6, density=True, label=group)
plt.xlabel("V (mV)")
plt.ylabel("Probability density")
plt.title("Voltage distribution comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "histogram_comparison.png"), dpi=150)
plt.close()

# ── Group statistics ──
print(f"\n=== Group Summary ===")
group_stdevs = {}
for group in sorted_groups:
    stdevs = [r["stdev"] for r in results if r["label"] in groups[group]]
    avg_std = np.mean(stdevs)
    group_stdevs[group] = avg_std
    print(f"{group} avg stdev: {avg_std:.6e} V")

# ── Linear fit: stdev vs cable length ──
lengths = np.array([int(re.match(r"(\d+)", g).group(1)) for g in sorted_groups])
stdevs_arr = np.array([group_stdevs[g] for g in sorted_groups])
slope, intercept = np.polyfit(lengths, stdevs_arr, 1)
print(f"\nLinear fit: stdev = {slope:.6e} * L + {intercept:.6e}")
ss_res = np.sum((stdevs_arr - (slope * lengths + intercept)) ** 2)
ss_tot = np.sum((stdevs_arr - np.mean(stdevs_arr)) ** 2)
r_squared = 1 - ss_res / ss_tot
print(f"  slope = {slope:.6e} V/cm,  intercept = {intercept:.6e} V,  R² = {r_squared:.6f}")

fit_x = np.linspace(lengths.min() * 0.9, lengths.max() * 1.1, 100)
fit_y = slope * fit_x + intercept

plt.figure(figsize=(8, 5))
plt.scatter(lengths, stdevs_arr * 1e3, zorder=5, label="Data")
plt.plot(fit_x, fit_y * 1e3, 'r--', label=f"Fit: {slope*1e6:.3f} μV/cm · L + {intercept*1e3:.4f} mV (R²={r_squared:.4f})")
plt.xlabel("Cable length (cm)")
plt.ylabel("Avg stdev (mV)")
plt.title("Noise (stdev) vs Cable Length")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "stdev_vs_length.png"), dpi=150)
plt.close()

# ── Write statistics.csv ──
csv_path = os.path.join(BASE_DIR, "statistics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "mean", "stdev"])
    for r in results:
        writer.writerow([r["label"], r["mean"], r["stdev"]])
    writer.writerow([])
    writer.writerow(["group", "avg_stdev"])
    for group in sorted_groups:
        writer.writerow([group, group_stdevs[group]])

print(f"\nSaved plots to {IMG_DIR}")
print(f"Saved statistics to {csv_path}")
