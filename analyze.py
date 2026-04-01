import numpy as np
import matplotlib.pyplot as plt
import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

labels = [f"long{i}" for i in range(1, 6)] + [f"short{i}" for i in range(1, 6)]

# ── Load all data ──
data = {}
for label in labels:
    filepath = os.path.join(BASE_DIR, f"{label}.txt")
    with open(filepath, "r") as f:
        lines = f.readlines()
    V = np.array([float(x) for x in lines[0].strip().split("\t")])
    t = np.array([float(x) for x in lines[1].strip().split("\t")])
    data[label] = {"V": V, "t": t}

# ── Per-file statistics and V(t) plots ──
results = []
for label in labels:
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
for label in labels:
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

# ── Overlay V(t) plots ──
for group in ["long", "short"]:
    plt.figure(figsize=(10, 4))
    for i in range(1, 6):
        label = f"{group}{i}"
        V, t = data[label]["V"], data[label]["t"]
        plt.plot(t * 1e3, V * 1e3, linewidth=0.4, alpha=0.7, label=label)
    plt.xlabel("t (ms)")
    plt.ylabel("V (mV)")
    plt.title(f"V(t) overlay — {group} cable")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, f"{group}_overlay.png"), dpi=150)
    plt.close()

# ── Average PSD comparison (long vs short) ──
avg_psd = {}
for group in ["long", "short"]:
    psds = []
    for i in range(1, 6):
        psds.append(psd_data[f"{group}{i}"]["psd"])
    avg_psd[group] = np.mean(psds, axis=0)

freq_ref = psd_data["long1"]["freq"]

plt.figure(figsize=(8, 5))
plt.loglog(freq_ref[1:], avg_psd["long"][1:], linewidth=1.0, label="Long cable (avg)")
plt.loglog(freq_ref[1:], avg_psd["short"][1:], linewidth=1.0, label="Short cable (avg)")
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"PSD (V$^2$/Hz)")
plt.title("Average PSD — Long vs Short cable")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "psd_comparison.png"), dpi=150)
plt.close()

# ── Histogram comparison ──
all_long_V = np.concatenate([data[f"long{i}"]["V"] for i in range(1, 6)])
all_short_V = np.concatenate([data[f"short{i}"]["V"] for i in range(1, 6)])

plt.figure(figsize=(8, 5))
bins = np.linspace(-3e-3, 3e-3, 100)
plt.hist(all_long_V * 1e3, bins=bins * 1e3, alpha=0.6, density=True, label="Long cable")
plt.hist(all_short_V * 1e3, bins=bins * 1e3, alpha=0.6, density=True, label="Short cable")
plt.xlabel("V (mV)")
plt.ylabel("Probability density")
plt.title("Voltage distribution — Long vs Short cable")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "histogram_comparison.png"), dpi=150)
plt.close()

# ── Group statistics and ratio ──
long_stdevs = [r["stdev"] for r in results if r["label"].startswith("long")]
short_stdevs = [r["stdev"] for r in results if r["label"].startswith("short")]
long_mean_std = np.mean(long_stdevs)
short_mean_std = np.mean(short_stdevs)
ratio = long_mean_std / short_mean_std

print(f"\n=== Group Summary ===")
print(f"Long  cable avg stdev: {long_mean_std:.6e} V")
print(f"Short cable avg stdev: {short_mean_std:.6e} V")
print(f"Ratio (long/short):    {ratio:.4f}")

# ── Write statistics.csv ──
csv_path = os.path.join(BASE_DIR, "statistics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "mean", "stdev"])
    for r in results:
        writer.writerow([r["label"], r["mean"], r["stdev"]])
    writer.writerow([])
    writer.writerow(["group", "avg_stdev", "ratio_long_short"])
    writer.writerow(["long", long_mean_std, ""])
    writer.writerow(["short", short_mean_std, ""])
    writer.writerow(["ratio", "", f"{ratio:.4f}"])

print(f"\nSaved plots to {IMG_DIR}")
print(f"Saved statistics to {csv_path}")
