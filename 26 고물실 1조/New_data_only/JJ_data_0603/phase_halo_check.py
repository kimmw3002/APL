"""정전기 halo 가설의 직접 검증: NC(JJ_AFM_2)의 Phase/Amplitude 채널에
topography 보다 옆으로 더 뻗는 lateral halo가 있는가?

물리: AM-AFM phase는 보존력 virial(정전기 포함) + 소산을 담는다. 장거리 정전기 halo면
z-피드백이 topo로 빼버려도 phase/amplitude-error 채널엔 남아, feature 가장자리 밖으로
뻗는 skirt로 보여야 한다.

같은 tall-object 단면(rows 5 좌표)에서 Z / Amplitude / Phase 를 함께 추출해
선형 baseline 제거 후 정규화 비교하고, 각 채널의 lateral 도달거리를 잰다.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)   # analyze_nc_vs_contact (sample_profile) 재사용
sys.path.insert(0, ROOT)   # measure_nid, export_leveled 재사용

from NSFopen.read import read                                   # noqa: E402
from measure_nid import pixel_nm                                # noqa: E402
from analyze_nc_vs_contact import sample_profile                # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

NID = os.path.join(HERE, "JJ_AFM_2.nid")
SEG = (586, 473, 422, 623)   # tall object (CSV row 5)
CHANNELS = ["Z-Axis", "Amplitude", "Phase"]


def detrend_linear(prof, dist, frac=0.25):
    """양끝 frac 영역에 직선 fit 후 빼기 (analyze 의 auto baseline 과 동일 개념)."""
    n = len(prof)
    k = max(1, round(n * frac))
    idx = [i for i in range(n) if i < k or i >= n - k]
    A = np.vstack([np.ones(len(idx)), dist[idx]]).T
    (b, s), *_ = np.linalg.lstsq(A, prof[idx], rcond=None)
    idx = np.asarray(idx)
    resid = prof - (b + s * dist)
    noise = np.std(prof[idx] - (b + s * dist[idx])) or 1e-12
    return resid, noise


def reach(dist, sig, noise, k=4.0):
    """|deviation| > k*noise 인 가장 바깥 교차점 사이 거리 (halo 도달폭)."""
    m = np.abs(sig) > k * noise
    if m.sum() < 2:
        return 0.0
    xs = dist[m]
    return xs.max() - xs.min()


def main():
    afm = read(NID)
    px = 0.5 * (pixel_nm(afm, "X", afm.data[("Image", "Forward", "Z-Axis")].shape[1])
                + pixel_nm(afm, "Y", afm.data[("Image", "Forward", "Z-Axis")].shape[0]))

    results = {}
    print("=" * 70)
    print("NC(JJ_AFM_2) tall-object 단면: 채널별 lateral 도달거리")
    print("=" * 70)
    for ch in CHANNELS:
        grid = np.asarray(afm.data[("Image", "Forward", ch)], dtype=float)
        # 단위 정리: Z->nm, Phase->deg(가정), Amplitude->그대로
        scale = 1e9 if ch == "Z-Axis" else 1.0
        prof, dist = sample_profile(grid * scale,
                                    pixel_nm(afm, "X", grid.shape[1]),
                                    pixel_nm(afm, "Y", grid.shape[0]), SEG)
        resid, noise = detrend_linear(prof, dist)
        # topo peak 중심으로 정렬
        zgrid = np.asarray(afm.data[("Image", "Forward", "Z-Axis")]) * 1e9
        zprof, _ = sample_profile(zgrid, pixel_nm(afm, "X", grid.shape[1]),
                                  pixel_nm(afm, "Y", grid.shape[0]), SEG)
        zres, _ = detrend_linear(zprof, dist)
        ctr = dist[int(np.argmax(np.abs(zres)))]
        results[ch] = dict(x=dist - ctr, sig=resid, noise=noise,
                           reach=reach(dist, resid, noise))
        print(f"  {ch:10s}: peak|dev|={np.abs(resid).max():8.3f}  "
              f"noise={noise:7.4f}  도달폭(|dev|>4σ)={results[ch]['reach']:7.0f} nm")

    rz = results["Z-Axis"]["reach"]
    print("\n  --- topography 대비 ---")
    for ch in ("Amplitude", "Phase"):
        r = results[ch]["reach"]
        print(f"  {ch}: {r:.0f} nm  (Z {rz:.0f} nm 의 {r/rz:.2f}배)"
              + ("  <- topo보다 넓음 = halo 방증" if r > rz * 1.1 else ""))

    # 그림
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    _fp = r"C:\Windows\Fonts\malgun.ttf"   # 한글 폰트 (Malgun Gothic)
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    titles = {"Z-Axis": "Z (topography)  [nm]",
              "Amplitude": "Amplitude (error)  [a.u.]",
              "Phase": "Phase  [deg]"}
    colors = {"Z-Axis": "tab:blue", "Amplitude": "tab:green", "Phase": "tab:red"}
    for ax, ch in zip(axes, CHANNELS):
        r = results[ch]
        ax.plot(r["x"], r["sig"], color=colors[ch], lw=1.3)
        ax.axhline(0, color="k", lw=0.5)
        for s in (1, -1):
            ax.axhline(s * 4 * r["noise"], color="gray", ls=":", lw=0.7)
        # topo FWHM 경계(±) 표시
        zr = results["Z-Axis"]
        # topo 반치폭 경계
        ax.axvspan(-1, 1, alpha=0)  # placeholder
        ax.set_ylabel(titles[ch]); ax.grid(alpha=0.3)
        ax.set_title(rf"{ch}:  도달폭(|dev|>4$\sigma$) = {r['reach']:.0f} nm", fontsize=9.5)
    # 모든 패널에 Z 도달폭 경계선
    zreach = results["Z-Axis"]["reach"]
    for ax in axes:
        ax.axvline(-zreach / 2, color="tab:blue", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(+zreach / 2, color="tab:blue", ls="--", lw=0.8, alpha=0.6)
    axes[-1].set_xlabel("lateral x (nm), topo peak = 0")
    fig.suptitle("NC tall-object: Phase/Amplitude halo가 topography(파란 점선)보다 옆으로 뻗는가?")
    fig.tight_layout()
    out = os.path.join(HERE, "PHASE_HALO.png")
    fig.savefig(out); plt.close(fig)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
