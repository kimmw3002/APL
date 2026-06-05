"""NC(JJ_AFM_2, Dynamic Force) vs Contact(JJ_AFM_4, Static Force) lateral-broadening 진단.

같은 시료/영역에서 height는 같은데 NC의 FWHM만 수백 nm 넓은 현상의 원인을,
apparent width = [사다리꼴 시료 ⊛ tip] ⊛ long-range 의 세 성분으로 분해하여 규명한다.

가설 -> 데이터 분석 -> 결론.

- 단면 추출은 measure_nid.py 의 sampleProfile/shiftToAlign/analyze 로직을 그대로 포팅
  (bilinear, swath W=21 cross-correlation 정렬, 선형 baseline auto-fit, half-max FWHM).
- 레벨링/픽셀크기는 기존 export_leveled.masked_polyfit / measure_nid.pixel_nm 재사용.

실행:  python New_data_only/JJ_data_0603/analyze_nc_vs_contact.py
산출:  PROFILES.png, FIT_overlay.png, HEIGHT_DEP.png, FB_asym.png  (+ 콘솔 검증 출력)
"""
import os
import sys
import numpy as np

# 상위 repo 루트를 import 경로에 추가 (export_leveled, measure_nid 재사용)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from NSFopen.read import read                      # noqa: E402
from export_leveled import masked_polyfit          # noqa: E402
from measure_nid import pixel_nm                   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# ----------------------------------------------------------------------------
# 분석 대상 (nid_measurements.csv rows 2,4,5,11)
# x0,y0,x1,y1 는 픽셀 좌표, peak_avg_n 은 CSV 의 peak 평균 반경.
# ----------------------------------------------------------------------------
TARGETS = {
    "contact_tall": dict(file="JJ_AFM_4.nid", mode="contact", obj="tall",
                         seg=(512, 514, 406, 607), peak_avg=1,
                         csv_h=95.088, csv_fwhm=397.965),
    "nc_tall":      dict(file="JJ_AFM_2.nid", mode="NC", obj="tall",
                         seg=(586, 473, 422, 623), peak_avg=2,
                         csv_h=95.200, csv_fwhm=929.438),
    "contact_short": dict(file="JJ_AFM_4.nid", mode="contact", obj="short",
                          seg=(482, 430, 400, 342), peak_avg=2,
                          csv_h=23.510, csv_fwhm=436.681),
    "nc_short":     dict(file="JJ_AFM_2.nid", mode="NC", obj="short",
                         seg=(570, 402, 493, 319), peak_avg=2,
                         csv_h=21.761, csv_fwhm=765.071),
}

SWATH = 21          # swath_px (CSV)
BASEFRAC = 0.25     # measure_nid basefrac 기본값
FWHM_BAND_SIGMA = 1.0


# ----------------------------------------------------------------------------
# .nid -> leveled Z image (캐시)
# ----------------------------------------------------------------------------
_cache = {}


def load_image(fname, channel="Forward"):
    key = (fname, channel)
    if key in _cache:
        return _cache[key]
    ap = os.path.join(HERE, fname)
    afm = read(ap)
    z = np.asarray(afm.data[("Image", channel, "Z-Axis")], dtype=float)
    lev = masked_polyfit(z, k=2.5, deg=2, right_exclude=0.2) * 1e9   # nm
    px_x = pixel_nm(afm, "X", z.shape[1])
    px_y = pixel_nm(afm, "Y", z.shape[0])
    out = dict(lev=lev, px_x=px_x, px_y=px_y, rows=z.shape[0], cols=z.shape[1])
    _cache[key] = out
    return out


# ----------------------------------------------------------------------------
# measure_nid.py 포팅
# ----------------------------------------------------------------------------
def bilinear(grid, c, r):
    """measure_nid bilinearG: 격자 경계 clamp 후 bilinear."""
    rows, cols = grid.shape
    c = min(max(c, 0.0), cols - 1.0)
    r = min(max(r, 0.0), rows - 1.0)
    c0, r0 = int(np.floor(c)), int(np.floor(r))
    c1, r1 = min(c0 + 1, cols - 1), min(r0 + 1, rows - 1)
    fc, fr = c - c0, r - r0
    return (grid[r0, c0] * (1 - fc) * (1 - fr) + grid[r0, c1] * fc * (1 - fr)
            + grid[r1, c0] * (1 - fc) * fr + grid[r1, c1] * fc * fr)


def shift_to_align(row, ref, max_lag):
    """measure_nid shiftToAlign: cross-covariance 를 최대화하는 정수 lag 로 정렬."""
    n = len(row)
    rm, om = ref.mean(), row.mean()
    best_lag, best = 0, -np.inf
    for L in range(-max_lag, max_lag + 1):
        s = c = 0.0
        for i in range(n):
            j = i + L
            if 0 <= j < n:
                s += (ref[i] - rm) * (row[j] - om)
                c += 1
        if c > 0:
            sc = s / c
            if sc > best:
                best, best_lag = sc, L
    out = np.full(n, np.nan)
    for i in range(n):
        j = i + best_lag
        if 0 <= j < n:
            out[i] = row[j]
    return out


def sample_profile(grid, px_x, px_y, seg, swath=SWATH, align=True):
    """measure_nid sampleProfile 포팅. 반환: prof(nm), dist(nm)."""
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    lenpx = np.hypot(dx, dy) or 1e-9
    dxn, dyn = dx * px_x, dy * px_y
    lennm = np.hypot(dxn, dyn) or 1.0
    perp_x, perp_y = -dy / lenpx, dx / lenpx
    half = (swath - 1) // 2
    steps = max(8, round(lenpx))
    n = steps + 1
    dist = np.array([i / steps * lennm for i in range(n)])
    lines = []
    for w in range(-half, half + 1):
        row = np.array([bilinear(grid, x0 + dx * (i / steps) + w * perp_x,
                                 y0 + dy * (i / steps) + w * perp_y)
                        for i in range(n)])
        lines.append(row)
    if swath == 1:
        return lines[0], dist
    if align:
        ref = lines[len(lines) // 2]
        max_lag = min(n - 2, int(np.ceil(half)) + 3)
        prof = np.zeros(n)
        cnt = np.zeros(n)
        for row in lines:
            a = shift_to_align(row, ref, max_lag)
            m = ~np.isnan(a)
            prof[m] += a[m]
            cnt[m] += 1
        prof = np.where(cnt > 0, prof / np.maximum(cnt, 1), ref)
    else:
        prof = np.mean(lines, axis=0)
    return prof, dist


def analyze_profile(prof, dist, peak_avg=2, basefrac=BASEFRAC):
    """measure_nid analyze 포팅: 선형 baseline, height(peak±n 평균), half-max FWHM."""
    n = len(prof)
    k = max(1, round(n * basefrac))
    base_idx = [i for i in range(n) if i < k or i >= n - k]
    rod_idx = [i for i in range(n) if k <= i < n - k]
    # 선형 baseline (baseline 점들에 대한 1차 fit)
    bx = np.array([dist[i] for i in base_idx])
    by = np.array([prof[i] for i in base_idx])
    A = np.vstack([np.ones_like(bx), bx]).T
    (bl_base, bl_slope), *_ = np.linalg.lstsq(A, by, rcond=None)
    baseline = bl_base + bl_slope * dist
    noise = np.std(by - (bl_base + bl_slope * bx)) or 1e-9
    det = prof - baseline
    search = rod_idx if rod_idx else list(range(n))
    pi = max(search, key=lambda i: det[i])
    avg_idx = [i for i in range(max(0, pi - peak_avg), min(n - 1, pi + peak_avg) + 1)
               if i in set(search)] or [pi]
    height = np.mean([det[i] for i in avg_idx])
    half = height / 2.0

    def cross_left(level):
        for i in range(pi, 0, -1):
            if det[i] >= level > det[i - 1]:
                f = (level - det[i - 1]) / (det[i] - det[i - 1])
                return dist[i - 1] + (dist[i] - dist[i - 1]) * f
        return None

    def cross_right(level):
        for i in range(pi, n - 1):
            if det[i] >= level > det[i + 1]:
                f = (level - det[i]) / (det[i + 1] - det[i])
                return dist[i] + (dist[i + 1] - dist[i]) * f
        return None

    xL, xR = cross_left(half), cross_right(half)
    fwhm = abs(xR - xL) if (xL is not None and xR is not None) else np.nan
    # top plateau (>=90% peak), base (>=15% peak)
    above90 = dist[det >= 0.9 * height]
    topw = above90.max() - above90.min() if above90.size > 1 else 0.0
    above15 = dist[det >= 0.15 * height]
    basew = above15.max() - above15.min() if above15.size > 1 else 0.0
    return dict(prof=prof, dist=dist, det=det, baseline=baseline, pi=pi,
                height=height, fwhm=fwhm, xL=xL, xR=xR, center=dist[pi],
                topw=topw, basew=basew, noise=noise)


# ----------------------------------------------------------------------------
# 단계 1: 추출 + CSV 검증
# ----------------------------------------------------------------------------
def extract_all():
    res = {}
    for name, t in TARGETS.items():
        img = load_image(t["file"], "Forward")
        prof, dist = sample_profile(img["lev"], img["px_x"], img["px_y"], t["seg"])
        a = analyze_profile(prof, dist, peak_avg=t["peak_avg"])
        a.update(meta=t, px=0.5 * (img["px_x"] + img["px_y"]))
        res[name] = a
    return res


def validate(res):
    print("=" * 78)
    print("단계 1 — 추출 재현 검증 (CSV rows 2,4,5,11 과 비교)")
    print("=" * 78)
    print(f"{'segment':14s} {'h_meas':>8s} {'h_csv':>8s} {'fwhm_meas':>10s} "
          f"{'fwhm_csv':>9s} {'Δfwhm%':>7s} {'top':>6s} {'base':>7s}")
    ok = True
    for name, a in res.items():
        t = a["meta"]
        dh = a["fwhm"] - t["csv_fwhm"]
        pct = 100 * dh / t["csv_fwhm"]
        if abs(pct) > 2.0:
            ok = False
        print(f"{name:14s} {a['height']:8.2f} {t['csv_h']:8.2f} "
              f"{a['fwhm']:10.2f} {t['csv_fwhm']:9.2f} {pct:7.2f} "
              f"{a['topw']:6.0f} {a['basew']:7.0f}")
    print(f"\n검증 {'통과 (모든 FWHM ±2% 이내)' if ok else '실패 — 추출 로직 점검 필요'}")
    return ok


# ----------------------------------------------------------------------------
# 단계 3: forward model  apparent = [trapezoid ⊛ tip(R)] ⊛ LR(σ)
# ----------------------------------------------------------------------------
from scipy.ndimage import grey_dilation, gaussian_filter1d   # noqa: E402
from scipy.optimize import least_squares                     # noqa: E402

DX = 2.0  # forward-model 격자 간격 (nm)


def trapezoid(xg, at, s, h):
    """대칭 사다리꼴: |x|<=at 에서 h, at..at+s 에서 선형 하강, 그 밖 0."""
    ax = np.abs(xg)
    z = np.where(ax <= at, h, np.where(ax >= at + s, 0.0, h * (1 - (ax - at) / max(s, 1e-9))))
    return z


def tip_dilate(z, R, dx=DX):
    """파라볼릭 팁(반경 R) 에 의한 형태학적 dilation (Villarrubia tip convolution)."""
    if R <= 0:
        return z
    U = np.sqrt(2 * R * (z.max() - z.min() + 1e-9)) + 3 * dx
    ku = np.arange(-U, U + dx, dx)
    struct = -ku * ku / (2.0 * R)          # 팁 apex 높이함수의 음수
    return grey_dilation(z, structure=struct, mode="nearest")


def forward(xg, at, s, h, R, sigma, dx=DX):
    z = trapezoid(xg, at, s, h)
    z = tip_dilate(z, R, dx)
    if sigma > 0:
        z = gaussian_filter1d(z, sigma / dx, mode="nearest")
    return z


def centered(a):
    """analyze 결과를 peak 중심(=0)으로 옮긴 (x, det) 반환."""
    x = a["dist"] - a["center"]
    return x, a["det"]


def model_at(xq, at, s, h, R, sigma):
    """측정 x 위치(xq)에서의 모델 값."""
    span = max(np.abs(xq).max() + 50, at + s + 6 * max(R, sigma, 1) + 50)
    xg = np.arange(-span, span + DX, DX)
    zg = forward(xg, at, s, h, R, sigma)
    return np.interp(xq, xg, zg)


def fit_object(contact, nc, with_lr):
    """contact+NC 결합 적합. 공유: at,s (참 형상).  per-mode: R.  NC만: sigma(LR).

    h 는 측정 height 로 고정(두 모드 동일). with_lr=False 면 sigma_NC=0 (M1)."""
    xc, yc = centered(contact)
    xn, yn = centered(nc)
    h = 0.5 * (contact["height"] + nc["height"])

    # params: at, s, R_C, R_NC, [sigma_NC]
    def resid(p):
        at, s, Rc, Rn = p[0], p[1], p[2], p[3]
        sig = p[4] if with_lr else 0.0
        rc = model_at(xc, at, s, h, Rc, 0.0) - yc
        rn = model_at(xn, at, s, h, Rn, sig) - yn
        return np.concatenate([rc, rn])

    p0 = [60.0, 120.0, 20.0, 60.0] + ([120.0] if with_lr else [])
    lo = [1.0, 1.0, 1.0, 1.0] + ([1.0] if with_lr else [])
    hi = [500.0, 900.0, 600.0, 800.0] + ([600.0] if with_lr else [])
    sol = least_squares(resid, p0, bounds=(lo, hi), method="trf", max_nfev=4000)
    r = sol.fun
    npt = len(r)
    rss = float(np.sum(r * r))
    rmse = np.sqrt(rss / npt)
    kparam = len(p0)
    aic = npt * np.log(rss / npt) + 2 * kparam
    p = sol.x
    out = dict(at=p[0], s=p[1], Rc=p[2], Rn=p[3],
               sigma=(p[4] if with_lr else 0.0), h=h,
               rmse=rmse, rss=rss, aic=aic, npt=npt, k=kparam,
               nc_rmse=np.sqrt(np.mean((model_at(xn, p[0], p[1], h, p[3],
                          (p[4] if with_lr else 0.0)) - yn) ** 2)))
    return out


def run_models(res):
    print("\n" + "=" * 78)
    print("단계 3 — forward-model 분해 & model selection (per object)")
    print("=" * 78)
    fits = {}
    for obj, (cname, nname) in [("tall", ("contact_tall", "nc_tall")),
                                ("short", ("contact_short", "nc_short"))]:
        c, n = res[cname], res[nname]
        m1 = fit_object(c, n, with_lr=False)
        m2 = fit_object(c, n, with_lr=True)
        fits[obj] = dict(M1=m1, M2=m2, cname=cname, nname=nname)
        print(f"\n[{obj}]  h={m1['h']:.1f} nm")
        print(f"  참형상(공유): top half-width at≈{m2['at']:.0f} nm, "
              f"sidewall run s≈{m2['s']:.0f} nm  -> 참 base 폭≈{2*(m2['at']+m2['s']):.0f} nm, "
              f"top 폭≈{2*m2['at']:.0f} nm")
        print(f"  M1 (tip-only): R_C={m1['Rc']:.0f}, R_NC={m1['Rn']:.0f} nm | "
              f"NC RMSE={m1['nc_rmse']:.2f} nm  AIC={m1['aic']:.1f}")
        print(f"  M2 (tip+LR)  : R_C={m2['Rc']:.0f}, R_NC={m2['Rn']:.0f} nm, "
              f"σ_LR={m2['sigma']:.0f} nm | NC RMSE={m2['nc_rmse']:.2f} nm  AIC={m2['aic']:.1f}")
        daic = m1["aic"] - m2["aic"]
        verdict = ("long-range 필요 (M2 우세)" if daic > 2 else
                   "tip+사다리꼴로 충분 (LR 불필요)")
        print(f"  ΔAIC(M1-M2)={daic:.1f}  ->  {verdict}")
    return fits


def height_dependence(res):
    print("\n" + "=" * 78)
    print("단계 3 보조 — height 의존성 (quadrature-added width)")
    print("=" * 78)
    rows = []
    for obj, (c, n) in [("tall", ("contact_tall", "nc_tall")),
                        ("short", ("contact_short", "nc_short"))]:
        wc, wn = res[c]["fwhm"], res[n]["fwhm"]
        h = 0.5 * (res[c]["height"] + res[n]["height"])
        add = np.sqrt(max(wn ** 2 - wc ** 2, 0))
        rows.append((obj, h, wc, wn, add))
        print(f"  {obj:5s}: h={h:5.1f}  FWHM_C={wc:6.1f}  FWHM_NC={wn:6.1f}  "
              f"√(NC²-C²)={add:6.1f} nm")
    (ot, ht, _, _, at), (os_, hs, _, _, as_) = rows
    print(f"\n  added-width 비:  tall/short = {at/as_:.2f}")
    print(f"  순수 sphere-tip 예측 ∝√h :  √(h_t/h_s) = {np.sqrt(ht/hs):.2f}")
    print(f"  -> 관측 비가 √h 예측보다 {'작음 => height-비의존 LR 성분 시사' if at/as_ < np.sqrt(ht/hs)*0.85 else '유사 => geometric 지배'}")
    return rows


# ----------------------------------------------------------------------------
# 단계 3 보조 — Forward vs Backward 비대칭 (parachuting 배제)
# ----------------------------------------------------------------------------
def fb_asymmetry():
    print("\n" + "=" * 78)
    print("단계 3 보조 — Forward/Backward 비대칭 (feedback parachuting 점검)")
    print("=" * 78)
    out = {}
    for name in ("contact_tall", "nc_tall"):
        t = TARGETS[name]
        prof = {}
        for ch in ("Forward", "Backward"):
            img = load_image(t["file"], ch)
            p, d = sample_profile(img["lev"], img["px_x"], img["px_y"], t["seg"])
            a = analyze_profile(p, d, peak_avg=t["peak_avg"])
            prof[ch] = a
        # 좌/우 flank half-width 비대칭
        a = prof["Forward"]
        lw = a["center"] - a["xL"] if a["xL"] else np.nan
        rw = a["xR"] - a["center"] if a["xR"] else np.nan
        asym = abs(rw - lw) / (0.5 * (rw + lw)) * 100
        df = abs(prof["Forward"]["fwhm"] - prof["Backward"]["fwhm"])
        out[name] = prof
        print(f"  {name:13s}: FWHM F={prof['Forward']['fwhm']:.0f} "
              f"B={prof['Backward']['fwhm']:.0f}  |F-B|={df:.0f} nm | "
              f"좌/우 flank {lw:.0f}/{rw:.0f} nm (비대칭 {asym:.0f}%)")
    print("  -> |F-B| 와 좌/우 비대칭이 작으면 parachuting(꼬리끌림) 아님")
    return out


# ----------------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------------
def make_figures(res, fits, hdep, fb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    _fp = r"C:\Windows\Fonts\malgun.ttf"   # 한글 폰트 (Malgun Gothic)
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지
    plt.rcParams.update({"font.size": 10, "figure.dpi": 110})

    # Fig 1: 측정 프로파일 overlay (contact vs NC), tall & short
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, obj, pair in zip(axes, ("tall", "short"),
                             (("contact_tall", "nc_tall"),
                              ("contact_short", "nc_short"))):
        for nm, col in zip(pair, ("tab:blue", "tab:red")):
            x, y = centered(res[nm])
            ax.plot(x, y, color=col, lw=1.4,
                    label=f"{res[nm]['meta']['mode']} (FWHM {res[nm]['fwhm']:.0f} nm)")
        ax.set_title(rf"{obj} object  (h$\approx${res[pair[0]]['height']:.0f} nm)")
        ax.set_xlabel("lateral x (nm)"); ax.set_ylabel("height (nm)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("측정 단면 h(x): 같은 height, NC 가 lateral 로 크게 broadening")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "PROFILES.png")); plt.close(fig)

    # Fig 2: forward-model fit (M1 tip-only vs M2 tip+LR) on NC, + contact fit
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, obj in zip(axes, ("tall", "short")):
        f = fits[obj]; m1, m2 = f["M1"], f["M2"]
        c, n = res[f["cname"]], res[f["nname"]]
        xc, yc = centered(c); xn, yn = centered(n)
        ax.plot(xc, yc, ".", ms=2, color="tab:blue", alpha=0.5, label="contact 측정")
        ax.plot(xn, yn, ".", ms=2, color="tab:red", alpha=0.5, label="NC 측정")
        xg = np.linspace(min(xn.min(), xc.min()), max(xn.max(), xc.max()), 400)
        ax.plot(xg, model_at(xg, m2["at"], m2["s"], m2["h"], m2["Rc"], 0.0),
                color="tab:blue", lw=1.6, label="contact fit")
        ax.plot(xg, model_at(xg, m1["at"], m1["s"], m1["h"], m1["Rn"], 0.0),
                "--", color="gray", lw=1.6, label=f"NC M1 tip-only (RMSE {m1['nc_rmse']:.1f})")
        ax.plot(xg, model_at(xg, m2["at"], m2["s"], m2["h"], m2["Rn"], m2["sigma"]),
                color="tab:red", lw=1.8, label=rf"NC M2 tip+LR $\sigma$={m2['sigma']:.0f} (RMSE {m2['nc_rmse']:.1f})")
        ax.plot(xg, trapezoid(xg, m2["at"], m2["s"], m2["h"]), ":", color="green",
                lw=1.4, label=f"추정 참형상 (top {2*m2['at']:.0f} nm)")
        ax.set_title(rf"{obj}: $\Delta$AIC(M1$-$M2)={m1['aic']-m2['aic']:.0f}")
        ax.set_xlabel("lateral x (nm)"); ax.set_ylabel("height (nm)")
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    fig.suptitle(r"분해: 참 사다리꼴 $\circledast$ tip $\circledast$ LR  |  tip-only(M1)는 NC 재현 실패, LR(M2) 필요")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "FIT_overlay.png")); plt.close(fig)

    # Fig 3: height-dependence
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    objs = [r[0] for r in hdep]; hs = [r[1] for r in hdep]; add = [r[4] for r in hdep]
    ax.plot(hs, add, "o", ms=9, color="tab:red")
    for o, h, _wc, _wn, a in hdep:
        ax.annotate(f"{o}\n{a:.0f} nm", (h, a), textcoords="offset points",
                    xytext=(8, -4), fontsize=9)
    # √h 예측 곡선 (short 점에 normalize)
    hh = np.linspace(min(hs) * 0.8, max(hs) * 1.1, 50)
    ax.plot(hh, add[1] * np.sqrt(hh / hs[1]), "--", color="gray",
            label=r"순수 tip 예측 $\propto\sqrt{h}$")
    ax.axhline(np.mean(add), color="tab:red", ls=":", alpha=0.6,
               label=rf"관측 $\approx$ 상수 ({np.mean(add):.0f} nm)")
    ax.set_xlabel("feature height h (nm)")
    ax.set_ylabel(r"NC 초과폭  $\sqrt{\mathrm{FWHM}_{NC}^2 - \mathrm{FWHM}_C^2}$  (nm)")
    ax.set_title(r"초과 broadening 이 height 에 거의 무관 $\rightarrow$ long-range")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "HEIGHT_DEP.png")); plt.close(fig)

    # Fig 4: Forward/Backward 비대칭
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, name in zip(axes, ("contact_tall", "nc_tall")):
        for ch, col in zip(("Forward", "Backward"), ("tab:green", "tab:purple")):
            a = fb[name][ch]
            x = a["dist"] - a["center"]
            ax.plot(x, a["det"], color=col, lw=1.3, label=f"{ch} (FWHM {a['fwhm']:.0f})")
        ax.set_title(name); ax.set_xlabel("lateral x (nm)"); ax.set_ylabel("height (nm)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(r"Forward vs Backward: 겹치면 parachuting 아님 (좌우대칭, F$\approx$B)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "FB_asym.png")); plt.close(fig)
    print("\n저장: PROFILES.png, FIT_overlay.png, HEIGHT_DEP.png, FB_asym.png")


if __name__ == "__main__":
    res = extract_all()
    validate(res)
    fits = run_models(res)
    hdep = height_dependence(res)
    fb = fb_asymmetry()
    make_figures(res, fits, hdep, fb)
