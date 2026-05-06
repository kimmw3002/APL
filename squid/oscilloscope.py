"""
Rigol DHO1104 오실로스코프 파형 데이터 캡처 스크립트
=====================================================
pyvisa를 사용하여 DHO1104에서 파형 데이터를 읽어오고
전압으로 변환한 뒤 그래프로 표시합니다.

사전 설치:
    pip install pyvisa pyvisa-py numpy matplotlib

USB 연결 시 NI-VISA 또는 pyvisa-py 백엔드 필요
LAN 연결 시 pyvisa-py만으로도 동작 가능
"""

import pyvisa
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from datetime import datetime

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 설정값 (사용 환경에 맞게 수정하세요)
# ============================================================
# USB 연결 예시: "USB0::0x1AB1::0x044C::DHO1A000000000::INSTR"
# LAN 연결 예시: "TCPIP0::192.168.1.100::INSTR"
VISA_ADDRESS = "USB0::0x1AB1::0x044C::DHO9S254703475::INSTR"

CHANNELS = (1, 2)     # 캡처할 채널 (XY 플롯: X=Ch2, Y=Ch1)
WAV_MODE = "NORMal"   # NORMal: 화면 포인트, RAW: 메모리 전체, MAXimum: 자동
WAV_FORMAT = "BYTE"   # BYTE: 8bit, WORD: 16bit(12bit 정밀도), ASCii: 텍스트
START_POINT = 1       # 읽기 시작점
STOP_POINT = 1000     # 읽기 종료점 (NORMal 모드 화면 포인트 = 1000)


def find_instruments():
    """연결 가능한 VISA 장비 목록을 검색합니다."""
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    print("=" * 60)
    print("검색된 VISA 장비 목록:")
    print("=" * 60)
    if not resources:
        print("  (장비가 발견되지 않았습니다)")
        print("  - USB 케이블 연결 확인")
        print("  - NI-VISA 드라이버 설치 확인")
        print("  - 오실로스코프 전원 확인")
    else:
        for i, res in enumerate(resources):
            print(f"  [{i}] {res}")
    print("=" * 60)
    rm.close()
    return resources


def connect(visa_address):
    """오실로스코프에 연결합니다."""
    rm = pyvisa.ResourceManager()
    try:
        inst = rm.open_resource(visa_address)
        inst.timeout = 10000      # 타임아웃 10초
        inst.read_termination = '\n'
        inst.write_termination = '\n'
        inst.chunk_size = 1024 * 1024  # 1MB 청크 (대용량 데이터용)

        # 장비 확인
        idn = inst.query("*IDN?")
        print(f"연결 성공: {idn.strip()}")
        return rm, inst
    except Exception as e:
        print(f"연결 실패: {e}")
        rm.close()
        sys.exit(1)


def _drain_errs(inst, label):
    """현재 에러 큐를 비우면서 0이 아닌 항목만 출력."""
    seen = []
    for _ in range(10):
        e = inst.query(":SYST:ERR?").strip()
        if e.startswith("0,") or not e:
            break
        seen.append(e)
    if seen:
        print(f"  [ERR after {label}] {seen}")


def get_waveform_params(inst):
    """파형 변환에 필요한 스케일링 파라미터를 읽습니다."""
    params = {}
    params['xinc'] = float(inst.query(":WAV:XINC?"))     # X축 시간 간격 (초)
    params['xor'] = float(inst.query(":WAV:XOR?"))       # X축 시작 시간
    params['xref'] = float(inst.query(":WAV:XREF?"))     # X축 레퍼런스
    params['yinc'] = float(inst.query(":WAV:YINC?"))     # Y축 전압 증분
    params['yor'] = float(inst.query(":WAV:YOR?"))       # Y축 전압 오프셋
    params['yref'] = float(inst.query(":WAV:YREF?"))     # Y축 레퍼런스

    print("\n--- 파형 파라미터 ---")
    print(f"  X Increment : {params['xinc']:.6e} s")
    print(f"  X Origin    : {params['xor']:.6e} s")
    print(f"  Y Increment : {params['yinc']:.6e} V")
    print(f"  Y Origin    : {params['yor']:.6e} V")
    print(f"  Y Reference : {params['yref']}")
    return params


def capture_waveform(inst, channel=1, mode="NORMal", fmt="BYTE",
                     start=1, stop=1200):
    """
    오실로스코프에서 파형 데이터를 읽어옵니다.

    Parameters
    ----------
    inst : pyvisa.Resource
        연결된 오실로스코프 리소스
    channel : int
        채널 번호 (1~4)
    mode : str
        "NORMal" / "RAW" / "MAXimum"
    fmt : str
        "BYTE" / "WORD" / "ASCii"
    start : int
        읽기 시작 포인트
    stop : int
        읽기 종료 포인트

    Returns
    -------
    raw_data : numpy.ndarray
        Raw 데이터 배열
    params : dict
        스케일링 파라미터
    """
    print(f"\n채널 {channel} 파형 데이터 캡처 시작...")

    inst.write(":STOP"); time.sleep(0.3)
    inst.write(f":WAV:SOUR CHAN{channel}")
    inst.write(f":WAV:MODE {mode}")
    inst.write(f":WAV:FORM {fmt}")
    inst.write(f":WAV:STAR {start}")
    inst.write(f":WAV:STOP {stop}")
    time.sleep(0.1)

    params = get_waveform_params(inst)

    fmt_u = fmt.upper()
    if fmt_u in ("ASCII", "ASC"):
        raw_str = inst.query(":WAV:DATA?")
        raw_data = np.array([float(x) for x in raw_str.split(',') if x.strip()])
        unit = "ASCii"
    elif fmt_u == "WORD":
        raw_data = inst.query_binary_values(
            ":WAV:DATA?", datatype='H',
            is_big_endian=False, container=np.array
        ).astype(np.float64)
        unit = "WORD/16bit"
    else:
        raw_data = inst.query_binary_values(
            ":WAV:DATA?", datatype='B',
            is_big_endian=False, container=np.array
        ).astype(np.float64)
        unit = "BYTE/8bit"

    err = inst.query(":SYST:ERR?").strip()
    print(f"  수신 포인트 수: {len(raw_data):,} ({unit})  err={err}")

    return raw_data, params


def convert_to_voltage(raw_data, params):
    """
    Raw 데이터를 실제 전압값으로 변환합니다.

    공식: Voltage = (Raw - YReference - YOrigin) × YIncrement
    """
    voltage = (raw_data - params['yref'] - params['yor']) * params['yinc']
    return voltage


def generate_time_axis(num_points, params):
    """
    시간축 배열을 생성합니다.

    공식: Time[i] = XIncrement × i + XOrigin
    """
    time_arr = np.arange(num_points) * params['xinc'] + params['xor']
    return time_arr


def capture_screen(inst, filename="screenshot.png"):
    """
    오실로스코프 화면을 PNG 이미지로 캡처합니다.

    Parameters
    ----------
    inst : pyvisa.Resource
        연결된 오실로스코프 리소스
    filename : str
        저장할 파일명
    """
    print(f"\n화면 캡처 중...")
    raw = inst.query_binary_values(
        ":DISP:DATA? PNG",
        datatype='B',
        container=bytearray
    )
    with open(filename, 'wb') as f:
        f.write(raw)
    print(f"화면 캡처 저장 완료: {filename}")


def _auto_scale(arr_max, base="V"):
    if base == "T":
        if arr_max < 1e-6: return 1e9, "ns"
        if arr_max < 1e-3: return 1e6, "μs"
        if arr_max < 1:    return 1e3, "ms"
        return 1, "s"
    if arr_max < 1e-3: return 1e6, f"μ{base}"
    if arr_max < 1:    return 1e3, f"m{base}"
    return 1, base


def _apply_dark_style():
    plt.gca().set_facecolor('#1a1a2e')
    plt.gcf().set_facecolor('#16213e')
    plt.tick_params(colors='white')
    for spine in plt.gca().spines.values():
        spine.set_color('#444')


def plot_waveform(time_arr, v1, i2, filename="waveform.png"):
    """Ch1(V) / Ch2(A) 시간 파형을 twin-y 로 겹쳐 그립니다."""
    t_scale, t_unit = _auto_scale(abs(time_arr[-1] - time_arr[0]), "T")
    v_scale, v_unit = _auto_scale(max(abs(v1.min()), abs(v1.max())), "V")
    a_scale, a_unit = _auto_scale(max(abs(i2.min()), abs(i2.max())), "A")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.set_facecolor('#16213e')
    ax1.set_facecolor('#1a1a2e')

    l1, = ax1.plot(time_arr * t_scale, v1 * v_scale, linewidth=0.8, color='#FFD700', label=f'Ch1 ({v_unit})')
    ax1.set_xlabel(f"Time ({t_unit})", fontsize=12, color='white')
    ax1.set_ylabel(f"Ch1 Voltage ({v_unit})", fontsize=12, color='#FFD700')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='#FFD700')
    ax1.grid(True, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_color('#444')

    ax2 = ax1.twinx()
    l2, = ax2.plot(time_arr * t_scale, i2 * a_scale, linewidth=0.8, color='#00CED1', label=f'Ch2 ({a_unit})')
    ax2.set_ylabel(f"Ch2 Current ({a_unit})", fontsize=12, color='#00CED1')
    ax2.tick_params(axis='y', colors='#00CED1')
    for spine in ax2.spines.values():
        spine.set_color('#444')

    plt.title("DHO924S - Ch1 (V) / Ch2 (A) Waveform", fontsize=14, color='white')
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()],
               facecolor='#1a1a2e', edgecolor='#444', labelcolor='white', loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='#16213e')
    plt.close()
    print(f"그래프 저장: {filename}")


def plot_xy(v1, i2, filename="waveform_xy.png"):
    """XY 플롯: x=Ch2 (A), y=Ch1 (V)."""
    v_scale, v_unit = _auto_scale(max(abs(v1.min()), abs(v1.max())), "V")
    a_scale, a_unit = _auto_scale(max(abs(i2.min()), abs(i2.max())), "A")

    plt.figure(figsize=(7, 7))
    plt.plot(i2 * a_scale, v1 * v_scale, linewidth=0.5, color='#FF6B9D', alpha=0.85)
    _apply_dark_style()
    plt.xlabel(f"Ch2 ({a_unit})", fontsize=12, color='white')
    plt.ylabel(f"Ch1 ({v_unit})", fontsize=12, color='white')
    plt.title("DHO924S - XY Plot (X=Ch2 [A], Y=Ch1 [V])", fontsize=14, color='white')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('auto')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, facecolor='#16213e')
    plt.close()
    print(f"XY 그래프 저장: {filename}")


def save_csv(time_arr, v1, i2, filename="waveform.csv"):
    """Time, Ch1(V), Ch2(A) 3열 CSV 저장."""
    data = np.column_stack((time_arr, v1, i2))
    np.savetxt(filename, data, delimiter=',',
               header='Time(s),Ch1(V),Ch2(A)', comments='')
    print(f"CSV 저장 완료: {filename} ({len(time_arr)} points)")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":

    # --- 저장 파일명 prefix 입력 ---
    name = input("저장 파일명 prefix (Enter = waveform): ").strip() or "waveform"

    # --- 장비 검색 ---
    resources = find_instruments()

    # --- 연결 (주소를 환경에 맞게 수정하세요) ---
    # 검색된 장비 중 첫 번째를 자동 사용하려면:
    #   address = resources[0] if resources else VISA_ADDRESS
    address = VISA_ADDRESS
    rm, inst = connect(address)
    inst.write("*CLS")

    try:
        # --- 파형 데이터 캡처 (두 채널, STOP 상태 유지) ---
        voltages = {}
        time_arr = None
        for ch in CHANNELS:
            raw_data, params = capture_waveform(
                inst,
                channel=ch,
                mode=WAV_MODE,
                fmt=WAV_FORMAT,
                start=START_POINT,
                stop=STOP_POINT
            )
            v = convert_to_voltage(raw_data, params)
            voltages[ch] = v
            if time_arr is None:
                time_arr = generate_time_axis(len(v), params)

        # 길이 정합
        n = min(len(voltages[1]), len(voltages[2]), len(time_arr))
        v1 = voltages[1][:n]
        v2 = voltages[2][:n]
        time_arr = time_arr[:n]

        print(f"\n--- 결과 요약 ---")
        print(f"  포인트 수 : {n}")
        print(f"  시간 범위 : {time_arr[0]:.6e} ~ {time_arr[-1]:.6e} s")
        for label, v, unit in (("Ch1", v1, "V"), ("Ch2", v2, "A")):
            print(f"  {label} : min={v.min():.6f}  max={v.max():.6f}  "
                  f"mean={v.mean():.6f}  std={np.std(v, ddof=1):.6f} {unit}")

        # --- 타임스탬프 기반 파일명 ---
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        csv_path = os.path.join(SAVE_DIR, f"{name}_{ts}.csv")
        png_path = os.path.join(SAVE_DIR, f"{name}_{ts}.png")
        xy_path  = os.path.join(SAVE_DIR, f"{name}_{ts}_xy.png")

        # --- CSV / 그래프 ---
        save_csv(time_arr, v1, v2, filename=csv_path)
        plot_waveform(time_arr, v1, v2, filename=png_path)
        plot_xy(v1, v2, filename=xy_path)

        # --- (선택) 화면 캡처 ---
        # capture_screen(inst, "screenshot.png")

    finally:
        # --- 연결 종료 ---
        inst.write(":RUN")   # 오실로스코프 다시 동작 시작
        print(inst.query(":SYST:ERR?"))
        inst.close()
        rm.close()
        print("\n연결 종료.")