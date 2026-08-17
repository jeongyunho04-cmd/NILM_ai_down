"""
NILM AI Configuration and Appliance Metadata
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

# ── 전기 계통 기본 상수 ──────────────────────────────────────────────────────────
SAMPLING_HZ = 60.0              # 주기별 데이터 해상도 (60Hz 1주기 = 1샘플)
DT_SECONDS = 1.0 / SAMPLING_HZ   # ~0.016667 초
NUM_HARMONICS = 15              # 1차~15차 고조파 (60Hz ~ 900Hz)
DEFAULT_FEATURE_DIM = 104       # 추출되는 V2 전기공학 피처 차원 수 (직교 페이저 기반)
GRID_NOMINAL_VOLTAGE = 220.0    # 계통 공칭 전압 (Vrms)
GRID_NOMINAL_FREQ = 60.0        # 계통 공칭 주파수 (Hz)

# ── 기기별 공칭 정격 전력 (W) (정규화 손실 가중치용) ───────────────────────
NOMINAL_POWERS = {
    "kettle": 1260.0,
    "fan": 32.0,
    "beam_projector": 48.0,
    "laptop_charger": 45.0,
    "minipc": 15.0
}

# ── 선로 임피던스 기본 파라미터 (Line Impedance for Voltage Sag) ─────────────
DEFAULT_LINE_RESISTANCE = 0.25   # R_line (Ohm, 일반 주택/사무실 옥내 배선 저항)
DEFAULT_LINE_INDUCTANCE = 0.15e-3 # L_line (Henry, 0.15 mH)

# ── 센서 노이즈 플로어 ────────────────────────────────────────────────────────
NOISE_FLOOR_POWER_W = 1.40      # 무부하 시 전력 베이스라인 (W)
NOISE_FLOOR_CURRENT_A = 0.0068  # 무부하 시 전류 베이스라인 (A)
APPLIANCE_ON_THRESHOLD_W = 3.0  # 기기 ON 판정 기본 임계값 (W)


@dataclass
class ApplianceInfo:
    id: int
    key: str
    name_ko: str
    category: str              # 'resistive', 'inductive_motor', 'smps_electronic', 'smps_computer'
    files: List[str]
    typical_power_range: tuple # (min_on_w, max_on_w)
    typical_pf: float          # typical Power Factor
    typical_ih3_ratio: float   # IH3 / IH1 ratio (%)
    nominal_voltage: float = 220.0
    notes: str = ""


# 5개 전자기기 메타데이터 정의
APPLIANCES: Dict[str, ApplianceInfo] = {
    "kettle": ApplianceInfo(
        id=0,
        key="kettle",
        name_ko="전기주전자",
        category="resistive",
        files=["electiric_kettle.csv"],
        typical_power_range=(1150.0, 1300.0),
        typical_pf=0.999,
        typical_ih3_ratio=5.1,
        notes="순수 저항성 대용량 히터 부하. 6A 고전류. High Range 스위칭."
    ),
    "fan": ApplianceInfo(
        id=1,
        key="fan",
        name_ko="선풍기",
        category="inductive_motor",
        files=["fan_1.csv", "fan_2.csv", "fan_3.csv"],
        typical_power_range=(18.0, 48.0),
        typical_pf=0.90,
        typical_ih3_ratio=11.2,
        notes="유도성 모터 부하. 미풍/약풍/강풍 다단 속도 모드."
    ),
    "beam_projector": ApplianceInfo(
        id=2,
        key="beam_projector",
        name_ko="빔프로젝터",
        category="smps_electronic",
        files=["beam_projector.csv"],
        typical_power_range=(40.0, 55.0),
        typical_pf=0.57,
        typical_ih3_ratio=85.7,
        notes="비선형 SMPS + 광원/팬. 3차/5차/7차 강한 고조파 왜곡."
    ),
    "laptop_charger": ApplianceInfo(
        id=3,
        key="laptop_charger",
        name_ko="노트북 충전기",
        category="smps_electronic",
        files=["laptop_charger_1.csv", "laptop_charger_2.csv"],
        typical_power_range=(25.0, 80.0),
        typical_pf=0.54,
        typical_ih3_ratio=92.8,
        notes="배터리 충전 프로파일에 따른 가변 전력 SMPS. 높은 고조파 왜곡."
    ),
    "minipc": ApplianceInfo(
        id=4,
        key="minipc",
        name_ko="미니PC",
        category="smps_computer",
        files=["minipc_1.csv", "minipc_2.csv"],
        typical_power_range=(9.0, 30.0),
        typical_pf=0.48,
        typical_ih3_ratio=82.0,
        notes="컴퓨터 SMPS 부하. CPU 로드에 따른 동적 변동."
    ),
}

# 기기 키 목록 및 ID 매핑
APPLIANCE_KEYS = list(APPLIANCES.keys())
NUM_APPLIANCES = len(APPLIANCE_KEYS)
APPLIANCE_TO_ID = {k: APPLIANCES[k].id for k in APPLIANCE_KEYS}
ID_TO_APPLIANCE = {APPLIANCES[k].id: k for k in APPLIANCE_KEYS}

# 노이즈 데이터 파일
NOISE_FILE = "noise_noselfpower.csv"
