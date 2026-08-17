r"""
ACPhysicsEngine: Physical AC Circuit Parallel Superposition & Harmonic Vector Summation Engine
=============================================================================================
물리적 AC 회로 병렬 중첩 & 고조파 벡터합 엔진
- KCL (키르히호프 전류 법칙) 기반 고조파 복소 전류 페이저(\dot{I}_h) 벡터합
- 유효전력(P), 무효전력(Q), 피상전력(S), 역률(PF), 전류 왜곡률(THD_I) 정밀 물리 합성
- 선로 임피던스(Line Impedance R_line + j*w*L_line)에 의한 전압 강하(Voltage Sag) 및 전압 고조파 왜곡 모델링
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from .config import (
    NUM_HARMONICS,
    GRID_NOMINAL_VOLTAGE,
    GRID_NOMINAL_FREQ,
    DEFAULT_LINE_RESISTANCE,
    DEFAULT_LINE_INDUCTANCE
)


class ACPhysicsEngine:
    """
    AC 전기 회로 물리 엔진.
    다중 전자기기 부하가 병렬 연결된 회로에서 발생하는 복소 전류 페이저 합성 및
    선로 임피던스에 따른 전압 강하(Voltage Sag)를 정밀 시뮬레이션합니다.
    """

    def __init__(
        self,
        r_line: float = DEFAULT_LINE_RESISTANCE,
        l_line: float = DEFAULT_LINE_INDUCTANCE,
        grid_v_nominal: float = GRID_NOMINAL_VOLTAGE,
        grid_f_nominal: float = GRID_NOMINAL_FREQ,
        num_harmonics: int = NUM_HARMONICS
    ):
        """
        Args:
            r_line: 선로 저항 (Ohm, 기본값 0.25 Ohm)
            l_line: 선로 인덕턴스 (Henry, 기본값 0.15 mH)
            grid_v_nominal: 계통 무부하 공칭 전압 (Vrms, 기본값 220.0V)
            grid_f_nominal: 계통 공칭 주파수 (Hz, 기본값 60.0Hz)
            num_harmonics: 고려할 고조파 차수 (기본값 15차)
        """
        self.r_line = float(r_line)
        self.l_line = float(l_line)
        self.grid_v_nominal = float(grid_v_nominal)
        self.grid_f_nominal = float(grid_f_nominal)
        self.num_harmonics = int(num_harmonics)
        self.eps = 1e-9

        # 각 고조파별 선로 리액턴스 X_line[h] = 2 * pi * f_0 * h * L_line
        self.omega_base = 2.0 * np.pi * self.grid_f_nominal
        self.harmonics_idx = np.arange(1, self.num_harmonics + 1)
        self.z_line = np.array([
            complex(self.r_line, self.omega_base * h * self.l_line)
            for h in self.harmonics_idx
        ]) # shape: (num_harmonics,)

    def compute_harmonic_phasors(
        self,
        ih_rms: np.ndarray,
        ih_deg: np.ndarray
    ) -> np.ndarray:
        r"""
        고조파 RMS 크기와 위상각(도)으로부터 복소 전류 페이저 \dot{I}_h 계산
        
        Args:
            ih_rms: (..., num_harmonics) 고조파 전류 RMS (A)
            ih_deg: (..., num_harmonics) 고조파 위상각 (Degrees)
            
        Returns:
            complex_phasors: (..., num_harmonics) 복소수 페이저 (Real = In-phase, Imag = Quadrature)
        """
        rad = np.radians(ih_deg)
        return ih_rms * (np.cos(rad) + 1j * np.sin(rad))

    def superimpose_parallel_loads(
        self,
        load_signals: List[Dict[str, np.ndarray]],
        noise_signal: Optional[Dict[str, np.ndarray]] = None,
        apply_voltage_sag: bool = True,
        v_source: Optional[Union[float, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        여러 기기의 시계열 전기 신호들을 KCL 및 KVL 기반으로 병렬 합성.
        
        Args:
            load_signals: 각 기기의 신호 딕셔너리 리스트. 각 dict는 다음 키를 포함:
                - 'p_w': (T,) 유효전력 (W)
                - 'ih_rms': (T, num_harmonics) 1~15차 고조파 실효값 (A)
                - 'ih_deg': (T, num_harmonics) 1~15차 고조파 위상각 (deg)
                - 'is_on': (T,) ON/OFF 상태 (0 or 1) [선택]
            noise_signal: 무부하 노이즈 신호 dict [선택]
            apply_voltage_sag: True일 경우 선로 임피던스에 의한 전압 강하 적용
            v_source: 무부하 계통 전압 (기본값 220.0V 또는 시계열)
            
        Returns:
            aggregate_dict: 합성된 종합 전기 신호 딕셔너리
                - 'p_w': 총 유효전력 (W)
                - 'q_var': 총 무효전력 (var)
                - 's_va': 총 피상전력 (VA)
                - 'power_factor': 총 역률 (PF)
                - 'irms': 총 실효 전류 (A)
                - 'vrms': 수전단 단자 전압 (V, 전압 강하 반영)
                - 'voltage_sag': 전압 강하량 (V)
                - 'thd_i': 총 전류 고조파 왜곡률
                - 'thd_v': 총 전압 고조파 왜곡률
                - 'ih_rms': (T, num_harmonics) 합성된 고조파 RMS (A)
                - 'ih_deg': (T, num_harmonics) 합성된 고조파 위상각 (deg)
                - 'vh_rms': (T, num_harmonics) 수전단 전압 고조파 RMS (V)
        """
        assert len(load_signals) > 0, "최소 1개 이상의 부하 신호가 필요합니다."
        T = len(load_signals[0]['p_w'])
        
        # 1. 초기 복소 고조파 전류 페이저 및 유효전력 합산
        total_p = np.zeros(T, dtype=np.float64)
        total_phasors = np.zeros((T, self.num_harmonics), dtype=np.complex128)
        
        for load in load_signals:
            p = load['p_w']
            ih = load['ih_rms']
            deg = load['ih_deg']
            
            total_p += p
            phasors = self.compute_harmonic_phasors(ih, deg)
            total_phasors += phasors
            
        # 노이즈 신호 추가
        if noise_signal is not None:
            total_p += noise_signal.get('p_w', 0.0)
            if 'ih_rms' in noise_signal and 'ih_deg' in noise_signal:
                noise_phasors = self.compute_harmonic_phasors(
                    noise_signal['ih_rms'], noise_signal['ih_deg']
                )
                total_phasors += noise_phasors
                
        # 2. 합성된 고조파 전류 RMS 및 위상각 (KCL)
        ih_rms_total = np.abs(total_phasors) # (T, num_harmonics)
        ih_deg_total = np.degrees(np.angle(total_phasors)) # (T, num_harmonics)
        
        # 기본파 전류 및 총 전류 RMS
        i1_rms_total = ih_rms_total[:, 0]
        irms_total = np.sqrt(np.sum(ih_rms_total**2, axis=-1)) # Parseval's theorem: I_rms = sqrt(sum(I_h^2))
        
        # 3. 전압 강하 (Voltage Sag) 및 전압 고조파 왜곡 계산
        if v_source is None:
            v_open_circuit = np.full(T, self.grid_v_nominal, dtype=np.float64)
        elif isinstance(v_source, (int, float)):
            v_open_circuit = np.full(T, float(v_source), dtype=np.float64)
        else:
            v_open_circuit = np.array(v_source, dtype=np.float64)
            
        # 각 고조파별 전압 강하 복소 페이저 \dot{V}_{drop, h} = \dot{I}_{total, h} * Z_{line, h}
        # total_phasors: (T, 15), z_line: (15,) -> v_drop_phasors: (T, 15)
        v_drop_phasors = total_phasors * self.z_line
        
        if apply_voltage_sag:
            # 기본파 전압 강하: V_terminal,1 = V_source - Re(V_drop,1)
            # (지상 유도성 부하 및 저항 부하 시 전류 페이저에 의해 전압이 강하됨)
            v1_sag = np.real(v_drop_phasors[:, 0])
            v1_terminal = np.maximum(50.0, v_open_circuit - v1_sag)
            
            # 고조파 전압 왜곡 (선로 임피던스를 통과하며 발생하는 전압 고조파)
            vh_rms_total = np.zeros((T, self.num_harmonics), dtype=np.float64)
            vh_rms_total[:, 0] = v1_terminal
            
            # 2차~15차 전압 고조파 실효값 V_h = |V_drop, h|
            vh_rms_total[:, 1:] = np.abs(v_drop_phasors[:, 1:])
            
            # 총 전압 RMS
            vrms_terminal = np.sqrt(np.sum(vh_rms_total**2, axis=-1))
            voltage_sag = v_open_circuit - vrms_terminal
        else:
            vrms_terminal = v_open_circuit
            voltage_sag = np.zeros(T, dtype=np.float64)
            vh_rms_total = np.zeros((T, self.num_harmonics), dtype=np.float64)
            vh_rms_total[:, 0] = vrms_terminal
            
        # 4. 전력 공학 종합 계산 (P, Q, S, PF, THD)
        # 기본파 위상차 phi_1 = theta_v1 - theta_i1 (전압 기준 위상각 0도) -> phi_1 = -theta_i1
        i1_phase_rad = np.radians(ih_deg_total[:, 0])
        # 기본파 무효전력 Q_1 = V1 * I1 * sin(-theta_i1) = -V1 * I1 * sin(theta_i1)
        # 수전공학 규약: 유도성(지상, Lagging, theta_i < 0) -> Q > 0 소비
        q_fundamental = -vrms_terminal * i1_rms_total * np.sin(i1_phase_rad)
        
        # Budeanu 총 무효전력 Q_total = sum_h (V_h * I_h * sin(phi_h))
        q_var_total = q_fundamental.copy()
        
        # 피상전력 S = V_rms * I_rms
        s_va_total = vrms_terminal * irms_total
        
        # 역률 Power Factor = P / S
        pf_total = np.clip(total_p / (s_va_total + self.eps), -1.0, 1.0)
        
        # 전류 고조파 왜곡률 THD_I = sqrt(sum_{h=2}^15 I_h^2) / I_1
        i_harm_sum_sq = np.sum(ih_rms_total[:, 1:]**2, axis=-1)
        thd_i_total = np.sqrt(i_harm_sum_sq) / (i1_rms_total + self.eps)
        
        # 전압 고조파 왜곡률 THD_V = sqrt(sum_{h=2}^15 V_h^2) / V_1
        v_harm_sum_sq = np.sum(vh_rms_total[:, 1:]**2, axis=-1)
        thd_v_total = np.sqrt(v_harm_sum_sq) / (vh_rms_total[:, 0] + self.eps)
        
        return {
            "p_w": total_p,
            "q_var": q_var_total,
            "s_va": s_va_total,
            "power_factor": pf_total,
            "irms": irms_total,
            "vrms": vrms_terminal,
            "voltage_sag": voltage_sag,
            "thd_i": thd_i_total,
            "thd_v": thd_v_total,
            "ih_rms": ih_rms_total,
            "ih_deg": ih_deg_total,
            "vh_rms": vh_rms_total,
            "phase_deg": ih_deg_total[:, 0] # 기본파 위상각
        }

    def convert_df_to_load_signal(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """DataFrame으로부터 물리 엔진 입력용 신호 딕셔너리 추출"""
        T = len(df)
        p_w = df['p_w'].values.astype(np.float64)
        
        ih_cols = [f"ih{h}" for h in range(1, self.num_harmonics + 1)]
        ihdeg_cols = [f"ihdeg{h}" for h in range(1, self.num_harmonics + 1)]
        
        if all(c in df.columns for c in ih_cols):
            ih_rms = df[ih_cols].values.astype(np.float64)
        else:
            # 고조파가 없으면 기본파만 irms로 채움
            ih_rms = np.zeros((T, self.num_harmonics), dtype=np.float64)
            ih_rms[:, 0] = df['irms'].values if 'irms' in df.columns else 0.0
            
        if all(c in df.columns for c in ihdeg_cols):
            ih_deg = df[ihdeg_cols].values.astype(np.float64)
        else:
            ih_deg = np.zeros((T, self.num_harmonics), dtype=np.float64)
            if 'phase_deg' in df.columns:
                ih_deg[:, 0] = df['phase_deg'].values
                
        is_on = df['is_on'].values.astype(np.int32) if 'is_on' in df.columns else (p_w > 5.0).astype(np.int32)
        
        return {
            "p_w": p_w,
            "ih_rms": ih_rms,
            "ih_deg": ih_deg,
            "is_on": is_on
        }
