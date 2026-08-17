"""
Feature Engineering for NILM (Power, Harmonics, Phasors, and Distortion)
"""
import pandas as pd
import numpy as np
from typing import List
from .config import NUM_HARMONICS


def extract_features(
    df: pd.DataFrame,
    include_phasors: bool = True,
    include_harmonic_ratios: bool = True,
    include_deltas: bool = True,
    delta_window_steps: int = 6 # 6 steps = 0.1s at 60Hz
) -> pd.DataFrame:
    """
    정제된 DataFrame으로부터 전기공학적 특성(P, Q, S, PF, THD_I, 직교 고조파 페이저 Re/Im, 비율 등)을 계산.
    [V2 개선]: 원시 위상각(Degrees)의 -180/+180도 불연속성 문제를 해결하기 위해
    순수 직교 좌표계 (cos/sin, Real/Imaginary) 페이저 성분으로 완벽 대체합니다.
    """
    res = df.copy()
    eps = 1e-7
    
    # 1. 기본 전력 특성
    vrms = res['vrms'] if 'vrms' in res.columns else 220.0
    irms = res['irms'] if 'irms' in res.columns else 0.0
    p_w = res['p_w'] if 'p_w' in res.columns else 0.0
    
    s_va = vrms * irms
    res['s_va'] = s_va
    res['power_factor'] = np.clip(p_w / (s_va + eps), -1.0, 1.0)
    
    # 2. 기본파 위상각의 직교 성분 (cos/sin)
    phase_deg = res['phase_deg'] if 'phase_deg' in res.columns else 0.0
    phase_rad = np.radians(phase_deg)
    res['v_i_cos_phi'] = np.cos(phase_rad)
    res['v_i_sin_phi'] = np.sin(phase_rad)
    
    # 3. 부호 있는 무효전력 Q (var)
    q_sign = np.sign(res['v_i_sin_phi'])
    q_sign = np.where(q_sign == 0, 1.0, q_sign)
    q_var = q_sign * np.sqrt(np.maximum(0.0, s_va**2 - p_w**2))
    res['q_var'] = q_var
    
    # 4. 고조파 및 THD_I
    ih1 = res['ih1'] if 'ih1' in res.columns else irms
    ih_cols = [f"ih{h}" for h in range(1, NUM_HARMONICS + 1) if f"ih{h}" in res.columns]
    
    if len(ih_cols) >= NUM_HARMONICS:
        harmonic_sum_sq = sum(res[f"ih{h}"]**2 for h in range(2, NUM_HARMONICS + 1))
        res['thd_i'] = np.sqrt(harmonic_sum_sq) / (ih1 + eps)
        
        # 주요 홀수 고조파 실효값 지문 (3, 5, 7, 9차)
        res['ih_odd_rms'] = np.sqrt(
            res['ih3']**2 + res['ih5']**2 + res['ih7']**2 + res['ih9']**2
        )
        
        # 5. 고조파 정규화 비율 (ih_ratio_h = ih_h / ih_1)
        if include_harmonic_ratios:
            for h in range(2, NUM_HARMONICS + 1):
                res[f"ih_ratio_{h}"] = res[f"ih{h}"] / (ih1 + eps)
                
        # 6. 고조파 복소 페이저 직교 성분 (In-Phase Real / Quadrature Imaginary)
        if include_phasors:
            for h in range(1, NUM_HARMONICS + 1):
                ih_val = res[f"ih{h}"]
                ih_deg = res[f"ihdeg{h}"] if f"ihdeg{h}" in res.columns else 0.0
                rad = np.radians(ih_deg)
                # 정규화된 Re/Im (IH_h * cos/sin)
                res[f"ih_re_{h}"] = ih_val * np.cos(rad)
                res[f"ih_im_{h}"] = ih_val * np.sin(rad)
                # 고조파 위상각 자체의 순환 직교 성분 (cos/sin of theta_h)
                res[f"ih_cos_{h}"] = np.cos(rad)
                res[f"ih_sin_{h}"] = np.sin(rad)
                
    # 7. 시계열 변화율 (Temporal Deltas)
    if include_deltas:
        p_smooth = res['p_w'].rolling(delta_window_steps, min_periods=1, center=True).median()
        res['delta_p'] = p_smooth.diff().fillna(0.0)
        
        q_smooth = res['q_var'].rolling(delta_window_steps, min_periods=1, center=True).median()
        res['delta_q'] = q_smooth.diff().fillna(0.0)
        
        i_smooth = res['irms'].rolling(delta_window_steps, min_periods=1, center=True).median()
        res['delta_i'] = i_smooth.diff().fillna(0.0)
        
    return res


def get_feature_column_names(
    include_phasors: bool = True,
    include_harmonic_ratios: bool = True,
    include_deltas: bool = True
) -> List[str]:
    """NILM AI V2 입력으로 사용될 표준 피처 컬럼 목록 반환 (불연속 각도 제거, 순수 직교 성분)"""
    base_cols = [
        "p_w", "q_var", "s_va", "power_factor", "v_i_cos_phi", "v_i_sin_phi",
        "irms", "vrms", "freq_hz", "thd_v", "thd_i", "ih_odd_rms"
    ]
    # 고조파 실효값 (ih1 ~ ih15)
    harm_cols = [f"ih{h}" for h in range(1, NUM_HARMONICS + 1)]
    
    cols = base_cols + harm_cols
    
    if include_harmonic_ratios:
        cols += [f"ih_ratio_{h}" for h in range(2, NUM_HARMONICS + 1)]
        
    if include_phasors:
        for h in range(1, NUM_HARMONICS + 1):
            cols.append(f"ih_re_{h}")
            cols.append(f"ih_im_{h}")
            cols.append(f"ih_cos_{h}")
            cols.append(f"ih_sin_{h}")
            
    if include_deltas:
        cols += ["delta_p", "delta_q", "delta_i"]
        
    return cols
