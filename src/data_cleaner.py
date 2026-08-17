"""
Data Cleaner and Signal Quality Processing Module
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from .config import (
    NUM_HARMONICS,
    NOISE_FLOOR_POWER_W,
    NOISE_FLOOR_CURRENT_A,
    SAMPLING_HZ
)


def normalize_angle_deg(angles: np.ndarray) -> np.ndarray:
    """위상각을 [-180.0, +180.0] 도로 정규화"""
    return (angles + 180.0) % 360.0 - 180.0


def clean_dataframe(
    df: pd.DataFrame,
    file_name: Optional[str] = None,
    subtract_noise_floor: bool = False,
    interpolate_pll_unlock: bool = True
) -> pd.DataFrame:
    """
    원시 60Hz CSV DataFrame을 정제하여 이상치, PLL 언락, 음수 스파이크를 보정.
    
    Args:
        df: 원본 DataFrame
        file_name: 파일명 (디버깅/로깅용)
        subtract_noise_floor: True일 경우 센서 회로 무부하 베이스라인 전력(~1.4W) 차감
        interpolate_pll_unlock: True일 경우 PLL 언락(동기 풀림) 구간을 보간
        
    Returns:
        정제된 DataFrame
    """
    cleaned = df.copy()
    
    # 1. 컬럼명 공백 제거
    cleaned.columns = [c.strip() for c in cleaned.columns]
    
    # 2. 음수 전력 및 전류 클리핑 (부하 기기는 발전원이 아니므로 음수 불가능)
    if 'p_w' in cleaned.columns:
        cleaned['p_w'] = np.maximum(0.0, cleaned['p_w'].astype(float))
    if 'irms' in cleaned.columns:
        cleaned['irms'] = np.maximum(0.0, cleaned['irms'].astype(float))
        
    # 3. 고조파 RMS 및 위상각 정제
    for h in range(1, NUM_HARMONICS + 1):
        ih_col = f"ih{h}"
        ihdeg_col = f"ihdeg{h}"
        vh_col = f"vh{h}"
        
        if ih_col in cleaned.columns:
            cleaned[ih_col] = np.maximum(0.0, cleaned[ih_col].astype(float))
        if vh_col in cleaned.columns:
            cleaned[vh_col] = np.maximum(0.0, cleaned[vh_col].astype(float))
        if ihdeg_col in cleaned.columns:
            cleaned[ihdeg_col] = normalize_angle_deg(cleaned[ihdeg_col].astype(float).values)
            
    if 'phase_deg' in cleaned.columns:
        cleaned['phase_deg'] = normalize_angle_deg(cleaned['phase_deg'].astype(float).values)
        
    # 4. PLL Unlock 처리
    if 'pll_locked' in cleaned.columns and interpolate_pll_unlock:
        unlock_mask = (cleaned['pll_locked'] == 0)
        if unlock_mask.any():
            # PLL 언락된 행의 고조파 및 위상, 전압/주파수 컬럼을 인접 정상값으로 선형 보간
            cols_to_interp = ['freq_hz', 'vrms', 'thd_v', 'phase_deg', 'p_w', 'irms'] + \
                             [f"ih{h}" for h in range(1, NUM_HARMONICS + 1)] + \
                             [f"ihdeg{h}" for h in range(1, NUM_HARMONICS + 1)]
            existing_cols = [c for c in cols_to_interp if c in cleaned.columns]
            
            # 언락 구간을 NaN으로 설정 후 보간
            cleaned.loc[unlock_mask, existing_cols] = np.nan
            cleaned[existing_cols] = cleaned[existing_cols].interpolate(method='linear', limit_direction='both')
            # 만약 전부 NaN인 경우(시작부 등) bfill/ffill
            cleaned[existing_cols] = cleaned[existing_cols].bfill().ffill()
            
    # 5. 무부하 노이즈 플로어 차감 옵션
    if subtract_noise_floor and 'p_w' in cleaned.columns:
        cleaned['p_w_raw'] = cleaned['p_w']
        cleaned['p_w'] = np.maximum(0.0, cleaned['p_w'] - NOISE_FLOOR_POWER_W)
        if 'irms' in cleaned.columns:
            cleaned['irms_raw'] = cleaned['irms']
            cleaned['irms'] = np.maximum(0.0, cleaned['irms'] - NOISE_FLOOR_CURRENT_A)
            
    # 6. 전압 기본값 및 THD_V 안전 범위 보정
    if 'vrms' in cleaned.columns:
        cleaned['vrms'] = cleaned['vrms'].clip(lower=150.0, upper=300.0)
    if 'freq_hz' in cleaned.columns:
        cleaned['freq_hz'] = cleaned['freq_hz'].clip(lower=50.0, upper=70.0)
        
    return cleaned
