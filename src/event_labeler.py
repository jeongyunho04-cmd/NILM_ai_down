"""
Event and Operating State Labeler for Appliance Ground Truth
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from .config import APPLIANCES, ApplianceInfo


# 기기별 ON/OFF 판정 전력 임계값 (Hysteresis: On Threshold / Off Threshold in Watts)
DEFAULT_APPLIANCE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "kettle": (500.0, 100.0),         # ON > 500W, OFF < 100W (실제 동작 ~1260W)
    "fan": (15.0, 8.0),               # ON > 15W, OFF < 8W (실제 동작 20~45W)
    "beam_projector": (20.0, 8.0),     # ON > 20W, OFF < 8W (실제 동작 ~48W)
    "laptop_charger": (15.0, 6.0),     # ON > 15W, OFF < 6W (실제 동작 30~75W)
    "minipc": (7.0, 4.0),             # ON > 7W, OFF < 4W (실제 동작 9.5~30W)
}


def label_appliance_state(
    df: pd.DataFrame,
    appliance_key: str,
    on_thresh: Optional[float] = None,
    off_thresh: Optional[float] = None,
    min_event_duration_steps: int = 15 # 15 steps = 0.25s at 60Hz
) -> pd.DataFrame:
    """
    단일 기기 DataFrame에서 히스테리시스 임계값과 지속 시간 필터를 이용해 ON/OFF 상태(0 or 1)를 라벨링.
    
    Args:
        df: 대상 DataFrame (반드시 'p_w' 포함)
        appliance_key: 기기 식별자 ('kettle', 'fan', 'beam_projector', etc.)
        on_thresh: ON 전환 임계 전력 (W)
        off_thresh: OFF 전환 임계 전력 (W)
        min_event_duration_steps: 채터링 방지를 위한 최소 지속 스텝 수
        
    Returns:
        'is_on', 'appliance_state', 'appliance_p_w' 컬럼이 추가된 DataFrame
    """
    res = df.copy()
    
    if on_thresh is None or off_thresh is None:
        th_on, th_off = DEFAULT_APPLIANCE_THRESHOLDS.get(appliance_key, (10.0, 5.0))
        if on_thresh is not None:
            th_on = on_thresh
        if off_thresh is not None:
            th_off = off_thresh
    else:
        th_on, th_off = on_thresh, off_thresh
        
    p = res['p_w'].values
    n = len(p)
    is_on = np.zeros(n, dtype=np.int32)
    
    # 1. 히스테리시스 상태 전이 (Schmitt Trigger Logic)
    curr_state = 0
    for i in range(n):
        val = p[i]
        if curr_state == 0:
            if val >= th_on:
                curr_state = 1
        else:
            if val <= th_off:
                curr_state = 0
        is_on[i] = curr_state
        
    # 2. 짧은 스파이크 및 순간 끊김 제거 (Duration Filtering)
    if min_event_duration_steps > 1:
        # 단기 ON 펄스 제거 (스파이크 필터)
        on_runs = np.split(np.where(is_on == 1)[0], np.where(np.diff(np.where(is_on == 1)[0]) != 1)[0] + 1)
        for run in on_runs:
            if len(run) > 0 and len(run) < min_event_duration_steps:
                is_on[run] = 0
                
        # 단기 OFF 갭 채우기 (순간 드롭아웃 보정)
        off_runs = np.split(np.where(is_on == 0)[0], np.where(np.diff(np.where(is_on == 0)[0]) != 1)[0] + 1)
        for run in off_runs:
            if len(run) > 0 and len(run) < min_event_duration_steps:
                if run[0] > 0 and run[-1] < n - 1: # 양 끝단 제외
                    is_on[run] = 1

    res['is_on'] = is_on
    res['appliance_key'] = appliance_key
    # ON 상태일 때만 유효한 개별 기기 전력
    res['appliance_p_w'] = np.where(is_on == 1, res['p_w'], 0.0)
    
    return res
