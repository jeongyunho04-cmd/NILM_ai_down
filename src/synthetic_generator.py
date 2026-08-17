"""
Synthetic Aggregate Dataset Generator for NILM AI
=================================================
5개 전자기기의 실제 측정 데이터와 노이즈를 물리 엔진(ACPhysicsEngine)을 통해
다양한 동시 가동 조합(Multi-appliance combinations) 및 현실적인 이벤트 패턴으로 합성하는 생성기.
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from .config import (
    DATA_DIR,
    APPLIANCES,
    APPLIANCE_KEYS,
    NUM_APPLIANCES,
    NOISE_FILE,
    SAMPLING_HZ
)
from .data_cleaner import clean_dataframe
from .event_labeler import label_appliance_state
from .ac_physics_engine import ACPhysicsEngine


# 기기별 현실적인 가동 지속시간 및 대기시간 범위 (초 단위)
REALISTIC_DURATION_RANGES = {
    "kettle": {"on": (30.0, 120.0), "off": (60.0, 240.0)},         # 물 끓이기 사이클
    "fan": {"on": (60.0, 300.0), "off": (45.0, 180.0)},           # 선풍기 지속 가동
    "beam_projector": {"on": (120.0, 600.0), "off": (60.0, 240.0)},# 빔프로젝터 시청 (2~10분)
    "laptop_charger": {"on": (180.0, 600.0), "off": (120.0, 300.0)},# 배터리 장기 충전 (3~10분)
    "minipc": {"on": (150.0, 600.0), "off": (60.0, 240.0)},       # PC 작업 (2.5~10분)
}


class SyntheticDatasetGenerator:
    """
    물리 기반 다중 기기 복합 신호(Aggregate Signal) 합성 생성기 (V2: 현실적 상태 머신 탑재)
    """

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        physics_engine: Optional[ACPhysicsEngine] = None,
        apply_voltage_sag: bool = True
    ):
        self.data_dir = data_dir
        self.engine = physics_engine if physics_engine is not None else ACPhysicsEngine()
        self.apply_voltage_sag = apply_voltage_sag
        
        # 기기별 정제된 데이터 및 세그먼트 풀 캐시
        self.appliance_data: Dict[str, List[pd.DataFrame]] = {}
        self.noise_df: Optional[pd.DataFrame] = None
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """원본 CSV 파일들을 로드하고 정제 및 라벨링하여 메모리에 캐싱"""
        print("[SyntheticGenerator] 원본 데이터 로드 및 전처리 시작...")
        
        # 1. 노이즈 데이터 로드
        noise_path = os.path.join(self.data_dir, NOISE_FILE)
        if os.path.exists(noise_path):
            raw_noise = pd.read_csv(noise_path)
            self.noise_df = clean_dataframe(raw_noise, file_name=NOISE_FILE)
            print(f"  - 노이즈 데이터 로드 완료: {len(self.noise_df)} 행")
        else:
            print(f"  [경고] 노이즈 파일({NOISE_FILE})을 찾을 수 없습니다.")
            
        # 2. 5개 기기 데이터 로드
        for app_key, app_info in APPLIANCES.items():
            self.appliance_data[app_key] = []
            for fname in app_info.files:
                fpath = os.path.join(self.data_dir, fname)
                if os.path.exists(fpath):
                    raw_df = pd.read_csv(fpath)
                    clean_df = clean_dataframe(raw_df, file_name=fname)
                    labeled_df = label_appliance_state(clean_df, appliance_key=app_key)
                    self.appliance_data[app_key].append(labeled_df)
                else:
                    print(f"  [경고] 파일 없음: {fname}")
                    
            total_rows = sum(len(d) for d in self.appliance_data[app_key])
            print(f"  - [{app_info.name_ko} ({app_key})] {len(self.appliance_data[app_key])}개 파일, 총 {total_rows} 행 로드 완료")

    def _get_random_appliance_slice(
        self,
        app_key: str,
        length: int,
        target_state: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None
    ) -> pd.DataFrame:
        """기기 풀에서 특정 길이(length)의 연속 시계열 슬라이스를 무작위 추출"""
        if rng is None:
            rng = np.random.RandomState()
            
        dfs = self.appliance_data[app_key]
        assert len(dfs) > 0, f"{app_key} 데이터가 로드되지 않았습니다."
        
        # 데이터프레임 랜덤 선택
        df = dfs[rng.randint(len(dfs))]
        n = len(df)
        
        if n <= length:
            repeats = int(np.ceil(length / n))
            tiled = pd.concat([df] * repeats, ignore_index=True)
            return tiled.iloc[:length].copy()
            
        # 특정 상태(ON=1 또는 OFF=0)를 만족하는 구간 탐색
        if target_state is not None:
            state_mask = (df['is_on'] == target_state)
            valid_indices = np.where(state_mask)[0]
            valid_starts = valid_indices[valid_indices <= n - length]
            if len(valid_starts) > 0:
                start_idx = rng.choice(valid_starts)
                return df.iloc[start_idx:start_idx + length].copy()
                
        # 일반 랜덤 슬라이스
        start_idx = rng.randint(0, n - length + 1)
        return df.iloc[start_idx:start_idx + length].copy()

    def _get_random_noise_slice(
        self,
        length: int,
        rng: Optional[np.random.RandomState] = None
    ) -> Optional[Dict[str, np.ndarray]]:
        """노이즈 데이터에서 무작위 슬라이스 추출"""
        if self.noise_df is None or len(self.noise_df) == 0:
            return None
        if rng is None:
            rng = np.random.RandomState()
            
        n = len(self.noise_df)
        if n <= length:
            repeats = int(np.ceil(length / n))
            tiled = pd.concat([self.noise_df] * repeats, ignore_index=True)
            slice_df = tiled.iloc[:length]
        else:
            start_idx = rng.randint(0, n - length + 1)
            slice_df = self.noise_df.iloc[start_idx:start_idx + length]
            
        return self.engine.convert_df_to_load_signal(slice_df)

    def generate_scenario(
        self,
        duration_seconds: float = 60.0,
        active_appliances: Optional[List[str]] = None,
        initial_on_appliances: Optional[List[str]] = None,
        on_probabilities: Optional[Dict[str, float]] = None,
        enable_power_scaling: bool = True,
        power_scale_range: Tuple[float, float] = (0.75, 1.25),
        voltage_variation_std: float = 2.5,
        harmonic_jitter_std: float = 0.03,
        phase_jitter_std: float = 2.0,
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        NILM 학습/평가용 현실적인 다중 기기 복합 시나리오 생성 (물리적 상태 머신 적용).
        
        Args:
            initial_on_appliances: 시작 시점(t=0s)부터 반드시 켜져 있는 기기 목록 (예: ['fan', 'beam_projector'])
        """
        rng = np.random.RandomState(seed)
        total_steps = int(duration_seconds * SAMPLING_HZ)
        
        if active_appliances is None:
            candidate_apps = list(APPLIANCES.keys())
        else:
            candidate_apps = [a for a in active_appliances if a in APPLIANCES]
            
        load_signals = []
        gt_states = {}
        gt_powers = {}
        
        for app_key in candidate_apps:
            # ── [물리적 상태 머신 타임라인 생성] ──
            dur_cfg = REALISTIC_DURATION_RANGES.get(
                app_key, {"on": (30.0, 120.0), "off": (30.0, 120.0)}
            )
            
            state_timeline = np.zeros(total_steps, dtype=np.int32)
            
            # 시작 상태 결정 (initial_on_appliances 명시 시 우선 반영)
            if initial_on_appliances is not None:
                curr_state = 1 if app_key in initial_on_appliances else 0
            elif on_probabilities is not None:
                p_on = on_probabilities.get(app_key, 0.5)
                curr_state = 1 if rng.rand() < p_on else 0
            else:
                curr_state = 1 if rng.rand() < 0.5 else 0
                
            curr_t = 0
            
            while curr_t < total_steps:
                if curr_state == 1:
                    dur_s = rng.uniform(dur_cfg["on"][0], dur_cfg["on"][1])
                else:
                    dur_s = rng.uniform(dur_cfg["off"][0], dur_cfg["off"][1])
                    
                dur_steps = max(15, int(dur_s * SAMPLING_HZ))
                end_t = min(total_steps, curr_t + dur_steps)
                state_timeline[curr_t:end_t] = curr_state
                curr_t = end_t
                curr_state = 1 - curr_state # 토글
                
            # 신호 조립
            app_p = np.zeros(total_steps, dtype=np.float64)
            app_ih_rms = np.zeros((total_steps, self.engine.num_harmonics), dtype=np.float64)
            app_ih_deg = np.zeros((total_steps, self.engine.num_harmonics), dtype=np.float64)
            
            splits = np.where(np.diff(state_timeline) != 0)[0] + 1
            segment_starts = np.concatenate(([0], splits))
            segment_ends = np.concatenate((splits, [total_steps]))
            
            for s_start, s_end in zip(segment_starts, segment_ends):
                seg_len = s_end - s_start
                st = state_timeline[s_start]
                if st == 1:
                    slice_df = self._get_random_appliance_slice(app_key, seg_len, target_state=1, rng=rng)
                    sig = self.engine.convert_df_to_load_signal(slice_df)
                    
                    if enable_power_scaling:
                        scale_factor = rng.uniform(power_scale_range[0], power_scale_range[1])
                    else:
                        scale_factor = 1.0
                        
                    scaled_p = sig['p_w'] * scale_factor
                    scaled_ih = sig['ih_rms'] * scale_factor
                    
                    if harmonic_jitter_std > 0:
                        h_noise = 1.0 + rng.normal(0.0, harmonic_jitter_std, size=scaled_ih.shape)
                        scaled_ih = np.maximum(0.0, scaled_ih * h_noise)
                        
                    scaled_deg = sig['ih_deg'].copy()
                    if phase_jitter_std > 0:
                        scaled_deg += rng.normal(0.0, phase_jitter_std, size=scaled_deg.shape)
                        
                    app_p[s_start:s_end] = scaled_p
                    app_ih_rms[s_start:s_end] = scaled_ih
                    app_ih_deg[s_start:s_end] = scaled_deg
                else:
                    app_p[s_start:s_end] = 0.0
                    app_ih_rms[s_start:s_end] = 0.0
                    app_ih_deg[s_start:s_end] = 0.0
                    
            load_signals.append({
                "app_key": app_key,
                "p_w": app_p,
                "ih_rms": app_ih_rms,
                "ih_deg": app_ih_deg,
                "is_on": state_timeline
            })
            
            gt_states[f"state_{app_key}"] = state_timeline
            gt_powers[f"power_{app_key}"] = app_p
            
        # 2. 노이즈 신호 슬라이스
        noise_sig = self._get_random_noise_slice(total_steps, rng=rng)
        
        # 3. 계통 전압 랜덤 변동 (220V +- jitter)
        v_source = self.engine.grid_v_nominal + rng.normal(0.0, voltage_variation_std, size=total_steps)
        
        # 4. ACPhysicsEngine 병렬 중첩 & 고조파 벡터합 & 전압 강하 적용
        agg_result = self.engine.superimpose_parallel_loads(
            load_signals=load_signals,
            noise_signal=noise_sig,
            apply_voltage_sag=self.apply_voltage_sag,
            v_source=v_source
        )
        
        # 4. 결과 DataFrame 조립
        agg_df = pd.DataFrame({
            "t_s": np.arange(total_steps) / SAMPLING_HZ,
            "p_w": agg_result["p_w"],
            "q_var": agg_result["q_var"],
            "s_va": agg_result["s_va"],
            "power_factor": agg_result["power_factor"],
            "irms": agg_result["irms"],
            "vrms": agg_result["vrms"],
            "voltage_sag": agg_result["voltage_sag"],
            "phase_deg": agg_result["phase_deg"],
            "thd_i": agg_result["thd_i"],
            "thd_v": agg_result["thd_v"],
        })
        
        # 고조파 RMS 및 위상각 컬럼 추가
        for h in range(1, self.engine.num_harmonics + 1):
            agg_df[f"ih{h}"] = agg_result["ih_rms"][:, h - 1]
            agg_df[f"ihdeg{h}"] = agg_result["ih_deg"][:, h - 1]
            agg_df[f"vh{h}"] = agg_result["vh_rms"][:, h - 1]
            
        # Ground Truth DataFrame
        gt_dict = {"t_s": agg_df["t_s"]}
        # 모든 표준 기기 키에 대해 컬럼 확보 (비활성화 기기는 0)
        for key in APPLIANCE_KEYS:
            gt_dict[f"state_{key}"] = gt_states.get(f"state_{key}", np.zeros(total_steps, dtype=np.int32))
            gt_dict[f"power_{key}"] = gt_powers.get(f"power_{key}", np.zeros(total_steps, dtype=np.float64))
            
        gt_df = pd.DataFrame(gt_dict)
        
        return agg_df, gt_df
