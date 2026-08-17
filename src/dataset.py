"""
PyTorch Compatible Dataset and Sliding Window Builder for NILM AI
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object # Fallback

from .config import (
    APPLIANCE_KEYS,
    NUM_APPLIANCES,
    SAMPLING_HZ
)
from .feature_engineering import extract_features, get_feature_column_names


class NILMDataset(Dataset):
    """
    NILM AI 모델 학습 및 평가를 위한 슬라이딩 윈도우 시계열 데이터셋.
    
    입력:
        - X: (Batch, Window_Size, Feature_Dim) 합성된 종합 전력/고조파 피처 텐서
    출력 (Multi-task Targets):
        - y_state: (Batch, Num_Appliances) 5개 기기의 ON/OFF 상태 (0 or 1)
        - y_power: (Batch, Num_Appliances) 5개 기기의 분해 소비전력 (W)
    """

    def __init__(
        self,
        aggregate_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame,
        window_size: int = 60, # 60 steps = 1초 at 60Hz
        step_stride: int = 10, # 슬라이딩 보폭 (스텝)
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
        scaler_stats: Optional[Dict[str, Tuple[float, float]]] = None,
        target_mode: str = "last" # "last" (마지막 스텝 기준) or "seq" (전체 윈도우 시퀀스)
    ):
        self.window_size = window_size
        self.step_stride = step_stride
        self.target_mode = target_mode
        self.normalize = normalize
        
        # 피처 엔지니어링 수행
        feat_df = extract_features(aggregate_df)
        
        if feature_cols is None:
            self.feature_cols = [c for c in get_feature_column_names() if c in feat_df.columns]
        else:
            self.feature_cols = [c for c in feature_cols if c in feat_df.columns]
            
        # 입력 데이터 추출
        X_raw = feat_df[self.feature_cols].values.astype(np.float32)
        
        # 정규화 처리 (Z-Score)
        self.scaler_stats = {} if scaler_stats is None else scaler_stats
        if normalize:
            if scaler_stats is None:
                means = np.nanmean(X_raw, axis=0)
                stds = np.nanstd(X_raw, axis=0) + 1e-6
                for i, c in enumerate(self.feature_cols):
                    self.scaler_stats[c] = (float(means[i]), float(stds[i]))
            else:
                means = np.array([self.scaler_stats[c][0] for c in self.feature_cols], dtype=np.float32)
                stds = np.array([self.scaler_stats[c][1] for c in self.feature_cols], dtype=np.float32)
                
            X_norm = (X_raw - means) / stds
            # 결측치 0 대체
            self.X = np.nan_to_num(X_norm, nan=0.0)
        else:
            self.X = np.nan_to_num(X_raw, nan=0.0)
            
        # Ground Truth 타겟 추출
        state_cols = [f"state_{k}" for k in APPLIANCE_KEYS]
        power_cols = [f"power_{k}" for k in APPLIANCE_KEYS]
        
        self.Y_state = ground_truth_df[state_cols].values.astype(np.float32)
        self.Y_power = ground_truth_df[power_cols].values.astype(np.float32)
        
        # 윈도우 인덱스 생성
        N = len(self.X)
        self.sample_indices = []
        for start_idx in range(0, N - window_size + 1, step_stride):
            self.sample_indices.append(start_idx)
            
    def __len__(self) -> int:
        return len(self.sample_indices)
        
    def __getitem__(self, idx: int):
        start = self.sample_indices[idx]
        end = start + self.window_size
        
        x_window = self.X[start:end] # (Window_Size, Feature_Dim)
        
        if self.target_mode == "last":
            # 윈도우의 마지막 시점의 상태 및 전력
            y_state = self.Y_state[end - 1] # (Num_Appliances,)
            y_power = self.Y_power[end - 1] # (Num_Appliances,)
        else:
            # 윈도우 전체 시퀀스
            y_state = self.Y_state[start:end] # (Window_Size, Num_Appliances)
            y_power = self.Y_power[start:end] # (Window_Size, Num_Appliances)
            
        if TORCH_AVAILABLE:
            return (
                torch.tensor(x_window, dtype=torch.float32),
                torch.tensor(y_state, dtype=torch.float32),
                torch.tensor(y_power, dtype=torch.float32)
            )
        return x_window, y_state, y_power

    @property
    def num_features(self) -> int:
        return len(self.feature_cols)
        
    @property
    def num_appliances(self) -> int:
        return NUM_APPLIANCES
