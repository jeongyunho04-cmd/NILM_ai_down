"""
NILM Real-Time and Batch Inference Engine
"""
import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from .config import DEFAULT_FEATURE_DIM, NUM_APPLIANCES, APPLIANCE_KEYS, APPLIANCES
from .feature_engineering import extract_features
from .models.nilm_net import MultiTaskConvBiGRUNet
from .power_allocator import PowerConservationAllocator


class NILMInferenceEngine:
    """
    NILM AI 실시간 추론 및 전력 보존 분해 엔진
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_nilm_model.pt",
        scaler_path: str = "checkpoints/scaler_stats.json",
        device: Optional[str] = None
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scaler_stats = {}
        self.feature_names = []
        self.model = None
        self.allocator = PowerConservationAllocator(unknown_appliance_threshold_w=20.0)

        # 1. 스케일러 로드
        if os.path.exists(scaler_path):
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.scaler_stats = json.load(f)
            self.feature_names = list(self.scaler_stats.keys())
            in_features = len(self.feature_names)
        else:
            in_features = DEFAULT_FEATURE_DIM

        # 2. 신경망 모델 초기화 및 가중치 로드
        self.model = MultiTaskConvBiGRUNet(in_features=in_features).to(self.device)
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)
            print(f"[InferenceEngine] 가중치 로드 완료: {checkpoint_path} (Device: {self.device})")
        else:
            print(f"[InferenceEngine] [경고] 가중치 파일을 찾을 수 없습니다: {checkpoint_path}")
        self.model.eval()

    def predict_window(
        self,
        window_df: pd.DataFrame,
        apply_power_conservation: bool = True
    ) -> Dict[str, Any]:
        """
        1개 슬라이딩 윈도우(예: 120스텝 = 2초) 데이터로부터 실시간 분해 수행
        
        Returns:
            dict with:
                - timestamp: float
                - states: Dict[str, bool]
                - state_probs: Dict[str, float]
                - powers: Dict[str, float] (5개 기기 분해 전력)
                - power_other: float (기타/잔여 전력)
                - power_total_measured: float (수전단 측정 총 전력)
                - is_unknown_active: bool
        """
        if self.model is None or len(self.feature_names) == 0:
            raise RuntimeError("InferenceEngine이 정상적으로 초기화되지 않았습니다.")

        # 1. 피처 추출
        feat_df = extract_features(window_df)
        
        # 2. 정규화 (Z-Score)
        feat_matrix = np.zeros((len(feat_df), len(self.feature_names)), dtype=np.float32)
        for col_idx, col in enumerate(self.feature_names):
            if col in feat_df.columns:
                stat = self.scaler_stats[col]
                mean = stat["mean"] if isinstance(stat, dict) else stat[0]
                std = stat["std"] if isinstance(stat, dict) else stat[1]
                feat_matrix[:, col_idx] = (feat_df[col].values - mean) / (std + 1e-8)
                
        # 3. 모델 순전파 (Tensor 변환)
        x_tensor = torch.tensor(feat_matrix, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, Time, Feat)
        
        with torch.no_grad():
            state_logits, power_pred = self.model(x_tensor)
            state_probs = torch.sigmoid(state_logits).squeeze(0).cpu().numpy()
            raw_powers = power_pred.squeeze(0).cpu().numpy()

        # 4. 결과 매핑
        states = {}
        probs = {}
        pred_dict = {}
        for idx, key in enumerate(APPLIANCE_KEYS):
            p = float(state_probs[idx])
            probs[key] = p
            states[key] = bool(p >= 0.5)
            pred_dict[key] = float(max(0.0, raw_powers[idx]))

        # 5. 물리적 전력 보존 및 기타 전력(Other Power) 할당
        p_total_measured = float(window_df['p_w'].iloc[-1]) if 'p_w' in window_df.columns else sum(pred_dict.values())
        
        if apply_power_conservation:
            final_powers, p_other, alloc_status = self.allocator.allocate_single(p_total_measured, pred_dict)
        else:
            final_powers = pred_dict
            p_other = max(0.0, p_total_measured - sum(final_powers.values()))
            alloc_status = {"is_unknown_appliance_active": p_other >= 20.0}

        return {
            "timestamp": float(window_df['t_s'].iloc[-1]) if 't_s' in window_df.columns else 0.0,
            "states": states,
            "state_probs": probs,
            "powers": final_powers,
            "power_other": p_other,
            "power_total_measured": p_total_measured,
            "is_unknown_appliance_active": alloc_status["is_unknown_appliance_active"]
        }
