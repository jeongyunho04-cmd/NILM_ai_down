"""
Physical Multi-Task Loss Functions for NILM AI
==============================================
- Binary Cross Entropy (ON/OFF State Classification)
- Smooth L1 (Power Disaggregation Regression)
- Physical Conservation Loss (|sum(P_pred) - P_total|^2)
- State-Power Consistency Constraint
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


from .config import NOMINAL_POWERS, APPLIANCE_KEYS


class PhysicalMultiTaskLoss(nn.Module):
    """
    물리 법칙(전력 보존 법칙) 및 상대 오차 정규화를 포함하는 NILM V2 멀티태스크 손실함수
    - 기기별 정격 전력으로 손실을 정규화하여 10W 소형 부하와 1260W 대형 부하를 공평하게 학습
    """

    def __init__(
        self,
        weight_cls: float = 1.0,
        weight_reg: float = 0.5,             # 정규화된 회귀 손실 가중치
        weight_conservation: float = 0.1,    # 상대 전력 보존 손실 가중치
        weight_state_consistency: float = 0.05
    ):
        super().__init__()
        self.weight_cls = weight_cls
        self.weight_reg = weight_reg
        self.weight_cons = weight_conservation
        self.weight_state_cons = weight_state_consistency
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)
        
        # 기기별 정규화 가중치 텐서 [sqrt(P_nominal)]
        nom_arr = [float(NOMINAL_POWERS.get(k, 50.0)) for k in APPLIANCE_KEYS]
        self.register_buffer("norm_weights", torch.sqrt(torch.tensor(nom_arr, dtype=torch.float32) + 5.0))

    def forward(
        self,
        state_logits: torch.Tensor,
        power_pred: torch.Tensor,
        y_state: torch.Tensor,
        y_power: torch.Tensor,
        total_p_true: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. 상태 다중분류 손실 (BCE)
        l_cls = self.bce_loss(state_logits, y_state)
        
        # 2. 정규화 상대 전력 회귀 손실 (Normalized Smooth L1)
        # norm_weights: (5,) -> (1, 5) 브로드캐스팅
        elem_reg = self.smooth_l1(power_pred, y_power) # (Batch, 5)
        norm_weights = self.norm_weights.to(power_pred.device)
        l_reg = torch.mean(elem_reg / (norm_weights + 1e-6))
        
        # 3. 상대 전력 보존 손실 (Relative Power Conservation Loss)
        if total_p_true is None:
            total_p_true = torch.sum(y_power, dim=-1) # (Batch,)
        pred_total_p = torch.sum(power_pred, dim=-1)  # (Batch,)
        
        # 큰 전력과 작은 전력에서 공평한 상대 오차 보존
        denom = torch.sqrt(total_p_true + 10.0)
        l_cons = torch.mean(((pred_total_p - total_p_true) / denom)**2)
        
        # 4. 상태-전력 일관성 페널티 (OFF인데 전력이 튀는 현상 억제)
        off_mask = (1.0 - y_state)
        l_state_cons = torch.mean((power_pred * off_mask / (norm_weights + 1e-6))**2)
        
        # 5. SMPS 상호 간섭 억제 손실 (Beam Projector vs Laptop Charger vs Mini PC 간섭 방지)
        # SMPS 인덱스: beam_projector(2), laptop_charger(3), minipc(4)
        smps_indices = [2, 3, 4]
        l_crosstalk = torch.tensor(0.0, device=power_pred.device)
        for i in smps_indices:
            for j in smps_indices:
                if i != j:
                    # i가 켜져있고 j가 꺼져있을 때 j의 예측 전력에 강력한 페널티
                    leakage_mask = y_state[:, i] * (1.0 - y_state[:, j])
                    l_crosstalk = l_crosstalk + torch.mean((power_pred[:, j] * leakage_mask / (norm_weights[j] + 1e-6))**2)
        
        # 종합 가중 손실
        loss_total = (
            self.weight_cls * l_cls +
            self.weight_reg * l_reg +
            self.weight_cons * l_cons +
            self.weight_state_cons * l_state_cons +
            0.15 * l_crosstalk
        )
        
        loss_dict = {
            "loss_total": float(loss_total.item()),
            "loss_cls": float(l_cls.item()),
            "loss_reg": float(l_reg.item()),
            "loss_conservation": float(l_cons.item()),
            "loss_state_consistency": float(l_state_cons.item()),
            "loss_crosstalk": float(l_crosstalk.item())
        }
        
        return loss_total, loss_dict
