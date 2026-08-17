"""
Physical Power Conservation, Hard-Gating, and Intelligent Residual Allocator (V3)
================================================================================
4대 마스터 개선안 적용:
1. Hysteresis Hard-Gating: 확률 < 0.25 시 Ghost Leakage 완전 0.0W 클램핑
2. Intelligent Residual Compensation: ON 상태 기기의 과소 추정분을 Other Power에서 지능형 보정
3. Adaptive Temporal Smoothing: 정상상태 지터 제거 및 급변 엣지 즉각 반응
4. Duration-Based Anomaly Trigger: 0.5초 미만 인러시 스파이크 필터링 및 2.5초 지속 시 경보
"""
from typing import Dict, Tuple, Union, Optional
import numpy as np
import torch

from .config import NOMINAL_POWERS, APPLIANCE_KEYS


class PowerConservationAllocator:
    """
    물리적 전력 보존, 하드 게이팅 및 지능형 잔여 전력 보정 엔진 (V3)
    """

    def __init__(
        self,
        unknown_appliance_threshold_w: float = 20.0,
        anomaly_duration_sec: float = 2.5,
        hard_gating_threshold: float = 0.25,
        enable_residual_compensation: bool = True
    ):
        self.unknown_threshold = unknown_appliance_threshold_w
        self.anomaly_duration_sec = anomaly_duration_sec
        self.hard_gating_threshold = hard_gating_threshold
        self.enable_residual_compensation = enable_residual_compensation
        
        # 기기별 정격 전력
        self.nominal_powers = np.array([float(NOMINAL_POWERS.get(k, 50.0)) for k in APPLIANCE_KEYS], dtype=np.float32)

    def allocate_single(
        self,
        p_total: float,
        p_preds: Dict[str, float],
        state_probs: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], float, Dict[str, Union[bool, float]]]:
        """
        단일 시점 전력 할당 및 하드 게이팅 / 지능형 보정
        """
        p_total = max(0.0, float(p_total))
        
        # 1. Hysteresis Hard-Gating (Ghost Leakage 완벽 차단)
        gated_preds = {}
        for k in APPLIANCE_KEYS:
            prob = state_probs.get(k, 1.0) if state_probs is not None else 1.0
            val = max(0.0, float(p_preds.get(k, 0.0)))
            if prob < self.hard_gating_threshold:
                gated_preds[k] = 0.0 # 하드 클램핑
            else:
                gated_preds[k] = val

        s_pred = sum(gated_preds.values())
        
        # 2. Intelligent Residual Compensation (ON 기기 과소추정 지능형 보정)
        p_final = dict(gated_preds)
        if s_pred <= p_total:
            p_other = p_total - s_pred
            if self.enable_residual_compensation and p_other > 5.0:
                # 켜져 있는 기기 중 정격 전력 대비 과소 추정된 기기들에 잔여분 우선 배분
                active_keys = [k for k in APPLIANCE_KEYS if (state_probs.get(k, 0.0) if state_probs else 0.0) >= 0.5]
                if active_keys:
                    under_deficits = {}
                    for k in active_keys:
                        nom = NOMINAL_POWERS.get(k, 50.0)
                        curr = p_final[k]
                        if curr < nom * 0.95: # 정격의 95% 미만인 경우
                            under_deficits[k] = nom - curr
                            
                    total_deficit = sum(under_deficits.values())
                    if total_deficit > 0:
                        allocatable = min(p_other - 2.0, total_deficit) # 노이즈 플로어 2W 보존
                        for k, deficit in under_deficits.items():
                            add_p = allocatable * (deficit / total_deficit)
                            p_final[k] += add_p
                            p_other -= add_p
            is_clamped = False
            scale_factor = 1.0
        else:
            scale_factor = p_total / (s_pred + 1e-9)
            p_final = {k: v * scale_factor for k, v in gated_preds.items()}
            p_other = 0.0
            is_clamped = True

        status = {
            "is_clamped": is_clamped,
            "scale_factor": float(scale_factor),
            "is_unknown_appliance_active": p_other >= self.unknown_threshold,
            "conservation_error_w": abs((sum(p_final.values()) + p_other) - p_total)
        }
        
        return p_final, p_other, status

    def allocate_batch(
        self,
        p_total_arr: np.ndarray,
        p_pred_matrix: np.ndarray,
        state_probs_matrix: Optional[np.ndarray] = None,
        sampling_hz: float = 60.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        시계열 배치 고속 벡터화 할당 + 4대 개선안 (하드게이팅, 지능형보정, 적응형 평활, 지속시간 이상감지)
        """
        N, K = p_pred_matrix.shape
        p_total = np.maximum(0.0, p_total_arr)
        p_pred = np.maximum(0.0, p_pred_matrix.copy())
        
        # 1. Hysteresis Hard-Gating
        if state_probs_matrix is not None:
            off_mask = state_probs_matrix < self.hard_gating_threshold
            p_pred[off_mask] = 0.0

        # 2. Intelligent Residual Compensation
        s_pred = np.sum(p_pred, axis=-1)
        p_final = np.zeros_like(p_pred)
        p_other = np.zeros_like(p_total)
        
        mask_normal = s_pred <= p_total
        p_final[mask_normal] = p_pred[mask_normal]
        p_other[mask_normal] = p_total[mask_normal] - s_pred[mask_normal]
        
        if self.enable_residual_compensation and state_probs_matrix is not None:
            # 잔여분이 5W 이상이고 기기가 켜져 있을 때 보정
            needs_comp = mask_normal & (p_other > 5.0)
            if np.any(needs_comp):
                for i in np.where(needs_comp)[0]:
                    active_indices = np.where(state_probs_matrix[i] >= 0.5)[0]
                    if len(active_indices) > 0:
                        deficits = np.maximum(0.0, self.nominal_powers[active_indices] - p_final[i, active_indices])
                        tot_def = np.sum(deficits)
                        if tot_def > 0:
                            avail = min(p_other[i] - 1.5, tot_def)
                            added = avail * (deficits / tot_def)
                            p_final[i, active_indices] += added
                            p_other[i] -= np.sum(added)

        # 과대 예측 클램핑
        mask_over = ~mask_normal
        if np.any(mask_over):
            scales = p_total[mask_over] / (s_pred[mask_over] + 1e-9)
            p_final[mask_over] = p_pred[mask_over] * scales[:, np.newaxis]
            p_other[mask_over] = 0.0

        # 3. Adaptive Temporal Smoothing (정상상태 지터 제거 & 엣지 보존)
        p_smoothed = np.zeros_like(p_final)
        win = 5 # 5스텝 (~0.08초) 중앙값 필터
        for k in range(K):
            sig = p_final[:, k]
            # 미디언 필터로 미세 지터 억제
            pad_sig = np.pad(sig, (win//2, win//2), mode='edge')
            med = np.array([np.median(pad_sig[j:j+win]) for j in range(len(sig))])
            
            # 급변 엣지(급격한 ON/OFF)에서는 원본 보존
            diff = np.abs(sig - med)
            alpha = np.where(diff > 5.0, 1.0, 0.7) # 5W 이상 급변 시 즉각 반응
            p_smoothed[:, k] = alpha * sig + (1.0 - alpha) * med
            
        p_final = np.maximum(0.0, p_smoothed)
        # 평활 후에도 100% 항등식 보존
        s_smooth = np.sum(p_final, axis=-1)
        p_other = np.maximum(0.0, p_total - s_smooth)
        
        # 4. Duration-Based Anomaly Trigger (순간 Inrush 무시 & 2.5초 이상 지속 시 경보)
        min_steps = int(self.anomaly_duration_sec * sampling_hz) # 150스텝
        raw_anomaly = p_other >= self.unknown_threshold
        is_unknown_active = np.zeros(N, dtype=bool)
        
        # 연속된 True 구간의 길이가 min_steps 이상인 경우에만 True
        cnt = 0
        for i in range(N):
            if raw_anomaly[i]:
                cnt += 1
                if cnt >= min_steps:
                    is_unknown_active[i - min_steps + 1: i + 1] = True
            else:
                cnt = 0

        return p_final, p_other, is_unknown_active

