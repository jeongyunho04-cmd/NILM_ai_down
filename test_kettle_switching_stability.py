"""
Stress Test: Kettle Rapid Switching & Small Load Immunity Test
==============================================================
1260W 대용량 전기주전자가 켜지고 꺼질 때(ON/OFF 반복 스위칭)
1) 주전자 자체의 예측이 흔들림 없이 정확한 사각 펄스로 추종하는지
2) 동시에 켜져 있는 소형 부하(선풍기, 빔프로젝터, 미니PC)가 전압강하/간섭에 흔들리지 않는지
정밀 검증하고 고해상도 시계열 차트를 생성합니다.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import DATA_DIR, SAMPLING_HZ, APPLIANCE_KEYS, APPLIANCES
from src.ac_physics_engine import ACPhysicsEngine
from src.synthetic_generator import SyntheticDatasetGenerator
from src.dataset import NILMDataset
from src.models.nilm_net import MultiTaskConvBiGRUNet
from src.power_allocator import PowerConservationAllocator

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "results", "charts")
ARTIFACT_DIR = r"C:\Users\yunho\.gemini\antigravity\brain\f5ae8871-964c-4bf1-9015-d565eaa1e471"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

APP_DISPLAY_NAMES = {
    "kettle": "Electric Kettle (~1260W)",
    "fan": "Electric Fan (~32W)",
    "beam_projector": "Beam Projector (~48W)",
    "laptop_charger": "Laptop Charger (~45W)",
    "minipc": "Mini PC (~15W)"
}

COLORS = {
    "kettle": "#e63946",
    "fan": "#2a9d8f",
    "beam_projector": "#e76f51",
    "laptop_charger": "#457b9d",
    "minipc": "#9b5de5",
    "other": "#6c757d"
}


def run_kettle_switching_test():
    print("=" * 80)
    print(" [스트레스 시험] 1260W 전기주전자 ON/OFF 반복 스위칭 및 소형 부하 안정성 시험")
    print("=" * 80)
    
    # 1. 모델 및 스케일러 로드
    scaler_path = "checkpoints/scaler_stats.json"
    ckpt_path = "checkpoints/best_nilm_model.pt"
    
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskConvBiGRUNet(in_features=len(scaler_stats)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    
    allocator = PowerConservationAllocator(unknown_appliance_threshold_w=20.0)
    
    # 2. 주전자 스위칭 전용 시나리오 생성 (120초)
    # 배경 부하(선풍기, 빔프로젝터, 미니PC)가 켜진 상태에서 주전자가 켜지고 꺼지는 시나리오
    engine = ACPhysicsEngine(r_line=0.25, l_line=0.15e-3)
    generator = SyntheticDatasetGenerator(data_dir=DATA_DIR, physics_engine=engine)
    
    initial_on_list = ["fan", "beam_projector", "minipc"]
    print(f"[*] 시작 시점 가동 기기: {initial_on_list}")
    
    agg_df, gt_df = generator.generate_scenario(
        duration_seconds=120.0,
        initial_on_appliances=initial_on_list,
        enable_power_scaling=True,
        power_scale_range=(0.9, 1.1),
        seed=1234
    )
    
    # 3. AI 모델 실시간 분해 추론
    dataset = NILMDataset(
        agg_df, gt_df,
        window_size=120,
        step_stride=1,
        normalize=True,
        scaler_stats=scaler_stats
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    all_preds_power = []
    all_preds_state = []
    with torch.no_grad():
        for batch in loader:
            x, _, _ = batch
            x = x.to(device)
            st_logits, pw_pred = model(x)
            all_preds_state.append(torch.sigmoid(st_logits).cpu().numpy())
            all_preds_power.append(pw_pred.cpu().numpy())
            
    raw_pred_power = np.concatenate(all_preds_power, axis=0)
    pred_state_probs = np.concatenate(all_preds_state, axis=0)
    
    sample_indices = dataset.sample_indices
    time_indices = [idx + dataset.window_size - 1 for idx in sample_indices]
    t_axis = agg_df['t_s'].iloc[time_indices].values
    gt_sub = gt_df.iloc[time_indices]
    agg_sub = agg_df.iloc[time_indices]
    p_total_measured = agg_sub['p_w'].values
    
    pred_power, p_other, _ = allocator.allocate_batch(
        p_total_measured, raw_pred_power, state_probs_matrix=pred_state_probs, sampling_hz=60.0
    )
    
    # 4. 주전자 ON 구간 vs OFF 구간에서 소형 부하들의 변동 분석
    kettle_gt = gt_sub["power_kettle"].values
    kettle_on_mask = kettle_gt > 500.0
    kettle_off_mask = ~kettle_on_mask
    
    print("\n" + "=" * 80)
    print(" [주전자 스위칭 시 소형 부하 안정성 분석]")
    print("=" * 80)
    for idx, k in enumerate(APPLIANCE_KEYS):
        disp = APP_DISPLAY_NAMES[k]
        p_pred = pred_power[:, idx]
        p_true = gt_sub[f"power_{k}"].values
        mae = np.mean(np.abs(p_pred - p_true))
        
        mean_on = np.mean(p_pred[kettle_on_mask]) if np.any(kettle_on_mask) else 0.0
        mean_off = np.mean(p_pred[kettle_off_mask]) if np.any(kettle_off_mask) else 0.0
        
        print(f"- {disp:<28} | 전체 MAE: {mae:5.2f}W | 주전자 ON 시 평균: {mean_on:5.2f}W | 주전자 OFF 시 평균: {mean_off:5.2f}W")
        
    # 5. 차트 렌더링
    fig, axes = plt.subplots(7, 1, figsize=(16, 21), sharex=True, dpi=150)
    fig.patch.set_facecolor('#fdfdfd')
    
    # (1) Total Aggregate Power
    ax0 = axes[0]
    total_reconstructed = np.sum(pred_power, axis=-1) + p_other
    ax0.plot(t_axis, p_total_measured, label='Measured Total Active Power P(t)', color='#1d3557', lw=1.6)
    ax0.plot(t_axis, total_reconstructed, label='AI Disaggregated Sum (100% Conserved)', color='#e63946', lw=1.2, linestyle='--')
    ax0.axvspan(20, 50, color='#e63946', alpha=0.08, label='Kettle 1st ON Period')
    ax0.axvspan(75, 105, color='#e63946', alpha=0.08, label='Kettle 2nd ON Period')
    ax0.set_title("[Stress Test: Kettle 1260W Switching] Total Power P(t) vs Reconstructed Sum", 
                  fontsize=12, fontweight='bold', pad=8)
    ax0.set_ylabel("Power (W)", fontsize=10)
    ax0.legend(loc='upper right', frameon=True, fontsize=9)
    ax0.set_ylim(bottom=0)
    
    # (2 ~ 6) 5개 개별 기기 플롯
    for idx, key in enumerate(APPLIANCE_KEYS):
        ax = axes[idx + 1]
        disp_name = APP_DISPLAY_NAMES.get(key, key)
        c = COLORS.get(key, '#333333')
        
        gt_p = gt_sub[f"power_{key}"].values
        p_pred = pred_power[:, idx]
        mae = np.mean(np.abs(p_pred - gt_p))
        
        # 주전자 가동 구간 음영 표시
        ax.axvspan(20, 50, color='#e63946', alpha=0.05)
        ax.axvspan(75, 105, color='#e63946', alpha=0.05)
        
        ax.fill_between(t_axis, gt_p, alpha=0.25, color=c, label=f'Ground Truth: {disp_name}')
        ax.plot(t_axis, gt_p, color=c, lw=1.2, linestyle=':', alpha=0.8)
        ax.plot(t_axis, p_pred, color=c, lw=1.8, label=f'AI Disaggregated (MAE: {mae:.2f}W)')
        
        ax.set_title(f"[{disp_name}] Disaggregation Stability (MAE: {mae:.2f}W)", 
                     fontsize=11, fontweight='bold', pad=6)
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.set_ylim(bottom=0)
        
    # (7) 기타/잔여 전력
    ax_other = axes[6]
    c_other = COLORS["other"]
    ax_other.fill_between(t_axis, p_other, alpha=0.3, color=c_other, label='Other / Residual Power')
    ax_other.plot(t_axis, p_other, color=c_other, lw=1.8)
    ax_other.axhline(y=20.0, color='#dc3545', linestyle='--', lw=1.0, label='Unknown Trigger (20W)')
    ax_other.set_title(f"[Other / Residual Power] Unallocated Residual (Mean: {np.mean(p_other):.2f}W)", 
                       fontsize=11, fontweight='bold', pad=6)
    ax_other.set_ylabel("Power (W)", fontsize=10)
    ax_other.legend(loc='upper right', frameon=True, fontsize=9)
    ax_other.set_ylim(bottom=0)
    
    axes[-1].set_xlabel("Time (Seconds)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out_res = os.path.join(RESULTS_DIR, "nilm_kettle_switching_stability.png")
    out_art = os.path.join(ARTIFACT_DIR, "nilm_kettle_switching_stability.png")
    plt.savefig(out_res, dpi=150, bbox_inches='tight')
    plt.savefig(out_art, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] 주전자 스위칭 안정성 차트 저장 완료: {out_res}")


if __name__ == "__main__":
    run_kettle_switching_test()
