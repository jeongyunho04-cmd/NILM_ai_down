"""
Test NILM AI Disaggregation with Initial-ON Appliances at t=0s
==============================================================
시작 시점(t=0초)부터 3개 전자기기(선풍기 + 빔프로젝터 + 미니PC)가 이미 켜져 있는 상태에서
AI 모델이 시작부터 각 기기별 소비전력을 정확히 분해해내는지 시험하고 그래프를 생성합니다.
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


def run_initial_on_test():
    print("=" * 80)
    print(" [시험] 시작 시점(t=0초) 기기 가동 상태(Initial-ON) 부하 분해 시험")
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
    
    # 2. 시작부터 3개 기기(선풍기, 빔프로젝터, 미니PC)가 켜져 있는 100초 시나리오 생성
    engine = ACPhysicsEngine()
    generator = SyntheticDatasetGenerator(data_dir=DATA_DIR, physics_engine=engine)
    
    initial_on_list = ["fan", "beam_projector", "minipc"]
    print(f"[*] 시작 시점(t=0s) 초기 가동 기기: {initial_on_list}")
    
    agg_df, gt_df = generator.generate_scenario(
        duration_seconds=100.0,
        initial_on_appliances=initial_on_list,
        enable_power_scaling=True,
        power_scale_range=(0.85, 1.15),
        seed=888
    )
    
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
    
    # 3. 지표 산출
    print("\n" + "=" * 80)
    print(" [Initial-ON 시험 결과: 기기별 분해 오차 및 정확도]")
    print("=" * 80)
    for idx, key in enumerate(APPLIANCE_KEYS):
        disp_name = APP_DISPLAY_NAMES.get(key, key)
        gt_p = gt_sub[f"power_{key}"].values
        p_pred = pred_power[:, idx]
        mae = np.mean(np.abs(p_pred - gt_p))
        init_on_flag = "(★ 시작부터 ON)" if key in initial_on_list else "(시작 시 OFF)"
        print(f"- {disp_name:<28} {init_on_flag:<15} | MAE: {mae:6.2f} W | Mean Pred: {np.mean(p_pred):6.2f}W (GT: {np.mean(gt_p):6.2f}W)")
    
    overall_mae = np.mean(np.abs(pred_power - gt_sub[[f"power_{k}" for k in APPLIANCE_KEYS]].values))
    print(f"\n[+] 전체 평균 전력 분해 오차 (MAE): {overall_mae:.2f} W")
    
    # 4. 차트 렌더링
    fig, axes = plt.subplots(7, 1, figsize=(16, 21), sharex=True, dpi=150)
    fig.patch.set_facecolor('#fdfdfd')
    
    # (1) Total Aggregate Power
    ax0 = axes[0]
    total_reconstructed = np.sum(pred_power, axis=-1) + p_other
    ax0.plot(t_axis, p_total_measured, label='Measured Total Active Power P(t)', color='#1d3557', lw=1.6)
    ax0.plot(t_axis, total_reconstructed, label='AI Sum (5 Appliances + Other Residual)', color='#e63946', lw=1.2, linestyle='--')
    ax0.set_title("[Test: Initial-ON at t=0s] Total Power P(t) vs Reconstructed Sum", 
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
        init_tag = " [Initial-ON at t=0s]" if key in initial_on_list else ""
        
        ax.fill_between(t_axis, gt_p, alpha=0.25, color=c, label=f'Ground Truth: {disp_name}{init_tag}')
        ax.plot(t_axis, gt_p, color=c, lw=1.2, linestyle=':', alpha=0.8)
        ax.plot(t_axis, p_pred, color=c, lw=1.8, label=f'AI Disaggregated (MAE: {mae:.2f}W)')
        
        ax.set_title(f"[{disp_name}]{init_tag} Disaggregation Result (MAE: {mae:.2f}W)", 
                     fontsize=11, fontweight='bold', pad=6)
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.set_ylim(bottom=0)
        
    # (7) 기타/잔여 전력
    ax_other = axes[6]
    c_other = COLORS["other"]
    ax_other.fill_between(t_axis, p_other, alpha=0.3, color=c_other, label='Other / Residual Power')
    ax_other.plot(t_axis, p_other, color=c_other, lw=1.8)
    ax_other.axhline(y=20.0, color='#dc3545', linestyle='--', lw=1.0, label='Unknown Appliance Trigger (20W)')
    ax_other.set_title(f"[Other / Residual Power] Unallocated Residual (Mean: {np.mean(p_other):.2f}W)", 
                       fontsize=11, fontweight='bold', pad=6)
    ax_other.set_ylabel("Power (W)", fontsize=10)
    ax_other.legend(loc='upper right', frameon=True, fontsize=9)
    ax_other.set_ylim(bottom=0)
    
    axes[-1].set_xlabel("Time (Seconds)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out_res = os.path.join(RESULTS_DIR, "nilm_disaggregation_initial_on.png")
    out_art = os.path.join(ARTIFACT_DIR, "nilm_disaggregation_initial_on.png")
    plt.savefig(out_res, dpi=150, bbox_inches='tight')
    plt.savefig(out_art, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Initial-ON 차트 저장 완료: {out_res}")


if __name__ == "__main__":
    run_initial_on_test()
