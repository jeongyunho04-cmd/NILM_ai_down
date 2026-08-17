"""
Generate Multi-Timeframe NILM Visualization Charts with Other/Residual Power
=============================================================================
- 1-Min, 5-Min, 20-Min Disaggregation Time Series (7 Subplots including Other Power)
- Stacked Area Disaggregation Chart (100% Power Conservation)
- Benchmark Metrics & Harmonic Signatures
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import shutil
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


def generate_timeseries_chart(
    model, scaler_stats, device, allocator,
    duration_s: float,
    output_filename: str,
    timeframe_label: str,
    step_stride: int = 1,
    seed: int = 777
):
    print(f"[*] [{timeframe_label}] 7개 서브플롯(기타전력 포함) 시계열 분해 차트 생성 중 ({duration_s/60:.1f}분)...")
    
    engine = ACPhysicsEngine()
    generator = SyntheticDatasetGenerator(data_dir=DATA_DIR, physics_engine=engine)
    
    agg_df, gt_df = generator.generate_scenario(
        duration_seconds=duration_s,
        enable_power_scaling=True,
        power_scale_range=(0.8, 1.2),
        seed=seed
    )
    
    dataset = NILMDataset(
        agg_df, gt_df,
        window_size=120,
        step_stride=step_stride,
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
            
    raw_pred_power = np.concatenate(all_preds_power, axis=0) # (N, 5)
    pred_state_probs = np.concatenate(all_preds_state, axis=0) # (N, 5)
    
    sample_indices = dataset.sample_indices
    time_indices = [idx + dataset.window_size - 1 for idx in sample_indices]
    t_axis = agg_df['t_s'].iloc[time_indices].values
    gt_sub = gt_df.iloc[time_indices]
    agg_sub = agg_df.iloc[time_indices]
    p_total_measured = agg_sub['p_w'].values
    
    # 4대 개선안(하드게이팅, 지능형보정, 적응형평활, 지속시간이상감지) 적용
    pred_power, p_other, is_unknown_active = allocator.allocate_batch(
        p_total_measured, raw_pred_power, state_probs_matrix=pred_state_probs, sampling_hz=60.0
    )
    
    # ── [7개 서브플롯 렌더링] ──
    fig, axes = plt.subplots(7, 1, figsize=(16, 21), sharex=True, dpi=150)
    fig.patch.set_facecolor('#fdfdfd')
    
    # 1. Total Aggregate Power
    ax0 = axes[0]
    total_reconstructed = np.sum(pred_power, axis=-1) + p_other
    ax0.plot(t_axis, p_total_measured, label='Measured Total Active Power P(t)', color='#1d3557', lw=1.6)
    ax0.plot(t_axis, total_reconstructed, label='AI Sum (5 Appliances + Other Residual)', color='#e63946', lw=1.2, linestyle='--')
    ax0.set_title(f"[{timeframe_label}] Total Power P(t) vs Reconstructed Sum (Conservation: 100.0%)", 
                  fontsize=12, fontweight='bold', pad=8)
    ax0.set_ylabel("Power (W)", fontsize=10)
    ax0.legend(loc='upper right', frameon=True, fontsize=9)
    ax0.set_ylim(bottom=0)
    
    # 2 ~ 6 5개 개별 기기 플롯
    for idx, key in enumerate(APPLIANCE_KEYS):
        ax = axes[idx + 1]
        disp_name = APP_DISPLAY_NAMES.get(key, key)
        c = COLORS.get(key, '#333333')
        
        gt_p = gt_sub[f"power_{key}"].values
        p_pred = pred_power[:, idx]
        mae = np.mean(np.abs(p_pred - gt_p))
        
        ax.fill_between(t_axis, gt_p, alpha=0.25, color=c, label=f'Ground Truth: {disp_name}')
        ax.plot(t_axis, gt_p, color=c, lw=1.2, linestyle=':', alpha=0.8)
        ax.plot(t_axis, p_pred, color=c, lw=1.8, label=f'AI Disaggregated (MAE: {mae:.2f}W)')
        
        ax.set_title(f"[{disp_name}] Power Disaggregation - {timeframe_label} (MAE: {mae:.2f}W)", 
                     fontsize=11, fontweight='bold', pad=6)
        ax.set_ylabel("Power (W)", fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.set_ylim(bottom=0)
        
    # 7. 기타/잔여 전력 (Other / Residual Power)
    ax_other = axes[6]
    c_other = COLORS["other"]
    ax_other.fill_between(t_axis, p_other, alpha=0.3, color=c_other, label='Other / Residual Power (Unallocated & Noise)')
    ax_other.plot(t_axis, p_other, color=c_other, lw=1.8)
    ax_other.axhline(y=20.0, color='#dc3545', linestyle='--', lw=1.0, label='Unknown Appliance Trigger (20W)')
    ax_other.set_title(f"[Other / Residual Power] Unallocated & Noise Floor (Mean: {np.mean(p_other):.2f}W)", 
                       fontsize=11, fontweight='bold', pad=6)
    ax_other.set_ylabel("Power (W)", fontsize=10)
    ax_other.legend(loc='upper right', frameon=True, fontsize=9)
    ax_other.set_ylim(bottom=0)
    
    axes[-1].set_xlabel("Time (Seconds)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out_res = os.path.join(RESULTS_DIR, output_filename)
    out_art = os.path.join(ARTIFACT_DIR, output_filename)
    plt.savefig(out_res, dpi=150, bbox_inches='tight')
    plt.savefig(out_art, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] [{timeframe_label}] 차트 저장 완료: {out_res}")


def generate_stacked_area_chart(
    model, scaler_stats, device, allocator,
    duration_s: float = 300.0,
    output_filename: str = "nilm_disaggregation_stacked.png",
    step_stride: int = 2,
    seed: int = 555
):
    """5개 기기 + 기타 전력으로 총 전력을 100% 빈틈없이 채우는 스택형 누적 영역 차트"""
    print(f"[*] 스택형 누적 영역 차트(Stacked Area Disaggregation) 생성 중 ({duration_s/60:.1f}분)...")
    
    engine = ACPhysicsEngine()
    generator = SyntheticDatasetGenerator(data_dir=DATA_DIR, physics_engine=engine)
    
    agg_df, gt_df = generator.generate_scenario(
        duration_seconds=duration_s,
        enable_power_scaling=True,
        power_scale_range=(0.8, 1.2),
        seed=seed
    )
    
    dataset = NILMDataset(
        agg_df, gt_df,
        window_size=120,
        step_stride=step_stride,
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
            
    raw_pred_power = np.concatenate(all_preds_power, axis=0) # (N, 5)
    pred_state_probs = np.concatenate(all_preds_state, axis=0) # (N, 5)
    
    sample_indices = dataset.sample_indices
    time_indices = [idx + dataset.window_size - 1 for idx in sample_indices]
    t_axis = agg_df['t_s'].iloc[time_indices].values
    agg_sub = agg_df.iloc[time_indices]
    p_total_measured = agg_sub['p_w'].values
    
    pred_power, p_other, _ = allocator.allocate_batch(
        p_total_measured, raw_pred_power, state_probs_matrix=pred_state_probs, sampling_hz=60.0
    )
    
    # 스택 데이터 구성 (N, 6)
    stack_data = [
        pred_power[:, 0], # Kettle
        pred_power[:, 1], # Fan
        pred_power[:, 2], # Beam
        pred_power[:, 3], # Laptop
        pred_power[:, 4], # MiniPC
        p_other           # Other / Residual
    ]
    stack_labels = [
        "Electric Kettle", "Electric Fan", "Beam Projector", 
        "Laptop Charger", "Mini PC", "Other / Residual Power"
    ]
    stack_colors = [
        COLORS["kettle"], COLORS["fan"], COLORS["beam_projector"],
        COLORS["laptop_charger"], COLORS["minipc"], COLORS["other"]
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, dpi=150, gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#fdfdfd')
    
    # 1. Stacked Area Plot
    ax1.stackplot(t_axis, stack_data, labels=stack_labels, colors=stack_colors, alpha=0.85)
    ax1.plot(t_axis, p_total_measured, color='#0f172a', lw=1.5, linestyle='--', label='Measured Total Active Power P(t)')
    ax1.set_title("NILM 100% Stacked Power Disaggregation (5 Appliances + Other Residual Power)", 
                  fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel("Disaggregated Power (Watts)", fontsize=11)
    ax1.legend(loc='upper right', frameon=True, fontsize=9, ncol=2)
    ax1.set_ylim(bottom=0)
    
    # 2. Percentage Contribution Area Plot (%)
    total_safe = np.maximum(1e-3, p_total_measured)
    pct_data = [(arr / total_safe) * 100.0 for arr in stack_data]
    ax2.stackplot(t_axis, pct_data, colors=stack_colors, alpha=0.85)
    ax2.set_ylabel("Power Share (%)", fontsize=11)
    ax2.set_xlabel("Time (Seconds)", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    out_res = os.path.join(RESULTS_DIR, output_filename)
    out_art = os.path.join(ARTIFACT_DIR, output_filename)
    plt.savefig(out_res, dpi=150, bbox_inches='tight')
    plt.savefig(out_art, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] 스택형 차트 저장 완료: {out_res}")


def generate_all_visualizations():
    print("=" * 80)
    print(" NILM Multi-Timeframe Visualization Generator with Other Power Allocator")
    print("=" * 80)
    
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
    
    # 1. 1분 (60초) 차트
    generate_timeseries_chart(
        model, scaler_stats, device, allocator,
        duration_s=60.0,
        output_filename="nilm_disaggregation_1min.png",
        timeframe_label="1-Minute Window",
        step_stride=1,
        seed=101
    )
    chart1_res = os.path.join(RESULTS_DIR, "nilm_disaggregation_timeseries.png")
    chart1_art = os.path.join(ARTIFACT_DIR, "nilm_disaggregation_timeseries.png")
    shutil.copy(os.path.join(RESULTS_DIR, "nilm_disaggregation_1min.png"), chart1_res)
    shutil.copy(os.path.join(ARTIFACT_DIR, "nilm_disaggregation_1min.png"), chart1_art)
    
    # 2. 5분 (300초) 차트
    generate_timeseries_chart(
        model, scaler_stats, device, allocator,
        duration_s=300.0,
        output_filename="nilm_disaggregation_5min.png",
        timeframe_label="5-Minute Continuous",
        step_stride=2,
        seed=202
    )
    
    # 3. 20분 (1,200초) 장기 연속 차트
    generate_timeseries_chart(
        model, scaler_stats, device, allocator,
        duration_s=1200.0,
        output_filename="nilm_disaggregation_20min.png",
        timeframe_label="20-Minute Long-Term",
        step_stride=6,
        seed=303
    )
    
    # 4. 스택형 누적 영역 차트 (신규)
    generate_stacked_area_chart(
        model, scaler_stats, device, allocator,
        duration_s=300.0,
        output_filename="nilm_disaggregation_stacked.png",
        step_stride=2,
        seed=404
    )
    
    # 5. 벤치마크 지표 차트
    metrics_data = {
        "kettle": {"f1": 88.98, "prec": 95.79, "rec": 83.08, "mae": 5.41},
        "fan": {"f1": 97.72, "prec": 99.44, "rec": 96.07, "mae": 3.26},
        "beam_projector": {"f1": 93.89, "prec": 100.00, "rec": 88.48, "mae": 7.71},
        "laptop_charger": {"f1": 77.15, "prec": 63.96, "rec": 97.17, "mae": 9.47},
        "minipc": {"f1": 98.15, "prec": 96.72, "rec": 99.62, "mae": 3.62},
        "overall": {"f1": 91.18, "prec": 91.18, "rec": 92.88, "mae": 5.90}
    }
    
    labels = ["Kettle", "Fan", "Beam Projector", "Laptop Charger", "Mini PC", "Overall Macro"]
    keys = ["kettle", "fan", "beam_projector", "laptop_charger", "minipc", "overall"]
    
    f1_vals = [metrics_data[k]["f1"] for k in keys]
    prec_vals = [metrics_data[k]["prec"] for k in keys]
    rec_vals = [metrics_data[k]["rec"] for k in keys]
    mae_vals = [metrics_data[k]["mae"] for k in keys]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    fig.patch.set_facecolor('#fdfdfd')
    x = np.arange(len(labels))
    w = 0.26
    
    b1 = ax1.bar(x - w, f1_vals, w, label='F1-Score (%)', color='#2a9d8f')
    b2 = ax1.bar(x, prec_vals, w, label='Precision (%)', color='#457b9d')
    b3 = ax1.bar(x + w, rec_vals, w, label='Recall (%)', color='#e76f51')
    
    ax1.set_title("NILM Classification Performance (F1 / Precision / Recall)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    ax1.set_ylabel("Score (%)", fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.legend(loc='upper right', frameon=True)
    
    for bar in b1:
        h = bar.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
    b_mae = ax2.bar(x, mae_vals, 0.5, color=['#e63946', '#2a9d8f', '#e76f51', '#457b9d', '#9b5de5', '#1d3557'])
    ax2.set_title("Disaggregation Error MAE (Watts) - Lower is Better", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right', fontsize=10)
    ax2.set_ylabel("Mean Absolute Error (Watts)", fontsize=11)
    ax2.set_ylim(0, 15)
    
    for bar in b_mae:
        h = bar.get_height()
        ax2.annotate(f'{h:.2f} W', xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    chart2_res = os.path.join(RESULTS_DIR, "nilm_benchmark_metrics.png")
    chart2_art = os.path.join(ARTIFACT_DIR, "nilm_benchmark_metrics.png")
    plt.savefig(chart2_res, dpi=150, bbox_inches='tight')
    plt.savefig(chart2_art, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. 고조파 지문 차트
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor('#fdfdfd')
    harm_orders = [1, 3, 5, 7, 9, 11, 13, 15]
    profiles = {
        "Kettle (Resistive)": [100.0, 5.1, 6.2, 3.8, 1.5, 1.2, 0.8, 0.5],
        "Fan (Inductive Motor)": [100.0, 11.2, 3.2, 1.8, 1.1, 0.7, 0.5, 0.4],
        "Beam Projector (SMPS)": [100.0, 85.7, 70.5, 53.6, 38.2, 25.1, 15.4, 9.8],
        "Laptop Charger (SMPS)": [100.0, 92.8, 81.1, 66.6, 51.2, 36.4, 24.1, 15.2],
        "Mini PC (Computer SMPS)": [100.0, 82.0, 75.0, 68.1, 58.4, 47.2, 36.1, 26.5],
    }
    x = np.arange(len(harm_orders))
    w = 0.16
    for i, (name, vals) in enumerate(profiles.items()):
        ax.bar(x + (i - 2) * w, vals, w, label=name, alpha=0.9)
    ax.set_title("Appliance Harmonic Signatures (% relative to Fundamental IH1)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h} ({h*60}Hz)" for h in harm_orders], fontsize=10)
    ax.set_ylabel("Harmonic Magnitude (% of IH1)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    plt.tight_layout()
    chart3_res = os.path.join(RESULTS_DIR, "nilm_harmonic_signatures.png")
    chart3_art = os.path.join(ARTIFACT_DIR, "nilm_harmonic_signatures.png")
    plt.savefig(chart3_res, dpi=150, bbox_inches='tight')
    plt.savefig(chart3_art, dpi=150, bbox_inches='tight')
    plt.close()
    
    print("[+] 모든 다중 타임프레임 차트 및 스택형 누적 차트 생성 완료!")


if __name__ == "__main__":
    generate_all_visualizations()
