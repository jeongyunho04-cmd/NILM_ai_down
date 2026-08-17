"""
NILM AI Preprocessing and Synthetic Physics Superposition Pipeline Runner
"""
import os
import argparse
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    PROCESSED_DIR,
    APPLIANCES,
    APPLIANCE_KEYS,
    SAMPLING_HZ
)
from src.ac_physics_engine import ACPhysicsEngine
from src.synthetic_generator import SyntheticDatasetGenerator
from src.feature_engineering import extract_features, get_feature_column_names
from src.dataset import NILMDataset


def main():
    parser = argparse.ArgumentParser(description="NILM AI Preprocessor & ACPhysics Superposition Engine")
    parser.add_argument("--duration", type=float, default=60.0, help="시나리오 생성 길이 (초, 기본값 60초)")
    parser.add_argument("--r_line", type=float, default=0.25, help="선로 저항 (Ohm)")
    parser.add_argument("--l_line", type=float, default=0.15e-3, help="선로 인덕턴스 (H)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--output_dir", type=str, default=PROCESSED_DIR, help="저장 디렉터리")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 80)
    print(" NILM AI 전처리기 & 물리적 AC 회로 병렬 중첩 엔진 (ACPhysicsEngine) 파이프라인")
    print("=" * 80)
    print(f"[*] 설정: 시나리오 길이={args.duration}초 ({int(args.duration * SAMPLING_HZ)} 스텝), 선로 R={args.r_line}Ω, L={args.l_line*1000:.2f}mH")
    
    # 1. ACPhysicsEngine 및 Synthetic Generator 초기화
    engine = ACPhysicsEngine(
        r_line=args.r_line,
        l_line=args.l_line,
        grid_v_nominal=220.0,
        grid_f_nominal=60.0
    )
    generator = SyntheticDatasetGenerator(
        data_dir=DATA_DIR,
        physics_engine=engine,
        apply_voltage_sag=True
    )
    
    # 2. 다중 기기 복합 시나리오 생성 (물리 기반 KCL/KVL 및 Voltage Sag 적용)
    print("\n[*] 5개 전자기기 복합 시계열 신호 합성 중...")
    agg_df, gt_df = generator.generate_scenario(
        duration_seconds=args.duration,
        seed=args.seed
    )
    
    # 3. 전기공학 피처 엔지니어링 (P, Q, S, PF, THD_I, 고조파 페이저 성분)
    print("[*] 고차원 전기 특성 엔지니어링 수행 중...")
    full_feat_df = extract_features(agg_df)
    
    # 4. 종합 요약 통계 출력
    print("\n" + "-" * 80)
    print(" [합성된 종합 수전단 신호 (Aggregate Signal) 요약 통계]")
    print("-" * 80)
    print(f"  - 총 샘플 수: {len(full_feat_df)} 스텝 (시간: {args.duration:.1f}초)")
    print(f"  - 유효전력 P(t) : 평균 {full_feat_df['p_w'].mean():.1f}W, 최소 {full_feat_df['p_w'].min():.1f}W, 최대 {full_feat_df['p_w'].max():.1f}W")
    print(f"  - 무효전력 Q(t) : 평균 {full_feat_df['q_var'].mean():.1f}var, 최대 {full_feat_df['q_var'].max():.1f}var")
    print(f"  - 피상전력 S(t) : 평균 {full_feat_df['s_va'].mean():.1f}VA, 최대 {full_feat_df['s_va'].max():.1f}VA")
    print(f"  - 종합 역률 PF  : 평균 {full_feat_df['power_factor'].mean():.3f}")
    print(f"  - 총 전류 Irms  : 평균 {full_feat_df['irms'].mean():.3f}A, 피크 {full_feat_df['irms'].max():.3f}A")
    print(f"  - 수전단 단자 전압 Vrms : 평균 {full_feat_df['vrms'].mean():.2f}V, 최저 {full_feat_df['vrms'].min():.2f}V (최대 전압강하: {full_feat_df['voltage_sag'].max():.2f}V)")
    print(f"  - 전류 왜곡률 THD_I : 평균 {full_feat_df['thd_i'].mean() * 100:.1f}%, 최대 {full_feat_df['thd_i'].max() * 100:.1f}%")
    print(f"  - 전압 왜곡률 THD_V : 평균 {full_feat_df['thd_v'].mean() * 100:.2f}%, 최대 {full_feat_df['thd_v'].max() * 100:.2f}%")
    
    print("\n" + "-" * 80)
    print(" [Ground Truth 개별 기기 상태 및 소비전력]")
    print("-" * 80)
    for key in APPLIANCE_KEYS:
        app_name = APPLIANCES[key].name_ko
        on_ratio = gt_df[f"state_{key}"].mean() * 100
        p_active = gt_df.loc[gt_df[f"state_{key}"] == 1, f"power_{key}"]
        mean_p = p_active.mean() if len(p_active) > 0 else 0.0
        print(f"  - [{app_name:10s} ({key:14s})]: 가동 비율 {on_ratio:5.1f}% | ON 시 평균전력 {mean_p:7.1f}W")
        
    # 5. PyTorch NILMDataset 생성 검증
    print("\n[*] 슬라이딩 윈도우 Dataset (Window=60 steps / Stride=10 steps) 생성 중...")
    dataset = NILMDataset(agg_df, gt_df, window_size=60, step_stride=10)
    print(f"  - 생성된 윈도우 샘플 수: {len(dataset)} 개")
    print(f"  - 추출된 피처 차원: {dataset.num_features} 차원")
    print(f"  - 타겟 기기 수: {dataset.num_appliances} 개")
    
    sample_x, sample_state, sample_power = dataset[0]
    print(f"  - 첫 번째 윈도우 텐서 크기:")
    print(f"      Input X: {tuple(sample_x.shape)} (Window, Features)")
    print(f"      Target State: {tuple(sample_state.shape)} (Appliances Multi-label)")
    print(f"      Target Power: {tuple(sample_power.shape)} (Appliances Disaggregation Power)")
    
    # 6. CSV 저장
    agg_save_path = os.path.join(args.output_dir, "synthetic_aggregate_features_demo.csv")
    gt_save_path = os.path.join(args.output_dir, "synthetic_ground_truth_demo.csv")
    full_feat_df.to_csv(agg_save_path, index=False)
    gt_df.to_csv(gt_save_path, index=False)
    print(f"\n[+] 전처리 및 합성 데이터 저장 완료:")
    print(f"    - 피처 데이터: {agg_save_path}")
    print(f"    - 라벨 데이터: {gt_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
