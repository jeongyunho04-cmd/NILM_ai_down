"""
NILM AI Model Evaluation and Disaggregation Report CLI Script
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import DATA_DIR, SAMPLING_HZ, APPLIANCE_KEYS, APPLIANCES
from src.ac_physics_engine import ACPhysicsEngine
from src.synthetic_generator import SyntheticDatasetGenerator
from src.dataset import NILMDataset
from src.models.nilm_net import MultiTaskConvBiGRUNet
from src.trainer import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained NILM Model on Independent Scenario")
    parser.add_argument("--test_duration", type=float, default=180.0, help="테스트 시나리오 시간 (초, 180초=10800스텝)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_nilm_model.pt", help="모델 체크포인트 경로")
    parser.add_argument("--scaler", type=str, default="checkpoints/scaler_stats.json", help="스케일러 통계 JSON 경로")
    parser.add_argument("--scale_min", type=float, default=0.70, help="테스트 시 전력 스케일 최소값")
    parser.add_argument("--scale_max", type=float, default=1.30, help="테스트 시 전력 스케일 최대값")
    parser.add_argument("--seed", type=int, default=9999, help="테스트 랜덤 시드")
    args = parser.parse_args()

    print("=" * 80)
    print(" NILM AI Model Evaluation & Disaggregation Benchmark")
    print("=" * 80)

    # 1. 스케일러 및 체크포인트 로드
    with open(args.scaler, "r", encoding="utf-8") as f:
        scaler_stats = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskConvBiGRUNet(in_features=len(scaler_stats)).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[*] 모델 로드 완료: {args.checkpoint} (Device: {device})")

    # 2. 미학습 독립 테스트 시나리오 생성 (±30% 광범위 전력 스케일 증강)
    physics_engine = ACPhysicsEngine()
    generator = SyntheticDatasetGenerator(data_dir=DATA_DIR, physics_engine=physics_engine)
    
    print(f"[*] 독립 테스트 시나리오 생성 중 ({args.test_duration}초, 스케일 [{args.scale_min}~{args.scale_max}])...")
    test_agg_df, test_gt_df = generator.generate_scenario(
        duration_seconds=args.test_duration,
        enable_power_scaling=True,
        power_scale_range=(args.scale_min, args.scale_max),
        seed=args.seed
    )

    test_dataset = NILMDataset(
        test_agg_df, test_gt_df,
        window_size=120,
        step_stride=2, # 고속 정밀 평가
        normalize=True,
        scaler_stats=scaler_stats
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 3. 모델 순전파 평가
    all_state_true, all_state_probs = [], []
    all_power_true, all_power_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, y_state, y_power = batch
            x = x.to(device)
            state_logits, power_pred = model(x)
            
            probs = torch.sigmoid(state_logits).cpu().numpy()
            all_state_probs.append(probs)
            all_state_true.append(y_state.numpy())
            all_power_pred.append(power_pred.cpu().numpy())
            all_power_true.append(y_power.numpy())

    y_st_true = np.concatenate(all_state_true, axis=0)
    y_st_prob = np.concatenate(all_state_probs, axis=0)
    y_pw_true = np.concatenate(all_power_true, axis=0)
    y_pw_pred = np.concatenate(all_power_pred, axis=0)

    # 4. 성능 지표 계산
    metrics = compute_metrics(y_st_true, y_st_prob, y_pw_true, y_pw_pred)

    print("\n" + "=" * 80)
    print(f" [독립 테스트 세트 종합 평가 결과 (총 {len(test_dataset)} 개 윈도우)]")
    print("=" * 80)
    print(f"{'기기명':<16s} | {'F1-Score':>8s} | {'Precision':>9s} | {'Recall':>8s} | {'Accuracy':>8s} | {'MAE (W)':>9s} | {'SAE (%)':>8s}")
    print("-" * 80)

    for key in APPLIANCE_KEYS:
        app_name = APPLIANCES[key].name_ko
        m = metrics[key]
        print(
            f"{app_name + ' (' + key + ')':<16s} | "
            f"{m['f1']*100:7.2f}% | "
            f"{m['precision']*100:8.2f}% | "
            f"{m['recall']*100:7.2f}% | "
            f"{m['accuracy']*100:7.2f}% | "
            f"{m['mae_w']:8.2f}W | "
            f"{m['sae_%']:7.2f}%"
        )

    ov = metrics["overall_macro"]
    print("-" * 80)
    print(
        f"{'전체 평균 (Macro)':<16s} | "
        f"{ov['f1']*100:7.2f}% | "
        f"{ov['precision']*100:8.2f}% | "
        f"{ov['recall']*100:7.2f}% | "
        f"{ov['accuracy']*100:7.2f}% | "
        f"{ov['mae_w']:8.2f}W | "
        f"{ov['sae_%']:7.2f}%"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
