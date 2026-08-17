"""
NILM AI Model Training CLI Script
=================================
5개 전자기기의 물리 기반 복합 데이터셋을 생성하고
MultiTaskConvBiGRUNet을 학습하여 최적의 모델 가중치를 저장합니다.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
import torch
from torch.utils.data import DataLoader

from src.config import DATA_DIR, SAMPLING_HZ, DEFAULT_FEATURE_DIM, NUM_APPLIANCES
from src.ac_physics_engine import ACPhysicsEngine
from src.synthetic_generator import SyntheticDatasetGenerator
from src.dataset import NILMDataset
from src.models.nilm_net import MultiTaskConvBiGRUNet
from src.losses import PhysicalMultiTaskLoss
from src.trainer import NILMTrainer


def main():
    parser = argparse.ArgumentParser(description="Train NILM Multi-Task Neural Network (V2: State-Gated & Physical State-Machine)")
    parser.add_argument("--epochs", type=int, default=25, help="학습 에포크 수 (기본값 25)")
    parser.add_argument("--batch_size", type=int, default=64, help="배치 크기 (기본값 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="학습률 (기본값 1e-3)")
    parser.add_argument("--train_duration", type=float, default=800.0, help="학습용 합성 시나리오 시간 (초, 800초=48000스텝)")
    parser.add_argument("--val_duration", type=float, default=200.0, help="검증용 합성 시나리오 시간 (초, 200초=12000스텝)")
    parser.add_argument("--window_size", type=int, default=120, help="슬라이딩 윈도우 크기 (120스텝 = 2초)")
    parser.add_argument("--step_stride", type=int, default=10, help="윈도우 슬라이딩 간격 (스텝)")
    parser.add_argument("--scale_min", type=float, default=0.75, help="전력 암기방지 스케일 최소값")
    parser.add_argument("--scale_max", type=float, default=1.25, help="전력 암기방지 스케일 최대값")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="체크포인트 저장 디렉터리")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "best_nilm_model.pt")
    scaler_path = os.path.join(args.checkpoint_dir, "scaler_stats.json")

    print("=" * 80)
    print(" NILM AI Deep Learning Model Training Pipeline")
    print("=" * 80)
    print(f"[*] 설정: Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")
    print(f"[*] 전력 암기방지 스케일링 범위: [{args.scale_min} ~ {args.scale_max}] (±25% 동적 스케일 증강)")
    print(f"[*] 데이터 생성: Train {args.train_duration}초 ({int(args.train_duration * SAMPLING_HZ)} 스텝), Val {args.val_duration}초 ({int(args.val_duration * SAMPLING_HZ)} 스텝)")

    # 1. 물리 엔진 및 합성 생성기 초기화
    physics_engine = ACPhysicsEngine(r_line=0.25, l_line=0.15e-3)
    generator = SyntheticDatasetGenerator(data_dir=DATA_DIR, physics_engine=physics_engine)

    # 2. 물리 기반 학습 및 검증 데이터셋 합성 (동적 전력 스케일 증강 적용)
    print("\n[*] 학습용 물리 합성 시나리오 생성 중...")
    train_agg_df, train_gt_df = generator.generate_scenario(
        duration_seconds=args.train_duration,
        enable_power_scaling=True,
        power_scale_range=(args.scale_min, args.scale_max),
        seed=1001
    )

    print("[*] 검증용 물리 합성 시나리오 생성 중 (독립 시드)...")
    val_agg_df, val_gt_df = generator.generate_scenario(
        duration_seconds=args.val_duration,
        enable_power_scaling=True,
        power_scale_range=(args.scale_min, args.scale_max),
        seed=2002
    )

    # 3. NILM Dataset 및 DataLoader 생성
    print("\n[*] 슬라이딩 윈도우 Dataset 구축 중...")
    train_dataset = NILMDataset(
        train_agg_df, train_gt_df,
        window_size=args.window_size,
        step_stride=args.step_stride,
        normalize=True
    )
    
    # 학습 셋의 스케일러 통계로 검증 셋 정규화
    val_dataset = NILMDataset(
        val_agg_df, val_gt_df,
        window_size=args.window_size,
        step_stride=args.step_stride,
        normalize=True,
        scaler_stats=train_dataset.scaler_stats
    )

    print(f"  - 학습 윈도우 샘플 수: {len(train_dataset)} 개")
    print(f"  - 검증 윈도우 샘플 수: {len(val_dataset)} 개")
    print(f"  - 입력 피처 차원: {train_dataset.num_features} 차원")

    # 스케일러 통계 JSON 저장
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset.scaler_stats, f, indent=2)
    print(f"  - 스케일러 통계 저장됨: {scaler_path}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 4. 모델 및 손실함수 생성
    model = MultiTaskConvBiGRUNet(
        in_features=train_dataset.num_features,
        num_appliances=NUM_APPLIANCES,
        conv_channels=128,
        gru_hidden_dim=64,
        gru_layers=2,
        attention_heads=4,
        dropout=0.15
    )
    
    criterion = PhysicalMultiTaskLoss(
        weight_cls=1.0,
        weight_reg=0.05,
        weight_conservation=0.01,
        weight_state_consistency=0.02
    )

    # 5. Trainer 실행
    trainer = NILMTrainer(
        model=model,
        criterion=criterion,
        lr=args.lr,
        weight_decay=1e-4
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_path=ckpt_path
    )

    print("\n" + "=" * 80)
    print(f"[+] NILM AI 모델 학습 완료! 체크포인트: {ckpt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
