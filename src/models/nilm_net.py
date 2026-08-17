"""
Multi-Task Conv-BiGRU Network with Harmonic Attention for NILM
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .layers import ConvBlock1D, HarmonicAttention, TemporalAttentionPooling
from ..config import NUM_APPLIANCES, DEFAULT_FEATURE_DIM


class MultiTaskConvBiGRUNet(nn.Module):
    """
    NILM AI Multi-Task Neural Network (V2: State-Gated Power Coupling & Multi-Scale Convolutions)
    - Multi-Scale 1D-CNN (k=3, 7, 15) + Harmonic Attention + Bi-GRU + State-Gated Power Coupling
    - 분류 확률과 분해 전력을 직접 곱해 OFF 상태 잔류 오차를 수학적으로 제거
    """

    def __init__(
        self,
        in_features: int = DEFAULT_FEATURE_DIM,
        num_appliances: int = NUM_APPLIANCES,
        conv_channels: int = 128,
        gru_hidden_dim: int = 64,
        gru_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.15
    ):
        super().__init__()
        self.in_features = in_features
        self.num_appliances = num_appliances
        
        # 1. Multi-Scale 1D Convolutional Blocks (다양한 시간 스케일의 전력/고조파 특성 포착)
        branch_dim = conv_channels // 3 # ~42
        self.conv_k3 = ConvBlock1D(in_features, branch_dim, kernel_size=3, dropout=dropout)
        self.conv_k7 = ConvBlock1D(in_features, branch_dim, kernel_size=7, dropout=dropout)
        self.conv_k15 = ConvBlock1D(in_features, conv_channels - 2 * branch_dim, kernel_size=15, dropout=dropout)
        
        self.fuse_conv = ConvBlock1D(conv_channels, conv_channels, kernel_size=3, dropout=dropout)
        
        # 2. Harmonic Multi-Head Attention
        self.harmonic_attn = HarmonicAttention(
            embed_dim=conv_channels,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # 3. Bidirectional GRU
        self.bi_gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        gru_out_dim = gru_hidden_dim * 2 # 128
        
        # 4. Temporal Attention Pooling
        self.attn_pooling = TemporalAttentionPooling(gru_out_dim)
        combined_dim = gru_out_dim * 2 # 256
        
        # 5. Dual Heads
        # (1) State Classifier (Logits)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_appliances)
        )
        
        # (2) Raw Power Regressor (Base Power in Watts)
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_appliances),
            nn.Softplus() # 부드러운 비음수 활성화 (>= 0W)
        )

    def forward(
        self,
        x: torch.Tensor,
        apply_gating: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, Time=120, in_features=104)
            apply_gating: True일 경우 Sigmoid(State) * Power 결합 적용
            
        Returns:
            state_logits: (Batch, num_appliances)
            power_pred: (Batch, num_appliances)
        """
        # (Batch, Time, Features) -> (Batch, Features, Time)
        x_conv = x.transpose(1, 2)
        
        # Multi-scale Convolutions
        c3 = self.conv_k3(x_conv)
        c7 = self.conv_k7(x_conv)
        c15 = self.conv_k15(x_conv)
        multi_scale = torch.cat([c3, c7, c15], dim=1) # (Batch, conv_channels, Time)
        feat = self.fuse_conv(multi_scale)
        
        # Time-domain Attention & GRU
        feat_time = feat.transpose(1, 2) # (Batch, Time, conv_channels)
        attn_out = self.harmonic_attn(feat_time)
        gru_out, _ = self.bi_gru(attn_out)
        
        # Temporal Pooling
        pooled = self.attn_pooling(gru_out)
        last_step = gru_out[:, -1, :]
        combined = torch.cat([pooled, last_step], dim=-1) # (Batch, 256)
        
        # Output Heads
        state_logits = self.classifier(combined)
        raw_power = self.regressor(combined)
        
        if apply_gating:
            # ── State-Gated Power Coupling (분류 확률과 전력의 물리적 결합) ──
            state_prob = torch.sigmoid(state_logits)
            power_pred = state_prob * raw_power
        else:
            power_pred = raw_power
            
        return state_logits, power_pred
