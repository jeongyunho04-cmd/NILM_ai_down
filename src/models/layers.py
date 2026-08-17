"""
Custom Neural Network Layers for NILM AI (Harmonic Attention & Temporal Pooling)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock1D(nn.Module):
    """1D Convolutional Residual Feature Extractor Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_channels, Time)
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.act2(out + res)
        return out


class HarmonicAttention(nn.Module):
    """
    고조파 성분 및 전기적 피처 간의 비선형 상호작용을 포착하는 Multi-Head Attention 모듈
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, embed_dim)
        B, T, C = x.shape
        residual = x
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v) # (B, H, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(context)
        out = self.norm(residual + self.dropout(out))
        return out


class TemporalAttentionPooling(nn.Module):
    """시계열 차원을 중요한 시점(스위칭, 전력 피크 등)에 가중치를 부여하여 벡터로 요약하는 어텐션 풀링"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, embed_dim)
        weights = self.attn_net(x) # (Batch, Time, 1)
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1) # (Batch, embed_dim)
        return pooled
