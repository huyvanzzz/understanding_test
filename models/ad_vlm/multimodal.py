from __future__ import annotations

import torch
from torch import nn


class MultiModalProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.d_model = d_model
        self.proj: nn.Module = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("Projector expects [B, S, C]")
        if features.shape[1] != self.seq_len or features.shape[2] != self.input_dim:
            raise ValueError(f"Expected [B,{self.seq_len},{self.input_dim}], got {tuple(features.shape)}")
        return self.proj(features)
