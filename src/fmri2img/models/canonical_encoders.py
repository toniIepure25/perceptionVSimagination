from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class ShallowBranchEncoder(nn.Module):
    """Shallow regularized encoder for low-SNR ROI branch inputs."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or max(output_dim, min(256, input_dim * 2))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ROISpecificHierarchicalEncoder(nn.Module):
    """
    Canonical branch encoder for early-visual, ventral, and metacognitive ROIs.
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        embedding_dim: int = 128,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        required = ("early_visual", "ventral_visual", "metacognitive")
        missing = [name for name in required if name not in input_dims]
        if missing:
            raise ValueError(f"Missing canonical ROI groups: {missing}")
        self.input_dims = dict(input_dims)
        self.embedding_dim = embedding_dim
        self.encoders = nn.ModuleDict(
            {
                name: ShallowBranchEncoder(
                    input_dim=input_dims[name],
                    output_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for name in required
            }
        )

    def forward(self, roi_features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.encoders[name](roi_features[name]) for name in self.encoders}
