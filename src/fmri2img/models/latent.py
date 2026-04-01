from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class SharedPrivateDisentanglementLayer(nn.Module):
    """Split visual branch features into shared and domain-private submanifolds."""

    def __init__(
        self,
        visual_input_dim: int,
        shared_dim: int = 128,
        private_dim: int = 64,
        mode: str = "shared_private",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.private_dim = private_dim
        self.mode = mode
        self.shared_proj = nn.Sequential(
            nn.LayerNorm(visual_input_dim),
            nn.Linear(visual_input_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(shared_dim),
        )
        self.perception_private_proj = nn.Sequential(
            nn.LayerNorm(visual_input_dim),
            nn.Linear(visual_input_dim, private_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(private_dim),
        )
        self.imagery_private_proj = nn.Sequential(
            nn.LayerNorm(visual_input_dim),
            nn.Linear(visual_input_dim, private_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(private_dim),
        )
        self.visual_reconstruction = nn.Sequential(
            nn.Linear(shared_dim + 2 * private_dim, visual_input_dim),
            nn.GELU(),
            nn.LayerNorm(visual_input_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> Mapping[str, torch.Tensor]:
        z_shared = self.shared_proj(visual_features)
        if self.mode == "shared_only":
            z_perc = torch.zeros(
                visual_features.size(0),
                self.private_dim,
                device=visual_features.device,
                dtype=visual_features.dtype,
            )
            z_imag = torch.zeros_like(z_perc)
        else:
            z_perc = self.perception_private_proj(visual_features)
            z_imag = self.imagery_private_proj(visual_features)
        reconstructed = self.visual_reconstruction(torch.cat([z_shared, z_perc, z_imag], dim=-1))
        return {
            "z_shared": z_shared,
            "z_perception_private": z_perc,
            "z_imagery_private": z_imag,
            "reconstructed_visual": reconstructed,
        }


def orthogonality_penalty(
    z_shared: torch.Tensor,
    z_perception_private: torch.Tensor | None = None,
    z_imagery_private: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = z_shared.new_tensor(0.0)
    if z_perception_private is not None:
        loss = loss + torch.linalg.matrix_norm(z_shared.transpose(0, 1) @ z_perception_private, ord="fro")
    if z_imagery_private is not None:
        loss = loss + torch.linalg.matrix_norm(z_shared.transpose(0, 1) @ z_imagery_private, ord="fro")
    return loss
