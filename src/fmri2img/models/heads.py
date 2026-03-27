from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z_shared), dim=-1)


class DomainHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, private_features: torch.Tensor) -> torch.Tensor:
        return self.net(private_features)


class VividnessHead(nn.Module):
    """
    Lightweight evidential-style head for vividness/confidence outputs.

    This is intentionally simple for the MVP: it exposes both a mean prediction
    and non-negative evidence term without claiming full Bayesian calibration.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1, mode: str = "evidential"):
        super().__init__()
        self.mode = mode
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.evidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(features)
        vividness = self.mean_head(h).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        evidence = F.softplus(self.evidence_head(h)).squeeze(-1)
        uncertainty = 1.0 / (1.0 + evidence)
        return {
            "vividness_pred": vividness,
            "confidence_pred": confidence,
            "evidence": evidence,
            "uncertainty": uncertainty,
        }


class MultiTaskHeads(nn.Module):
    def __init__(
        self,
        shared_dim: int,
        private_dim: int,
        metacognitive_dim: int,
        target_dim: int = 768,
        use_domain_head: bool = True,
        use_vividness_head: bool = True,
        vividness_mode: str = "evidential",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.content_head = ContentHead(shared_dim, output_dim=target_dim, dropout=dropout)
        self.domain_head = (
            DomainHead(private_dim * 2, hidden_dim=max(32, private_dim), dropout=dropout)
            if use_domain_head
            else None
        )
        vivid_input_dim = metacognitive_dim + 2
        self.vividness_head = (
            VividnessHead(vivid_input_dim, hidden_dim=max(32, metacognitive_dim), dropout=dropout, mode=vividness_mode)
            if use_vividness_head
            else None
        )

    def forward(
        self,
        z_shared: torch.Tensor,
        z_perception_private: torch.Tensor,
        z_imagery_private: torch.Tensor,
        metacognitive_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {"content_pred": self.content_head(z_shared)}
        if self.domain_head is not None:
            private_concat = torch.cat([z_perception_private, z_imagery_private], dim=-1)
            outputs["domain_logits"] = self.domain_head(private_concat)
        if self.vividness_head is not None:
            shared_summary = torch.stack(
                [
                    torch.linalg.norm(z_shared, dim=-1),
                    torch.var(z_shared, dim=-1, unbiased=False),
                ],
                dim=-1,
            )
            vivid_input = torch.cat([metacognitive_embedding, shared_summary], dim=-1)
            outputs.update(self.vividness_head(vivid_input))
        return outputs
