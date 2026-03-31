from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class DecoderTargets:
    clip_target_768: torch.Tensor
    domain_label: Optional[torch.Tensor] = None
    vividness: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None

    def to_device(self, device: str | torch.device) -> "DecoderTargets":
        return DecoderTargets(
            clip_target_768=self.clip_target_768.to(device),
            domain_label=None if self.domain_label is None else self.domain_label.to(device),
            vividness=None if self.vividness is None else self.vividness.to(device),
            confidence=None if self.confidence is None else self.confidence.to(device),
        )


@dataclass
class DecoderBatch:
    fmri: Optional[torch.Tensor]
    roi_features: dict[str, torch.Tensor]
    condition: torch.Tensor
    nsd_ids: torch.Tensor
    pair_ids: torch.Tensor
    targets: DecoderTargets
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def to_device(self, device: str | torch.device) -> "DecoderBatch":
        return DecoderBatch(
            fmri=None if self.fmri is None else self.fmri.to(device),
            roi_features={k: v.to(device) for k, v in self.roi_features.items()},
            condition=self.condition.to(device),
            nsd_ids=self.nsd_ids.to(device),
            pair_ids=self.pair_ids.to(device),
            targets=self.targets.to_device(device),
            metadata=self.metadata,
        )


@dataclass
class DecoderOutputs:
    z_shared: torch.Tensor
    z_perception_private: torch.Tensor
    z_imagery_private: torch.Tensor
    content_pred: torch.Tensor
    branch_embeddings: dict[str, torch.Tensor]
    reconstructed_visual: Optional[torch.Tensor] = None
    visual_target: Optional[torch.Tensor] = None
    domain_logits: Optional[torch.Tensor] = None
    vividness_pred: Optional[torch.Tensor] = None
    confidence_pred: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    evidence: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DecoderConfig:
    target_dim: int = 768
    branch_embedding_dim: int = 128
    shared_dim: int = 128
    private_dim: int = 64
    dropout: float = 0.1
    use_domain_head: bool = True
    use_vividness_head: bool = True
    vividness_mode: str = "evidential"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_version: str
    target_spec: dict[str, Any]
    preprocessing_spec: dict[str, Any]
    roi_spec: dict[str, Any]
    checkpoint_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDecoderModel(nn.Module, ABC):
    config: DecoderConfig

    @abstractmethod
    def forward(self, batch: DecoderBatch) -> DecoderOutputs:
        raise NotImplementedError

    def describe_artifacts(self) -> dict[str, Any]:
        return {
            "model_class": self.__class__.__name__,
            "config": self.config.__dict__,
        }
