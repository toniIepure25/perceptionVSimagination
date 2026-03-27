from __future__ import annotations

from typing import Mapping

import torch

from .canonical_encoders import ROISpecificHierarchicalEncoder
from .heads import MultiTaskHeads
from .interfaces import BaseDecoderModel, DecoderBatch, DecoderConfig, DecoderOutputs
from .latent import SharedPrivateDisentanglementLayer


class SharedPrivateMultitaskDecoder(BaseDecoderModel):
    """Canonical shared/private decoder for perception-imagery disentanglement."""

    def __init__(self, roi_input_dims: Mapping[str, int], config: DecoderConfig | None = None):
        super().__init__()
        self.config = config or DecoderConfig()
        self.encoder = ROISpecificHierarchicalEncoder(
            input_dims=roi_input_dims,
            embedding_dim=self.config.branch_embedding_dim,
            dropout=self.config.dropout,
        )
        visual_input_dim = self.config.branch_embedding_dim * 2
        self.disentangler = SharedPrivateDisentanglementLayer(
            visual_input_dim=visual_input_dim,
            shared_dim=self.config.shared_dim,
            private_dim=self.config.private_dim,
            dropout=self.config.dropout,
        )
        self.heads = MultiTaskHeads(
            shared_dim=self.config.shared_dim,
            private_dim=self.config.private_dim,
            metacognitive_dim=self.config.branch_embedding_dim,
            target_dim=self.config.target_dim,
            use_domain_head=self.config.use_domain_head,
            use_vividness_head=self.config.use_vividness_head,
            vividness_mode=self.config.vividness_mode,
            dropout=self.config.dropout,
        )

    def forward(self, batch: DecoderBatch) -> DecoderOutputs:
        branch_embeddings = self.encoder(batch.roi_features)
        visual_target = torch.cat(
            [branch_embeddings["early_visual"], branch_embeddings["ventral_visual"]],
            dim=-1,
        )
        latent_outputs = self.disentangler(visual_target)
        head_outputs = self.heads(
            z_shared=latent_outputs["z_shared"],
            z_perception_private=latent_outputs["z_perception_private"],
            z_imagery_private=latent_outputs["z_imagery_private"],
            metacognitive_embedding=branch_embeddings["metacognitive"],
        )
        return DecoderOutputs(
            z_shared=latent_outputs["z_shared"],
            z_perception_private=latent_outputs["z_perception_private"],
            z_imagery_private=latent_outputs["z_imagery_private"],
            content_pred=head_outputs["content_pred"],
            branch_embeddings=branch_embeddings,
            reconstructed_visual=latent_outputs["reconstructed_visual"],
            visual_target=visual_target,
            domain_logits=head_outputs.get("domain_logits"),
            vividness_pred=head_outputs.get("vividness_pred"),
            confidence_pred=head_outputs.get("confidence_pred"),
            uncertainty=head_outputs.get("uncertainty"),
            evidence=head_outputs.get("evidence"),
        )
