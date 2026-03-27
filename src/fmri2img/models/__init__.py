"""
fMRI-to-Image Models
===================

Neural and linear models for mapping fMRI activity to image representations.
"""

from .ridge import RidgeEncoder
from .mlp import MLPEncoder, save_mlp, load_mlp
from .clip_adapter import CLIPAdapter, save_adapter, load_adapter
from .adapters import (
    LinearAdapter,
    MLPAdapter,
    ConditionEmbedding,
    AdaptedModel,
    create_adapter,
    save_adapter as save_imagery_adapter,
    load_adapter as load_imagery_adapter
)
from .encoders import (
    ResidualBlock,
    ResidualMLPEncoder,
    CLIPMappingHead,
    TwoStageEncoder,
    MultiLayerTwoStageEncoder,
    SelfSupervisedPretrainer,
    save_two_stage_encoder,
    load_two_stage_encoder
)
from .multi_target_decoder import (
    IPAdapterTokenHead,
    SDLatentHead,
    MultiTargetDecoder,
    MultiTaskLoss,
    save_multi_target_decoder,
    load_multi_target_decoder
)
from .encoding_model import (
    ImageEncoder,
    EncodingModel,
    save_encoding_model,
    load_encoding_model
)
from .interfaces import (
    ArtifactSpec,
    BaseDecoderModel,
    DecoderBatch,
    DecoderConfig,
    DecoderOutputs,
    DecoderTargets,
)
from .canonical_encoders import ROISpecificHierarchicalEncoder, ShallowBranchEncoder
from .canonical_decoder import SharedPrivateMultitaskDecoder
from .latent import SharedPrivateDisentanglementLayer, orthogonality_penalty
from .heads import ContentHead, DomainHead, MultiTaskHeads, VividnessHead
from .losses import (
    cosine_loss,
    mse_loss,
    infonce_loss,
    compose_loss,
    ComposedLoss
)

__all__ = [
    # Baseline models
    "RidgeEncoder",
    "MLPEncoder",
    "save_mlp",
    "load_mlp",
    "CLIPAdapter",
    "save_adapter",
    "load_adapter",
    # Imagery adapters
    "LinearAdapter",
    "MLPAdapter",
    "ConditionEmbedding",
    "AdaptedModel",
    "create_adapter",
    "save_imagery_adapter",
    "load_imagery_adapter",
    # Two-stage encoder
    "ResidualBlock",
    "ResidualMLPEncoder",
    "CLIPMappingHead",
    "TwoStageEncoder",
    "MultiLayerTwoStageEncoder",
    "SelfSupervisedPretrainer",
    "save_two_stage_encoder",
    "load_two_stage_encoder",
    # Multi-target decoder (novel)
    "IPAdapterTokenHead",
    "SDLatentHead",
    "MultiTargetDecoder",
    "MultiTaskLoss",
    "save_multi_target_decoder",
    "load_multi_target_decoder",
    # Encoding model (for BOI-lite)
    "ImageEncoder",
    "EncodingModel",
    "save_encoding_model",
    "load_encoding_model",
    # Canonical shared/private decoder
    "ArtifactSpec",
    "BaseDecoderModel",
    "DecoderBatch",
    "DecoderConfig",
    "DecoderOutputs",
    "DecoderTargets",
    "ShallowBranchEncoder",
    "ROISpecificHierarchicalEncoder",
    "SharedPrivateDisentanglementLayer",
    "SharedPrivateMultitaskDecoder",
    "orthogonality_penalty",
    "ContentHead",
    "DomainHead",
    "MultiTaskHeads",
    "VividnessHead",
    # Loss functions (novel)
    "cosine_loss",
    "mse_loss",
    "infonce_loss",
    "compose_loss",
    "ComposedLoss"
]
