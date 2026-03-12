"""
Models Package
==============

Neural and linear decoders for mapping fMRI activity to image
representations, plus imagery adapters for cross-domain transfer.
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
    save_imagery_adapter,
    load_imagery_adapter,
)
from .encoders import (
    ResidualMLPEncoder,
    TwoStageEncoder,
    MultiLayerTwoStageEncoder,
    SelfSupervisedPretrainer,
    save_two_stage_encoder,
    load_two_stage_encoder,
)
from .multi_target_decoder import (
    MultiTargetDecoder,
    MultiTaskLoss,
    save_multi_target_decoder,
    load_multi_target_decoder,
)
from .encoding_model import (
    EncodingModel,
    save_encoding_model,
    load_encoding_model,
)
from .losses import (
    cosine_loss,
    mse_loss,
    info_nce_loss,
    infonce_loss,
    compose_loss,
    ComposedLoss,
)

from .external_loader import (
    ExternalModelLoader,
    ExternalModelConfig,
    load_fmri2images_model,
    compare_predictions,
)

try:
    from .lora_adapter import (
        LoRALinear,
        LoRAAdapter,
        MultiRankLoRA,
        LoRAAdaptedModel,
        save_lora_adapter,
        load_lora_adapter,
    )
except ImportError:
    pass  # torch not available

__all__ = [
    # Baseline decoders
    "RidgeEncoder",
    "MLPEncoder",
    "save_mlp",
    "load_mlp",
    # CLIP adapter
    "CLIPAdapter",
    "save_adapter",
    "load_adapter",
    # Imagery adapters (perception -> imagination transfer)
    "LinearAdapter",
    "MLPAdapter",
    "ConditionEmbedding",
    "AdaptedModel",
    "create_adapter",
    "save_imagery_adapter",
    "load_imagery_adapter",
    # Two-stage encoder
    "ResidualMLPEncoder",
    "TwoStageEncoder",
    "MultiLayerTwoStageEncoder",
    "SelfSupervisedPretrainer",
    "save_two_stage_encoder",
    "load_two_stage_encoder",
    # Multi-target decoder (semantic-structural dissociation)
    "MultiTargetDecoder",
    "MultiTaskLoss",
    "save_multi_target_decoder",
    "load_multi_target_decoder",
    # Encoding model
    "EncodingModel",
    "save_encoding_model",
    "load_encoding_model",
    # Loss functions
    "cosine_loss",
    "mse_loss",
    "info_nce_loss",
    "infonce_loss",
    "compose_loss",
    "ComposedLoss",
    # External model loader (FMRI2images bridge)
    "ExternalModelLoader",
    "ExternalModelConfig",
    "load_fmri2images_model",
    "compare_predictions",
    # LoRA adapters (parameter-efficient domain adaptation)
    "LoRALinear",
    "LoRAAdapter",
    "MultiRankLoRA",
    "LoRAAdaptedModel",
    "save_lora_adapter",
    "load_lora_adapter",
]
