"""
Multi-Objective Loss Functions for CLIP Alignment
=================================================

Implements SOTA loss functions for training fMRI → CLIP encoders:
1. MSE loss: L2 distance in CLIP space
2. Cosine similarity loss: Directional alignment
3. InfoNCE contrastive loss: Batch-wise discrimination

Scientific Rationale:
- MSE captures magnitude alignment (Euclidean distance)
- Cosine captures directional alignment (angular distance)
- InfoNCE provides contrastive learning signal (discrimination)
- Combining all three improves representation quality (Radford et al. 2021, Chen et al. 2020)

References:
- Radford et al. (2021): CLIP - contrastive learning of visual representations
- Chen et al. (2020): SimCLR - simple framework for contrastive learning
- Oord et al. (2018): Representation learning with contrastive predictive coding (InfoNCE)
- MindEye2 (Scotti et al. 2024): Multi-objective loss for fMRI decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error loss in CLIP space.
    
    Measures L2 distance between predicted and target embeddings.
    Captures magnitude alignment (how close predictions are in Euclidean space).
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D)
    
    Returns:
        Scalar loss (averaged over batch)
    """
    return F.mse_loss(pred, target)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine distance loss: 1 - cosine_similarity(pred, target).
    
    Measures angular distance between predicted and target embeddings.
    Captures directional alignment (same direction in embedding space).
    
    IMPORTANT: Both pred and target should be L2-normalized for proper cosine computation.
    If not normalized, this still works but cosine similarity is not in [-1, 1].
    
    Args:
        pred: Predicted embeddings (B, D), ideally L2-normalized
        target: Target embeddings (B, D), ideally L2-normalized
    
    Returns:
        Scalar loss (averaged over batch)
    
    Scientific Context:
    - Cosine loss is standard for CLIP alignment (Radford et al. 2021)
    - Directional alignment often more important than magnitude for retrieval
    """
    # Compute cosine similarity: dot product of normalized vectors
    # If inputs are L2-normalized: cos_sim = (pred * target).sum(dim=-1)
    # Otherwise: use F.cosine_similarity which normalizes internally
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (B,)
    
    # Cosine loss: 1 - similarity (minimizing distance)
    # Range: [0, 2] if normalized (0 = perfect match, 2 = opposite direction)
    return (1.0 - cos_sim).mean()


def info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style) contrastive loss.
    
    Computes bidirectional contrastive loss: averages pred→target and
    target→pred cross-entropy directions. L2-normalizes inputs internally
    for proper cosine similarity computation.
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D)
        temperature: Temperature scaling parameter (default: 0.07)
                    Lower temperature = harder discrimination
                    Typical range: [0.05, 0.2]
    
    Returns:
        Scalar loss (averaged over batch and both directions)
    
    Scientific Context:
    - Symmetric InfoNCE following CLIP (Radford et al. 2021)
    - Averaging both directions provides stronger, more stable gradients
    - Internal L2-normalization ensures proper cosine similarity
    - Temperature 0.07 is CLIP's default; 0.05 can cause gradient collapse
    """
    batch_size = pred.shape[0]
    
    if batch_size < 2:
        logger.warning(f"InfoNCE loss requires batch_size >= 2, got {batch_size}. Returning zero.")
        return torch.tensor(0.0, device=pred.device)
    
    # L2-normalize for proper cosine similarity
    pred_norm = F.normalize(pred, p=2, dim=-1)
    target_norm = F.normalize(target, p=2, dim=-1)
    
    # Cosine similarity matrix scaled by temperature
    logits = torch.matmul(pred_norm, target_norm.T) / temperature  # (B, B)
    
    labels = torch.arange(batch_size, device=pred.device)
    
    # Symmetric loss (CLIP-style): average both directions
    loss_p2t = F.cross_entropy(logits, labels)      # pred → target
    loss_t2p = F.cross_entropy(logits.T, labels)     # target → pred
    
    return (loss_p2t + loss_t2p) / 2.0


class MultiLoss(nn.Module):
    """
    Combined multi-objective loss for CLIP alignment with optional brain-consistency.
    
    Combines MSE, cosine, and InfoNCE losses with configurable weights:
        L_total = w_mse * L_mse + w_cos * L_cos + w_nce * L_nce + w_brain * L_brain
    
    Args:
        mse_weight: Weight for MSE loss (default: 0.3)
        cosine_weight: Weight for cosine loss (default: 0.3)
        info_nce_weight: Weight for InfoNCE loss (default: 0.4)
        brain_consistency_weight: Weight for brain cycle loss (default: 0.0)
        temperature: Temperature for InfoNCE (default: 0.05)
        log_components: Whether to return individual loss components (default: False)
        clip_to_fmri_encoder: Optional frozen CLIP→fMRI encoder for cycle loss
    
    Scientific Rationale:
    - MSE: magnitude alignment
    - Cosine: directional alignment
    - InfoNCE: discriminative learning
    - Brain consistency: cycle-consistency regularization
    - Balanced weights (0.3/0.3/0.4) prioritize discrimination slightly
    - Can adjust weights via config for ablation studies
    
    Example:
        >>> # Without brain consistency
        >>> criterion = MultiLoss(mse_weight=0.3, cosine_weight=0.3, 
        ...                       info_nce_weight=0.4)
        >>> 
        >>> # With brain consistency
        >>> criterion = MultiLoss(
        ...     mse_weight=0.3, cosine_weight=0.3, info_nce_weight=0.4,
        ...     brain_consistency_weight=0.1,
        ...     clip_to_fmri_encoder=encoder
        ... )
        >>> 
        >>> # In training loop
        >>> loss, components = criterion(
        ...     pred_clip, true_clip, 
        ...     fmri_input=fmri_pca,  # Required for brain loss
        ...     return_components=True
        ... )
    """
    
    def __init__(
        self,
        mse_weight: float = 0.3,
        cosine_weight: float = 0.3,
        info_nce_weight: float = 0.4,
        brain_consistency_weight: float = 0.0,
        temperature: float = 0.07,
        log_components: bool = False,
        clip_to_fmri_encoder: Optional[nn.Module] = None
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.info_nce_weight = info_nce_weight
        self.brain_consistency_weight = brain_consistency_weight
        self.temperature = temperature
        self.log_components = log_components
        self.clip_to_fmri_encoder = clip_to_fmri_encoder
        
        # Validate brain consistency setup
        if brain_consistency_weight > 0 and clip_to_fmri_encoder is None:
            raise ValueError(
                "brain_consistency_weight > 0 requires clip_to_fmri_encoder. "
                "Either set weight to 0 or provide the encoder."
            )
        
        # Freeze encoder if provided
        if clip_to_fmri_encoder is not None:
            for param in clip_to_fmri_encoder.parameters():
                param.requires_grad = False
            clip_to_fmri_encoder.eval()
        
        # Validate weights
        total_weight = mse_weight + cosine_weight + info_nce_weight
        if not torch.isclose(torch.tensor(total_weight), torch.tensor(1.0), atol=1e-3):
            logger.warning(f"CLIP loss weights sum to {total_weight:.3f}, not 1.0. This is okay but may affect learning rate tuning.")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fmri_input: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted CLIP embeddings (B, D), L2-normalized
            target: Target CLIP embeddings (B, D), L2-normalized
            fmri_input: Original fMRI PCA features (B, fmri_dim), required if brain_weight > 0
            return_components: If True, return (total_loss, components_dict)
        
        Returns:
            If return_components=False: total_loss (scalar)
            If return_components=True: (total_loss, components_dict)
                components_dict = {"mse": scalar, "cosine": scalar, "info_nce": scalar, "brain": scalar}
        """
        # Compute CLIP alignment losses
        loss_mse = mse_loss(pred, target)
        loss_cos = cosine_loss(pred, target)
        loss_nce = info_nce_loss(pred, target, temperature=self.temperature)
        
        # Weighted combination for CLIP losses
        total_loss = (
            self.mse_weight * loss_mse +
            self.cosine_weight * loss_cos +
            self.info_nce_weight * loss_nce
        )
        
        # Add brain-consistency loss if enabled
        loss_brain = torch.tensor(0.0, device=pred.device)
        if self.brain_consistency_weight > 0:
            if fmri_input is None:
                raise ValueError(
                    "fmri_input required for brain-consistency loss. "
                    "Pass the original fMRI PCA features."
                )
            loss_brain = brain_consistency_loss(pred, fmri_input, self.clip_to_fmri_encoder)
            total_loss = total_loss + self.brain_consistency_weight * loss_brain
        
        if return_components or self.log_components:
            components = {
                "mse": loss_mse.item() if isinstance(loss_mse, torch.Tensor) else loss_mse,
                "cosine": loss_cos.item() if isinstance(loss_cos, torch.Tensor) else loss_cos,
                "info_nce": loss_nce.item() if isinstance(loss_nce, torch.Tensor) else loss_nce,
                "brain": loss_brain.item() if isinstance(loss_brain, torch.Tensor) else loss_brain,
                "total": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            }
            
            if return_components:
                return total_loss, components
            else:
                # Just log internally
                if self.log_components:
                    logger.debug(f"Loss components: MSE={loss_mse:.4f}, Cos={loss_cos:.4f}, NCE={loss_nce:.4f}, Brain={loss_brain:.4f}")
        
        return total_loss


def compute_multiloss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: Optional[Dict[str, float]] = None
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Functional interface for multi-objective loss (no nn.Module).
    
    Convenience function for computing multi-loss without creating a module.
    Useful for simple training scripts or one-off evaluations.
    
    Args:
        pred: Predicted embeddings (B, D), L2-normalized
        target: Target embeddings (B, D), L2-normalized
        config: Dictionary with keys:
                - mse_weight (default: 0.3)
                - cosine_weight (default: 0.3)
                - info_nce_weight (default: 0.4)
                - temperature (default: 0.05)
    
    Returns:
        total_loss: Scalar loss
        components: Dictionary with individual loss values
    
    Example:
        >>> config = {"mse_weight": 0.3, "cosine_weight": 0.3, 
        ...           "info_nce_weight": 0.4, "temperature": 0.05}
        >>> loss, components = compute_multiloss(pred, target, config)
    """
    if config is None:
        config = {}
    
    mse_weight = config.get("mse_weight", 0.3)
    cosine_weight = config.get("cosine_weight", 0.3)
    info_nce_weight = config.get("info_nce_weight", 0.4)
    temperature = config.get("temperature", 0.05)
    
    # Compute individual losses
    loss_mse = mse_loss(pred, target)
    loss_cos = cosine_loss(pred, target)
    loss_nce = info_nce_loss(pred, target, temperature=temperature)
    
    # Weighted combination
    total_loss = (
        mse_weight * loss_mse +
        cosine_weight * loss_cos +
        info_nce_weight * loss_nce
    )
    
    components = {
        "mse": loss_mse.item(),
        "cosine": loss_cos.item(),
        "info_nce": loss_nce.item(),
        "total": total_loss.item()
    }
    
    return total_loss, components


# Backward compatibility: keep old compose_loss function
def compose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 0.5
) -> torch.Tensor:
    """
    Legacy combined cosine + MSE loss (for backward compatibility).
    
    This is the original loss function from train_utils.py.
    Kept for backward compatibility with existing training scripts.
    
    For new code, prefer MultiLoss or compute_multiloss which include InfoNCE.
    
    Args:
        pred: Predicted embeddings (B, D), L2-normalized
        target: Target embeddings (B, D), L2-normalized
        mse_weight: Weight for MSE term (default: 0.5)
    
    Returns:
        Scalar loss
    """
    loss_cos = cosine_loss(pred, target)
    loss_mse = mse_loss(pred, target)
    return loss_cos + mse_weight * loss_mse


def brain_consistency_loss(
    clip_pred: torch.Tensor,
    fmri_true: torch.Tensor,
    clip_to_fmri_encoder: torch.nn.Module
) -> torch.Tensor:
    """
    Brain-consistency (cycle) loss using CLIP→fMRI encoder.
    
    Measures how well predicted CLIP embeddings can reconstruct the original fMRI:
    
        fMRI_true → Decoder → CLIP_pred → Encoder → fMRI_reconstructed
        Loss = MSE(fMRI_reconstructed, fMRI_true)
    
    This acts as a regularizer: predicted CLIP embeddings must be brain-plausible,
    i.e., they should map back to valid fMRI patterns consistent with the input.
    
    Scientific Rationale:
    - Inspired by cycle-consistency in CycleGAN (Zhu et al. 2017)
    - Ensures predictions lie in a brain-realistic subspace of CLIP space
    - Acts as implicit regularization without explicit constraints
    - Novel application: most fMRI→image papers don't use cycle loss
    
    Args:
        clip_pred: Predicted CLIP embeddings from decoder (B, 512)
        fmri_true: Original fMRI PCA features (B, fmri_dim)
        clip_to_fmri_encoder: Frozen CLIP→fMRI encoder
    
    Returns:
        Scalar MSE loss between reconstructed and true fMRI
    
    Example:
        >>> # In decoder training loop
        >>> z_pred = decoder(fmri_pca)  # Predict CLIP
        >>> 
        >>> # Standard losses
        >>> loss_clip = compute_multiloss(z_pred, z_true, config)
        >>> 
        >>> # Add brain-consistency loss
        >>> loss_brain = brain_consistency_loss(
        >>>     z_pred, fmri_pca, clip_to_fmri_encoder
        >>> )
        >>> 
        >>> # Combined loss
        >>> total_loss = loss_clip + brain_weight * loss_brain
    """
    # Reconstruct fMRI from predicted CLIP
    with torch.no_grad():
        # Encoder is frozen, no gradients needed for its parameters
        clip_to_fmri_encoder.eval()
    
    fmri_reconstructed = clip_to_fmri_encoder(clip_pred)
    
    # MSE between reconstructed and true fMRI
    loss = torch.nn.functional.mse_loss(fmri_reconstructed, fmri_true)
    
    return loss


class MultiLayerLoss(nn.Module):
    """
    Multi-layer CLIP supervision loss for hierarchical feature learning.
    
    Computes weighted cosine similarity loss across multiple ViT layers:
        L_total = Σ w_i * (1 - cos_sim(pred_i, target_i))
    
    Where i ∈ {layer_4, layer_8, layer_12, final} and w_i are layer weights.
    
    **Phase 1 Enhancement**: Supports both fixed and learnable layer weights.
    - Fixed weights: Manually specified via config (backward-compatible)
    - Learnable weights: Optimized via softmax-normalized parameters during training
    
    **Phase 3 Enhancement**: Optional multi-layer InfoNCE for contrastive learning.
    - Standard: InfoNCE only on final layer (if enabled)
    - Multi-layer: InfoNCE on combined representation from all layers
    - Provides richer contrastive signal (+2-3% expected improvement)
    
    Architecture Rationale:
    - Early layers (4, 8): Low-level visual features, high spatial resolution
    - Late layers (12): Semantic features, more abstract
    - Final: Global image representation, CLIP embedding space
    - Multi-level supervision improves gradient flow and feature learning
    
    Args:
        layer_weights: Dict mapping layer names to weights (default: uniform)
        use_mse: If True, also include MSE term (default: False)
        mse_weight: Weight for MSE component if enabled (default: 0.1)
        use_learnable_weights: If True, learn layer weights via gradient descent (default: False)
        use_multilayer_infonce: If True, add InfoNCE on combined multi-layer representation (default: False)
        infonce_weight: Weight for InfoNCE loss if enabled (default: 0.2)
        infonce_temperature: Temperature for InfoNCE (default: 0.05)
        infonce_combination: Strategy for combining layers ("weighted_pool", "concat_project", "average")
    
    Example:
        >>> # Fixed weights (backward-compatible)
        >>> criterion = MultiLayerLoss(layer_weights={
        ...     'layer_4': 0.15,
        ...     'layer_8': 0.2,
        ...     'layer_12': 0.25,
        ...     'final': 0.4
        ... })
        >>> 
        >>> # Learnable weights
        >>> criterion = MultiLayerLoss(use_learnable_weights=True)
        >>> # Weights are learned during training, logged periodically
        >>> 
        >>> # Multi-layer InfoNCE (Phase 3)
        >>> criterion = MultiLayerLoss(
        ...     use_learnable_weights=True,
        ...     use_multilayer_infonce=True,
        ...     infonce_weight=0.2,
        ...     infonce_combination="weighted_pool"
        ... )
        >>> 
        >>> # In training loop
        >>> pred_dict = model(fmri)
        >>> target_dict = load_multilayer_targets(batch)
        >>> # Pass model for InfoNCE (needs get_infonce_representation method)
        >>> loss, components = criterion(
        ...     pred_dict, target_dict,
        ...     model=model,  # Required if use_multilayer_infonce=True
        ...     return_components=True
        ... )
        >>> 
        >>> # Get current effective weights (fixed or learned)
        >>> current_weights = criterion.get_effective_weights()
    
    Scientific Background:
    - Feature Pyramid Networks (Lin et al. 2017): Multi-scale supervision
    - U-Net (Ronneberger et al. 2015): Skip connections with multi-level loss
    - ViT Analysis (Raghu et al. 2021): Different layers capture different semantics
    - Task-dependent weighting (Kendall et al. 2018): Learning optimal task balance
    - Expected improvement: +5-10% embedding similarity (Li et al. 2023)
    """
    
    def __init__(
        self,
        layer_weights: Optional[Dict[str, float]] = None,
        layer_names: Optional[list] = None,
        use_mse: bool = False,
        mse_weight: float = 0.1,
        use_learnable_weights: bool = False,
        use_multilayer_infonce: bool = False,
        infonce_weight: float = 0.2,
        infonce_temperature: float = 0.07,
        infonce_combination: str = "weighted_pool",
        text_clip_weight: float = 0.3  # Phase 2: Weight for text-CLIP loss
    ):
        super().__init__()
        
        self.use_mse = use_mse
        self.mse_weight = mse_weight
        self.use_learnable_weights = use_learnable_weights
        
        # Phase 2: Text-CLIP weighting
        self.text_clip_weight = text_clip_weight
        
        # Phase 3: Multi-layer InfoNCE
        self.use_multilayer_infonce = use_multilayer_infonce
        self.infonce_weight = infonce_weight
        self.infonce_temperature = infonce_temperature
        self.infonce_combination = infonce_combination
        
        # Determine layer order (use provided layer_weights keys or defaults)
        if layer_weights is not None:
            self.layer_names = list(layer_weights.keys())
        else:
            self.layer_names = layer_names or ['layer_12', 'layer_18', 'final']
        
        if use_learnable_weights:
            # Initialize learnable weight parameters (raw logits)
            # Will be softmax-normalized to sum to 1.0
            init_logits = torch.zeros(len(self.layer_names))
            
            # If fixed weights provided, use them as initialization
            if layer_weights is not None:
                for i, name in enumerate(self.layer_names):
                    if name in layer_weights:
                        # Convert weight to logit: w = exp(logit) / sum(exp(logits))
                        # For init, use log(w) as rough approximation
                        init_logits[i] = torch.log(torch.tensor(layer_weights[name]) + 1e-8)
            
            self.weight_logits = nn.Parameter(init_logits)
            self.layer_weights = None  # Will be computed dynamically
            logger.info(f"MultiLayerLoss initialized with LEARNABLE weights for layers {self.layer_names}")
        else:
            # Fixed weights (backward-compatible)
            if layer_weights is None:
                # Uniform weights across enabled layers
                n = len(self.layer_names)
                self.layer_weights = {name: 1.0 / n for name in self.layer_names}
            else:
                self.layer_weights = layer_weights
            
            # Validate and normalize fixed weights
            total = sum(self.layer_weights.values())
            if not torch.isclose(torch.tensor(total), torch.tensor(1.0), atol=1e-3):
                logger.warning(f"Layer weights sum to {total:.3f}, not 1.0. Normalizing.")
                self.layer_weights = {k: v/total for k, v in self.layer_weights.items()}
            
            self.weight_logits = None
            logger.info(f"MultiLayerLoss initialized with FIXED weights: {self.layer_weights}")
    
    def get_effective_weights(self) -> Dict[str, float]:
        """
        Get current effective layer weights (fixed or learned).
        
        Returns:
            Dict mapping layer names to current weights (sum to 1.0)
        """
        if self.use_learnable_weights:
            # Compute softmax-normalized weights
            weights_tensor = torch.softmax(self.weight_logits, dim=0)
            return {name: weights_tensor[i].item() for i, name in enumerate(self.layer_names)}
        else:
            return self.layer_weights.copy()
    
    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None,
        return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-layer loss.
        
        Args:
            pred_dict: Predicted features {layer_name: (B, D)}
            target_dict: Target features {layer_name: (B, D)}
            model: Model instance (required if use_multilayer_infonce=True)
            return_components: If True, return (total_loss, components_dict)
        
        Returns:
            If return_components=False: total_loss (scalar)
            If return_components=True: (total_loss, components_dict)
                components_dict = {layer_name: scalar_loss, "infonce": scalar_loss, ...}
        """
        # Get current effective weights
        if self.use_learnable_weights:
            # Compute softmax-normalized weights dynamically
            weights_tensor = torch.softmax(self.weight_logits, dim=0)
            effective_weights = {name: weights_tensor[i] for i, name in enumerate(self.layer_names)}
        else:
            effective_weights = self.layer_weights
        
        total_loss = 0.0
        components = {}
        
        # Phase 2: Separate text-CLIP loss if present
        text_loss = None
        if 'text' in pred_dict and 'text' in target_dict:
            pred_text = pred_dict['text']
            target_text = target_dict['text']
            
            # Cosine similarity loss
            cos_sim = torch.nn.functional.cosine_similarity(pred_text, target_text, dim=-1)
            text_loss = (1.0 - cos_sim).mean()
            
            # Optional MSE
            if self.use_mse:
                mse_loss = torch.nn.functional.mse_loss(pred_text, target_text)
                text_loss = text_loss + self.mse_weight * mse_loss
            
            components['text'] = text_loss.item()
        
        # Image-CLIP layers (layer_4, layer_8, layer_12, final)
        for layer_name in self.layer_names:
            if layer_name not in pred_dict or layer_name not in target_dict:
                continue
            
            weight = effective_weights[layer_name]
            pred = pred_dict[layer_name]  # (B, D)
            target = target_dict[layer_name]  # (B, D)
            
            # Cosine similarity loss: 1 - cos_sim
            cos_sim = torch.nn.functional.cosine_similarity(pred, target, dim=-1)  # (B,)
            cos_loss = (1.0 - cos_sim).mean()
            
            # Optional MSE component
            if self.use_mse:
                mse_loss = torch.nn.functional.mse_loss(pred, target)
                layer_loss = cos_loss + self.mse_weight * mse_loss
            else:
                layer_loss = cos_loss
            
            # Weighted sum (use tensor weight for learnable, float for fixed)
            if self.use_learnable_weights:
                total_loss = total_loss + weight * layer_loss
            else:
                total_loss = total_loss + weight * layer_loss
            
            components[layer_name] = layer_loss.item()
        
        # Phase 3: Multi-layer InfoNCE
        if self.use_multilayer_infonce:
            if model is None:
                raise ValueError(
                    "use_multilayer_infonce=True requires passing model to forward(). "
                    "Model must have get_infonce_representation() method."
                )
            
            # Get combined representation from model
            z_pred = model.get_infonce_representation(
                pred_dict,
                strategy=self.infonce_combination
            )
            z_target = model.get_infonce_representation(
                target_dict,
                strategy=self.infonce_combination
            )
            
            # Compute InfoNCE loss
            loss_infonce = info_nce_loss(z_pred, z_target, temperature=self.infonce_temperature)
            total_loss = total_loss + self.infonce_weight * loss_infonce
            components['infonce'] = loss_infonce.item()
        
        # Phase 2: Combine image-CLIP and text-CLIP losses
        # If text loss exists, use weighted combination: (1-w)*image + w*text
        if text_loss is not None:
            image_loss = total_loss  # Store image loss
            components['image_total'] = image_loss.item()
            total_loss = (1.0 - self.text_clip_weight) * image_loss + self.text_clip_weight * text_loss
        
        if return_components:
            return total_loss, components
        return total_loss


class ProbabilisticMultiLayerLoss(nn.Module):
    """
    Phase 3: Probabilistic loss with KL divergence regularization.
    
    Extends MultiLayerLoss to support probabilistic predictions:
        L_total = L_reconstruction + β * L_KL
    
    Where:
    - L_reconstruction: Multi-layer cosine + MSE loss (like MultiLayerLoss)
    - L_KL: KL divergence KL(q(z|x) || N(0,I)) to regularize distributions
    - β: KL weight (annealed during training, e.g., 0 → 0.01 over 20 epochs)
    
    **Annealing Schedule:**
    - Epochs 1-10: β = 0 (learn good μ first, ignore variance)
    - Epochs 11-30: β linearly increases 0 → β_max (gradually add regularization)
    - Epochs 30+: β = β_max (full VAE training)
    
    **Scientific Motivation:**
    - Uncertainty quantification: Model can express confidence in predictions
    - Better generalization: KL regularization prevents overfitting
    - Principled Bayesian inference: Variational lower bound on log p(target|fmri)
    - Enables confidence-aware decoding: Weight predictions by certainty
    
    **Usage:**
        # Create loss with annealing
        criterion = ProbabilisticMultiLayerLoss(
            kl_weight_max=0.01,
            kl_anneal_epochs=20,
            layer_weights={'layer_4': 0.15, ..., 'final': 0.4}
        )
        
        # In training loop
        pred_dict, kl_loss = model(fmri, sample=True, return_kl=True)
        target_dict = load_targets(batch)
        
        # Pass current epoch for annealing
        loss, components = criterion(
            pred_dict, target_dict, kl_loss,
            current_epoch=epoch,
            return_components=True
        )
        
        # components = {
        #     'layer_4': 0.12, 'layer_8': 0.15, ...,
        #     'kl': 0.05, 'kl_weight': 0.005  # Annealed weight
        # }
    
    Args:
        layer_weights: Dict mapping layer names to weights (like MultiLayerLoss)
        use_mse: If True, also include MSE term
        mse_weight: Weight for MSE component
        kl_weight_max: Maximum KL weight after annealing (default: 0.01)
        kl_anneal_epochs: Number of epochs to anneal from 0 to kl_weight_max (default: 20)
        kl_anneal_start: Epoch to start annealing (default: 10, learn μ first)
        text_clip_weight: Weight for text-CLIP loss (Phase 2)
    
    Example:
        >>> # Standard probabilistic training
        >>> criterion = ProbabilisticMultiLayerLoss(
        ...     kl_weight_max=0.01,
        ...     kl_anneal_epochs=20
        ... )
        >>> 
        >>> # With text-CLIP (Phase 2 + Phase 3)
        >>> criterion = ProbabilisticMultiLayerLoss(
        ...     kl_weight_max=0.01,
        ...     kl_anneal_epochs=20,
        ...     text_clip_weight=0.3
        ... )
        >>> 
        >>> # Training loop
        >>> for epoch in range(50):
        ...     pred_dict, kl_loss = model(fmri, sample=True, return_kl=True)
        ...     loss, comp = criterion(pred_dict, target, kl_loss, current_epoch=epoch, return_components=True)
        ...     print(f"Epoch {epoch}: KL weight = {comp['kl_weight']:.4f}, KL loss = {comp['kl']:.4f}")
    
    References:
        - Kingma & Welling (2014): VAE with β-annealing
        - Bowman et al. (2016): Generating Sentences from a Continuous Space (KL annealing)
        - Higgins et al. (2017): β-VAE for disentangled representations
        - Sønderby et al. (2016): Ladder VAE with annealing schedules
    """
    
    def __init__(
        self,
        layer_weights: Optional[Dict[str, float]] = None,
        use_mse: bool = False,
        mse_weight: float = 0.1,
        kl_weight_max: float = 0.01,
        kl_anneal_epochs: int = 20,
        kl_anneal_start: int = 10,
        text_clip_weight: float = 0.3
    ):
        super().__init__()
        
        # Use MultiLayerLoss for reconstruction term
        self.reconstruction_loss = MultiLayerLoss(
            layer_weights=layer_weights,
            use_mse=use_mse,
            mse_weight=mse_weight,
            use_learnable_weights=False,  # Keep weights fixed for simplicity
            text_clip_weight=text_clip_weight
        )
        
        # KL annealing parameters
        self.kl_weight_max = kl_weight_max
        self.kl_anneal_epochs = kl_anneal_epochs
        self.kl_anneal_start = kl_anneal_start
        
        logger.info(
            f"ProbabilisticMultiLayerLoss initialized: "
            f"KL weight {0:.3f} → {kl_weight_max:.3f} over epochs {kl_anneal_start}-{kl_anneal_start + kl_anneal_epochs}"
        )
    
    def get_kl_weight(self, current_epoch: int) -> float:
        """
        Compute current KL weight based on annealing schedule.
        
        Annealing schedule:
        - epoch < kl_anneal_start: β = 0 (no KL loss, learn good μ)
        - kl_anneal_start ≤ epoch < kl_anneal_start + kl_anneal_epochs:
            β = kl_weight_max * (epoch - kl_anneal_start) / kl_anneal_epochs
        - epoch ≥ kl_anneal_start + kl_anneal_epochs: β = kl_weight_max
        
        Args:
            current_epoch: Current training epoch (0-indexed)
        
        Returns:
            kl_weight: Current KL weight β ∈ [0, kl_weight_max]
        """
        if current_epoch < self.kl_anneal_start:
            return 0.0
        elif current_epoch >= self.kl_anneal_start + self.kl_anneal_epochs:
            return self.kl_weight_max
        else:
            # Linear annealing
            progress = (current_epoch - self.kl_anneal_start) / self.kl_anneal_epochs
            return self.kl_weight_max * progress
    
    def forward(
        self,
        pred_dict: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        kl_loss: torch.Tensor,
        current_epoch: int = 0,
        return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute probabilistic multi-layer loss.
        
        Args:
            pred_dict: Predicted features {layer_name: (B, D)}
            target_dict: Target features {layer_name: (B, D)}
            kl_loss: KL divergence from model forward pass (scalar)
            current_epoch: Current training epoch for KL annealing
            return_components: If True, return (total_loss, components_dict)
        
        Returns:
            If return_components=False: total_loss (scalar)
            If return_components=True: (total_loss, components_dict)
                components_dict includes all layer losses + 'kl', 'kl_weight'
        """
        # Reconstruction loss (multi-layer cosine/MSE)
        recon_loss, recon_components = self.reconstruction_loss(
            pred_dict, target_dict, return_components=True
        )
        
        # KL loss with annealing
        kl_weight = self.get_kl_weight(current_epoch)
        weighted_kl_loss = kl_weight * kl_loss
        
        # Total loss
        total_loss = recon_loss + weighted_kl_loss
        
        if return_components:
            components = recon_components.copy()
            components['kl'] = kl_loss.item()
            components['kl_weight'] = kl_weight
            components['weighted_kl'] = weighted_kl_loss.item()
            components['reconstruction'] = recon_loss.item()
            return total_loss, components
        
        return total_loss


# ==========================================================================
# Non-Contrastive Losses — VICReg & Barlow Twins
# ==========================================================================

class VICRegLoss(nn.Module):
    """
    VICReg (Variance-Invariance-Covariance Regularization) loss.

    Non-contrastive self-supervised loss that avoids representation collapse
    via three explicit regularization terms instead of negative pairs:

    L = λ·invariance + μ·variance + ν·covariance

    where:
    - Invariance: MSE between paired representations (alignment)
    - Variance: hinge loss on per-dimension std (prevents collapse)
    - Covariance: penalizes off-diagonal covariance (decorrelation)

    Advantages over InfoNCE for fMRI decoding:
    1. No negative pairs → no batch-size sensitivity
    2. No temperature hyperparameter → more stable training
    3. Explicit variance constraint → prevents the dimension collapse
       observed with strong InfoNCE (Section 6.2 of EXPERIMENT_RESULTS.md)
    4. Covariance term → whitening effect promotes information spread

    References:
        Bardes, Ponce, LeCun (2022). "VICReg: Variance-Invariance-Covariance
            Regularization for Self-Supervised Learning." ICLR.
    """

    def __init__(
        self,
        sim_weight: float = 25.0,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        var_target: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_target = var_target
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VICReg loss.

        Args:
            pred: Predicted embeddings (B, D)
            target: Target embeddings (B, D)
            return_components: If True, return (loss, components_dict)

        Returns:
            Total loss, or (total_loss, components) if return_components=True
        """
        batch_size, dim = pred.shape

        # --- Invariance: MSE between prediction and target ---
        sim_loss = F.mse_loss(pred, target)

        # --- Variance: hinge loss on per-dimension std ---
        # Prevents collapse by forcing each dimension to have std >= target
        pred_std = torch.sqrt(pred.var(dim=0) + self.eps)
        target_std = torch.sqrt(target.var(dim=0) + self.eps)
        var_loss = (
            F.relu(self.var_target - pred_std).mean()
            + F.relu(self.var_target - target_std).mean()
        )

        # --- Covariance: penalize off-diagonal elements ---
        # Decorrelates dimensions, promoting information spread
        pred_centered = pred - pred.mean(dim=0)
        target_centered = target - target.mean(dim=0)
        cov_pred = (pred_centered.T @ pred_centered) / max(batch_size - 1, 1)
        cov_target = (target_centered.T @ target_centered) / max(batch_size - 1, 1)

        # Zero out diagonal (we only penalize off-diagonal)
        cov_pred = cov_pred - torch.diag(torch.diag(cov_pred))
        cov_target = cov_target - torch.diag(torch.diag(cov_target))

        cov_loss = (cov_pred.pow(2).sum() + cov_target.pow(2).sum()) / dim

        # Total
        total = (
            self.sim_weight * sim_loss
            + self.var_weight * var_loss
            + self.cov_weight * cov_loss
        )

        if return_components:
            components = {
                "invariance": sim_loss.item(),
                "variance": var_loss.item(),
                "covariance": cov_loss.item(),
                "total": total.item(),
            }
            return total, components

        return total


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss — redundancy reduction via cross-correlation identity.

    Computes the cross-correlation matrix C between predicted and target
    embeddings (both batch-normalized), and pushes C toward identity:

    L = Σ_i(1 - C_ii)² + λ · Σ_{i≠j} C_ij²

    The first term forces alignment (on-diagonal → 1), the second term
    forces decorrelation (off-diagonal → 0). This is equivalent to
    maximizing mutual information while minimizing redundancy.

    Advantages for fMRI decoding:
    1. Batch-norm on features acts as implicit whitening
    2. No collapse without explicit variance regularization
    3. Scale-invariant (important when CLIP embedding norms vary)
    4. Naturally encourages full-rank representations

    References:
        Zbontar et al. (2021). "Barlow Twins: Self-Supervised Learning via
            Redundancy Reduction." ICML.
    """

    def __init__(self, lambda_bt: float = 0.005):
        super().__init__()
        self.lambda_bt = lambda_bt

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Barlow Twins loss.

        Args:
            pred: Predicted embeddings (B, D)
            target: Target embeddings (B, D)
            return_components: If True, return (loss, components_dict)

        Returns:
            Total loss, or (total_loss, components) if return_components=True
        """
        batch_size, dim = pred.shape

        # Batch-normalize along feature dimension
        pred_bn = (pred - pred.mean(dim=0)) / (pred.std(dim=0) + 1e-5)
        target_bn = (target - target.mean(dim=0)) / (target.std(dim=0) + 1e-5)

        # Cross-correlation matrix C: (D, D)
        c = (pred_bn.T @ target_bn) / batch_size

        # On-diagonal: push toward 1
        on_diag = ((1 - torch.diag(c)).pow(2)).sum()

        # Off-diagonal: push toward 0
        off_diag_mask = ~torch.eye(dim, dtype=torch.bool, device=c.device)
        off_diag = (c[off_diag_mask].pow(2)).sum()

        total = on_diag + self.lambda_bt * off_diag

        if return_components:
            components = {
                "on_diag": on_diag.item(),
                "off_diag": off_diag.item(),
                "total": total.item(),
            }
            return total, components

        return total


# ==========================================================================
# Triplet Loss with Hard-Negative Mining
# ==========================================================================

def triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Triplet margin loss using cosine distance.

    L = max(0, d(a, p) - d(a, n) + margin)

    where d(x, y) = 1 - cos(x, y).

    Args:
        anchor: Anchor embeddings (B, D)
        positive: Positive embeddings (B, D)
        negative: Negative embeddings (B, D)
        margin: Minimum margin between pos and neg distances

    Returns:
        Scalar loss averaged over batch
    """
    d_ap = 1.0 - F.cosine_similarity(anchor, positive, dim=-1)
    d_an = 1.0 - F.cosine_similarity(anchor, negative, dim=-1)
    return F.relu(d_ap - d_an + margin).mean()


class HardNegativeMiner:
    """
    Mines hard and semi-hard negatives from a similarity matrix.

    Mining strategies:
    - **hardest**: closest wrong match (most confusing negative)
    - **semi-hard**: within margin of positive but further than positive
    - **curriculum**: start random → semi-hard → hard over training

    Semi-hard mining (Schroff et al., 2015) is preferred because:
    - Hard negatives can cause training collapse early on
    - Semi-hard negatives provide informative but stable gradients
    - Curriculum bridges random→hard smoothly

    References:
        Schroff, Kalenichenko, Philbin (2015). "FaceNet: A Unified Embedding
            for Face Recognition and Clustering." CVPR.
    """

    def __init__(
        self,
        strategy: str = "semi_hard",
        margin: float = 0.2,
    ):
        self.strategy = strategy
        self.margin = margin

    def mine(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        gallery: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mine negative samples from gallery.

        Args:
            anchor: Anchor embeddings (B, D)
            positive: Positive (matched) embeddings (B, D)
            gallery: Full gallery of negatives (N, D)

        Returns:
            Mined negative embeddings (B, D)
        """
        # Similarity between anchors and all gallery items
        sim_ag = F.cosine_similarity(
            anchor.unsqueeze(1), gallery.unsqueeze(0), dim=-1
        )  # (B, N)

        # Positive similarity
        sim_ap = F.cosine_similarity(anchor, positive, dim=-1)  # (B,)

        if self.strategy == "hardest":
            # Hardest negative: highest similarity to anchor (closest wrong match)
            neg_idx = sim_ag.argmax(dim=-1)  # (B,)

        elif self.strategy == "semi_hard":
            # Semi-hard: closer than positive but beyond margin
            # d(a,n) < d(a,p) + margin AND d(a,n) > d(a,p)
            # In similarity space: sim(a,n) > sim(a,p) - margin AND sim(a,n) < sim(a,p)
            upper = sim_ap.unsqueeze(1)  # (B, 1)
            lower = sim_ap.unsqueeze(1) - self.margin
            mask = (sim_ag < upper) & (sim_ag > lower)

            # For samples with no valid semi-hard negatives, fall back to hardest
            has_valid = mask.any(dim=-1)

            # Among valid, pick the one with highest similarity (hardest semi-hard)
            sim_masked = sim_ag.clone()
            sim_masked[~mask] = -float("inf")
            neg_idx = sim_masked.argmax(dim=-1)

            # Fallback: hardest negative for samples with no semi-hard
            if not has_valid.all():
                fallback_idx = sim_ag.argmax(dim=-1)
                neg_idx[~has_valid] = fallback_idx[~has_valid]

        else:
            # Random negatives
            neg_idx = torch.randint(0, gallery.shape[0], (anchor.shape[0],),
                                    device=anchor.device)

        return gallery[neg_idx]


class TripletInfoNCELoss(nn.Module):
    """
    Hybrid loss: InfoNCE + triplet with hard-negative mining.

    Combines the breadth of InfoNCE (all-pairs within batch) with the
    targeted precision of triplet loss (mined hard negatives). This
    addresses the Pareto front between alignment and retrieval observed
    in the project:

    L = α · L_InfoNCE + β · L_triplet_hard

    - InfoNCE provides broad discriminative signal
    - Triplet with hard negatives focuses on decision boundary
    - Together they improve R@1 without sacrificing cosine alignment

    Args:
        infonce_weight: Weight for InfoNCE component (default: 0.5)
        triplet_weight: Weight for triplet component (default: 0.5)
        temperature: InfoNCE temperature (default: 0.07)
        margin: Triplet margin (default: 0.2)
        mining_strategy: 'semi_hard', 'hardest', or 'random'
    """

    def __init__(
        self,
        infonce_weight: float = 0.5,
        triplet_weight: float = 0.5,
        temperature: float = 0.07,
        margin: float = 0.2,
        mining_strategy: str = "semi_hard",
    ):
        super().__init__()
        self.infonce_weight = infonce_weight
        self.triplet_weight = triplet_weight
        self.temperature = temperature
        self.margin = margin
        self.miner = HardNegativeMiner(strategy=mining_strategy, margin=margin)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        gallery: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid InfoNCE + triplet loss.

        Args:
            pred: Predicted embeddings (B, D)
            target: Target embeddings (B, D)
            gallery: Optional external gallery for hard-negative mining.
                     If None, uses in-batch targets as gallery.
            return_components: If True, return (loss, components)

        Returns:
            Total loss, or (total_loss, components) if return_components=True
        """
        # InfoNCE component
        loss_nce = info_nce_loss(pred, target, temperature=self.temperature)

        # Triplet component with hard-negative mining
        if gallery is None:
            # Use in-batch targets as gallery (exclude matched pairs)
            gallery = target.detach()

        # Mine hard negatives
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        gallery_norm = F.normalize(gallery, p=2, dim=-1)

        negatives = self.miner.mine(pred_norm, target_norm, gallery_norm)
        loss_triplet = triplet_margin_loss(
            pred_norm, target_norm, negatives, margin=self.margin
        )

        total = self.infonce_weight * loss_nce + self.triplet_weight * loss_triplet

        if return_components:
            components = {
                "info_nce": loss_nce.item(),
                "triplet": loss_triplet.item(),
                "total": total.item(),
            }
            return total, components

        return total