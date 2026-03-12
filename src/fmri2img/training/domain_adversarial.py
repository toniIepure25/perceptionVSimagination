"""
Domain-Adversarial Training for Perception→Imagery Transfer
============================================================

Implements gradient reversal (Ganin et al., 2016) for learning
domain-invariant fMRI representations. The key idea: a domain
classifier tries to distinguish perception from imagery latent
representations, while the encoder is trained to *fool* the
classifier via gradient reversal — forcing the latent space
to be domain-invariant.

This directly tests the hypothesis that a domain-invariant
representation preserves decodable information across both
conditions while discarding condition-specific noise.

Architecture:
    fMRI input → [Encoder] → latent h ─┬─→ [CLIP Head]  → ŷ_clip
                                        │
                                        └─→ [GRL] → [Domain Classifier] → ŷ_domain
                                              ↑
                                      gradient reversal: -α·∇

The gradient reversal layer (GRL) acts as identity during forward
pass but negates and scales gradients during backprop. This creates
an adversarial game:
    - Domain classifier minimizes domain classification loss
    - Encoder maximizes domain classification loss (via GRL)
    - Result: encoder learns representations that can't distinguish
      perception from imagery

Lambda scheduling follows DANN (Ganin et al., 2016):
    λ(p) = 2 / (1 + exp(-γ·p)) - 1
where p ∈ [0, 1] is training progress and γ = 10.

References:
    Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks."
        JMLR 17(1). https://arxiv.org/abs/1505.07818
    Tzeng et al. (2017). "Adversarial Discriminative Domain Adaptation."
        CVPR.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradientReversalFunction(torch.autograd.Function):
    """
    Autograd function for gradient reversal.

    Forward: identity (pass-through).
    Backward: multiply gradients by -alpha.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (GRL) — identity on forward, negates on backward.

    The scaling factor `alpha` controls the strength of gradient reversal:
    - alpha=0: no adversarial effect (standard forward/backward)
    - alpha=1: full gradient negation
    - alpha>1: amplified adversarial gradient (aggressive domain confusion)

    In practice, alpha is scheduled from 0 → 1 over training to stabilize
    early optimization (the encoder hasn't learned useful features yet).

    Args:
        alpha: Gradient reversal strength (default: 1.0)

    Example:
        >>> grl = GradientReversalLayer(alpha=1.0)
        >>> x = torch.randn(32, 512, requires_grad=True)
        >>> y = grl(x)  # Forward: y = x
        >>> y.sum().backward()  # Backward: x.grad = -1.0 * ones
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        """Update reversal strength (call per-epoch)."""
        self.alpha = alpha


# ---------------------------------------------------------------------------
# Domain Classifier
# ---------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """
    Binary domain classifier: perception (0) vs imagery (1).

    Architecture: GRL → Linear → ReLU → Dropout → Linear → Sigmoid

    The classifier receives encoder latent representations and predicts
    the domain (condition) label. Connected through GRL so encoder
    gradients push representations toward domain-invariance.

    Design choices:
    - 2-layer MLP (following DANN paper) — sufficient for binary classification
    - Dropout for regularization (domain classification is an easy task)
    - Small hidden dim relative to input (compression forces generalization)
    - Separate from encoder to allow independent learning rate

    Args:
        input_dim: Dimension of encoder latent (default: 512 for ResidualMLP)
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout rate (default: 0.3)
        alpha: Initial GRL alpha (default: 1.0)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize with small weights for stable start
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict domain probability from encoder latent.

        Args:
            x: Encoder latent representations, shape (B, input_dim)

        Returns:
            Domain logits (pre-sigmoid), shape (B, 1)
        """
        return self.classifier(self.grl(x))

    def set_alpha(self, alpha: float) -> None:
        """Update GRL reversal strength."""
        self.grl.set_alpha(alpha)

    def get_alpha(self) -> float:
        """Get current GRL alpha."""
        return self.grl.alpha


# ---------------------------------------------------------------------------
# Lambda scheduling (DANN-style)
# ---------------------------------------------------------------------------

def dann_lambda_schedule(
    current_epoch: int,
    total_epochs: int,
    gamma: float = 10.0,
) -> float:
    """
    DANN lambda schedule: smooth sigmoid ramp from 0 to 1.

    λ(p) = 2 / (1 + exp(-γ·p)) - 1

    where p = current_epoch / total_epochs ∈ [0, 1].

    This is superior to linear ramp because:
    - Slow start: encoder learns useful features before adversarial kicks in
    - Fast middle: adversarial pressure ramps up once features are stable
    - Saturating end: full strength for final fine-tuning

    Args:
        current_epoch: Current training epoch (0-indexed)
        total_epochs: Total training epochs
        gamma: Steepness of sigmoid (default: 10.0, per DANN paper)

    Returns:
        Lambda value in [0, 1]

    Example:
        >>> for epoch in range(100):
        ...     lam = dann_lambda_schedule(epoch, 100)
        ...     classifier.set_alpha(lam)
    """
    p = float(current_epoch) / max(total_epochs, 1)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def linear_lambda_schedule(
    current_epoch: int,
    warmup_epochs: int = 10,
    lambda_max: float = 1.0,
) -> float:
    """
    Simple linear warmup schedule for GRL alpha.

    More interpretable than DANN sigmoid; useful for debugging.

    Args:
        current_epoch: Current epoch
        warmup_epochs: Epochs to ramp from 0 to lambda_max
        lambda_max: Maximum lambda value

    Returns:
        Lambda value in [0, lambda_max]
    """
    if current_epoch >= warmup_epochs:
        return lambda_max
    return lambda_max * current_epoch / max(warmup_epochs, 1)


# ---------------------------------------------------------------------------
# Domain-Adversarial Loss
# ---------------------------------------------------------------------------

class DomainAdversarialLoss(nn.Module):
    """
    Combined loss: CLIP alignment + adversarial domain confusion.

    L_total = L_clip + λ · L_domain

    where L_domain is binary cross-entropy on domain prediction, and λ
    is scheduled via DANN sigmoid ramp. The gradient reversal layer
    inside DomainClassifier ensures gradients from L_domain push the
    encoder toward domain-invariant representations.

    Tracks domain classification accuracy as a monitor:
    - High accuracy → domain is still distinguishable → λ too low or
      representations still domain-specific
    - ~50% accuracy → perfect domain confusion → encoder is domain-invariant
    - The sweet spot is reached when domain accuracy drops from ~100% to ~50-60%

    Args:
        domain_classifier: DomainClassifier module
        lambda_max: Maximum adversarial weight (default: 1.0)
        schedule: 'dann' (sigmoid) or 'linear' warmup
        warmup_epochs: For linear schedule, epochs to reach lambda_max
        total_epochs: For DANN schedule, total training epochs
        gamma: DANN schedule steepness (default: 10.0)

    Example:
        >>> encoder = TwoStageEncoder(input_dim=3072, latent_dim=512)
        >>> domain_cls = DomainClassifier(input_dim=512)
        >>> criterion = DomainAdversarialLoss(domain_cls, total_epochs=50)
        >>> # In training loop:
        >>> latent = encoder.backbone(fmri_input)  # (B, 512)
        >>> domain_labels = torch.cat([torch.zeros(B_perc), torch.ones(B_imag)])
        >>> loss, components = criterion(latent, domain_labels, epoch=20)
    """

    def __init__(
        self,
        domain_classifier: DomainClassifier,
        lambda_max: float = 1.0,
        schedule: str = "dann",
        warmup_epochs: int = 10,
        total_epochs: int = 50,
        gamma: float = 10.0,
    ):
        super().__init__()
        self.domain_classifier = domain_classifier
        self.lambda_max = lambda_max
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def get_lambda(self, epoch: int) -> float:
        """Get adversarial weight for current epoch."""
        if self.schedule == "dann":
            return self.lambda_max * dann_lambda_schedule(
                epoch, self.total_epochs, self.gamma
            )
        elif self.schedule == "linear":
            return linear_lambda_schedule(epoch, self.warmup_epochs, self.lambda_max)
        else:
            return self.lambda_max

    def forward(
        self,
        latent: torch.Tensor,
        domain_labels: torch.Tensor,
        epoch: int = 0,
        return_components: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute domain adversarial loss.

        Args:
            latent: Encoder latent representations, shape (B, latent_dim)
                    Mixed batch of perception and imagery samples.
            domain_labels: Binary labels, shape (B,). 0=perception, 1=imagery.
            epoch: Current training epoch for lambda scheduling.
            return_components: If True, return loss components dict.

        Returns:
            (domain_loss, components_dict)
            - domain_loss: Weighted adversarial loss (λ · L_domain)
            - components_dict: {
                "domain_loss_raw": unweighted BCE,
                "domain_lambda": current λ,
                "domain_loss_weighted": λ · BCE,
                "domain_accuracy": classification accuracy (monitor),
              }
        """
        # Update GRL alpha
        lam = self.get_lambda(epoch)
        self.domain_classifier.set_alpha(lam)

        # Domain prediction
        domain_logits = self.domain_classifier(latent).squeeze(-1)  # (B,)
        domain_loss = self.bce(domain_logits, domain_labels.float())

        # Weighted loss
        weighted_loss = lam * domain_loss

        # Monitor: domain classification accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(domain_logits) > 0.5).float()
            accuracy = (preds == domain_labels.float()).float().mean().item()

        components = {
            "domain_loss_raw": domain_loss.item(),
            "domain_lambda": lam,
            "domain_loss_weighted": weighted_loss.item(),
            "domain_accuracy": accuracy,
        }

        if return_components:
            return weighted_loss, components
        return weighted_loss, components


# ---------------------------------------------------------------------------
# Helper: build DANN components from config
# ---------------------------------------------------------------------------

def build_domain_adversarial(
    latent_dim: int = 512,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    lambda_max: float = 1.0,
    schedule: str = "dann",
    total_epochs: int = 50,
    warmup_epochs: int = 10,
    gamma: float = 10.0,
) -> Tuple[DomainClassifier, DomainAdversarialLoss]:
    """
    Factory function: build domain classifier and adversarial loss.

    Returns:
        (domain_classifier, adversarial_loss) ready for training.
    """
    classifier = DomainClassifier(
        input_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    loss = DomainAdversarialLoss(
        domain_classifier=classifier,
        lambda_max=lambda_max,
        schedule=schedule,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        gamma=gamma,
    )
    return classifier, loss
