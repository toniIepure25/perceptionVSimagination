"""
Gradient-Based Interpretability for fMRI Decoders
=================================================

Maps decoder decisions back to brain space using attribution methods,
revealing which brain regions (voxels) the encoder relies on. This is
essential for neuroscience claims about perception vs imagery.

Key research question:
    "Does the encoder rely on different brain regions for perceiving
     vs. imagining the same stimulus?"

Methods implemented:
1. Integrated Gradients (Sundararajan et al., 2017)
2. SmoothGrad (Smilkov et al., 2017)
3. Gradient × Input (simple baseline)
4. PCA-space → Voxel-space mapping (inverse transform)

All methods produce attribution maps in PCA feature space, then
optionally map back to voxel space via PCA inverse transform.

References:
    Sundararajan, Taly, Yan (2017). "Axiomatic Attribution for Deep
        Networks." ICML.
    Smilkov et al. (2017). "SmoothGrad: removing noise by adding noise."
        Workshop on Visualization for Deep Learning, ICML.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Container for attribution analysis results."""

    attributions: np.ndarray  # (n_samples, n_features) in PCA space
    method: str
    target_type: str = "cosine"  # what was differentiated

    # Optional voxel-space attributions
    voxel_attributions: Optional[np.ndarray] = None  # (n_samples, n_voxels)

    # Summary statistics
    top_k_features: Optional[np.ndarray] = None  # indices of top-K features
    feature_importance: Optional[np.ndarray] = None  # mean |attribution| per feature


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
    target_fn: Optional[callable] = None,
) -> np.ndarray:
    """
    Compute Integrated Gradients for fMRI decoder.

    IG attributes the output to each input feature by integrating
    gradients along the path from a baseline to the input:

        IG_i(x) = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x')) / ∂x_i dα

    where x' is the baseline (default: zero vector), and the integral
    is approximated via Riemann sum with n_steps.

    Properties (axiomatic):
    - Sensitivity: if feature i differs and changes output, IG_i ≠ 0
    - Implementation invariance: same IG for functionally identical models
    - Completeness: Σ_i IG_i = F(x) - F(x')
    - Symmetry: equal IG for equally-contributing features

    Args:
        model: fMRI encoder (maps input → embedding)
        inputs: Input fMRI features, shape (B, input_dim)
        targets: Target embeddings for cosine similarity (B, embed_dim).
                 If None, uses L2 norm of output as scalar target.
        baseline: Baseline input (default: zero vector)
        n_steps: Number of interpolation steps (default: 50)
        target_fn: Custom scalar function of model output.
                   Signature: target_fn(output, targets) → scalar

    Returns:
        Attribution map, shape (B, input_dim)
    """
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    if baseline is None:
        baseline = torch.zeros_like(inputs)
    else:
        baseline = baseline.to(device)

    if targets is not None:
        targets = targets.to(device)

    # Interpolation: x' + α·(x - x') for α ∈ [0, 1]
    alphas = torch.linspace(0, 1, n_steps + 1, device=device)
    delta = inputs - baseline  # (B, D)

    # Accumulate gradients
    integrated_grads = torch.zeros_like(inputs)

    for alpha in alphas:
        # Interpolated input
        x_interp = baseline + alpha * delta
        x_interp = x_interp.detach().requires_grad_(True)

        # Forward pass
        output = model(x_interp)

        # Compute scalar target
        if target_fn is not None:
            scalar = target_fn(output, targets)
        elif targets is not None:
            # Cosine similarity with target
            out_norm = torch.nn.functional.normalize(output, p=2, dim=-1)
            tgt_norm = torch.nn.functional.normalize(targets, p=2, dim=-1)
            scalar = (out_norm * tgt_norm).sum()
        else:
            # L2 norm of output
            scalar = output.norm(p=2)

        # Backward
        scalar.backward()
        integrated_grads += x_interp.grad.detach()
        x_interp.grad = None

    # Average and multiply by delta
    integrated_grads = (integrated_grads / (n_steps + 1)) * delta

    return integrated_grads.detach().cpu().numpy()


def smooth_grad(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    n_samples: int = 50,
    noise_scale: float = 0.1,
    target_fn: Optional[callable] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute SmoothGrad attributions.

    Averages gradients over noisy copies of the input:

        SG(x) = (1/N) Σ_n ∇_x F(x + ε_n)   where ε_n ~ N(0, σ²I)

    This smooths out noisy single-sample gradients, producing more
    visually interpretable and stable attribution maps.

    Args:
        model: fMRI encoder
        inputs: Input features, shape (B, input_dim)
        targets: Target embeddings (optional)
        n_samples: Number of noisy samples (default: 50)
        noise_scale: Noise std as fraction of input range (default: 0.1)
        target_fn: Custom scalar function of model output
        seed: Random seed

    Returns:
        Attribution map, shape (B, input_dim)
    """
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    if targets is not None:
        targets = targets.to(device)

    torch.manual_seed(seed)
    noise_std = noise_scale * (inputs.max() - inputs.min()).item()
    accumulated_grads = torch.zeros_like(inputs)

    for _ in range(n_samples):
        noisy_input = inputs + torch.randn_like(inputs) * noise_std
        noisy_input = noisy_input.detach().requires_grad_(True)

        output = model(noisy_input)

        if target_fn is not None:
            scalar = target_fn(output, targets)
        elif targets is not None:
            out_norm = torch.nn.functional.normalize(output, p=2, dim=-1)
            tgt_norm = torch.nn.functional.normalize(targets, p=2, dim=-1)
            scalar = (out_norm * tgt_norm).sum()
        else:
            scalar = output.norm(p=2)

        scalar.backward()
        accumulated_grads += noisy_input.grad.detach()
        noisy_input.grad = None

    return (accumulated_grads / n_samples).cpu().numpy()


def gradient_x_input(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    target_fn: Optional[callable] = None,
) -> np.ndarray:
    """
    Gradient × Input attribution (simplest method).

    Computes ∇_x F(x) ⊙ x, which weights gradients by input magnitude.
    Fast but less principled than IG or SmoothGrad.

    Args:
        model: fMRI encoder
        inputs: Input features (B, input_dim)
        targets: Target embeddings (optional)
        target_fn: Custom scalar function

    Returns:
        Attribution map, shape (B, input_dim)
    """
    model.eval()
    device = next(model.parameters()).device
    inputs = inputs.to(device).detach().requires_grad_(True)

    if targets is not None:
        targets = targets.to(device)

    output = model(inputs)

    if target_fn is not None:
        scalar = target_fn(output, targets)
    elif targets is not None:
        out_norm = torch.nn.functional.normalize(output, p=2, dim=-1)
        tgt_norm = torch.nn.functional.normalize(targets, p=2, dim=-1)
        scalar = (out_norm * tgt_norm).sum()
    else:
        scalar = output.norm(p=2)

    scalar.backward()
    grads = inputs.grad.detach()

    return (grads * inputs.detach()).cpu().numpy()


# ---------------------------------------------------------------------------
# PCA-space → Voxel-space mapping
# ---------------------------------------------------------------------------

def attribution_to_brain_map(
    attributions: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map PCA-space attributions back to voxel space.

    Uses PCA inverse transform to convert feature-space gradients
    into voxel-space importance maps:

        g_voxel = W_PCA^T · g_PCA

    where W_PCA is the (n_components, n_voxels) PCA components matrix.

    The resulting voxel-space map can be overlaid on anatomical images
    to identify which brain regions drive the decoder's predictions.

    Args:
        attributions: PCA-space attributions (n_samples, n_components)
        pca_components: PCA components matrix (n_components, n_voxels)
        pca_mean: PCA mean vector (optional, not used for gradients)

    Returns:
        Voxel-space attributions (n_samples, n_voxels)
    """
    # Gradient mapping: just multiply by PCA components (linear chain rule)
    return attributions @ pca_components


# ---------------------------------------------------------------------------
# Cross-condition attribution comparison
# ---------------------------------------------------------------------------

def compare_condition_attributions(
    model: nn.Module,
    perception_inputs: torch.Tensor,
    imagery_inputs: torch.Tensor,
    perception_targets: Optional[torch.Tensor] = None,
    imagery_targets: Optional[torch.Tensor] = None,
    method: str = "integrated_gradients",
    pca_components: Optional[np.ndarray] = None,
    **kwargs,
) -> Dict[str, AttributionResult]:
    """
    Compare attributions between perception and imagery conditions.

    This is the key analysis: do perception and imagery trials rely on
    the same brain features? Large differences indicate the encoder
    uses different computational pathways for each condition.

    Computes attributions for both conditions, then compares:
    - Mean attribution maps (which features are important on average)
    - Attribution variance (which features are variable)
    - Feature-wise divergence (KL or L2 between attribution distributions)
    - Top-K feature overlap (do the same features dominate?)

    Args:
        model: fMRI encoder
        perception_inputs: Perception fMRI features (N_p, D)
        imagery_inputs: Imagery fMRI features (N_i, D)
        perception_targets: Optional CLIP targets for perception
        imagery_targets: Optional CLIP targets for imagery
        method: Attribution method ('integrated_gradients', 'smooth_grad',
                'gradient_x_input')
        pca_components: If provided, map to voxel space
        **kwargs: Passed to attribution function

    Returns:
        Dict with keys: 'perception', 'imagery', 'divergence'
    """
    # Select attribution method
    attr_fn = {
        "integrated_gradients": integrated_gradients,
        "smooth_grad": smooth_grad,
        "gradient_x_input": gradient_x_input,
    }[method]

    # Compute attributions
    logger.info(f"Computing {method} attributions for perception ({len(perception_inputs)} samples)")
    attr_perc = attr_fn(model, perception_inputs, perception_targets, **kwargs)

    logger.info(f"Computing {method} attributions for imagery ({len(imagery_inputs)} samples)")
    attr_imag = attr_fn(model, imagery_inputs, imagery_targets, **kwargs)

    # Summary statistics
    mean_perc = np.mean(np.abs(attr_perc), axis=0)
    mean_imag = np.mean(np.abs(attr_imag), axis=0)

    # Feature importance ranking
    top_k = 100
    top_perc = np.argsort(-mean_perc)[:top_k]
    top_imag = np.argsort(-mean_imag)[:top_k]

    # Top-K overlap
    overlap = len(set(top_perc) & set(top_imag))
    logger.info(f"Top-{top_k} feature overlap: {overlap}/{top_k}")

    # Feature-wise L2 divergence
    divergence = np.sqrt(np.mean((mean_perc - mean_imag) ** 2))
    logger.info(f"Feature-wise L2 divergence: {divergence:.6f}")

    # Build results
    perc_result = AttributionResult(
        attributions=attr_perc,
        method=method,
        feature_importance=mean_perc,
        top_k_features=top_perc,
    )
    imag_result = AttributionResult(
        attributions=attr_imag,
        method=method,
        feature_importance=mean_imag,
        top_k_features=top_imag,
    )

    # Optional voxel-space mapping
    if pca_components is not None:
        perc_result.voxel_attributions = attribution_to_brain_map(
            attr_perc, pca_components
        )
        imag_result.voxel_attributions = attribution_to_brain_map(
            attr_imag, pca_components
        )

    return {
        "perception": perc_result,
        "imagery": imag_result,
        "divergence": {
            "l2": float(divergence),
            "top_k_overlap": overlap,
            "top_k": top_k,
            "cosine_of_importance": float(
                np.dot(mean_perc, mean_imag)
                / (np.linalg.norm(mean_perc) * np.linalg.norm(mean_imag) + 1e-10)
            ),
        },
    }
