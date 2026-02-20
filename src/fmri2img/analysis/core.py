"""
Shared utilities for novel perception-vs-imagery analyses.

Provides embedding collection from both conditions, model loading,
and synthetic data generation for testing without real NSD-Imagery data.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingBundle:
    """Paired perception and imagery embeddings for analysis."""

    perception: np.ndarray  # (N_p, D)
    imagery: np.ndarray  # (N_i, D)
    perception_targets: np.ndarray  # (N_p, D) ground-truth CLIP
    imagery_targets: np.ndarray  # (N_i, D) ground-truth CLIP
    embed_dim: int = 512
    subject: str = "subj01"

    # Per-trial metadata
    perception_meta: List[Dict] = field(default_factory=list)
    imagery_meta: List[Dict] = field(default_factory=list)

    @property
    def perception_cosines(self) -> np.ndarray:
        p = _l2(self.perception)
        t = _l2(self.perception_targets)
        return np.sum(p * t, axis=1)

    @property
    def imagery_cosines(self) -> np.ndarray:
        p = _l2(self.imagery)
        t = _l2(self.imagery_targets)
        return np.sum(p * t, axis=1)


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


def load_model_for_analysis(
    checkpoint_path: str,
    model_type: str = "two_stage",
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a trained encoder for analysis (inference only)."""
    from fmri2img.models.encoders import load_two_stage_encoder
    from fmri2img.models.mlp import load_mlp

    if model_type == "two_stage":
        model, meta = load_two_stage_encoder(checkpoint_path, map_location=device)
    elif model_type == "mlp":
        model, meta = load_mlp(checkpoint_path, map_location=device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model = model.to(device)
    model.eval()
    return model


def collect_embeddings(
    model,
    perception_dataset,
    imagery_dataset,
    device: str = "cpu",
    max_samples: Optional[int] = None,
) -> EmbeddingBundle:
    """
    Run a model on both perception and imagery datasets, collecting
    predicted embeddings and ground-truth CLIP targets.
    """

    def _run(dataset, label: str):
        preds, targets, metas = [], [], []
        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            voxels = sample["voxels"]
            if isinstance(voxels, np.ndarray):
                voxels = torch.from_numpy(voxels).float().unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(voxels).cpu().numpy().squeeze(0)
            preds.append(pred)

            clip_target = sample.get("clip_target")
            if clip_target is not None:
                if isinstance(clip_target, torch.Tensor):
                    clip_target = clip_target.numpy()
                targets.append(clip_target)
            else:
                targets.append(np.zeros(pred.shape[-1], dtype=np.float32))

            metas.append({
                "trial_id": sample.get("trial_id"),
                "stimulus_type": sample.get("stimulus_type", "unknown"),
                "condition": sample.get("condition", label),
            })

        return np.array(preds), np.array(targets), metas

    logger.info("Collecting perception embeddings...")
    p_preds, p_targets, p_meta = _run(perception_dataset, "perception")
    logger.info(f"  Perception: {p_preds.shape[0]} samples")

    logger.info("Collecting imagery embeddings...")
    i_preds, i_targets, i_meta = _run(imagery_dataset, "imagery")
    logger.info(f"  Imagery: {i_preds.shape[0]} samples")

    return EmbeddingBundle(
        perception=p_preds,
        imagery=i_preds,
        perception_targets=p_targets,
        imagery_targets=i_targets,
        embed_dim=p_preds.shape[1],
        perception_meta=p_meta,
        imagery_meta=i_meta,
    )


def generate_synthetic_embeddings(
    n_perception: int = 500,
    n_imagery: int = 200,
    embed_dim: int = 512,
    imagery_dim_fraction: float = 0.6,
    imagery_noise_scale: float = 0.3,
    seed: int = 42,
) -> EmbeddingBundle:
    """
    Generate synthetic perception/imagery embeddings that exhibit the
    expected neuroscience properties for testing analysis pipelines.

    The synthetic imagery embeddings are constructed to occupy a
    lower-dimensional subspace with added noise, mimicking the
    hypothesized "dimensionality collapse" of mental imagery.
    """
    rng = np.random.RandomState(seed)

    # Ground-truth CLIP targets span the full space
    gt_targets = rng.randn(n_perception + n_imagery, embed_dim).astype(np.float32)
    gt_targets /= np.linalg.norm(gt_targets, axis=1, keepdims=True)

    perc_targets = gt_targets[:n_perception]
    imag_targets = gt_targets[n_perception:]

    # Perception predictions: noisy but high-dimensional
    perc_preds = perc_targets + rng.randn(n_perception, embed_dim).astype(np.float32) * 0.15
    perc_preds /= np.linalg.norm(perc_preds, axis=1, keepdims=True)

    # Imagery predictions: projected onto a lower-dimensional subspace
    effective_dim = max(10, int(embed_dim * imagery_dim_fraction))
    projection = rng.randn(embed_dim, effective_dim).astype(np.float32)
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)
    back_proj = projection @ projection.T / effective_dim

    imag_preds = (imag_targets @ back_proj) + rng.randn(n_imagery, embed_dim).astype(np.float32) * imagery_noise_scale
    imag_preds /= np.linalg.norm(imag_preds, axis=1, keepdims=True)

    # Synthetic stimulus types
    stim_types = ["simple", "complex", "conceptual"]
    perc_meta = [
        {"trial_id": i, "stimulus_type": stim_types[i % 3], "condition": "perception"}
        for i in range(n_perception)
    ]
    imag_meta = [
        {"trial_id": n_perception + i, "stimulus_type": stim_types[i % 3], "condition": "imagery"}
        for i in range(n_imagery)
    ]

    logger.info(
        f"Generated synthetic embeddings: perception={n_perception}, "
        f"imagery={n_imagery}, dim={embed_dim}, "
        f"imagery_effective_dim={effective_dim}"
    )

    return EmbeddingBundle(
        perception=perc_preds,
        imagery=imag_preds,
        perception_targets=perc_targets,
        imagery_targets=imag_targets,
        embed_dim=embed_dim,
        perception_meta=perc_meta,
        imagery_meta=imag_meta,
    )
