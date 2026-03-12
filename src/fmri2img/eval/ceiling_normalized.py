"""
Noise-Ceiling Normalized Evaluation
====================================

Provides noise-ceiling normalization for fMRI decoding metrics,
converting raw metrics into fraction-of-ceiling-achieved scores.
This is essential for neuroscience venues (Nature Neuroscience, NeuroImage)
where reviewers expect results to be contextualized against the
theoretical maximum decodable information.

The noise ceiling represents the best possible decoding performance
given the signal-to-noise ratio of the fMRI data. It is derived from
the NSD's NCSNR (noise-ceiling signal-to-noise ratio) per voxel:

    ceiling_voxel = √(SNR / (1 + SNR))

For embedding-space metrics, we propagate the voxel-level ceiling
through the PCA transform to get an embedding-space ceiling.

Normalized metric:
    m_norm = m_raw / m_ceiling

where m_norm ∈ [0, 1] indicates fraction of achievable performance.

References:
    Kay et al. (2013). "Identifying natural images from human brain
        activity." Nature.
    Allen et al. (2022). "A massive 7T fMRI dataset to bridge cognitive
        neuroscience and artificial intelligence." Nature Neuroscience.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CeilingResult:
    """Container for noise-ceiling analysis results."""

    raw_metric: float
    ceiling: float
    normalized: float  # raw / ceiling
    metric_name: str
    method: str = "ncsnr"

    @property
    def percent_of_ceiling(self) -> float:
        """Express normalized metric as percentage."""
        return self.normalized * 100.0


class NoiseCeilingNormalizer:
    """
    Normalizes decoding metrics against noise ceiling.

    Loads per-voxel NCSNR from NSD data, computes ceiling, and
    provides normalization methods for various metric types.

    Usage:
        >>> normalizer = NoiseCeilingNormalizer(subject="subj01")
        >>> result = normalizer.normalize("cosine", raw_cosine=0.81)
        >>> print(f"Achieved {result.percent_of_ceiling:.1f}% of ceiling")

    Args:
        subject: NSD subject ID (e.g., "subj01")
        ncsnr: Pre-loaded NCSNR array (optional, will load from disk if None)
        data_root: Root directory for NSD data
        method: Ceiling computation method ('standard', 'correlation', 'linear')
    """

    def __init__(
        self,
        subject: str = "subj01",
        ncsnr: Optional[np.ndarray] = None,
        data_root: str = "data",
        method: str = "standard",
    ):
        self.subject = subject
        self.method = method
        self.data_root = Path(data_root)

        if ncsnr is not None:
            self.ncsnr = ncsnr
        else:
            self.ncsnr = self._load_ncsnr()

        # Compute per-voxel ceiling
        if self.ncsnr is not None:
            self.voxel_ceiling = self._compute_voxel_ceiling()
            self.aggregate_ceiling = float(np.mean(self.voxel_ceiling))
        else:
            self.voxel_ceiling = None
            self.aggregate_ceiling = None
            logger.warning(
                "No NCSNR data available. Using theoretical bounds for normalization."
            )

    def _load_ncsnr(self) -> Optional[np.ndarray]:
        """Attempt to load NCSNR from standard NSD paths."""
        try:
            from ..reliability.noise_ceiling import load_ncsnr
            ncsnr = load_ncsnr(self.subject, data_root=str(self.data_root))
            if ncsnr is not None:
                logger.info(f"Loaded NCSNR for {self.subject}: {len(ncsnr)} voxels")
            return ncsnr
        except Exception as e:
            logger.warning(f"Could not load NCSNR: {e}")
            return None

    def _compute_voxel_ceiling(self) -> np.ndarray:
        """
        Compute per-voxel noise ceiling from NCSNR.

        ceiling = √(SNR / (1 + SNR))

        This represents the maximum correlation between a single-trial
        response and the true underlying signal.
        """
        snr = self.ncsnr ** 2  # NCSNR is √SNR
        ceiling = np.sqrt(snr / (1.0 + snr))
        return np.clip(ceiling, 0.0, 1.0)

    def normalize(
        self,
        metric_name: str,
        raw_value: float,
        ceiling_override: Optional[float] = None,
    ) -> CeilingResult:
        """
        Normalize a raw metric against noise ceiling.

        For cosine similarity and correlation metrics, the ceiling is
        the aggregate voxel ceiling. For retrieval metrics (R@K), the
        ceiling is estimated from the embedding-space ceiling.

        Args:
            metric_name: Name of the metric (e.g., 'cosine', 'R@1')
            raw_value: Raw metric value
            ceiling_override: Use this ceiling instead of computed one

        Returns:
            CeilingResult with raw, ceiling, and normalized values
        """
        if ceiling_override is not None:
            ceiling = ceiling_override
        elif self.aggregate_ceiling is not None:
            ceiling = self._estimate_metric_ceiling(metric_name)
        else:
            # Theoretical bounds
            ceiling = self._theoretical_ceiling(metric_name)

        normalized = raw_value / max(ceiling, 1e-8)
        normalized = min(normalized, 1.0)  # Cap at 1.0

        return CeilingResult(
            raw_metric=raw_value,
            ceiling=ceiling,
            normalized=normalized,
            metric_name=metric_name,
            method=self.method,
        )

    def _estimate_metric_ceiling(self, metric_name: str) -> float:
        """Estimate ceiling for specific metric types."""
        if metric_name.lower() in ("cosine", "correlation", "clip_score"):
            return self.aggregate_ceiling
        elif metric_name.lower().startswith("r@"):
            # For retrieval: ceiling is harder to estimate analytically
            # Use a conservative estimate based on embedding-space ceiling
            return min(self.aggregate_ceiling ** 2, 1.0)
        elif metric_name.lower() == "mrr":
            return min(self.aggregate_ceiling ** 2, 1.0)
        else:
            return self.aggregate_ceiling

    def _theoretical_ceiling(self, metric_name: str) -> float:
        """Fallback theoretical ceilings when no NCSNR available."""
        # Conservative estimates from NSD literature
        ceilings = {
            "cosine": 0.95,
            "correlation": 0.95,
            "clip_score": 0.95,
            "r@1": 0.80,
            "r@5": 0.95,
            "r@10": 0.98,
            "mrr": 0.85,
        }
        return ceilings.get(metric_name.lower(), 0.95)

    def normalize_results_dict(
        self,
        results: Dict[str, float],
    ) -> Dict[str, CeilingResult]:
        """
        Normalize all metrics in a results dictionary.

        Args:
            results: Dict mapping metric names to raw values

        Returns:
            Dict mapping metric names to CeilingResult objects
        """
        return {
            name: self.normalize(name, value)
            for name, value in results.items()
        }


def compute_embedding_noise_ceiling(
    ncsnr: np.ndarray,
    pca_components: np.ndarray,
    method: str = "propagated",
) -> float:
    """
    Propagate voxel-level noise ceiling through PCA to embedding space.

    Given per-voxel NCSNR and PCA transform W, the embedding-space
    noise ceiling is estimated by propagating voxel-level signal variance
    through the linear PCA transform.

    Method 'propagated':
        σ²_emb = W^T · diag(σ²_signal_voxel) · W
        ceiling ≈ mean(σ_emb / (σ_emb + σ_noise_emb))

    Method 'weighted_mean':
        ceiling ≈ Σ_v |w_v|² · ceiling_v / Σ_v |w_v|²

    Args:
        ncsnr: Per-voxel NCSNR, shape (n_voxels,)
        pca_components: PCA components matrix, shape (n_components, n_voxels)
        method: 'propagated' or 'weighted_mean'

    Returns:
        Scalar embedding-space noise ceiling in [0, 1]
    """
    snr = ncsnr ** 2
    voxel_ceiling = np.sqrt(snr / (1.0 + snr))

    if method == "weighted_mean":
        # Weight each voxel's ceiling by its PCA loading magnitude
        weights = np.sum(pca_components ** 2, axis=0)  # (n_voxels,)
        weights = weights / (weights.sum() + 1e-10)
        return float(np.sum(weights * voxel_ceiling))

    elif method == "propagated":
        # Propagate signal variance through PCA
        signal_var = snr / (1.0 + snr)  # fraction of variance that is signal
        # PCA-space signal variance per component
        pca_signal_var = np.sum(
            pca_components ** 2 * signal_var[np.newaxis, :], axis=1
        )
        pca_total_var = np.sum(pca_components ** 2, axis=1)
        # Per-component ceiling
        component_ceiling = np.sqrt(
            pca_signal_var / (pca_total_var + 1e-10)
        )
        return float(np.mean(np.clip(component_ceiling, 0.0, 1.0)))

    else:
        raise ValueError(f"Unknown method: {method}")
