"""
Centered Kernel Alignment (CKA) for Representational Geometry
=============================================================

Implements CKA (Kornblith et al., 2019) for comparing neural representations
across decoder layers and brain regions. Key novelty: applying CKA to compare
fMRI decoder layer hierarchy with cortical hierarchy, and quantifying how
this alignment changes between perception and mental imagery.

CKA measures representational similarity between two sets of activations
X ∈ R^{n×p} and Y ∈ R^{n×q} using the Hilbert-Schmidt Independence
Criterion (HSIC):

    CKA(K, L) = HSIC(K, L) / √(HSIC(K, K) · HSIC(L, L))

where K = X X^T and L = Y Y^T are Gram matrices (linear kernel).

Key properties:
- Invariant to orthogonal transforms and isotropic scaling
- Not invariant to invertible linear transforms (unlike CCA)
- Ranges from [0, 1] with 1 meaning identical representations

Includes:
- Linear CKA (O(n²p + n²q) time, efficient for large feature dims)
- Debiased CKA (Nguyen et al., 2021) for finite-sample correction
- RBF kernel CKA for nonlinear comparisons
- Cross-condition CKA for perception vs imagery hierarchy comparison
- Permutation-based significance testing

References:
    Kornblith et al. (2019). "Similarity of Neural Network Representations
        Revisited." ICML. https://arxiv.org/abs/1905.00414
    Nguyen et al. (2021). "Do Wide Neural Networks Suffer from the Curse
        of Dimensionality?" NeurIPS.
    Raghu et al. (2021). "Do Vision Transformers See Like Convolutional
        Neural Networks?" NeurIPS.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core CKA implementations
# ---------------------------------------------------------------------------

def _center_gram(K: np.ndarray) -> np.ndarray:
    """Center Gram matrix: H K H where H = I - 1/n 11^T."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic_biased(K: np.ndarray, L: np.ndarray) -> float:
    """
    Biased HSIC estimator: (1/n²) tr(HKHKL).

    This is the standard estimator used in Kornblith et al. (2019).
    """
    n = K.shape[0]
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    return float(np.trace(Kc @ Lc) / (n * n))


def _hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """
    Unbiased HSIC estimator (Song et al., 2012).

    Corrects for finite-sample bias in HSIC estimation, critical when
    n < 2p (common in fMRI where n_samples << n_voxels).

    HSIC_1 = [tr(K̃L̃) + 1^T K̃ 1 · 1^T L̃ 1 / ((n-1)(n-2))
              - 2 · 1^T K̃ L̃ 1 / (n-2)] / (n(n-3))

    where K̃ = K with diagonal zeroed out.
    """
    n = K.shape[0]
    if n < 4:
        logger.warning("Unbiased HSIC requires n >= 4, falling back to biased.")
        return _hsic_biased(K, L)

    # Zero out diagonals
    Kt = K.copy()
    Lt = L.copy()
    np.fill_diagonal(Kt, 0.0)
    np.fill_diagonal(Lt, 0.0)

    # Three terms of the unbiased estimator
    term1 = np.trace(Kt @ Lt)
    term2 = (Kt.sum() * Lt.sum()) / ((n - 1) * (n - 2))
    term3 = 2.0 * (Kt @ Lt).sum() / (n - 2)

    return float((term1 + term2 - term3) / (n * (n - 3)))


def linear_cka(
    X: np.ndarray,
    Y: np.ndarray,
    debiased: bool = False,
) -> float:
    """
    Compute linear CKA between two representation matrices.

    Uses the efficient formulation that avoids forming n×n Gram matrices
    when feature dimensions p, q << n:

        CKA_linear = ‖Y^T X‖²_F / (‖X^T X‖_F · ‖Y^T Y‖_F)

    For the biased estimator. Falls back to Gram-based computation
    for the debiased estimator.

    Args:
        X: First representation matrix, shape (n_samples, p_features)
        Y: Second representation matrix, shape (n_samples, q_features)
        debiased: If True, use unbiased HSIC (Nguyen et al., 2021)

    Returns:
        CKA similarity in [0, 1]. Returns 1.0 for identical representations.

    Example:
        >>> X = np.random.randn(100, 768)
        >>> Y = X @ np.random.randn(768, 512)  # linear transform of X
        >>> cka = linear_cka(X, Y)  # Should be ~1.0
    """
    assert X.shape[0] == Y.shape[0], (
        f"Sample dimension mismatch: X has {X.shape[0]}, Y has {Y.shape[0]}"
    )
    n = X.shape[0]

    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    if debiased:
        # Must form Gram matrices for unbiased HSIC
        K = X @ X.T
        L = Y @ Y.T
        hsic_xy = _hsic_unbiased(K, L)
        hsic_xx = _hsic_unbiased(K, K)
        hsic_yy = _hsic_unbiased(L, L)
    else:
        # Efficient: avoid n×n Gram matrices
        # HSIC_biased ∝ ‖Y^T X‖²_F  (up to centering normalization)
        # After column-centering, K = XX^T is already centered
        hsic_xy = np.linalg.norm(Y.T @ X, "fro") ** 2 / (n - 1) ** 2
        hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2 / (n - 1) ** 2
        hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2 / (n - 1) ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        logger.warning("CKA denominator near zero — one representation is constant.")
        return 0.0

    return float(np.clip(hsic_xy / denom, 0.0, 1.0))


def rbf_cka(
    X: np.ndarray,
    Y: np.ndarray,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
    debiased: bool = False,
) -> float:
    """
    Compute CKA with RBF (Gaussian) kernels for nonlinear comparison.

    K_ij = exp(-‖x_i - x_j‖² / (2σ²))

    Uses median heuristic for bandwidth selection when sigma is None:
        σ = median(pairwise distances) / √2

    The RBF kernel captures nonlinear representational structure that
    linear CKA misses. Important for detecting:
    - Nonlinear manifold structure in imagery vs perception embeddings
    - Cluster-level similarity when point-wise alignment is poor

    Args:
        X: First representations, shape (n_samples, p_features)
        Y: Second representations, shape (n_samples, q_features)
        sigma_x: RBF bandwidth for X (None = median heuristic)
        sigma_y: RBF bandwidth for Y (None = median heuristic)
        debiased: If True, use unbiased HSIC estimator

    Returns:
        CKA similarity in [0, 1]
    """
    assert X.shape[0] == Y.shape[0]

    def _rbf_gram(Z: np.ndarray, sigma: Optional[float]) -> np.ndarray:
        dists = squareform(pdist(Z, "sqeuclidean"))
        if sigma is None:
            # Median heuristic
            triu = dists[np.triu_indices_from(dists, k=1)]
            sigma = float(np.sqrt(np.median(triu) / 2.0))
            if sigma < 1e-10:
                sigma = 1.0
        return np.exp(-dists / (2.0 * sigma ** 2))

    K = _rbf_gram(X, sigma_x)
    L = _rbf_gram(Y, sigma_y)

    hsic_fn = _hsic_unbiased if debiased else _hsic_biased
    hsic_xy = hsic_fn(K, L)
    hsic_xx = hsic_fn(K, K)
    hsic_yy = hsic_fn(L, L)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0

    return float(np.clip(hsic_xy / denom, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Matrix & cross-condition analysis
# ---------------------------------------------------------------------------

@dataclass
class CKAResult:
    """Result container for CKA analysis."""

    matrix: np.ndarray  # (n_layers, n_layers) CKA similarity matrix
    layer_names: List[str]
    pvalue_matrix: Optional[np.ndarray] = None  # permutation p-values
    debiased: bool = False
    kernel: str = "linear"

    # Cross-condition results (when comparing perception vs imagery)
    perception_matrix: Optional[np.ndarray] = None
    imagery_matrix: Optional[np.ndarray] = None
    cross_condition_matrix: Optional[np.ndarray] = None

    # Summary statistics
    diagonal_mean: float = 0.0  # mean CKA on diagonal (self-similarity)
    off_diagonal_mean: float = 0.0  # mean CKA off-diagonal
    hierarchy_score: float = 0.0  # how "hierarchical" the matrix is

    def compute_summary(self) -> None:
        """Compute summary statistics from the CKA matrix."""
        n = self.matrix.shape[0]
        diag = np.diag(self.matrix)
        mask = ~np.eye(n, dtype=bool)
        self.diagonal_mean = float(np.mean(diag))
        self.off_diagonal_mean = float(np.mean(self.matrix[mask]))

        # Hierarchy score: how much does CKA decay with layer distance?
        # Higher = more hierarchically organized (nearby layers more similar)
        if n > 2:
            distances = []
            cka_values = []
            for i in range(n):
                for j in range(i + 1, n):
                    distances.append(abs(i - j))
                    cka_values.append(self.matrix[i, j])
            from scipy.stats import spearmanr
            corr, _ = spearmanr(distances, cka_values)
            # Negative correlation = hierarchical (nearby = high CKA)
            self.hierarchy_score = float(-corr)


def compute_cka_matrix(
    representations: Dict[str, np.ndarray],
    debiased: bool = False,
    kernel: str = "linear",
) -> CKAResult:
    """
    Compute pairwise CKA matrix across a set of named representations.

    Produces an (n_layers × n_layers) similarity matrix suitable for
    heatmap visualization. Can be used to compare:
    - Decoder layers with each other
    - Decoder layers with brain ROI activations
    - Perception representations with imagery representations

    Args:
        representations: Dict mapping layer names to activation matrices,
            each of shape (n_samples, n_features). All must have the same
            n_samples dimension.
        debiased: Use debiased CKA estimator
        kernel: 'linear' or 'rbf'

    Returns:
        CKAResult with the similarity matrix and metadata.

    Example:
        >>> reps = {
        ...     'layer_12': model_l12_activations,  # (N, 768)
        ...     'layer_18': model_l18_activations,  # (N, 768)
        ...     'final': model_final_activations,    # (N, 768)
        ...     'V1': brain_v1_activations,          # (N, n_v1)
        ...     'V4': brain_v4_activations,          # (N, n_v4)
        ... }
        >>> result = compute_cka_matrix(reps, debiased=True)
        >>> print(result.matrix)  # 5×5 CKA symmetric matrix
    """
    names = list(representations.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    cka_fn = rbf_cka if kernel == "rbf" else linear_cka

    for i in range(n):
        for j in range(i, n):
            cka_val = cka_fn(
                representations[names[i]],
                representations[names[j]],
                debiased=debiased,
            )
            matrix[i, j] = cka_val
            matrix[j, i] = cka_val

    result = CKAResult(
        matrix=matrix,
        layer_names=names,
        debiased=debiased,
        kernel=kernel,
    )
    result.compute_summary()
    return result


def compute_cka_significance(
    representations: Dict[str, np.ndarray],
    n_permutations: int = 1000,
    debiased: bool = False,
    kernel: str = "linear",
    seed: int = 42,
) -> CKAResult:
    """
    Compute CKA matrix with permutation-based p-values.

    For each pair (i, j), shuffles samples in representation j and
    recomputes CKA n_permutations times. The p-value is the fraction
    of permuted CKA values ≥ observed CKA.

    This is essential for establishing that observed CKA between
    decoder layers and brain ROIs is above chance.

    Args:
        representations: Dict of named representations
        n_permutations: Number of permutations (default: 1000)
        debiased: Use debiased CKA
        kernel: 'linear' or 'rbf'
        seed: Random seed for reproducibility

    Returns:
        CKAResult with both similarity matrix and p-value matrix.
    """
    rng = np.random.RandomState(seed)
    result = compute_cka_matrix(representations, debiased=debiased, kernel=kernel)

    names = result.layer_names
    n_layers = len(names)
    n_samples = representations[names[0]].shape[0]
    pvalues = np.zeros((n_layers, n_layers))
    cka_fn = rbf_cka if kernel == "rbf" else linear_cka

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            observed = result.matrix[i, j]
            count = 0
            X_i = representations[names[i]]
            X_j = representations[names[j]]
            for _ in range(n_permutations):
                perm = rng.permutation(n_samples)
                perm_cka = cka_fn(X_i, X_j[perm], debiased=debiased)
                if perm_cka >= observed:
                    count += 1
            p = (count + 1) / (n_permutations + 1)  # +1 for conservative estimate
            pvalues[i, j] = p
            pvalues[j, i] = p

    result.pvalue_matrix = pvalues
    return result


# ---------------------------------------------------------------------------
# Cross-condition CKA (perception vs imagery)
# ---------------------------------------------------------------------------

def compute_cross_condition_cka(
    bundle: EmbeddingBundle,
    debiased: bool = True,
    kernel: str = "linear",
) -> CKAResult:
    """
    Compare representational hierarchy between perception and imagery.

    This is the key novel analysis: computes CKA matrices separately
    for perception and imagery multi-layer outputs, then computes a
    cross-condition CKA matrix. The difference reveals how imagery
    "disrupts" the decoder's representational hierarchy.

    Hypothesis (Dijkstra et al., 2019): Imagery representations are
    more similar to late-layer (semantic) representations, while
    perception engages a full hierarchy from early (structural) to
    late (semantic) layers.

    Requires `bundle.multilayer_perception` and `bundle.multilayer_imagery`
    to be populated.

    Args:
        bundle: EmbeddingBundle with multi-layer embeddings
        debiased: Use debiased CKA (recommended for fMRI)
        kernel: 'linear' or 'rbf'

    Returns:
        CKAResult with three matrices:
        - perception_matrix: layer×layer CKA within perception
        - imagery_matrix: layer×layer CKA within imagery
        - cross_condition_matrix: layer×layer CKA across conditions
        - matrix: set to the cross_condition_matrix for convenience
    """
    if bundle.multilayer_perception is None or bundle.multilayer_imagery is None:
        raise ValueError(
            "EmbeddingBundle must have multilayer_perception and multilayer_imagery "
            "populated. Use a multi-layer encoder model."
        )

    perc = bundle.multilayer_perception
    imag = bundle.multilayer_imagery

    # Ensure same layer names
    common_layers = sorted(set(perc.keys()) & set(imag.keys()))
    if not common_layers:
        raise ValueError("No common layers between perception and imagery embeddings.")

    logger.info(
        f"Computing cross-condition CKA for {len(common_layers)} layers: {common_layers}"
    )

    # Within-condition CKA
    perc_result = compute_cka_matrix(
        {k: perc[k] for k in common_layers}, debiased=debiased, kernel=kernel
    )
    imag_result = compute_cka_matrix(
        {k: imag[k] for k in common_layers}, debiased=debiased, kernel=kernel
    )

    # Cross-condition CKA: CKA(perc_layer_i, imag_layer_j)
    n = len(common_layers)
    cross_matrix = np.zeros((n, n))
    cka_fn = rbf_cka if kernel == "rbf" else linear_cka

    # Need matched samples for cross-condition CKA
    # Use shared stimuli if available, otherwise use minimum N
    if bundle.perception_nsd_ids is not None and bundle.imagery_nsd_ids is not None:
        pairs = bundle.get_shared_stimulus_pairs()
        if pairs is not None:
            shared_ids, p_idx, i_idx = pairs
            logger.info(f"Using {len(shared_ids)} shared stimuli for cross-condition CKA")
            for i_layer in range(n):
                for j_layer in range(n):
                    X = perc[common_layers[i_layer]][p_idx]
                    Y = imag[common_layers[j_layer]][i_idx]
                    cross_matrix[i_layer, j_layer] = cka_fn(X, Y, debiased=debiased)
        else:
            # No shared stimuli — use min N with random pairing
            n_p = perc[common_layers[0]].shape[0]
            n_i = imag[common_layers[0]].shape[0]
            n_use = min(n_p, n_i)
            for i_layer in range(n):
                for j_layer in range(n):
                    X = perc[common_layers[i_layer]][:n_use]
                    Y = imag[common_layers[j_layer]][:n_use]
                    cross_matrix[i_layer, j_layer] = cka_fn(X, Y, debiased=debiased)
    else:
        n_p = perc[common_layers[0]].shape[0]
        n_i = imag[common_layers[0]].shape[0]
        n_use = min(n_p, n_i)
        for i_layer in range(n):
            for j_layer in range(n):
                X = perc[common_layers[i_layer]][:n_use]
                Y = imag[common_layers[j_layer]][:n_use]
                cross_matrix[i_layer, j_layer] = cka_fn(X, Y, debiased=debiased)

    result = CKAResult(
        matrix=cross_matrix,
        layer_names=common_layers,
        debiased=debiased,
        kernel=kernel,
        perception_matrix=perc_result.matrix,
        imagery_matrix=imag_result.matrix,
        cross_condition_matrix=cross_matrix,
    )
    result.compute_summary()
    return result


# ---------------------------------------------------------------------------
# Utility: Mini CKA between two single representations
# ---------------------------------------------------------------------------

def cka_similarity(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
    debiased: bool = False,
) -> float:
    """
    Convenience wrapper: compute CKA between two representation matrices.

    Args:
        X: shape (n_samples, p_features)
        Y: shape (n_samples, q_features)
        kernel: 'linear' or 'rbf'
        debiased: Use debiased estimator

    Returns:
        CKA similarity in [0, 1]
    """
    if kernel == "rbf":
        return rbf_cka(X, Y, debiased=debiased)
    return linear_cka(X, Y, debiased=debiased)
