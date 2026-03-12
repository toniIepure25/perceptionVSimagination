"""
Embedding Space Visualization — UMAP / t-SNE Pipeline
=====================================================

Publication-quality visualizations of perception vs imagery embeddings
in 2D projection space. These figures are essential for any perception-
vs-imagery paper: they directly show how the two manifolds overlap,
diverge, and relate in semantic space.

Key visualizations:
1. Perception vs imagery scatter (colored by condition)
2. Category-colored overlays (MS-COCO supercategories)
3. Shared-stimulus connecting lines (paired stimuli)
4. Layer progression (how gap changes across decoder layers)
5. Density comparison (KDE of manifold overlap)
6. Ground-truth CLIP overlay as reference frame

Follows the matplotlib style of make_novel_figures.py for consistency.

Dependencies:
    - umap-learn (optional, for UMAP projections)
    - scikit-learn (for t-SNE, PCA fallback)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_HAS_UMAP = None
_HAS_MATPLOTLIB = None


def _check_umap():
    global _HAS_UMAP
    if _HAS_UMAP is None:
        try:
            import umap
            _HAS_UMAP = True
        except ImportError:
            _HAS_UMAP = False
    return _HAS_UMAP


def _check_matplotlib():
    global _HAS_MATPLOTLIB
    if _HAS_MATPLOTLIB is None:
        try:
            import matplotlib
            _HAS_MATPLOTLIB = True
        except ImportError:
            _HAS_MATPLOTLIB = False
    return _HAS_MATPLOTLIB


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute UMAP projection of embeddings.

    UMAP preserves both local and global structure better than t-SNE
    for high-dimensional neural embeddings. Uses cosine metric by
    default since CLIP embeddings are L2-normalized.

    Args:
        embeddings: Input array, shape (n_samples, n_features)
        n_neighbors: Size of local neighborhood (default: 15)
        min_dist: Minimum distance in projection (default: 0.1)
        n_components: Output dimensionality (default: 2)
        metric: Distance metric (default: 'cosine')
        random_state: Seed for reproducibility

    Returns:
        Projected coordinates, shape (n_samples, n_components)
    """
    if not _check_umap():
        raise ImportError(
            "umap-learn is required for UMAP projections. "
            "Install with: pip install umap-learn"
        )
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    n_components: int = 2,
    random_state: int = 42,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Compute t-SNE projection of embeddings.

    t-SNE is better at preserving local cluster structure but can
    distort global distances. Use alongside UMAP for robustness.

    Args:
        embeddings: Input array, shape (n_samples, n_features)
        perplexity: Perplexity parameter (default: 30)
        n_components: Output dimensionality (default: 2)
        random_state: Seed for reproducibility
        metric: Distance metric (default: 'cosine')

    Returns:
        Projected coordinates, shape (n_samples, n_components)
    """
    from sklearn.manifold import TSNE
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        metric=metric,
        random_state=random_state,
        init="pca" if metric != "cosine" else "random",
    )
    return reducer.fit_transform(embeddings)


def compute_pca_2d(
    embeddings: np.ndarray,
) -> np.ndarray:
    """PCA projection to 2D (fast fallback when UMAP unavailable)."""
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

# Publication-quality color scheme
PERCEPTION_COLOR = "#2196F3"  # Blue
IMAGERY_COLOR = "#FF5722"  # Deep Orange
GT_COLOR = "#4CAF50"  # Green


def visualize_perception_vs_imagery(
    perception_embeddings: np.ndarray,
    imagery_embeddings: np.ndarray,
    method: str = "umap",
    gt_embeddings: Optional[np.ndarray] = None,
    shared_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Perception vs Imagery Embedding Space",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    alpha: float = 0.5,
    point_size: int = 15,
) -> Optional[object]:
    """
    Main visualization: perception vs imagery embeddings in 2D.

    Produces a multi-panel figure:
    - Main panel: scatter of perception (blue) and imagery (orange)
    - Optional: ground-truth CLIP overlay (green)
    - Optional: connecting lines between shared stimuli

    Args:
        perception_embeddings: Predicted perception embeddings (N_p, D)
        imagery_embeddings: Predicted imagery embeddings (N_i, D)
        method: 'umap', 'tsne', or 'pca'
        gt_embeddings: Optional ground-truth CLIP embeddings
        shared_pairs: Optional (perc_indices, imag_indices) for connecting lines
        title: Figure title
        save_path: If provided, save figure to this path
        figsize: Figure size
        alpha: Point transparency
        point_size: Scatter point size

    Returns:
        Matplotlib figure object (or None if matplotlib unavailable)
    """
    if not _check_matplotlib():
        logger.warning("matplotlib not available, skipping visualization")
        return None

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Combine embeddings for joint projection
    all_emb = np.vstack([perception_embeddings, imagery_embeddings])
    if gt_embeddings is not None:
        all_emb = np.vstack([all_emb, gt_embeddings])

    n_perc = len(perception_embeddings)
    n_imag = len(imagery_embeddings)

    # Compute projection
    if method == "umap" and _check_umap():
        coords = compute_umap(all_emb)
    elif method == "tsne":
        coords = compute_tsne(all_emb)
    else:
        coords = compute_pca_2d(all_emb)

    perc_coords = coords[:n_perc]
    imag_coords = coords[n_perc : n_perc + n_imag]
    gt_coords = coords[n_perc + n_imag :] if gt_embeddings is not None else None

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Shared-stimulus connecting lines
    if shared_pairs is not None:
        p_idx, i_idx = shared_pairs
        for pi, ii in zip(p_idx[:100], i_idx[:100]):  # Cap at 100 lines
            ax.plot(
                [perc_coords[pi, 0], imag_coords[ii, 0]],
                [perc_coords[pi, 1], imag_coords[ii, 1]],
                color="gray", alpha=0.15, linewidth=0.5, zorder=1,
            )

    # Ground-truth overlay
    if gt_coords is not None:
        ax.scatter(
            gt_coords[:, 0], gt_coords[:, 1],
            c=GT_COLOR, s=point_size * 0.5, alpha=alpha * 0.5,
            label="Ground Truth CLIP", marker="x", zorder=2,
        )

    # Main scatter
    ax.scatter(
        perc_coords[:, 0], perc_coords[:, 1],
        c=PERCEPTION_COLOR, s=point_size, alpha=alpha,
        label=f"Perception (n={n_perc})", zorder=3,
    )
    ax.scatter(
        imag_coords[:, 0], imag_coords[:, 1],
        c=IMAGERY_COLOR, s=point_size, alpha=alpha,
        label=f"Imagery (n={n_imag})", zorder=3,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{method.upper()} 1", fontsize=11)
    ax.set_ylabel(f"{method.upper()} 2", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {save_path}")

    return fig


def visualize_layer_progression(
    multilayer_perception: Dict[str, np.ndarray],
    multilayer_imagery: Dict[str, np.ndarray],
    layers: Optional[List[str]] = None,
    method: str = "umap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (5, 4),
) -> Optional[object]:
    """
    Side-by-side UMAP/t-SNE showing perception-imagery gap per layer.

    Creates a row of panels, one per decoder layer, showing how the
    two manifolds progressively separate or overlap at different
    levels of the representational hierarchy.

    This figure directly addresses: "At which layer does imagery
    diverge from perception?"

    Args:
        multilayer_perception: Dict {layer_name: (N_p, D)}
        multilayer_imagery: Dict {layer_name: (N_i, D)}
        layers: Ordered list of layer names to plot (default: sorted keys)
        method: Projection method
        save_path: Optional save path
        figsize: Per-panel figure size

    Returns:
        Matplotlib figure
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt

    if layers is None:
        layers = sorted(
            set(multilayer_perception.keys()) & set(multilayer_imagery.keys())
        )

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(figsize[0] * n_layers, figsize[1]))
    if n_layers == 1:
        axes = [axes]

    for idx, layer in enumerate(layers):
        perc = multilayer_perception[layer]
        imag = multilayer_imagery[layer]
        combined = np.vstack([perc, imag])

        if method == "umap" and _check_umap():
            coords = compute_umap(combined, n_neighbors=min(15, len(combined) - 1))
        elif method == "tsne":
            coords = compute_tsne(combined, perplexity=min(30, len(combined) // 4))
        else:
            coords = compute_pca_2d(combined)

        n_p = len(perc)
        ax = axes[idx]
        ax.scatter(coords[:n_p, 0], coords[:n_p, 1],
                   c=PERCEPTION_COLOR, s=10, alpha=0.4, label="Perception")
        ax.scatter(coords[n_p:, 0], coords[n_p:, 1],
                   c=IMAGERY_COLOR, s=10, alpha=0.4, label="Imagery")
        ax.set_title(layer, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Layer Progression: Perception vs Imagery", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def embedding_density_comparison(
    perception_embeddings: np.ndarray,
    imagery_embeddings: np.ndarray,
    method: str = "umap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> Optional[Tuple[object, float]]:
    """
    KDE density comparison between perception and imagery manifolds.

    Creates three panels:
    1. Perception density (blue contours)
    2. Imagery density (orange contours)
    3. Overlap (both overlaid with difference highlight)

    Also computes the overlap coefficient (Bhattacharyya coefficient)
    as a scalar measure of manifold overlap.

    Args:
        perception_embeddings: shape (N_p, D)
        imagery_embeddings: shape (N_i, D)
        method: Projection method before KDE
        save_path: Optional save path
        figsize: Figure size

    Returns:
        (figure, overlap_coefficient) or None
    """
    if not _check_matplotlib():
        return None

    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    # Project to 2D
    combined = np.vstack([perception_embeddings, imagery_embeddings])
    if method == "umap" and _check_umap():
        coords = compute_umap(combined)
    elif method == "tsne":
        coords = compute_tsne(combined)
    else:
        coords = compute_pca_2d(combined)

    n_p = len(perception_embeddings)
    perc_2d = coords[:n_p]
    imag_2d = coords[n_p:]

    # KDE on 2D projections
    try:
        kde_p = gaussian_kde(perc_2d.T)
        kde_i = gaussian_kde(imag_2d.T)
    except Exception as e:
        logger.warning(f"KDE failed: {e}")
        return None

    # Create evaluation grid
    x_min = coords[:, 0].min() - 1
    x_max = coords[:, 0].max() + 1
    y_min = coords[:, 1].min() - 1
    y_max = coords[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100),
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])

    z_p = kde_p(grid).reshape(xx.shape)
    z_i = kde_i(grid).reshape(xx.shape)

    # Overlap coefficient (Bhattacharyya)
    # Discretized integral of √(p·q)
    dp = z_p / (z_p.sum() + 1e-10)
    di = z_i / (z_i.sum() + 1e-10)
    overlap = float(np.sum(np.sqrt(dp * di)))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, z, color, label in [
        (axes[0], z_p, PERCEPTION_COLOR, "Perception"),
        (axes[1], z_i, IMAGERY_COLOR, "Imagery"),
    ]:
        ax.contourf(xx, yy, z, levels=15, cmap="Blues" if "2196" in color else "Oranges")
        ax.set_title(f"{label} Density", fontsize=11)
        ax.set_aspect("equal")

    # Overlap panel
    axes[2].contour(xx, yy, z_p, levels=5, colors=PERCEPTION_COLOR, alpha=0.6)
    axes[2].contour(xx, yy, z_i, levels=5, colors=IMAGERY_COLOR, alpha=0.6)
    axes[2].set_title(f"Overlap (BC={overlap:.3f})", fontsize=11)
    axes[2].set_aspect("equal")

    fig.suptitle("Manifold Density Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, overlap


# ---------------------------------------------------------------------------
# Utility: batch visualization from EmbeddingBundle
# ---------------------------------------------------------------------------

def visualize_bundle(
    bundle,  # EmbeddingBundle (avoid circular import)
    method: str = "umap",
    output_dir: Optional[str] = None,
    prefix: str = "embed_viz",
) -> Dict[str, object]:
    """
    Generate a complete visualization suite from an EmbeddingBundle.

    Produces:
    1. Main perception vs imagery scatter
    2. Density comparison with overlap coefficient
    3. Layer progression (if multi-layer data available)

    Args:
        bundle: EmbeddingBundle instance
        method: Projection method ('umap', 'tsne', 'pca')
        output_dir: Directory to save figures (optional)
        prefix: Filename prefix

    Returns:
        Dict of figure objects keyed by name
    """
    from pathlib import Path

    figs = {}
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # 1. Main scatter
    shared = None
    if hasattr(bundle, "get_shared_stimulus_pairs"):
        pairs = bundle.get_shared_stimulus_pairs()
        if pairs is not None:
            shared = (pairs[1], pairs[2])  # (perc_indices, imag_indices)

    fig = visualize_perception_vs_imagery(
        bundle.perception, bundle.imagery,
        method=method,
        gt_embeddings=bundle.perception_targets[:min(500, len(bundle.perception_targets))],
        shared_pairs=shared,
        save_path=str(out / f"{prefix}_scatter.png") if out else None,
    )
    figs["scatter"] = fig

    # 2. Density
    result = embedding_density_comparison(
        bundle.perception, bundle.imagery,
        method=method,
        save_path=str(out / f"{prefix}_density.png") if out else None,
    )
    if result is not None:
        figs["density"] = result[0]
        logger.info(f"Manifold overlap (Bhattacharyya coefficient): {result[1]:.4f}")

    # 3. Layer progression
    if bundle.multilayer_perception and bundle.multilayer_imagery:
        fig = visualize_layer_progression(
            bundle.multilayer_perception,
            bundle.multilayer_imagery,
            method=method,
            save_path=str(out / f"{prefix}_layers.png") if out else None,
        )
        figs["layer_progression"] = fig

    return figs
