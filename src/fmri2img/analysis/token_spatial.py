"""
Token-Level Spatial Decomposition of Imagery Fidelity
=====================================================

Novel analysis module that exploits the 257-token spatial structure of
ViT-bigG/14 outputs from the FMRI2images model. The 256 patch tokens
correspond to a 16×16 spatial grid (for 224×224 input), where each patch
represents a 14×14 pixel region.

This enables the first-ever **spatial map** of the perception-imagery gap:
which image regions are best preserved during mental imagery?

Key analyses:
    1. Token Fidelity Map: Per-token cosine similarity → 16×16 heatmap
    2. Center-vs-Periphery: Tests Kosslyn's "spotlight" theory of imagery
    3. Token Information Content: Per-token dimensionality and variance
    4. Token Correlation Structure: Spatial coherence across domains

References:
    - Kosslyn, S.M. (1994). Image and Brain. MIT Press.
    - Dijkstra, N., Bosch, S.E., & van Gerven, M.A. (2019). Shared neural
      mechanisms of visual perception and imagery. Trends in Cognitive Sciences.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

# ViT-bigG/14 spatial layout
N_TOKENS = 257  # 256 patches + 1 CLS
N_PATCHES = 256
GRID_SIZE = 16  # 16×16 = 256 patches
TOKEN_DIM = 1280


def _l2_tokens(x: np.ndarray) -> np.ndarray:
    """L2-normalize along last axis. x: (..., D) -> (..., D)."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x / norms


def _per_token_cosine(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute per-token cosine similarity averaged over samples.

    Parameters
    ----------
    preds : (N, T, D) predicted token embeddings
    targets : (N, T, D) target token embeddings

    Returns
    -------
    (T,) mean cosine similarity per token position
    """
    p = _l2_tokens(preds)
    t = _l2_tokens(targets)
    # Per-sample, per-token cosine: (N, T)
    cos = np.sum(p * t, axis=-1)
    return cos.mean(axis=0)  # (T,)


def _tokens_to_grid(token_values: np.ndarray) -> np.ndarray:
    """Reshape 256 patch token values to 16×16 spatial grid.

    Parameters
    ----------
    token_values : (256,) or (256, ...) values for patch tokens (excludes CLS)

    Returns
    -------
    (16, 16) or (16, 16, ...) spatial grid
    """
    if token_values.shape[0] != N_PATCHES:
        raise ValueError(f"Expected {N_PATCHES} patch tokens, got {token_values.shape[0]}")
    return token_values.reshape(GRID_SIZE, GRID_SIZE, *token_values.shape[1:])


def _get_spatial_rings() -> Dict[str, np.ndarray]:
    """Define concentric rings on the 16×16 grid for center-periphery analysis.

    Returns dict mapping ring name to boolean mask over the 16×16 grid.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    center_y, center_x = GRID_SIZE / 2, GRID_SIZE / 2

    rings = {}

    # Center: inner 4×4 (rows 6-9, cols 6-9)
    center = np.zeros_like(grid)
    center[6:10, 6:10] = True
    rings["center"] = center

    # Near periphery: 8×8 minus center
    near = np.zeros_like(grid)
    near[4:12, 4:12] = True
    near[6:10, 6:10] = False  # exclude center
    rings["near_periphery"] = near

    # Far periphery: everything outside 8×8
    far = np.ones_like(grid)
    far[4:12, 4:12] = False
    rings["far_periphery"] = far

    return rings


def analyze_token_fidelity_map(
    perception_tokens: np.ndarray,
    imagery_tokens: np.ndarray,
    perception_targets: np.ndarray,
    imagery_targets: np.ndarray,
) -> Dict:
    """Compute per-token fidelity map and perception-imagery gap map.

    This is the core novel analysis: a 16×16 spatial heatmap showing where
    imagery representation is preserved vs degraded relative to perception.

    Parameters
    ----------
    perception_tokens : (N_p, 257, 1280) perception token predictions
    imagery_tokens : (N_i, 257, 1280) imagery token predictions
    perception_targets : (N_p, 257, 1280) perception CLIP targets
    imagery_targets : (N_i, 257, 1280) imagery CLIP targets

    Returns
    -------
    dict with keys:
        perception_fidelity : (16, 16) mean cosine per spatial position (perception)
        imagery_fidelity : (16, 16) mean cosine per spatial position (imagery)
        gap_map : (16, 16) perception_fidelity - imagery_fidelity (positive = imagery worse)
        ratio_map : (16, 16) imagery_fidelity / perception_fidelity
        cls_perception_cosine : float, CLS token perception cosine
        cls_imagery_cosine : float, CLS token imagery cosine
        cls_gap : float, CLS perception - imagery gap
        mean_gap : float, mean spatial gap across all patches
        max_gap_position : (row, col) position of largest gap
        min_gap_position : (row, col) position of smallest gap
    """
    # Per-token cosine: (257,)
    perc_cosines = _per_token_cosine(perception_tokens, perception_targets)
    imag_cosines = _per_token_cosine(imagery_tokens, imagery_targets)

    # CLS token (index 0)
    cls_perc = float(perc_cosines[0])
    cls_imag = float(imag_cosines[0])

    # Patch tokens (indices 1:257)
    perc_patches = perc_cosines[1:]  # (256,)
    imag_patches = imag_cosines[1:]  # (256,)
    gap_patches = perc_patches - imag_patches  # positive = imagery worse

    # Reshape to spatial grid
    perc_grid = _tokens_to_grid(perc_patches)
    imag_grid = _tokens_to_grid(imag_patches)
    gap_grid = _tokens_to_grid(gap_patches)

    # Ratio map (imagery / perception), clipped to avoid div by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_grid = np.where(
            perc_grid > 0.01,
            imag_grid / perc_grid,
            1.0,
        )

    # Find extreme positions
    max_gap_pos = np.unravel_index(gap_grid.argmax(), gap_grid.shape)
    min_gap_pos = np.unravel_index(gap_grid.argmin(), gap_grid.shape)

    return {
        "perception_fidelity": perc_grid,
        "imagery_fidelity": imag_grid,
        "gap_map": gap_grid,
        "ratio_map": ratio_grid,
        "cls_perception_cosine": cls_perc,
        "cls_imagery_cosine": cls_imag,
        "cls_gap": cls_perc - cls_imag,
        "mean_gap": float(gap_patches.mean()),
        "std_gap": float(gap_patches.std()),
        "max_gap_position": list(max_gap_pos),
        "max_gap_value": float(gap_grid.max()),
        "min_gap_position": list(min_gap_pos),
        "min_gap_value": float(gap_grid.min()),
    }


def analyze_center_vs_periphery(
    perception_tokens: np.ndarray,
    imagery_tokens: np.ndarray,
    perception_targets: np.ndarray,
    imagery_targets: np.ndarray,
) -> Dict:
    """Test the "spotlight" hypothesis: does imagery preserve center better?

    Tests Kosslyn's (1994) visual buffer theory: mental imagery has a
    resolution gradient with highest fidelity at the center of the visual
    field, degrading toward the periphery.

    Groups 256 patch tokens into three concentric rings:
        - Center: inner 4×4 (16 tokens)
        - Near periphery: 8×8 minus center (48 tokens)
        - Far periphery: everything outside 8×8 (192 tokens)

    Returns per-ring transfer gap and statistical test for monotonic gradient.
    """
    perc_cosines = _per_token_cosine(perception_tokens, perception_targets)
    imag_cosines = _per_token_cosine(imagery_tokens, imagery_targets)

    # Patch tokens only (exclude CLS at index 0)
    perc_patches = perc_cosines[1:]
    imag_patches = imag_cosines[1:]
    gap_patches = perc_patches - imag_patches

    perc_grid = _tokens_to_grid(perc_patches)
    imag_grid = _tokens_to_grid(imag_patches)
    gap_grid = _tokens_to_grid(gap_patches)

    rings = _get_spatial_rings()
    results = {"rings": {}}

    ring_gaps = []
    ring_names = ["center", "near_periphery", "far_periphery"]

    for name in ring_names:
        mask = rings[name]
        n_tokens = mask.sum()
        perc_mean = float(perc_grid[mask].mean())
        imag_mean = float(imag_grid[mask].mean())
        gap_mean = float(gap_grid[mask].mean())
        gap_std = float(gap_grid[mask].std())

        results["rings"][name] = {
            "n_tokens": int(n_tokens),
            "perception_cosine": perc_mean,
            "imagery_cosine": imag_mean,
            "gap": gap_mean,
            "gap_std": gap_std,
            "transfer_ratio": imag_mean / perc_mean if perc_mean > 0.01 else 1.0,
        }
        ring_gaps.append(gap_mean)

    # Test for monotonic gradient: gap should increase from center to periphery
    # (if spotlight hypothesis holds)
    is_monotonic = ring_gaps[0] <= ring_gaps[1] <= ring_gaps[2]

    # Jonckheere-Terpstra-like test: simple ordered comparison
    # Spearman correlation between ring index [0,1,2] and gap
    if len(set(ring_gaps)) > 1:
        rho, p_val = scipy_stats.spearmanr([0, 1, 2], ring_gaps)
    else:
        rho, p_val = 0.0, 1.0

    results["gradient_test"] = {
        "is_monotonic": bool(is_monotonic),
        "center_to_periphery_gaps": ring_gaps,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "hypothesis": "spotlight" if is_monotonic and rho > 0 else "uniform",
    }

    return results


def analyze_token_information_content(
    perception_tokens: np.ndarray,
    imagery_tokens: np.ndarray,
) -> Dict:
    """Analyze per-token information content: dimensionality and variance.

    Computes at each spatial position:
        - Participation ratio (effective dimensionality)
        - Total variance (signal strength)
        - Token-token correlation (spatial coherence)

    Parameters
    ----------
    perception_tokens : (N_p, 257, 1280) perception predictions
    imagery_tokens : (N_i, 257, 1280) imagery predictions

    Returns
    -------
    dict with per-token and spatial grid analysis results
    """

    def _token_stats(tokens: np.ndarray, label: str) -> Dict:
        """Compute per-token statistics for one domain."""
        # tokens: (N, 257, 1280)
        N, T, D = tokens.shape
        patch_tokens = tokens[:, 1:, :]  # (N, 256, D) exclude CLS

        # Per-token participation ratio
        prs = np.zeros(N_PATCHES)
        variances = np.zeros(N_PATCHES)

        for t in range(N_PATCHES):
            t_data = patch_tokens[:, t, :]  # (N, D)
            variances[t] = t_data.var()

            # PCA eigenvalues for participation ratio
            if N >= 2:
                centered = t_data - t_data.mean(axis=0)
                # Use economy SVD for efficiency
                try:
                    _, s, _ = np.linalg.svd(centered, full_matrices=False)
                    eigenvalues = (s ** 2) / (N - 1)
                    eigenvalues = eigenvalues[eigenvalues > 1e-10]
                    if len(eigenvalues) > 0:
                        p = eigenvalues / eigenvalues.sum()
                        prs[t] = 1.0 / (p ** 2).sum()
                except np.linalg.LinAlgError:
                    prs[t] = 0.0

        # Token-token correlation matrix (spatial coherence)
        # Average each token across samples → (256, D), then compute (256, 256) correlation
        mean_tokens = patch_tokens.mean(axis=0)  # (256, D)
        mean_tokens_norm = _l2_tokens(mean_tokens)  # (256, D)
        spatial_corr = mean_tokens_norm @ mean_tokens_norm.T  # (256, 256)

        pr_grid = _tokens_to_grid(prs)
        var_grid = _tokens_to_grid(variances)

        return {
            f"{label}_participation_ratio": pr_grid.tolist(),
            f"{label}_variance": var_grid.tolist(),
            f"{label}_mean_pr": float(prs.mean()),
            f"{label}_mean_variance": float(variances.mean()),
            f"{label}_spatial_coherence_mean": float(
                spatial_corr[np.triu_indices(N_PATCHES, k=1)].mean()
            ),
        }

    results = {}
    results.update(_token_stats(perception_tokens, "perception"))
    results.update(_token_stats(imagery_tokens, "imagery"))

    # Dimensionality ratio per token
    perc_pr = np.array(results["perception_participation_ratio"])
    imag_pr = np.array(results["imagery_participation_ratio"])
    with np.errstate(divide="ignore", invalid="ignore"):
        pr_ratio = np.where(perc_pr > 0.01, imag_pr / perc_pr, 1.0)
    results["pr_ratio_grid"] = pr_ratio.tolist()
    results["mean_pr_ratio"] = float(pr_ratio.mean())

    return results


def analyze_token_transfer_by_trial(
    perception_tokens: np.ndarray,
    imagery_tokens: np.ndarray,
    perception_targets: np.ndarray,
    imagery_targets: np.ndarray,
) -> Dict:
    """Per-trial token analysis for variance and confidence intervals.

    Returns per-trial spatial gap maps for statistical testing.
    """
    N_p = perception_tokens.shape[0]
    N_i = imagery_tokens.shape[0]

    # Per-trial, per-token cosine
    perc_cos = np.sum(
        _l2_tokens(perception_tokens) * _l2_tokens(perception_targets), axis=-1
    )  # (N_p, 257)
    imag_cos = np.sum(
        _l2_tokens(imagery_tokens) * _l2_tokens(imagery_targets), axis=-1
    )  # (N_i, 257)

    # Per-token mean and std across trials
    perc_patch_mean = perc_cos[:, 1:].mean(axis=0)  # (256,)
    perc_patch_std = perc_cos[:, 1:].std(axis=0)
    imag_patch_mean = imag_cos[:, 1:].mean(axis=0)
    imag_patch_std = imag_cos[:, 1:].std(axis=0)

    # Per-token t-test (two-sample, unequal variance)
    t_stats = np.zeros(N_PATCHES)
    p_values = np.ones(N_PATCHES)
    for t in range(N_PATCHES):
        perc_vals = perc_cos[:, t + 1]
        imag_vals = imag_cos[:, t + 1]
        if len(perc_vals) >= 2 and len(imag_vals) >= 2:
            stat, pval = scipy_stats.ttest_ind(perc_vals, imag_vals, equal_var=False)
            t_stats[t] = stat
            p_values[t] = pval

    # FDR correction (Benjamini-Hochberg)
    sorted_idx = np.argsort(p_values)
    n_tests = N_PATCHES
    fdr_threshold = 0.05
    fdr_significant = np.zeros(N_PATCHES, dtype=bool)
    for rank, idx in enumerate(sorted_idx):
        if p_values[idx] <= fdr_threshold * (rank + 1) / n_tests:
            fdr_significant[idx] = True
        else:
            break

    return {
        "perception_cosine_mean": _tokens_to_grid(perc_patch_mean).tolist(),
        "perception_cosine_std": _tokens_to_grid(perc_patch_std).tolist(),
        "imagery_cosine_mean": _tokens_to_grid(imag_patch_mean).tolist(),
        "imagery_cosine_std": _tokens_to_grid(imag_patch_std).tolist(),
        "t_statistics": _tokens_to_grid(t_stats).tolist(),
        "p_values": _tokens_to_grid(p_values).tolist(),
        "fdr_significant": _tokens_to_grid(fdr_significant).tolist(),
        "n_significant_tokens": int(fdr_significant.sum()),
        "n_perception_trials": N_p,
        "n_imagery_trials": N_i,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_imagery_fidelity_map(
    gap_map: np.ndarray,
    save_path: str,
    title: str = "Imagery Fidelity Gap (Perception − Imagery)",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Create a publication-quality heatmap of the spatial transfer gap.

    Cool colors (blue) = imagery matches or exceeds perception.
    Warm colors (red) = imagery degrades relative to perception.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if vmin is None or vmax is None:
        absmax = max(abs(gap_map.min()), abs(gap_map.max()), 0.01)
        vmin, vmax = -absmax, absmax

    im = ax.imshow(gap_map, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Patch Column (→ right of image)")
    ax.set_ylabel("Patch Row (↓ bottom of image)")

    # Add concentric ring boundaries
    rings = _get_spatial_rings()
    for ring_name, style in [("center", "w--"), ("near_periphery", "w:")]:
        mask = rings[ring_name]
        # Draw boundary using contour
        ax.contour(mask.astype(float), levels=[0.5], colors="white",
                   linewidths=1.0, linestyles="dashed" if "center" in ring_name else "dotted")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Gap (P − I)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved fidelity map to {save_path}")


def plot_center_periphery(
    ring_results: Dict,
    save_path: str,
    title: str = "Center vs. Periphery Transfer Gap",
) -> None:
    """Bar chart of transfer gap by spatial ring."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rings_data = ring_results["rings"]
    names = ["center", "near_periphery", "far_periphery"]
    labels = ["Center\n(4×4)", "Near Periphery\n(8×8 ring)", "Far Periphery\n(outer ring)"]

    perc_cosines = [rings_data[n]["perception_cosine"] for n in names]
    imag_cosines = [rings_data[n]["imagery_cosine"] for n in names]
    gaps = [rings_data[n]["gap"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Perception vs Imagery cosines
    ax1.bar(x - width / 2, perc_cosines, width, label="Perception", color="#2196F3", alpha=0.8)
    ax1.bar(x + width / 2, imag_cosines, width, label="Imagery", color="#FF9800", alpha=0.8)
    ax1.set_xlabel("Spatial Region")
    ax1.set_ylabel("Mean Cosine Similarity")
    ax1.set_title("Token Fidelity by Region")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend()

    # Right panel: Gap by ring
    colors = ["#4CAF50" if g <= 0 else "#F44336" for g in gaps]
    ax2.bar(x, gaps, 0.6, color=colors, alpha=0.8)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Spatial Region")
    ax2.set_ylabel("Transfer Gap (P − I)")
    ax2.set_title(title)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)

    # Annotate hypothesis test
    hyp = ring_results["gradient_test"]
    annotation = (
        f"Gradient: {'Monotonic' if hyp['is_monotonic'] else 'Non-monotonic'}\n"
        f"ρ = {hyp['spearman_rho']:.3f}, p = {hyp['spearman_p']:.3f}"
    )
    ax2.annotate(annotation, xy=(0.95, 0.95), xycoords="axes fraction",
                 ha="right", va="top", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    plt.suptitle("Spatial Decomposition of Perception–Imagery Transfer", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved center-periphery plot to {save_path}")


def run_full_token_analysis(bundle: "EmbeddingBundle", output_dir: str) -> Dict:
    """Run all token-level analyses on an EmbeddingBundle.

    Requires bundle.has_tokens == True.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bundle.has_tokens:
        logger.warning("Bundle has no token data — skipping token analysis")
        return {"error": "no_token_data"}

    results = {}

    # 1. Fidelity map
    logger.info("Running token fidelity map analysis...")
    fidelity = analyze_token_fidelity_map(
        bundle.perception_tokens, bundle.imagery_tokens,
        bundle.perception_token_targets, bundle.imagery_token_targets,
    )
    results["fidelity_map"] = {
        k: v for k, v in fidelity.items()
        if not isinstance(v, np.ndarray)
    }
    # Save raw arrays
    np.save(output_dir / "perception_fidelity_grid.npy", fidelity["perception_fidelity"])
    np.save(output_dir / "imagery_fidelity_grid.npy", fidelity["imagery_fidelity"])
    np.save(output_dir / "gap_map.npy", fidelity["gap_map"])

    # Plot
    plot_imagery_fidelity_map(
        fidelity["gap_map"],
        str(output_dir / "token_fidelity_map.png"),
    )
    plot_imagery_fidelity_map(
        fidelity["gap_map"],
        str(output_dir / "token_fidelity_map.pdf"),
    )

    # 2. Center-periphery
    logger.info("Running center-periphery analysis...")
    cp = analyze_center_vs_periphery(
        bundle.perception_tokens, bundle.imagery_tokens,
        bundle.perception_token_targets, bundle.imagery_token_targets,
    )
    results["center_periphery"] = cp

    plot_center_periphery(
        cp, str(output_dir / "center_periphery.png"),
    )
    plot_center_periphery(
        cp, str(output_dir / "center_periphery.pdf"),
    )

    # 3. Information content
    logger.info("Running token information content analysis...")
    info = analyze_token_information_content(
        bundle.perception_tokens, bundle.imagery_tokens,
    )
    results["information_content"] = {
        k: v for k, v in info.items()
        if not isinstance(v, (list,)) or len(str(v)) < 1000
    }

    # 4. Per-trial statistical testing
    logger.info("Running per-trial token statistics...")
    trial_stats = analyze_token_transfer_by_trial(
        bundle.perception_tokens, bundle.imagery_tokens,
        bundle.perception_token_targets, bundle.imagery_token_targets,
    )
    results["trial_statistics"] = {
        "n_significant_tokens": trial_stats["n_significant_tokens"],
        "n_perception_trials": trial_stats["n_perception_trials"],
        "n_imagery_trials": trial_stats["n_imagery_trials"],
    }
    np.save(output_dir / "token_p_values.npy", np.array(trial_stats["p_values"]))
    np.save(output_dir / "token_fdr_significant.npy", np.array(trial_stats["fdr_significant"]))

    logger.info(
        f"Token analysis complete: mean_gap={fidelity['mean_gap']:.4f}, "
        f"n_significant={trial_stats['n_significant_tokens']}/{N_PATCHES}"
    )

    return results
