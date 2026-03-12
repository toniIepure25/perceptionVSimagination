"""
ROI-Specific Decoding Analysis
===============================

Tests brain decoding at the level of individual cortical regions of
interest (ROIs), revealing which brain areas carry decodable imagery
information. This directly addresses the neuroscience question:

    "Does imagery information survive better in higher visual cortex
     (IT/PFC) than early visual cortex (V1-V2)?"

Based on the NSD ROI definitions (nsdgeneral + functional ROIs).
Trains independent Ridge regression models per ROI and compares
decoding performance to establish a "cortical hierarchy of imagery
preservation."

NSD ROI hierarchy (approximate, from early to late):
    V1 → V2 → V3 → hV4 → LOC → EBA → OPA → RSC → PPA → FFA

Expected result (Dijkstra et al., 2019): Late visual areas (IT complex,
PPA/FFA) should decode imagery better than early areas (V1-V3), because
imagery representations are top-down and engage semantic areas more
than structural/retinotopic areas.

References:
    Allen et al. (2022). NSD ROI definitions in fsaverage space.
    Dijkstra et al. (2019). "Differential temporal dynamics during visual
        imagery and perception." eLife.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# NSD standard ROIs in approximately early-to-late order
NSD_ROIS = [
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4",
    "EBA", "FBA-1", "FBA-2", "mTL-bodies",
    "OFA", "FFA-1", "FFA-2", "mTL-faces",
    "OPA", "PPA", "RSC",
    "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
]

# Grouped for clearer hierarchy
ROI_GROUPS = {
    "early_visual": ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d"],
    "mid_visual": ["hV4"],
    "body_selective": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
    "face_selective": ["OFA", "FFA-1", "FFA-2", "mTL-faces"],
    "scene_selective": ["OPA", "PPA", "RSC"],
    "word_selective": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
}

# Simplified hierarchy (merging sub-ROIs)
SIMPLE_HIERARCHY = ["V1", "V2", "V3", "hV4", "LOC", "EBA", "OPA", "PPA", "FFA", "RSC"]


@dataclass
class ROIDecodingResult:
    """Results from ROI-specific decoding analysis."""

    roi_name: str
    n_voxels: int
    cosine: float = 0.0
    r_at_1: float = 0.0
    r_at_5: float = 0.0
    median_rank: float = 0.0
    mrr: float = 0.0
    alpha: float = 0.0  # Ridge regularization used


@dataclass
class ROIHierarchyResult:
    """Complete ROI hierarchy analysis result."""

    roi_results: List[ROIDecodingResult]
    condition: str = "perception"  # or "imagery"
    subject: str = "subj01"

    def to_dict_list(self) -> List[Dict]:
        """Convert to list of dicts for DataFrame creation."""
        return [
            {
                "roi": r.roi_name,
                "n_voxels": r.n_voxels,
                "cosine": r.cosine,
                "r_at_1": r.r_at_1,
                "r_at_5": r.r_at_5,
                "median_rank": r.median_rank,
                "mrr": r.mrr,
                "alpha": r.alpha,
                "condition": self.condition,
                "subject": self.subject,
            }
            for r in self.roi_results
        ]

    def get_ranked(self, metric: str = "cosine") -> List[ROIDecodingResult]:
        """Return ROI results sorted by metric (descending)."""
        return sorted(
            self.roi_results,
            key=lambda r: getattr(r, metric),
            reverse=True,
        )


def load_nsd_roi_masks(
    subject: str = "subj01",
    data_root: str = "data",
    rois: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load NSD ROI masks for a subject.

    ROI masks are boolean arrays indicating which voxels belong to
    each brain region. NSD provides these in the functional space.

    Args:
        subject: NSD subject ID
        data_root: Root data directory
        rois: List of ROI names to load (default: all available)

    Returns:
        Dict mapping ROI names to boolean mask arrays.
        Each mask has shape (n_total_voxels,) where True indicates
        membership in that ROI.
    """
    root = Path(data_root)
    masks = {}

    if rois is None:
        rois = NSD_ROIS

    # Try loading from standard NSD paths
    for roi_name in rois:
        # Check multiple possible file patterns
        possible_paths = [
            root / subject / "roi" / f"{roi_name}.npy",
            root / subject / "rois" / f"{roi_name}.npy",
            root / "nsd" / subject / "roi" / f"{roi_name}.npy",
            root / "rois" / subject / f"{roi_name}.npy",
        ]
        for p in possible_paths:
            if p.exists():
                masks[roi_name] = np.load(p).astype(bool)
                logger.debug(f"Loaded ROI {roi_name}: {masks[roi_name].sum()} voxels")
                break

    if not masks:
        logger.warning(
            f"No ROI masks found for {subject} in {data_root}. "
            "Expected .npy files in {data_root}/{subject}/roi/"
        )

    return masks


def extract_roi_features(
    betas: np.ndarray,
    roi_mask: np.ndarray,
    reliability_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract fMRI features for a specific brain ROI.

    Applies the ROI mask and optional reliability mask to select
    voxels, then extracts the corresponding columns from the
    beta weight matrix.

    Args:
        betas: Full brain betas, shape (n_trials, n_voxels)
        roi_mask: Boolean ROI mask, shape (n_voxels,)
        reliability_mask: Optional additional mask (e.g., reliability threshold)

    Returns:
        ROI-specific features, shape (n_trials, n_roi_voxels)
    """
    if reliability_mask is not None:
        combined_mask = roi_mask & reliability_mask
    else:
        combined_mask = roi_mask

    n_voxels = combined_mask.sum()
    if n_voxels == 0:
        logger.warning("ROI mask selects 0 voxels after reliability filtering")
        return np.zeros((betas.shape[0], 0))

    return betas[:, combined_mask]


def train_ridge_per_roi(
    roi_features: Dict[str, np.ndarray],
    clip_targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alphas: Optional[np.ndarray] = None,
    min_voxels: int = 10,
) -> Dict[str, Tuple]:
    """
    Train independent Ridge regression models per ROI.

    Uses cross-validated alpha selection (same as main Ridge baseline)
    for each ROI independently. Skips ROIs with too few voxels.

    Args:
        roi_features: Dict {roi_name: (n_trials, n_voxels)}
        clip_targets: CLIP target embeddings (n_trials, 768)
        train_idx: Training sample indices
        val_idx: Validation sample indices
        alphas: Ridge regularization values to try (default: logspace)
        min_voxels: Minimum voxels required per ROI

    Returns:
        Dict {roi_name: (model, best_alpha, val_cosine)}
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    if alphas is None:
        alphas = np.logspace(1, 6, 20)

    results = {}

    for roi_name, features in roi_features.items():
        n_voxels = features.shape[1] if features.ndim > 1 else 0
        if n_voxels < min_voxels:
            logger.info(f"Skipping {roi_name}: only {n_voxels} voxels (< {min_voxels})")
            continue

        logger.info(f"Training Ridge for {roi_name} ({n_voxels} voxels)")

        # Split
        X_train = features[train_idx]
        X_val = features[val_idx]
        Y_train = clip_targets[train_idx]
        Y_val = clip_targets[val_idx]

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # CV over alphas
        best_alpha = alphas[0]
        best_cosine = -1.0

        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_train, Y_train)
            preds = model.predict(X_val)

            # L2-normalize
            norms = np.linalg.norm(preds, axis=1, keepdims=True)
            preds = preds / np.maximum(norms, 1e-8)

            norms_t = np.linalg.norm(Y_val, axis=1, keepdims=True)
            targets = Y_val / np.maximum(norms_t, 1e-8)

            cosine = float(np.mean(np.sum(preds * targets, axis=1)))

            if cosine > best_cosine:
                best_cosine = cosine
                best_alpha = alpha

        # Retrain with best alpha on full training set
        final_model = Ridge(alpha=best_alpha, fit_intercept=True)
        final_model.fit(X_train, Y_train)

        results[roi_name] = (final_model, scaler, best_alpha, best_cosine)
        logger.info(f"  {roi_name}: α={best_alpha:.0f}, val_cos={best_cosine:.4f}")

    return results


def evaluate_roi_hierarchy(
    roi_models: Dict[str, Tuple],
    roi_features: Dict[str, np.ndarray],
    clip_targets: np.ndarray,
    test_idx: np.ndarray,
) -> ROIHierarchyResult:
    """
    Evaluate all ROI models on test set to build hierarchy.

    Args:
        roi_models: Dict from train_ridge_per_roi()
        roi_features: Dict {roi_name: (n_trials, n_voxels)}
        clip_targets: CLIP targets (n_trials, 768)
        test_idx: Test sample indices

    Returns:
        ROIHierarchyResult with per-ROI metrics
    """
    results = []
    Y_test = clip_targets[test_idx]

    # L2-normalize targets
    t_norms = np.linalg.norm(Y_test, axis=1, keepdims=True)
    Y_test_norm = Y_test / np.maximum(t_norms, 1e-8)

    for roi_name, (model, scaler, alpha, _) in roi_models.items():
        features = roi_features[roi_name]
        X_test = scaler.transform(features[test_idx])

        preds = model.predict(X_test)
        p_norms = np.linalg.norm(preds, axis=1, keepdims=True)
        preds_norm = preds / np.maximum(p_norms, 1e-8)

        # Cosine similarity
        cosines = np.sum(preds_norm * Y_test_norm, axis=1)
        mean_cosine = float(np.mean(cosines))

        # Retrieval metrics
        sim_matrix = preds_norm @ Y_test_norm.T  # (N, N)
        n = len(Y_test)
        gt_indices = np.arange(n)
        ranks = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(-sim_matrix[i])
            rank = np.where(sorted_idx == gt_indices[i])[0]
            ranks[i] = rank[0] + 1 if len(rank) > 0 else n

        r_at_1 = float(np.mean(ranks <= 1))
        r_at_5 = float(np.mean(ranks <= 5))
        median_rank = float(np.median(ranks))
        mrr = float(np.mean(1.0 / ranks))

        results.append(ROIDecodingResult(
            roi_name=roi_name,
            n_voxels=features.shape[1],
            cosine=mean_cosine,
            r_at_1=r_at_1,
            r_at_5=r_at_5,
            median_rank=median_rank,
            mrr=mrr,
            alpha=alpha,
        ))

    return ROIHierarchyResult(roi_results=results)


def plot_roi_hierarchy(
    result: ROIHierarchyResult,
    metric: str = "cosine",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Optional[object]:
    """
    Create publication-quality ROI hierarchy barplot.

    Shows decoding performance per brain region, ordered by the
    cortical hierarchy (early visual → late visual → category-selective).

    Args:
        result: ROIHierarchyResult from evaluate_roi_hierarchy()
        metric: Metric to plot ('cosine', 'r_at_1', 'mrr')
        save_path: Optional file path for saving
        figsize: Figure size

    Returns:
        Matplotlib figure or None
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return None

    ranked = result.get_ranked(metric)
    names = [r.roi_name for r in ranked]
    values = [getattr(r, metric) for r in ranked]
    voxels = [r.n_voxels for r in ranked]

    # Color by ROI group
    group_colors = {
        "early_visual": "#42A5F5",
        "mid_visual": "#66BB6A",
        "body_selective": "#FFA726",
        "face_selective": "#EF5350",
        "scene_selective": "#AB47BC",
        "word_selective": "#78909C",
    }

    def _get_group(roi_name):
        for group, members in ROI_GROUPS.items():
            if roi_name in members:
                return group
        return "other"

    colors = [group_colors.get(_get_group(n), "#9E9E9E") for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", height=0.7)

    # Add voxel count annotations
    for i, (v, n_v) in enumerate(zip(values, voxels)):
        ax.text(v + 0.002, i, f"n={n_v}", va="center", fontsize=8, color="gray")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(
        f"ROI Hierarchy — {result.condition.title()} Decoding ({result.subject})",
        fontsize=13, fontweight="bold",
    )
    ax.invert_yaxis()

    # Legend
    patches = [
        mpatches.Patch(color=c, label=g.replace("_", " ").title())
        for g, c in group_colors.items()
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
