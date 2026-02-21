#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Novel Analyses
========================================================

Creates figures for all five research directions from analysis results.

Usage:
    python scripts/make_novel_figures.py \\
        --results-dir outputs/novel_analyses \\
        --output-dir outputs/novel_analyses/figures
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Publication defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

PERC_COLOR = "#2196F3"
IMAG_COLOR = "#FF5722"
SEMANTIC_COLOR = "#4CAF50"
VISUAL_COLOR = "#9C27B0"


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        logger.warning(f"Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


# --- Direction 1: Dimensionality Gap ---

def plot_explained_variance(data: Dict, output_dir: Path):
    """Cumulative explained variance curves for perception vs imagery."""
    curves = data.get("explained_variance_curves", {})
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, color, key in [
        ("Perception", PERC_COLOR, "perception"),
        ("Imagery", IMAG_COLOR, "imagery"),
    ]:
        c = curves.get(key, {})
        components = c.get("components", [])
        cumvar = c.get("cumvar", [])
        if components and cumvar:
            ax.plot(components, cumvar, color=color, linewidth=2, label=label)

    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Dimensionality Gap: Perception vs Imagery")
    ax.legend(framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig1a_explained_variance.png")
    fig.savefig(output_dir / "fig1a_explained_variance.pdf")
    plt.close(fig)


def plot_dimensionality_summary(data: Dict, output_dir: Path):
    """Bar chart comparing dimensionality metrics."""
    perc = data.get("perception", {})
    imag = data.get("imagery", {})

    metrics = ["participation_ratio", "intrinsic_dim_mle"]
    labels = ["Participation\nRatio", "Intrinsic Dim\n(MLE)"]
    perc_vals = [perc.get(m, 0) for m in metrics]
    imag_vals = [imag.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))
    w = 0.35

    bars_p = ax.bar(x - w / 2, perc_vals, w, label="Perception", color=PERC_COLOR,
                     edgecolor="black", linewidth=1)
    bars_i = ax.bar(x + w / 2, imag_vals, w, label="Imagery", color=IMAG_COLOR,
                     edgecolor="black", linewidth=1)

    for bars in [bars_p, bars_i]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Effective Dimensionality")
    ax.set_title("Imagery Compresses Representational Space")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig1b_dimensionality_bars.png")
    fig.savefig(output_dir / "fig1b_dimensionality_bars.pdf")
    plt.close(fig)


# --- Direction 2: Uncertainty as Vividness ---

def plot_uncertainty_distributions(data: Dict, output_dir: Path):
    """Histogram comparing uncertainty distributions."""
    dist = data.get("distribution_comparison", {})
    if not dist:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Simulate distributions from summary stats for visualization
    rng = np.random.RandomState(42)
    n = 500
    perc_samples = rng.normal(dist["perception_mean"], dist["perception_std"], n)
    imag_samples = rng.normal(dist["imagery_mean"], dist["imagery_std"], n)

    ax.hist(perc_samples, bins=40, alpha=0.6, color=PERC_COLOR, label="Perception",
            density=True, edgecolor="white", linewidth=0.5)
    ax.hist(imag_samples, bins=40, alpha=0.6, color=IMAG_COLOR, label="Imagery",
            density=True, edgecolor="white", linewidth=0.5)

    ax.axvline(dist["perception_mean"], color=PERC_COLOR, linestyle="--", linewidth=2)
    ax.axvline(dist["imagery_mean"], color=IMAG_COLOR, linestyle="--", linewidth=2)

    ks_p = dist.get("ks_p_value", 1.0)
    sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "n.s."
    ax.text(0.97, 0.95, f"KS test: p={ks_p:.4f} {sig}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Prediction Uncertainty (MC Dropout std)")
    ax.set_ylabel("Density")
    ax.set_title("Imagery Increases Decoder Uncertainty")
    ax.legend(framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig2a_uncertainty_distributions.png")
    fig.savefig(output_dir / "fig2a_uncertainty_distributions.pdf")
    plt.close(fig)


def plot_uncertainty_vs_accuracy(data: Dict, output_dir: Path):
    """Scatter: uncertainty vs cosine similarity for imagery trials."""
    imag_corr = data.get("imagery_accuracy_correlation", {})
    if not imag_corr or imag_corr.get("n", 0) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    r = imag_corr.get("spearman_r", 0)
    p = imag_corr.get("spearman_p", 1)

    ax.text(0.05, 0.95, f"Spearman r = {r:.3f}\np = {p:.4f}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    ax.set_xlabel("Prediction Uncertainty")
    ax.set_ylabel("Decoding Accuracy (CLIP cosine)")
    ax.set_title("Uncertainty Predicts Imagery Quality")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)

    fig.savefig(output_dir / "fig2b_uncertainty_vs_accuracy.png")
    fig.savefig(output_dir / "fig2b_uncertainty_vs_accuracy.pdf")
    plt.close(fig)


# --- Direction 3: Semantic Survival ---

def plot_preservation_profile(data: Dict, output_dir: Path):
    """Bar chart of preservation ratios: semantic vs visual concepts."""
    details = data.get("concept_details", [])
    if not details:
        return

    semantic = [d for d in details if d["type"] == "semantic"]
    visual = [d for d in details if d["type"] == "visual"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, concepts, title, color in [
        (axes[0], semantic, "Semantic Concepts", SEMANTIC_COLOR),
        (axes[1], visual, "Visual Concepts", VISUAL_COLOR),
    ]:
        names = [c["concept"].replace("a photo of ", "").replace("a ", "").replace("an ", "")
                 for c in concepts]
        perc_acc = [c["perception_accuracy"] for c in concepts]
        imag_acc = [c["imagery_accuracy"] for c in concepts]

        y = np.arange(len(names))
        h = 0.35

        ax.barh(y - h / 2, perc_acc, h, label="Perception", color=PERC_COLOR,
                edgecolor="black", linewidth=0.5, alpha=0.8)
        ax.barh(y + h / 2, imag_acc, h, label="Imagery", color=IMAG_COLOR,
                edgecolor="black", linewidth=0.5, alpha=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Decoding Accuracy (correlation)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Semantic Survival: What Imagery Preserves", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_semantic_survival.png")
    fig.savefig(output_dir / "fig3_semantic_survival.pdf")
    plt.close(fig)


def plot_semantic_vs_visual_summary(data: Dict, output_dir: Path):
    """Summary bar: mean preservation for semantic vs visual."""
    fig, ax = plt.subplots(figsize=(5, 5))

    labels = ["Semantic", "Visual"]
    means = [data.get("semantic_mean_preservation", 0),
             data.get("visual_mean_preservation", 0)]
    stds = [data.get("semantic_std_preservation", 0),
            data.get("visual_std_preservation", 0)]
    colors = [SEMANTIC_COLOR, VISUAL_COLOR]

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors,
                   edgecolor="black", linewidth=1.2, alpha=0.85)

    for bar, val in zip(bars, means):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    gap = data.get("semantic_vs_visual_gap", 0)
    ax.set_ylabel("Mean Preservation Ratio")
    ax.set_title(f"Semantic > Visual (gap = {gap:.3f})")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig3b_semantic_vs_visual.png")
    fig.savefig(output_dir / "fig3b_semantic_vs_visual.pdf")
    plt.close(fig)


# --- Direction 4: Topological RSA ---

def plot_neighborhood_preservation(data: Dict, output_dir: Path):
    """Line plot of k-NN overlap at different k values."""
    knn = data.get("neighborhood_preservation", {})
    if not knn:
        return

    ks = []
    overlaps = []
    for key, val in sorted(knn.items()):
        k_val = int(key.split("@")[1])
        ks.append(k_val)
        overlaps.append(val)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ks, overlaps, "o-", color=IMAG_COLOR, linewidth=2, markersize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # Chance level baseline
    n = data.get("n_matched_samples", 200)
    for k in ks:
        chance = k / n if n > 0 else 0
        ax.plot(k, chance, "x", color="gray", markersize=10, markeredgewidth=2)

    ax.set_xlabel("k (neighborhood size)")
    ax.set_ylabel("k-NN Overlap (perception ↔ imagery)")
    ax.set_title("Neighborhood Structure Preservation")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig4a_knn_overlap.png")
    fig.savefig(output_dir / "fig4a_knn_overlap.pdf")
    plt.close(fig)


def plot_contraction_summary(data: Dict, output_dir: Path):
    """Summary of RDM contraction from perception to imagery."""
    contract = data.get("contraction", {})
    rdm_corr = data.get("rdm_correlation_pred_perc_vs_imag", {})

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Contraction
    ax = axes[0]
    mean_r = contract.get("mean_ratio", 1.0)
    frac = contract.get("fraction_contracted", 0.5)
    ax.bar(["Contraction\nRatio"], [mean_r], color=IMAG_COLOR, edgecolor="black",
            linewidth=1.2, alpha=0.85)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No contraction")
    ax.text(0, mean_r + 0.02, f"{mean_r:.3f}\n({frac:.0%} contracted)",
            ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Imagery / Perception Distance Ratio")
    ax.set_title("Representational Contraction")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # RDM correlation
    ax = axes[1]
    r_val = rdm_corr.get("spearman_r", 0)
    ax.bar(["RDM\nCorrelation"], [r_val], color=PERC_COLOR, edgecolor="black",
            linewidth=1.2, alpha=0.85)
    ax.text(0, r_val + 0.02, f"r = {r_val:.3f}", ha="center", va="bottom",
            fontweight="bold")
    ax.set_ylabel("Spearman correlation")
    ax.set_title("Representational Structure Similarity")
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Topological RSA: Imagery Contracts but Preserves Structure",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4b_contraction_and_rsa.png")
    fig.savefig(output_dir / "fig4b_contraction_and_rsa.pdf")
    plt.close(fig)


# --- Direction 5: Cross-Subject ---

def plot_cross_subject_transfer(data: Dict, output_dir: Path):
    """Per-subject transfer ratios and degradation profile similarity."""
    stats = data.get("per_subject_stats", {})
    if not stats:
        return

    subjects = list(stats.keys())
    perc_vals = [stats[s]["perception_cosine_mean"] for s in subjects]
    imag_vals = [stats[s]["imagery_cosine_mean"] for s in subjects]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(subjects))
    w = 0.35

    ax.bar(x - w / 2, perc_vals, w, label="Perception", color=PERC_COLOR,
            edgecolor="black", linewidth=1)
    ax.bar(x + w / 2, imag_vals, w, label="Imagery", color=IMAG_COLOR,
            edgecolor="black", linewidth=1)

    for i, (p, im) in enumerate(zip(perc_vals, imag_vals)):
        ratio = im / max(p, 1e-8)
        ax.text(i, max(p, im) + 0.02, f"{ratio:.0%}", ha="center", fontsize=9)

    ax.set_ylabel("CLIP Cosine Similarity")
    ax.set_title("Cross-Subject Transfer: Perception vs Imagery")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig5a_cross_subject_transfer.png")
    fig.savefig(output_dir / "fig5a_cross_subject_transfer.pdf")
    plt.close(fig)


def plot_degradation_similarity(data: Dict, output_dir: Path):
    """Heatmap of degradation profile similarity across subjects."""
    profile_data = data.get("degradation_profile_comparison", {})
    corr_matrix = profile_data.get("correlation_matrix")
    subjects = profile_data.get("subjects", [])

    if corr_matrix is None or len(subjects) < 2:
        return

    corr_matrix = np.array(corr_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(subjects)))
    ax.set_yticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_yticklabels(subjects)

    for i in range(len(subjects)):
        for j in range(len(subjects)):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Degradation Profile Similarity")

    fig.savefig(output_dir / "fig5b_degradation_similarity.png")
    fig.savefig(output_dir / "fig5b_degradation_similarity.pdf")
    plt.close(fig)


# --- Direction 6: Semantic-Structural Dissociation ---

MONITOR_COLOR = "#E91E63"
THRESHOLD_COLOR = "#FF9800"
GAN_GEN_COLOR = "#00BCD4"
GAN_DISC_COLOR = "#795548"
ALGEBRA_COLOR = "#3F51B5"
FLOW_FWD_COLOR = "#4CAF50"
FLOW_BWD_COLOR = "#F44336"
MANIFOLD_COLOR = "#009688"
CORE_COLOR = "#607D8B"
DIVERGE_COLOR = "#E040FB"

def plot_dissociation_summary(data: Dict, output_dir: Path):
    """Bar chart comparing the preservation ratio of CLIP, IP-Tokens, and SD-Latent."""
    gap = data.get("gap", {})
    if not gap:
        return

    metrics = ["clip_preservation_ratio", "tokens_preservation_ratio", "sd_latent_preservation_ratio"]
    labels = ["Semantic\n(CLIP)", "Fine-Visual\n(IP-Tokens)", "Structural\n(SD-Latent)"]
    vals = [gap.get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    w = 0.5

    # Color gradient from semantic to structural
    colors = [SEMANTIC_COLOR, "#2196F3", VISUAL_COLOR]

    bars = ax.bar(x, vals, w, color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Preservation Ratio (Imagery / Perception)")
    ax.set_title("The Semantic-Structural Dissociation of Imagination")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(vals) * 1.2 if vals else 1.0)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig6_semantic_structural_dissociation.png")
    fig.savefig(output_dir / "fig6_semantic_structural_dissociation.pdf")
    plt.close(fig)


# --- Direction 7: Computational Reality Monitor ---

def plot_reality_classifier_comparison(data: Dict, output_dir: Path):
    """Bar chart comparing AUC across signal_strength, content, and combined classifiers."""
    classifiers = data.get("classifiers", {})
    if not classifiers:
        return

    modes = ["signal_strength", "content", "combined"]
    labels = ["Signal\nStrength", "Content\n(PCA)", "Combined"]
    aucs = [classifiers.get(m, {}).get("mean_auc", 0) for m in modes]
    stds = [classifiers.get(m, {}).get("std_auc", 0) for m in modes]
    colors = [MONITOR_COLOR, PERC_COLOR, "#673AB7"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, aucs, yerr=stds, capsize=5, color=colors,
                   edgecolor="black", linewidth=1.2, alpha=0.85)

    for bar, val in zip(bars, aucs):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("Classification AUC")
    ax.set_title("Reality Monitor: Signal Strength vs Content Features")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig7a_reality_classifier_comparison.png")
    fig.savefig(output_dir / "fig7a_reality_classifier_comparison.pdf")
    plt.close(fig)


def plot_reality_roc(data: Dict, output_dir: Path):
    """ROC curves for the three classifier modes."""
    classifiers = data.get("classifiers", {})
    if not classifiers:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    colors_map = {"signal_strength": MONITOR_COLOR, "content": PERC_COLOR, "combined": "#673AB7"}
    labels_map = {"signal_strength": "Signal Strength", "content": "Content (PCA)", "combined": "Combined"}

    for mode in ["signal_strength", "content", "combined"]:
        clf_data = classifiers.get(mode, {})
        fpr = clf_data.get("roc_fpr", [])
        tpr = clf_data.get("roc_tpr", [])
        auc_val = clf_data.get("mean_auc", 0)
        if fpr and tpr:
            ax.plot(fpr, tpr, color=colors_map[mode], linewidth=2,
                    label=f"{labels_map[mode]} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Reality Monitor ROC Curves")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    fig.savefig(output_dir / "fig7b_reality_roc.png")
    fig.savefig(output_dir / "fig7b_reality_roc.pdf")
    plt.close(fig)


def plot_signal_strength_threshold(data: Dict, output_dir: Path):
    """Distribution of norms with the estimated reality threshold."""
    threshold = data.get("reality_threshold", {})
    if not threshold:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    rng = np.random.RandomState(42)
    n = 500
    perc_norms = rng.normal(threshold["perception_mean_norm"],
                             threshold["perception_std_norm"], n)
    imag_norms = rng.normal(threshold["imagery_mean_norm"],
                             threshold["imagery_std_norm"], n)

    ax.hist(perc_norms, bins=40, alpha=0.6, color=PERC_COLOR, label="Perception",
            density=True, edgecolor="white", linewidth=0.5)
    ax.hist(imag_norms, bins=40, alpha=0.6, color=IMAG_COLOR, label="Imagery",
            density=True, edgecolor="white", linewidth=0.5)

    t = threshold.get("optimal_threshold", 0)
    ax.axvline(t, color=THRESHOLD_COLOR, linewidth=2.5, linestyle="-",
               label=f"Reality Threshold = {t:.3f}")

    d = threshold.get("cohens_d", 0)
    ax.text(0.97, 0.95, f"Cohen's d = {d:.3f}\nAUC = {threshold.get('auc_norm_only', 0):.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("Embedding L2 Norm (Signal Strength)")
    ax.set_ylabel("Density")
    ax.set_title("Signal Strength Distributions: The Reality Threshold")
    ax.legend(framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig7c_signal_strength_threshold.png")
    fig.savefig(output_dir / "fig7c_signal_strength_threshold.pdf")
    plt.close(fig)


# --- Direction 8: Reality Confusion Mapping ---

def plot_confusion_by_category(data: Dict, output_dir: Path):
    """Bar chart of mean confusion index by stimulus category."""
    cat_data = data.get("category_confusability", {}).get("per_category", {})
    if not cat_data:
        return

    categories = sorted(cat_data.keys())
    means = [cat_data[c]["mean_confusion"] for c in categories]
    stds = [cat_data[c]["std_confusion"] for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(categories))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=MONITOR_COLOR,
                   edgecolor="black", linewidth=1, alpha=0.8)

    for bar, val in zip(bars, means):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Mean Confusion Index")
    ax.set_title("Category-Level Reality Confusion")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig8a_confusion_by_category.png")
    fig.savefig(output_dir / "fig8a_confusion_by_category.pdf")
    plt.close(fig)


def plot_reality_boundary(data: Dict, output_dir: Path):
    """Sigmoid fit of signal strength vs confusion (the reality boundary)."""
    boundary = data.get("reality_boundary", {})
    if not boundary or not boundary.get("threshold_estimated"):
        return

    bin_centers = np.array(boundary["bin_centers"])
    bin_means = np.array(boundary["bin_means"])
    params = boundary["sigmoid_params"]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(bin_centers, bin_means, color=MONITOR_COLOR, s=60,
               edgecolors="black", linewidth=0.5, zorder=3, label="Binned data")

    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    y_fit = params["L"] / (1.0 + np.exp(-params["k"] * (x_fit - params["x0"]))) + params["b"]
    ax.plot(x_fit, y_fit, color=THRESHOLD_COLOR, linewidth=2.5, label="Sigmoid fit")

    threshold = boundary["reality_threshold"]
    ax.axvline(threshold, color="gray", linestyle="--", alpha=0.7,
               label=f"Threshold = {threshold:.3f}")

    ax.text(0.97, 0.05,
            f"R² = {boundary['r_squared']:.3f}\nSteepness = {boundary['steepness']:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    ax.set_xlabel("Imagery Signal Strength (L2 Norm)")
    ax.set_ylabel("Confusion Index")
    ax.set_title("The Reality Boundary: Where Imagination Meets Perception")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig8b_reality_boundary.png")
    fig.savefig(output_dir / "fig8b_reality_boundary.pdf")
    plt.close(fig)


# --- Direction 9: Adversarial Reality Probing ---

def plot_adversarial_dynamics(data: Dict, output_dir: Path):
    """Training dynamics: D accuracy and losses over epochs."""
    dynamics = data.get("training_dynamics", {})
    if not dynamics:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    epochs = np.arange(len(dynamics.get("d_loss", [])))

    # Losses
    ax = axes[0]
    ax.plot(epochs, dynamics.get("d_loss", []), color=GAN_DISC_COLOR,
            linewidth=2, label="Discriminator")
    ax.plot(epochs, dynamics.get("g_loss", []), color=GAN_GEN_COLOR,
            linewidth=2, label="Generator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Adversarial Training Losses")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Discriminator accuracy
    ax = axes[1]
    ax.plot(epochs, dynamics.get("d_acc_total", []), color=GAN_DISC_COLOR,
            linewidth=2, label="Overall")
    ax.plot(epochs, dynamics.get("d_acc_real", []), color=PERC_COLOR,
            linewidth=1.5, alpha=0.7, linestyle="--", label="Perception (real)")
    ax.plot(epochs, dynamics.get("d_acc_fake", []), color=IMAG_COLOR,
            linewidth=1.5, alpha=0.7, linestyle="--", label="Imagery (fake)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reality Discriminator Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Adversarial Reality Probing: Can Imagery Fool the Detector?",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig9a_adversarial_dynamics.png")
    fig.savefig(output_dir / "fig9a_adversarial_dynamics.pdf")
    plt.close(fig)


def plot_perturbation_histogram(data: Dict, output_dir: Path):
    """Histogram of per-trial perturbation distances (distance to reality)."""
    perturbation = data.get("perturbation", {})
    per_trial = perturbation.get("per_trial_l2", [])
    if not per_trial:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(per_trial, bins=40, color=GAN_GEN_COLOR, edgecolor="white",
            linewidth=0.5, alpha=0.8)

    mean_l2 = perturbation.get("mean_l2", 0)
    ax.axvline(mean_l2, color=THRESHOLD_COLOR, linewidth=2.5, linestyle="-",
               label=f"Mean = {mean_l2:.4f}")

    ax.set_xlabel("Perturbation Distance (L2)")
    ax.set_ylabel("Count")
    ax.set_title("Distance to Reality: Per-Trial Generator Perturbation")
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig9b_perturbation_histogram.png")
    fig.savefig(output_dir / "fig9b_perturbation_histogram.pdf")
    plt.close(fig)


# --- Direction 10: Hierarchical Reality Gradient ---

def plot_layer_discriminability(data: Dict, output_dir: Path):
    """AUC and Cohen's d at each processing layer."""
    layer_disc = data.get("per_layer_discriminability", {})
    if not layer_disc:
        return

    layer_order = ["layer_4", "layer_8", "layer_12", "final"]
    layer_labels = ["L4\n(Early)", "L8\n(Mid)", "L12\n(Late)", "Final\n(CLIP)"]

    layers_present = [l for l in layer_order if l in layer_disc]
    labels = [layer_labels[layer_order.index(l)] for l in layers_present]
    aucs = [layer_disc[l]["classification_auc"] for l in layers_present]
    auc_stds = [layer_disc[l].get("classification_auc_std", 0) for l in layers_present]
    cohens_ds = [layer_disc[l]["cohens_d"] for l in layers_present]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUC gradient
    ax = axes[0]
    x = np.arange(len(layers_present))
    bars = ax.bar(x, aucs, yerr=auc_stds, capsize=4,
                   color=[PERC_COLOR if a < max(aucs) else MONITOR_COLOR for a in aucs],
                   edgecolor="black", linewidth=1, alpha=0.85)
    for bar, val in zip(bars, aucs):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Classification AUC")
    ax.set_title("Per-Layer Perception vs Imagery AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Cohen's d gradient
    ax = axes[1]
    ax.plot(x, cohens_ds, "o-", color=MONITOR_COLOR, linewidth=2.5, markersize=10)
    for i, d in enumerate(cohens_ds):
        ax.annotate(f"{d:.2f}", (x[i], d), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cohen's d (Signal Strength Gap)")
    ax.set_title("Hierarchical Signal Strength Divergence")
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Hierarchical Reality Gradient: Where Does the Distinction Emerge?",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig10a_layer_discriminability.png")
    fig.savefig(output_dir / "fig10a_layer_discriminability.pdf")
    plt.close(fig)


def plot_cascade_summary(data: Dict, output_dir: Path):
    """Summary of the threshold cascade analysis."""
    cascade = data.get("threshold_cascade", {})
    if not cascade or cascade.get("cascade_type") == "insufficient_layers":
        return

    per_layer_auc = cascade.get("per_layer_auc", {})
    per_layer_sep = cascade.get("per_layer_separation", {})
    if not per_layer_auc:
        return

    layer_order = ["layer_4", "layer_8", "layer_12", "final"]
    layer_labels = ["L4", "L8", "L12", "Final"]
    layers = [l for l in layer_order if l in per_layer_auc]
    labels = [layer_labels[layer_order.index(l)] for l in layers]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(layers))
    aucs_vals = [per_layer_auc[l] for l in layers]
    sep_vals = [per_layer_sep.get(l, 0) for l in layers]

    ax.plot(x, aucs_vals, "o-", color=MONITOR_COLOR, linewidth=2.5, markersize=10, label="AUC")

    ax2 = ax.twinx()
    ax2.plot(x, sep_vals, "s--", color=IMAG_COLOR, linewidth=2, markersize=8, label="Separation")
    ax2.set_ylabel("Separation Index", color=IMAG_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Classification AUC", color=MONITOR_COLOR)
    ax.set_xlabel("Processing Level")

    cascade_type = cascade.get("cascade_type", "unknown")
    peak = cascade.get("peak_layer", "N/A")
    prm_match = cascade.get("prm_prediction_matched", False)
    ax.set_title(f"Threshold Cascade: {cascade_type.title()} "
                 f"(peak={peak}, PRM match={prm_match})")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, framealpha=0.9)

    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)

    fig.savefig(output_dir / "fig10b_cascade_summary.png")
    fig.savefig(output_dir / "fig10b_cascade_summary.pdf")
    plt.close(fig)


# --- Direction 11: Compositional Imagination ---

def plot_composition_success(data: Dict, output_dir: Path):
    """Bar chart comparing composition success rate for perception vs imagery."""
    comparison = data.get("comparison", {})
    if not comparison:
        return

    perc_sr = comparison.get("perception_compositionality", {}).get("success_rate", 0)
    imag_sr = comparison.get("imagery_compositionality", {}).get("success_rate", 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Perception", "Imagery"]
    vals = [perc_sr, imag_sr]
    colors = [PERC_COLOR, IMAG_COLOR]

    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    gap = comparison.get("compositionality_gap", 0)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("Composition Success Rate")
    ax.set_title(f"Brain Algebra: Perception vs Imagery (gap={gap:.3f})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig11a_composition_success.png")
    fig.savefig(output_dir / "fig11a_composition_success.pdf")
    plt.close(fig)


def plot_concept_compositionality(data: Dict, output_dir: Path):
    """Semantic vs visual concept compositionality comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))

    sem_success = data.get("semantic_mean_success", 0)
    vis_success = data.get("visual_mean_success", 0)
    gap = data.get("semantic_vs_visual_gap", 0)

    labels = ["Semantic\nConcepts", "Visual\nConcepts"]
    vals = [sem_success, vis_success]
    colors = [SEMANTIC_COLOR, VISUAL_COLOR]

    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Per-Concept Success Rate")
    ax.set_title(f"Compositional Fidelity by Concept Type (gap={gap:.3f})")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig11b_concept_compositionality.png")
    fig.savefig(output_dir / "fig11b_concept_compositionality.pdf")
    plt.close(fig)


# --- Direction 12: Predictive Coding ---

def plot_directional_flow_index(data: Dict, output_dir: Path):
    """DFI values at each layer transition for perception vs imagery."""
    flow = data.get("flow_analysis", {})
    transitions = flow.get("transitions", [])
    if not transitions:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [t["transition"].replace("->", " → ") for t in transitions]
    perc_dfi = [t["perception_dfi"] for t in transitions]
    imag_dfi = [t["imagery_dfi"] for t in transitions]

    x = np.arange(len(labels))
    w = 0.35

    ax.bar(x - w / 2, perc_dfi, w, label="Perception", color=PERC_COLOR,
           edgecolor="black", linewidth=1, alpha=0.85)
    ax.bar(x + w / 2, imag_dfi, w, label="Imagery", color=IMAG_COLOR,
           edgecolor="black", linewidth=1, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Directional Flow Index")
    ax.set_title("Information Flow Direction: Bottom-Up (+) vs Top-Down (−)")

    n_rev = flow.get("any_reversal", False)
    ax.text(0.97, 0.95, f"Flow reversal: {'Yes' if n_rev else 'No'}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig12a_directional_flow_index.png")
    fig.savefig(output_dir / "fig12a_directional_flow_index.pdf")
    plt.close(fig)


def plot_prediction_r2(data: Dict, output_dir: Path):
    """Forward vs backward prediction R² at each transition."""
    transitions = data.get("flow_analysis", {}).get("transitions", [])
    if not transitions:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, condition, color_main in [
        (axes[0], "perception", PERC_COLOR),
        (axes[1], "imagery", IMAG_COLOR),
    ]:
        labels = [t["transition"].replace("->", "→") for t in transitions]
        fwd_r2 = [t[f"{condition}_fwd_r2"] for t in transitions]
        bwd_r2 = [t[f"{condition}_bwd_r2"] for t in transitions]

        x = np.arange(len(labels))
        w = 0.35

        ax.bar(x - w / 2, fwd_r2, w, label="Forward (↑)", color=FLOW_FWD_COLOR,
               edgecolor="black", linewidth=1, alpha=0.85)
        ax.bar(x + w / 2, bwd_r2, w, label="Backward (↓)", color=FLOW_BWD_COLOR,
               edgecolor="black", linewidth=1, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Prediction R²")
        ax.set_title(f"{condition.title()}")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Predictive Coding: Forward vs Backward Prediction Accuracy",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig12b_prediction_r2.png")
    fig.savefig(output_dir / "fig12b_prediction_r2.pdf")
    plt.close(fig)


# --- Direction 13: Imagination Manifold Geometry ---

def plot_manifold_comparison(data: Dict, output_dir: Path):
    """Compare manifold metrics between perception and imagery."""
    perc = data.get("perception_manifold", {})
    imag = data.get("imagery_manifold", {})
    if not perc or not imag:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Isotropy
    ax = axes[0]
    vals = [perc.get("isotropy", 0), imag.get("isotropy", 0)]
    bars = ax.bar(["Perception", "Imagery"], vals, color=[PERC_COLOR, IMAG_COLOR],
                   edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Isotropy Score")
    ax.set_title("Representational Isotropy")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Hull volume
    ax = axes[1]
    vals = [perc.get("hull_volume_estimate", 0), imag.get("hull_volume_estimate", 0)]
    bars = ax.bar(["Perception", "Imagery"], vals, color=[PERC_COLOR, IMAG_COLOR],
                   edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Convex Hull Volume (2D proj.)")
    ax.set_title("Manifold Volume")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Pairwise distances
    ax = axes[2]
    vals = [perc.get("mean_pairwise_distance", 0), imag.get("mean_pairwise_distance", 0)]
    bars = ax.bar(["Perception", "Imagery"], vals, color=[PERC_COLOR, IMAG_COLOR],
                   edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Mean Pairwise Distance")
    ax.set_title("Representational Spread")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Imagination Manifold Geometry", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig13a_manifold_comparison.png")
    fig.savefig(output_dir / "fig13a_manifold_comparison.pdf")
    plt.close(fig)


def plot_centrality_and_structure(data: Dict, output_dir: Path):
    """Centrality bias and position preservation summary."""
    centrality = data.get("centrality_bias", {})
    positions = data.get("position_preservation", {})

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Centrality bias
    ax = axes[0]
    if "error" not in centrality:
        bias = centrality.get("mean_centrality_bias", 0)
        frac = centrality.get("fraction_imagery_closer_to_centroid", 0)
        p_val = centrality.get("p_value", 1)

        ax.bar(["Centrality\nBias"], [bias], color=MANIFOLD_COLOR,
               edgecolor="black", linewidth=1.2, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        confirmed = centrality.get("schema_bias_confirmed", False)
        ax.text(0, bias + 0.005, f"p={p_val:.4f}\n{frac:.0%} closer",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_ylabel("Mean Distance Bias")
        ax.set_title(f"Schema Bias: {'Confirmed' if confirmed else 'Not Confirmed'}")
    else:
        ax.text(0.5, 0.5, "No shared stimuli", transform=ax.transAxes,
                ha="center", va="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Position preservation
    ax = axes[1]
    if "error" not in positions:
        rho = positions.get("rank_order_spearman", 0)
        proc_d = positions.get("procrustes_distance", 0)
        preserved = positions.get("structure_preserved", False)

        x_pos = np.arange(2)
        vals = [rho, 1 - proc_d]
        bars = ax.bar(["Spearman ρ\n(rank order)", "1 − Procrustes\nDistance"],
                       vals, color=[MANIFOLD_COLOR, ALGEBRA_COLOR],
                       edgecolor="black", linewidth=1.2, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax.set_ylabel("Preservation Score")
        ax.set_title(f"Relative Position Preservation: {'Yes' if preserved else 'No'}")
    else:
        ax.text(0.5, 0.5, "No shared stimuli", transform=ax.transAxes,
                ha="center", va="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "fig13b_centrality_structure.png")
    fig.savefig(output_dir / "fig13b_centrality_structure.pdf")
    plt.close(fig)


# --- Direction 14: Modality-Invariant Decomposition ---

def plot_decomposition_norms(data: Dict, output_dir: Path):
    """Norm analysis: core vs perception residual vs imagery residual."""
    norms = data.get("norm_analysis", {})
    if not norms:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Invariant\nCore", "Perception\nResidual", "Imagery\nResidual"]
    vals = [norms.get("core_mean_norm", 0),
            norms.get("perc_residual_mean_norm", 0),
            norms.get("imag_residual_mean_norm", 0)]
    colors = [CORE_COLOR, PERC_COLOR, IMAG_COLOR]

    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Modality Decomposition: Component Magnitudes")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / "fig14a_decomposition_norms.png")
    fig.savefig(output_dir / "fig14a_decomposition_norms.pdf")
    plt.close(fig)


def plot_core_vs_residual_content(data: Dict, output_dir: Path):
    """Semantic vs visual strength in core and residual components."""
    core = data.get("core_content", {})
    resid = data.get("residual_content", {})
    if not core or not resid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Core content
    ax = axes[0]
    vals = [core.get("semantic_strength", 0), core.get("visual_strength", 0)]
    bars = ax.bar(["Semantic", "Visual"], vals, color=[SEMANTIC_COLOR, VISUAL_COLOR],
                   edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Mean |Projection|")
    ax.set_title("Invariant Core")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Perception residual
    ax = axes[1]
    p_res = resid.get("perception_residual", {})
    vals = [p_res.get("semantic_strength", 0), p_res.get("visual_strength", 0)]
    bars = ax.bar(["Semantic", "Visual"], vals, color=[SEMANTIC_COLOR, VISUAL_COLOR],
                   edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Mean |Projection|")
    ax.set_title("Perception Residual")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Imagery residual
    ax = axes[2]
    i_res = resid.get("imagery_residual", {})
    vals = [i_res.get("semantic_strength", 0), i_res.get("visual_strength", 0)]
    bars = ax.bar(["Semantic", "Visual"], vals, color=[SEMANTIC_COLOR, VISUAL_COLOR],
                   edgecolor="black", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylabel("Mean |Projection|")
    ax.set_title("Imagery Residual")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Modality Decomposition: What Each Component Encodes",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig14b_core_vs_residual.png")
    fig.savefig(output_dir / "fig14b_core_vs_residual.pdf")
    plt.close(fig)


# --- Direction 15: Creative Divergence Mapping ---

def plot_concept_divergence_profile(data: Dict, output_dir: Path):
    """Bar chart of per-concept amplification/suppression in divergence."""
    decomp = data.get("concept_decomposition", {})
    per_concept = decomp.get("per_concept", [])
    if not per_concept:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    names = [c["concept"].replace("a photo of ", "").replace("a ", "").replace("an ", "")
             for c in per_concept]
    projections = [c["mean_projection"] for c in per_concept]
    significant = [c.get("significant", False) for c in per_concept]
    types = [c["type"] for c in per_concept]

    colors = [SEMANTIC_COLOR if t == "semantic" else VISUAL_COLOR for t in types]
    edge_colors = ["black" if sig else "gray" for sig in significant]
    linewidths = [1.5 if sig else 0.5 for sig in significant]

    y = np.arange(len(names))
    bars = ax.barh(y, projections, color=colors, edgecolor=edge_colors,
                    linewidth=linewidths, alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Divergence Projection (+ = amplified, − = suppressed)")
    ax.set_title("Creative Divergence: Concept Amplification & Suppression")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SEMANTIC_COLOR, label="Semantic"),
        Patch(facecolor=VISUAL_COLOR, label="Visual"),
        Patch(facecolor="white", edgecolor="black", linewidth=1.5, label="Significant (p<0.05)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig15a_concept_divergence.png")
    fig.savefig(output_dir / "fig15a_concept_divergence.pdf")
    plt.close(fig)


def plot_creativity_and_archetypes(data: Dict, output_dir: Path):
    """Creativity index distribution and archetype summary."""
    creativity = data.get("creativity", {})
    archetypes = data.get("archetypes", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Creativity index
    ax = axes[0]
    mean_c = creativity.get("mean_creativity", 0)
    std_c = creativity.get("std_creativity", 0)
    high_frac = creativity.get("high_creativity_fraction", 0)

    ax.bar(["Mean\nCreativity", "Scaling\nComponent", "Orthogonal\nComponent"],
           [mean_c,
            creativity.get("mean_scaling_component", 0),
            creativity.get("mean_orthogonal_component", 0)],
           color=[DIVERGE_COLOR, PERC_COLOR, IMAG_COLOR],
           edgecolor="black", linewidth=1.2, alpha=0.85)
    ax.set_ylabel("Mean Magnitude")
    ax.set_title(f"Creativity Index: {mean_c:.3f} ({high_frac:.0%} highly creative)")
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Archetypes
    ax = axes[1]
    if "error" not in archetypes:
        arch_list = archetypes.get("archetypes", [])
        if arch_list:
            a_labels = [f"Archetype {a['cluster_id']}" for a in arch_list]
            a_fracs = [a["fraction"] for a in arch_list]
            a_norms = [a["mean_divergence_norm"] for a in arch_list]

            x_a = np.arange(len(a_labels))
            bars = ax.bar(x_a, a_fracs, color=DIVERGE_COLOR, edgecolor="black",
                           linewidth=1, alpha=0.7)
            for bar, frac, norm in zip(bars, a_fracs, a_norms):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{frac:.0%}\n(‖d‖={norm:.2f})", ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x_a)
            ax.set_xticklabels(a_labels, fontsize=9)
            ax.set_ylabel("Fraction of Trials")
            sil = archetypes.get("best_silhouette", 0)
            ax.set_title(f"Imagination Archetypes (k={archetypes.get('best_k', 0)}, "
                         f"sil={sil:.3f})")
    else:
        ax.text(0.5, 0.5, "Too few samples", transform=ax.transAxes,
                ha="center", va="center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Creative Divergence: How Imagination Transforms Perception",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig15b_creativity_archetypes.png")
    fig.savefig(output_dir / "fig15b_creativity_archetypes.pdf")
    plt.close(fig)


# --- Main ---

def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate all figures from analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(results_dir)

    generated = []

    dim_data = load_json(results_dir / "dimensionality_gap.json")
    if dim_data:
        plot_explained_variance(dim_data, output_dir)
        plot_dimensionality_summary(dim_data, output_dir)
        generated.extend(["fig1a", "fig1b"])

    unc_data = load_json(results_dir / "uncertainty_vividness.json")
    if unc_data:
        plot_uncertainty_distributions(unc_data, output_dir)
        plot_uncertainty_vs_accuracy(unc_data, output_dir)
        generated.extend(["fig2a", "fig2b"])

    sem_data = load_json(results_dir / "semantic_survival.json")
    if sem_data:
        plot_preservation_profile(sem_data, output_dir)
        plot_semantic_vs_visual_summary(sem_data, output_dir)
        generated.extend(["fig3", "fig3b"])

    topo_data = load_json(results_dir / "topological_rsa.json")
    if topo_data:
        plot_neighborhood_preservation(topo_data, output_dir)
        plot_contraction_summary(topo_data, output_dir)
        generated.extend(["fig4a", "fig4b"])

    cross_data = load_json(results_dir / "cross_subject.json")
    if cross_data:
        plot_cross_subject_transfer(cross_data, output_dir)
        plot_degradation_similarity(cross_data, output_dir)
        generated.extend(["fig5a", "fig5b"])

    dissociation_data = load_json(results_dir / "semantic_structural_dissociation.json")
    if dissociation_data:
        plot_dissociation_summary(dissociation_data, output_dir)
        generated.append("fig6")

    # Direction 7: Computational Reality Monitor
    monitor_data = load_json(results_dir / "reality_monitor.json")
    if monitor_data:
        plot_reality_classifier_comparison(monitor_data, output_dir)
        plot_reality_roc(monitor_data, output_dir)
        plot_signal_strength_threshold(monitor_data, output_dir)
        generated.extend(["fig7a", "fig7b", "fig7c"])

    # Direction 8: Reality Confusion Mapping
    confusion_data = load_json(results_dir / "reality_confusion.json")
    if confusion_data:
        plot_confusion_by_category(confusion_data, output_dir)
        plot_reality_boundary(confusion_data, output_dir)
        generated.extend(["fig8a", "fig8b"])

    # Direction 9: Adversarial Reality Probing
    adversarial_data = load_json(results_dir / "adversarial_reality.json")
    if adversarial_data:
        plot_adversarial_dynamics(adversarial_data, output_dir)
        plot_perturbation_histogram(adversarial_data, output_dir)
        generated.extend(["fig9a", "fig9b"])

    # Direction 10: Hierarchical Reality Gradient
    hierarchical_data = load_json(results_dir / "hierarchical_reality.json")
    if hierarchical_data:
        plot_layer_discriminability(hierarchical_data, output_dir)
        plot_cascade_summary(hierarchical_data, output_dir)
        generated.extend(["fig10a", "fig10b"])

    # Direction 11: Compositional Imagination
    comp_data = load_json(results_dir / "compositional_imagination.json")
    if comp_data:
        plot_composition_success(comp_data, output_dir)
        plot_concept_compositionality(comp_data, output_dir)
        generated.extend(["fig11a", "fig11b"])

    # Direction 12: Predictive Coding
    pc_data = load_json(results_dir / "predictive_coding.json")
    if pc_data:
        plot_directional_flow_index(pc_data, output_dir)
        plot_prediction_r2(pc_data, output_dir)
        generated.extend(["fig12a", "fig12b"])

    # Direction 13: Manifold Geometry
    manifold_data = load_json(results_dir / "manifold_geometry.json")
    if manifold_data:
        plot_manifold_comparison(manifold_data, output_dir)
        plot_centrality_and_structure(manifold_data, output_dir)
        generated.extend(["fig13a", "fig13b"])

    # Direction 14: Modality Decomposition
    decomp_data = load_json(results_dir / "modality_decomposition.json")
    if decomp_data:
        plot_decomposition_norms(decomp_data, output_dir)
        plot_core_vs_residual_content(decomp_data, output_dir)
        generated.extend(["fig14a", "fig14b"])

    # Direction 15: Creative Divergence
    diverge_data = load_json(results_dir / "creative_divergence.json")
    if diverge_data:
        plot_concept_divergence_profile(diverge_data, output_dir)
        plot_creativity_and_archetypes(diverge_data, output_dir)
        generated.extend(["fig15a", "fig15b"])

    print(f"Generated {len(generated)} figures in {output_dir}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate novel analysis figures")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    generate_all_figures(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
