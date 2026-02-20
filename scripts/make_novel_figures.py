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
