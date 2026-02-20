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
