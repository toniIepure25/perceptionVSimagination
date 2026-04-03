from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER1_DIR = ROOT / "docs" / "paper1"
ASSETS_DIR = PAPER1_DIR / "assets"
FIGURES_DIR = PAPER1_DIR / "figures"
TABLES_DIR = PAPER1_DIR / "tables"
SOURCE_DATA_PATH = ASSETS_DIR / "paper1_source_data.json"


COLORS = {
    "ridge": "#2E3440",
    "shared_only": "#1F77B4",
    "shared_private_primary": "#C97A00",
    "shared_private_secondary": "#C9A227",
    "shared_private_base": "#9C6644",
    "shared_private_control": "#8A8D91",
    "scarcity": "#C44E52",
    "neutral_fill": "#F5F7FA",
    "neutral_edge": "#D9E1EA",
    "success": "#2A9D8F",
}


def _ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> dict:
    return json.loads(SOURCE_DATA_PATH.read_text())


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _fmt_float(value: float | None, digits: int = 5) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _role_short(role_label: str) -> str:
    mapping = {
        "external baseline": "External baseline",
        "canonical neural baseline": "Canonical neural baseline",
        "exploratory hypothesis model": "Exploratory hypothesis model",
        "canonical hypothesis-family baseline": "Hypothesis-family baseline",
        "diagnostic recovery variant": "Diagnostic recovery variant",
        "diagnostic control": "Diagnostic control",
    }
    return mapping.get(role_label, role_label)


def build_benchmark_ladder_figure(data: dict) -> None:
    results = data["benchmark_results"]
    order = [
        "Ridge",
        "Shared-only",
        "Shared-private p16",
        "Shared-private p8",
        "Shared-private",
        "Shared-private no-domain",
    ]
    rows = [next(item for item in results if item["model"] == model) for model in order]
    y = np.arange(len(rows))
    cosines = [item["cosine"] for item in rows]
    colors = [
        COLORS["ridge"],
        COLORS["shared_only"],
        COLORS["shared_private_primary"],
        COLORS["shared_private_secondary"],
        COLORS["shared_private_base"],
        COLORS["shared_private_control"],
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.barh(y, cosines, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_yticks(y, [item["model"] for item in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Test cosine similarity", fontsize=11)
    ax.set_title("Figure 1. Frozen benchmark ladder on the current low-overlap dataset", fontsize=13, weight="bold")
    ax.grid(axis="x", color="#E6E9EF", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#AAB2BD")
    ax.set_xlim(0.0, max(cosines) * 1.18)

    for idx, item in enumerate(rows):
        ax.text(
            item["cosine"] + 0.01,
            idx,
            f"{item['cosine']:.5f}  |  {_role_short(item['role_label'])}",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    fig.text(
        0.01,
        -0.02,
        "Higher is better. Shared-only is the strongest canonical neural baseline; Ridge remains the strongest overall reference baseline.",
        fontsize=9,
        color="#444444",
    )
    _save_figure(fig, "figure1_benchmark_ladder")


def build_overlap_scarcity_figure(data: dict) -> None:
    dataset = data["dataset"]
    fig = plt.figure(figsize=(10.5, 5.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.25)

    def stat_box(ax, value: str, label: str, color: str) -> None:
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle((0.02, 0.08), 0.96, 0.84, facecolor=COLORS["neutral_fill"], edgecolor=COLORS["neutral_edge"], linewidth=1.5)
        )
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=24, weight="bold", color=color)
        ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=11, color="#333333", wrap=True)

    stat_box(fig.add_subplot(gs[0, 0]), str(dataset["rows"]), "Total benchmark rows", COLORS["scarcity"])
    stat_box(fig.add_subplot(gs[0, 1]), str(dataset["shared_paired_nsd_ids"]), "Shared paired stimulus IDs", COLORS["scarcity"])
    stat_box(fig.add_subplot(gs[0, 2]), str(dataset["held_out_paired_groups"]), "Held-out paired evaluation groups", COLORS["scarcity"])

    ax = fig.add_subplot(gs[1, :])
    subjects = [item["subject"] for item in dataset["subject_breakdown"]]
    rows = [item["rows"] for item in dataset["subject_breakdown"]]
    overlap_ids = [item["overlap_ids"] for item in dataset["subject_breakdown"]]
    bars = ax.bar(subjects, rows, color=COLORS["success"], edgecolor="white", linewidth=1.0)
    ax.set_ylabel("Mixed rows", fontsize=11)
    ax.set_title(
        "Figure 2. The current benchmark is constrained by severe overlap scarcity",
        fontsize=13,
        weight="bold",
        pad=12,
    )
    ax.grid(axis="y", color="#E6E9EF", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#AAB2BD")
    ax.spines["bottom"].set_color("#AAB2BD")
    ax.set_ylim(0, max(rows) + 9)
    for bar, overlap in zip(bars, overlap_ids):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{int(bar.get_height())} rows\n{overlap} paired ids",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.text(
        0.01,
        0.01,
        "Public NSD-Imagery is already integrated. The current evidence boundary is shaped by data scarcity, not by a missing acquisition path.",
        fontsize=9,
        color="#444444",
    )
    _save_figure(fig, "figure2_overlap_scarcity")


def build_shared_only_vs_shared_private_figure(data: dict) -> None:
    results = {item["model"]: item for item in data["benchmark_results"]}
    order = ["Shared-only", "Shared-private p16", "Shared-private p8", "Shared-private", "Shared-private no-domain"]
    rows = [results[name] for name in order]
    colors = [
        COLORS["shared_only"],
        COLORS["shared_private_primary"],
        COLORS["shared_private_secondary"],
        COLORS["shared_private_base"],
        COLORS["shared_private_control"],
    ]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={"wspace": 0.3})

    cosines = [item["cosine"] for item in rows]
    axes[0].barh(y, cosines, color=colors, edgecolor="white", linewidth=1.0)
    axes[0].set_yticks(y, [item["model"] for item in rows], fontsize=10)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test cosine similarity", fontsize=11)
    axes[0].set_title("A. Higher is better", fontsize=11, weight="bold")
    axes[0].grid(axis="x", color="#E6E9EF", linewidth=0.8)
    axes[0].set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axes[0].spines[spine].set_visible(False)
    axes[0].spines["bottom"].set_color("#AAB2BD")
    for idx, item in enumerate(rows):
        axes[0].text(item["cosine"] + 0.008, idx, f"{item['cosine']:.5f}", va="center", fontsize=9)

    mses = [item["mse"] for item in rows]
    axes[1].barh(y, mses, color=colors, edgecolor="white", linewidth=1.0)
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Test mean squared error", fontsize=11)
    axes[1].set_title("B. Lower is better", fontsize=11, weight="bold")
    axes[1].grid(axis="x", color="#E6E9EF", linewidth=0.8)
    axes[1].set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axes[1].spines[spine].set_visible(False)
    axes[1].spines["bottom"].set_color("#AAB2BD")
    for idx, item in enumerate(rows):
        axes[1].text(item["mse"] + 0.00003, idx, f"{item['mse']:.6f}", va="center", fontsize=9)

    fig.suptitle("Figure 3. Shared-only currently beats every shared-private variant tested", fontsize=13, weight="bold")
    fig.text(
        0.01,
        -0.02,
        "Reduced private capacity improves the shared-private family, but the best exploratory variant (p16) still remains below shared-only.",
        fontsize=9,
        color="#444444",
    )
    _save_figure(fig, "figure3_shared_only_vs_shared_private")


def build_threshold_hypothesis_figure(data: dict) -> None:
    current_overlap = data["dataset"]["shared_paired_nsd_ids"]
    x = np.linspace(0, 100, 300)
    ridge_curve = 0.78 - 0.0022 * x
    shared_only_curve = 0.22 + 0.0035 * x - 0.000015 * (x ** 2)
    shared_private_curve = 0.07 + 0.0012 * x + 0.0028 * np.maximum(0, x - 35)
    shared_private_curve = np.clip(shared_private_curve, 0, 0.82)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, ridge_curve, color=COLORS["ridge"], linewidth=2.2, label="Ridge (reference baseline)")
    ax.plot(x, shared_only_curve, color=COLORS["shared_only"], linewidth=2.2, label="Shared-only (current practical neural lane)")
    ax.plot(x, shared_private_curve, color=COLORS["shared_private_primary"], linewidth=2.6, linestyle="--", label="Shared-private (threshold hypothesis)")
    ax.axvline(current_overlap, color=COLORS["scarcity"], linestyle=":", linewidth=2.0)
    ax.text(current_overlap + 1.5, 0.06, "Current benchmark\n(5 paired IDs)", color=COLORS["scarcity"], fontsize=9, va="bottom")
    ax.axvspan(0, 20, color="#FDECEC", alpha=0.7)
    ax.text(7, 0.8, "Low-overlap\nscarcity regime", ha="center", va="center", fontsize=10, color="#6B2D2F")
    ax.axvspan(20, 55, color="#FFF7E6", alpha=0.6)
    ax.text(37.5, 0.8, "Hypothesized transition zone", ha="center", va="center", fontsize=10, color="#7A5A00")
    ax.axvspan(55, 100, color="#ECF8F4", alpha=0.7)
    ax.text(77.5, 0.8, "Potential larger-overlap regime", ha="center", va="center", fontsize=10, color="#216F63")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("Paired overlap scale (schematic, not empirical)", fontsize=11)
    ax.set_ylabel("Expected relative suitability", fontsize=11)
    ax.set_title("Figure 4. Threshold hypothesis schematic for future paired-data expansion", fontsize=13, weight="bold")
    ax.grid(color="#E6E9EF", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(
        0.01,
        -0.02,
        "This figure is conceptual only. It visualizes the program hypothesis that explicit shared/private structure may help only after materially larger paired overlap is available.",
        fontsize=9,
        color="#444444",
    )
    _save_figure(fig, "figure4_threshold_hypothesis")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body]) + "\n"


def build_tables(data: dict) -> None:
    results = data["benchmark_results"]
    results_rows = [
        [
            item["model"],
            _role_short(item["role_label"]),
            _fmt_float(item["cosine"]),
            _fmt_float(item["mse"], digits=6),
            _fmt_float(item["imagery_mean_cosine"]),
            _fmt_float(item["perception_mean_cosine"]),
            str(item["paired_eval_groups"]),
        ]
        for item in results
    ]
    (TABLES_DIR / "TABLE_1_MAIN_RESULTS.md").write_text(
        "# Table 1. Main benchmark results\n\n"
        + _markdown_table(
            ["Model", "Role", "Cosine", "MSE", "Imagery mean", "Perception mean", "Paired groups"],
            results_rows,
        )
        + "\nMissing imagery/perception means indicate that the currently frozen evidence bundle does not report those condition-specific summaries for that variant.\n"
    )

    claims_rows = []
    for item in data["claims_boundary"]:
        status = item["status"].replace("_", " ")
        claims_rows.append([status, item["claim"], item["missing_for_stronger_claim"] or "—"])
    (TABLES_DIR / "TABLE_2_CLAIMS_EVIDENCE_BOUNDARY.md").write_text(
        "# Table 2. Claims and evidence boundary\n\n"
        + _markdown_table(["Status", "Claim", "What is still missing"], claims_rows)
    )

    repro_rows = [
        [item["stage"], item["config"], item["workflow"], item["artifact"]]
        for item in data["reproducibility_contract"]
    ]
    (TABLES_DIR / "TABLE_3_REPRODUCIBILITY_ARTIFACT_CONTRACT.md").write_text(
        "# Table 3. Reproducibility and artifact contract\n\n"
        + _markdown_table(["Stage", "Config", "Workflow", "Primary artifact"], repro_rows)
    )


def main() -> int:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    _ensure_dirs()
    data = _load_data()
    build_benchmark_ladder_figure(data)
    build_overlap_scarcity_figure(data)
    build_shared_only_vs_shared_private_figure(data)
    build_threshold_hypothesis_figure(data)
    build_tables(data)
    print("Paper 1 assets generated:")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"  Tables:  {TABLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
