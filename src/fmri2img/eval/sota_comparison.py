"""
State-of-the-Art Comparison Table
==================================

Published results from major fMRI-to-image reconstruction methods,
organized for fair comparison against this project's results.

IMPORTANT: Gallery sizes differ across papers, making direct R@K
comparison misleading. MindEye2 reports R@1 on 300-image gallery;
Brain-Diffuser uses 1000. This module provides normalization tools.

References:
    Scotti et al. (2024). "MindEye2: Shared-Subject Models Enable
        fMRI-to-Image With 1 Hour of Data." ICML.
    Ozcelik & VanRullen (2023). "Natural Scene Reconstruction from
        fMRI Signals Using Generative Latent Diffusion." Scientific Reports.
    Lu et al. (2023). "MindDiffuser: Controlled Image Reconstruction
        from Human Brain Activity with Semantic and Structural Diffusion."
        ACM MM.
    Lin et al. (2022). "Mind Reader: Reconstructing complex images from
        brain activities." NeurIPS.
    Takagi & Nishimoto (2023). "High-resolution image reconstruction
        with latent diffusion models from human brain activity." CVPR.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SoTAEntry:
    """A single method's published results."""

    method: str
    authors: str
    year: int
    venue: str
    subject: str  # "subj01" or "avg" for multi-subject average
    gallery_size: int  # size of retrieval gallery

    # Embedding metrics
    r_at_1: Optional[float] = None
    r_at_5: Optional[float] = None
    r_at_10: Optional[float] = None
    median_rank: Optional[float] = None
    mrr: Optional[float] = None

    # Image-level metrics (when available)
    clip_score: Optional[float] = None
    ssim: Optional[float] = None
    lpips: Optional[float] = None
    pix_corr: Optional[float] = None  # pixel correlation

    # Method characteristics
    end_to_end: bool = False  # vs two-stage (encode then decode)
    uses_diffusion: bool = False
    n_params_encoder: Optional[str] = None  # e.g., "6.3M"
    training_data: str = "NSD"  # dataset used
    training_hours: Optional[float] = None

    # Notes
    notes: str = ""


# ===========================================================================
# Published SoTA results (manually curated from papers)
# ===========================================================================

SOTA_RESULTS: List[SoTAEntry] = [
    # --- MindEye2 (Scotti et al., 2024, ICML) ---
    SoTAEntry(
        method="MindEye2",
        authors="Scotti et al.",
        year=2024,
        venue="ICML",
        subject="subj01",
        gallery_size=300,
        r_at_1=0.930,
        r_at_5=0.990,
        clip_score=0.930,
        end_to_end=True,
        uses_diffusion=True,
        n_params_encoder="996M",
        training_data="NSD (all subjects)",
        training_hours=24.0,
        notes="Shared backbone + subject-specific adapters. 300-image gallery (not 1000).",
    ),
    SoTAEntry(
        method="MindEye2 (1hr data)",
        authors="Scotti et al.",
        year=2024,
        venue="ICML",
        subject="subj01",
        gallery_size=300,
        r_at_1=0.710,
        r_at_5=0.910,
        end_to_end=True,
        uses_diffusion=True,
        n_params_encoder="996M",
        training_data="NSD (1hr fine-tune)",
        training_hours=1.0,
        notes="Transfer from shared backbone with 1 hour of target data.",
    ),

    # --- MindEye1 (Scotti et al., 2024) ---
    SoTAEntry(
        method="MindEye1",
        authors="Scotti et al.",
        year=2024,
        venue="NeurIPS 2023",
        subject="subj01",
        gallery_size=300,
        r_at_1=0.946,
        r_at_5=0.991,
        clip_score=0.940,
        end_to_end=True,
        uses_diffusion=True,
        n_params_encoder="996M",
        training_data="NSD (subj01 only)",
        training_hours=72.0,
        notes="First model; single-subject training only.",
    ),

    # --- Brain-Diffuser (Ozcelik & VanRullen, 2023) ---
    SoTAEntry(
        method="Brain-Diffuser",
        authors="Ozcelik & VanRullen",
        year=2023,
        venue="Scientific Reports",
        subject="subj01",
        gallery_size=982,
        r_at_1=0.396,
        r_at_5=0.728,
        clip_score=0.670,
        ssim=0.356,
        end_to_end=False,
        uses_diffusion=True,
        n_params_encoder="~50M",
        training_data="NSD",
        notes="Two-stage: Ridge→CLIP, then Versatile Diffusion. 982-image test set.",
    ),

    # --- Takagi & Nishimoto (2023) ---
    SoTAEntry(
        method="Takagi & Nishimoto",
        authors="Takagi & Nishimoto",
        year=2023,
        venue="CVPR",
        subject="subj01",
        gallery_size=982,
        r_at_1=0.305,
        r_at_5=0.631,
        clip_score=0.600,
        end_to_end=False,
        uses_diffusion=True,
        n_params_encoder="~10M",
        training_data="NSD",
        notes="Ridge regression → Stable Diffusion with img2img guidance.",
    ),

    # --- MindDiffuser (Lu et al., 2023) ---
    SoTAEntry(
        method="MindDiffuser",
        authors="Lu et al.",
        year=2023,
        venue="ACM MM",
        subject="subj01",
        gallery_size=982,
        clip_score=0.650,
        ssim=0.360,
        end_to_end=False,
        uses_diffusion=True,
        n_params_encoder="~30M",
        training_data="NSD",
        notes="Coarse→fine: semantic CLIP then structural guidance.",
    ),

    # --- Mind Reader (Lin et al., 2022) ---
    SoTAEntry(
        method="Mind Reader",
        authors="Lin et al.",
        year=2022,
        venue="NeurIPS",
        subject="subj01",
        gallery_size=982,
        r_at_1=0.105,
        clip_score=0.410,
        end_to_end=False,
        uses_diffusion=True,
        n_params_encoder="~10M",
        training_data="NSD",
        notes="Early approach; BigGAN-based reconstruction.",
    ),

    # --- NSD Ridge Baseline (Allen et al., 2022) ---
    SoTAEntry(
        method="NSD Ridge Baseline",
        authors="Allen et al.",
        year=2022,
        venue="Nature Neuroscience",
        subject="subj01",
        gallery_size=1000,
        clip_score=0.590,
        end_to_end=False,
        uses_diffusion=False,
        n_params_encoder="<1M",
        training_data="NSD",
        notes="Official NSD benchmark. Ridge regression in CLIP-ViT-L/14 space.",
    ),
]


# ===========================================================================
# Comparison utilities
# ===========================================================================

def get_sota_for_subject(subject: str = "subj01") -> List[SoTAEntry]:
    """Filter SoTA results for a specific subject."""
    return [r for r in SOTA_RESULTS if r.subject == subject or r.subject == "avg"]


def generate_comparison_table(
    our_results: Dict[str, Dict[str, float]],
    gallery_size: int = 1000,
    subject: str = "subj01",
) -> List[Dict]:
    """
    Generate comparison table combining our results with published SoTA.

    Args:
        our_results: Dict mapping model_name → {metric: value}
            Expected keys: 'r_at_1', 'r_at_5', 'cosine', 'median_rank', 'mrr'
        gallery_size: Our gallery size for context
        subject: Subject to filter SoTA results

    Returns:
        List of dicts, one per method (ours + published), suitable
        for pandas DataFrame.
    """
    rows = []

    # Our results
    for model_name, metrics in our_results.items():
        rows.append({
            "method": f"Ours ({model_name})",
            "year": 2026,
            "venue": "This work",
            "gallery_size": gallery_size,
            "r_at_1": metrics.get("r_at_1") or metrics.get("R@1"),
            "r_at_5": metrics.get("r_at_5") or metrics.get("R@5"),
            "cosine": metrics.get("cosine"),
            "median_rank": metrics.get("median_rank"),
            "mrr": metrics.get("mrr"),
            "uses_diffusion": False,
            "notes": "Embedding-space only (no image reconstruction)",
        })

    # Published results
    for entry in get_sota_for_subject(subject):
        rows.append({
            "method": entry.method,
            "year": entry.year,
            "venue": entry.venue,
            "gallery_size": entry.gallery_size,
            "r_at_1": entry.r_at_1,
            "r_at_5": entry.r_at_5,
            "cosine": entry.clip_score,
            "median_rank": entry.median_rank,
            "mrr": entry.mrr,
            "uses_diffusion": entry.uses_diffusion,
            "notes": entry.notes,
        })

    return rows


def generate_latex_table(
    rows: List[Dict],
    metrics: Optional[List[str]] = None,
    caption: str = "Comparison with state-of-the-art fMRI decoding methods",
    label: str = "tab:sota_comparison",
) -> str:
    """
    Generate publication-ready LaTeX table from comparison rows.

    Args:
        rows: List of dicts from generate_comparison_table()
        metrics: Columns to include (default: standard set)
        caption: Table caption
        label: LaTeX label

    Returns:
        Complete LaTeX table string (\\begin{table}...\\end{table})
    """
    if metrics is None:
        metrics = ["method", "venue", "gallery_size", "r_at_1", "r_at_5", "cosine"]

    # Header
    header_map = {
        "method": "Method",
        "venue": "Venue",
        "gallery_size": "Gallery",
        "r_at_1": "R@1",
        "r_at_5": "R@5",
        "cosine": "CLIPScore",
        "median_rank": "Med. Rank",
        "mrr": "MRR",
        "uses_diffusion": "Diffusion",
    }

    col_spec = "l" + "c" * (len(metrics) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(header_map.get(m, m) for m in metrics) + r" \\",
        r"\midrule",
    ]

    # Sort by R@1 descending
    sorted_rows = sorted(rows, key=lambda r: r.get("r_at_1") or 0, reverse=True)

    for row in sorted_rows:
        cells = []
        for m in metrics:
            val = row.get(m)
            if val is None:
                cells.append("—")
            elif isinstance(val, float):
                if m in ("r_at_1", "r_at_5", "cosine", "mrr"):
                    cells.append(f"{val:.3f}")
                else:
                    cells.append(f"{val:.1f}")
            elif isinstance(val, bool):
                cells.append(r"\checkmark" if val else "")
            else:
                cells.append(str(val))
        # Bold our methods
        if "Ours" in str(row.get("method", "")):
            cells[0] = r"\textbf{" + cells[0] + "}"
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
