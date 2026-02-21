"""
Direction 12: Predictive Coding Residual Analysis
===================================================

Tests whether information flow direction reverses between perception
(bottom-up) and imagery (top-down) by training inter-layer predictors
on multi-layer decoder embeddings and measuring directional residuals.

Core hypothesis: Perception is bottom-up (large forward prediction errors
at higher layers — new information added). Imagery reverses this flow
(large backward prediction errors at lower layers — top-down generation).

References:
    "Top-down perceptual inference shaping activity of early visual cortex"
    (Nature Communications, 2025)
    "Neural dynamics of perceptual inference and its reversal during imagery"
    (eLife, 2020)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from .core import EmbeddingBundle

logger = logging.getLogger(__name__)

LAYER_ORDER = ["layer_4", "layer_8", "layer_12", "final"]


def _align_dimensions(source: np.ndarray, target: np.ndarray, rng) -> np.ndarray:
    """Project source to match target dimensionality via random projection if needed."""
    if source.shape[1] == target.shape[1]:
        return source
    proj = rng.randn(source.shape[1], target.shape[1]).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8
    return source @ proj


def train_inter_layer_predictors(
    multilayer: Dict[str, np.ndarray],
    alpha: float = 1.0,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    For each adjacent layer pair, train Ridge regressors in both directions:
      forward:  layer_i -> layer_{i+1}
      backward: layer_{i+1} -> layer_i

    Returns dict keyed by transition name (e.g. "layer_4->layer_8").
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    rng = np.random.RandomState(seed)
    available = [l for l in LAYER_ORDER if l in multilayer]
    if len(available) < 2:
        logger.warning("Need at least 2 layers for predictive coding analysis")
        return {}

    predictors = {}
    for i in range(len(available) - 1):
        src_name, tgt_name = available[i], available[i + 1]
        src = multilayer[src_name]
        tgt = multilayer[tgt_name]

        src_aligned = _align_dimensions(src, tgt, rng)
        tgt_aligned = _align_dimensions(tgt, src, rng)

        fwd_model = Ridge(alpha=alpha)
        fwd_model.fit(src_aligned, tgt)
        fwd_pred = fwd_model.predict(src_aligned)
        fwd_r2 = float(1 - np.mean((tgt - fwd_pred) ** 2) / (np.var(tgt) + 1e-8))

        bwd_model = Ridge(alpha=alpha)
        bwd_model.fit(tgt_aligned, src)
        bwd_pred = bwd_model.predict(tgt_aligned)
        bwd_r2 = float(1 - np.mean((src - bwd_pred) ** 2) / (np.var(src) + 1e-8))

        transition = f"{src_name}->{tgt_name}"
        predictors[transition] = {
            "forward_model": fwd_model,
            "backward_model": bwd_model,
            "forward_r2": fwd_r2,
            "backward_r2": bwd_r2,
            "src_name": src_name,
            "tgt_name": tgt_name,
        }
        logger.info(f"  Transition {transition}: "
                     f"fwd R²={fwd_r2:.4f}, bwd R²={bwd_r2:.4f}")

    return predictors


def compute_prediction_residuals(
    multilayer: Dict[str, np.ndarray],
    predictors: Dict[str, Dict],
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    For each transition and trial, compute per-trial residual magnitudes
    in both forward and backward directions.
    """
    rng = np.random.RandomState(seed)
    residuals = {}

    for transition, pdict in predictors.items():
        src = multilayer[pdict["src_name"]]
        tgt = multilayer[pdict["tgt_name"]]

        src_aligned = _align_dimensions(src, tgt, rng)
        tgt_aligned = _align_dimensions(tgt, src, rng)

        fwd_pred = pdict["forward_model"].predict(src_aligned)
        fwd_residual = np.linalg.norm(tgt - fwd_pred, axis=1)

        bwd_pred = pdict["backward_model"].predict(tgt_aligned)
        bwd_residual = np.linalg.norm(src - bwd_pred, axis=1)

        residuals[transition] = {
            "forward": fwd_residual,
            "backward": bwd_residual,
        }

    return residuals


def compute_directional_flow_index(
    forward_residuals: np.ndarray,
    backward_residuals: np.ndarray,
) -> float:
    """
    DFI = (mean_forward - mean_backward) / (mean_forward + mean_backward)
    Positive => bottom-up (perception-like): more unpredicted info at higher layers.
    Negative => top-down (imagery-like): more unpredicted info at lower layers.
    """
    fwd_mean = float(np.mean(forward_residuals))
    bwd_mean = float(np.mean(backward_residuals))
    denom = fwd_mean + bwd_mean
    if denom < 1e-8:
        return 0.0
    return (fwd_mean - bwd_mean) / denom


def analyze_flow_by_layer_transition(
    bundle: EmbeddingBundle,
    alpha: float = 1.0,
) -> Dict:
    """
    Compute DFI at each layer transition for both perception and imagery.
    Tests whether the information flow direction reverses.
    """
    if bundle.multilayer_perception is None or bundle.multilayer_imagery is None:
        logger.warning("Multi-layer embeddings required for predictive coding analysis")
        return {"error": "no_multilayer_data"}

    perc_ml = bundle.multilayer_perception
    imag_ml = bundle.multilayer_imagery

    logger.info("  Training inter-layer predictors on perception data...")
    perc_predictors = train_inter_layer_predictors(perc_ml, alpha=alpha)

    logger.info("  Training inter-layer predictors on imagery data...")
    imag_predictors = train_inter_layer_predictors(imag_ml, alpha=alpha)

    logger.info("  Computing residuals...")
    perc_residuals = compute_prediction_residuals(perc_ml, perc_predictors)
    imag_residuals = compute_prediction_residuals(imag_ml, imag_predictors)

    transition_results = []
    for transition in perc_predictors:
        perc_dfi = compute_directional_flow_index(
            perc_residuals[transition]["forward"],
            perc_residuals[transition]["backward"],
        )
        imag_dfi = compute_directional_flow_index(
            imag_residuals[transition]["forward"],
            imag_residuals[transition]["backward"],
        )

        # Statistical test: paired comparison of DFI shift
        perc_fwd = perc_residuals[transition]["forward"]
        perc_bwd = perc_residuals[transition]["backward"]
        imag_fwd = imag_residuals[transition]["forward"]
        imag_bwd = imag_residuals[transition]["backward"]

        min_n = min(len(perc_fwd), len(imag_fwd))
        perc_diff = perc_fwd[:min_n] - perc_bwd[:min_n]
        imag_diff = imag_fwd[:min_n] - imag_bwd[:min_n]

        stat, p_val = scipy_stats.mannwhitneyu(
            perc_diff, imag_diff, alternative="two-sided",
        )

        flow_reversed = (perc_dfi > 0 and imag_dfi < 0) or (perc_dfi < 0 and imag_dfi > 0)

        transition_results.append({
            "transition": transition,
            "perception_dfi": float(perc_dfi),
            "imagery_dfi": float(imag_dfi),
            "dfi_shift": float(imag_dfi - perc_dfi),
            "flow_reversed": bool(flow_reversed),
            "mann_whitney_stat": float(stat),
            "p_value": float(p_val),
            "perception_fwd_r2": float(perc_predictors[transition]["forward_r2"]),
            "perception_bwd_r2": float(perc_predictors[transition]["backward_r2"]),
            "imagery_fwd_r2": float(imag_predictors[transition]["forward_r2"]),
            "imagery_bwd_r2": float(imag_predictors[transition]["backward_r2"]),
        })

        logger.info(f"    {transition}: perc_DFI={perc_dfi:.4f}, "
                     f"imag_DFI={imag_dfi:.4f}, reversed={flow_reversed}")

    return {
        "transitions": transition_results,
        "any_reversal": any(t["flow_reversed"] for t in transition_results),
    }


def analyze_predictive_coding(
    bundle: EmbeddingBundle,
    alpha: float = 1.0,
) -> Dict:
    """
    Full predictive coding residual analysis.
    """
    logger.info("Running Predictive Coding Residual Analysis...")

    flow_results = analyze_flow_by_layer_transition(bundle, alpha=alpha)

    if "error" in flow_results:
        return flow_results

    transitions = flow_results["transitions"]

    # Aggregate DFI across all transitions
    perc_dfis = [t["perception_dfi"] for t in transitions]
    imag_dfis = [t["imagery_dfi"] for t in transitions]
    mean_perc_dfi = float(np.mean(perc_dfis))
    mean_imag_dfi = float(np.mean(imag_dfis))

    n_reversals = sum(1 for t in transitions if t["flow_reversed"])
    n_significant = sum(1 for t in transitions if t["p_value"] < 0.05)

    results = {
        "flow_analysis": flow_results,
        "mean_perception_dfi": mean_perc_dfi,
        "mean_imagery_dfi": mean_imag_dfi,
        "global_dfi_shift": float(mean_imag_dfi - mean_perc_dfi),
        "n_transitions": len(transitions),
        "n_reversals": n_reversals,
        "n_significant_differences": n_significant,
        "supports_flow_reversal": bool(n_reversals > 0),
    }

    logger.info(f"  Global DFI: perception={mean_perc_dfi:.4f}, "
                f"imagery={mean_imag_dfi:.4f}")
    logger.info(f"  Flow reversals: {n_reversals}/{len(transitions)} transitions")
    logger.info(f"  Significant differences: {n_significant}/{len(transitions)}")

    return results
