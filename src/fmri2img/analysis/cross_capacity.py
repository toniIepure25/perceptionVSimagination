"""
Cross-Capacity Consistency Analysis
====================================

Tests whether perception-imagery findings are model-independent (neural
phenomena) or model-dependent (artifacts of decoder capacity).

Runs the same suite of analyses on both weak (Ridge/MLP, 6M params,
ViT-L/14 768-d) and strong (FMRI2images, 825M params, ViT-bigG/14 1280-d)
model predictions, then measures consistency of effect sizes.

If findings are consistent across a 130× capacity range, they reflect
genuine neural properties of perception vs imagery. If inconsistent,
the finding depends on decoder resolution.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def _safe_extract_metric(result: Dict, *keys: str) -> Optional[float]:
    """Extract a nested metric from an analysis result dict."""
    d = result
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    if isinstance(d, (int, float)):
        return float(d)
    return None


def compare_analysis_results(
    weak_result: Dict,
    strong_result: Dict,
    metric_paths: List[Tuple[str, ...]],
) -> Dict:
    """Compare corresponding metrics between weak and strong model results.

    Parameters
    ----------
    weak_result : dict from analysis run on weak model
    strong_result : dict from analysis run on strong model
    metric_paths : list of tuples, each a path to a scalar metric

    Returns
    -------
    dict with per-metric comparison and overall consistency score
    """
    comparisons = []

    for path in metric_paths:
        w = _safe_extract_metric(weak_result, *path)
        s = _safe_extract_metric(strong_result, *path)
        name = ".".join(path)

        if w is not None and s is not None:
            comparisons.append({
                "metric": name,
                "weak_value": w,
                "strong_value": s,
                "difference": s - w,
                "sign_agreement": (w > 0) == (s > 0) if w != 0 else True,
            })

    if not comparisons:
        return {"error": "no_comparable_metrics", "comparisons": []}

    # Overall consistency
    n_agree = sum(1 for c in comparisons if c["sign_agreement"])
    weak_values = [c["weak_value"] for c in comparisons]
    strong_values = [c["strong_value"] for c in comparisons]

    if len(weak_values) >= 3:
        rho, p_val = scipy_stats.spearmanr(weak_values, strong_values)
    else:
        rho, p_val = float("nan"), float("nan")

    return {
        "comparisons": comparisons,
        "n_metrics": len(comparisons),
        "sign_agreement_rate": n_agree / len(comparisons),
        "rank_correlation": float(rho) if not np.isnan(rho) else None,
        "rank_correlation_p": float(p_val) if not np.isnan(p_val) else None,
    }


def run_cross_capacity_consistency(
    bundle_weak: "EmbeddingBundle",
    bundle_strong: "EmbeddingBundle",
    output_dir: Optional[str] = None,
    analyses: Optional[Dict[str, Callable]] = None,
) -> Dict:
    """Run all available analyses on both bundles and compare.

    Parameters
    ----------
    bundle_weak : EmbeddingBundle with weak model predictions (768-d)
    bundle_strong : EmbeddingBundle with strong model predictions (hifi 1280-d)
    output_dir : where to save results
    analyses : optional dict of {name: callable(bundle) -> dict}
        If None, imports and runs the standard 13 analysis modules.

    Returns
    -------
    dict with per-analysis consistency and overall summary
    """
    if analyses is None:
        analyses = _get_default_analyses()

    results = {}
    weak_results = {}
    strong_results = {}

    for name, func in analyses.items():
        logger.info(f"Running {name} on weak model...")
        try:
            weak_results[name] = func(bundle_weak)
        except Exception as e:
            logger.warning(f"  {name} failed on weak: {e}")
            weak_results[name] = {"error": str(e)}

        logger.info(f"Running {name} on strong model...")
        try:
            strong_results[name] = func(bundle_strong)
        except Exception as e:
            logger.warning(f"  {name} failed on strong: {e}")
            strong_results[name] = {"error": str(e)}

    # Define which metrics to compare per analysis
    metric_registry = {
        "dimensionality": [("pr_ratio",), ("imagery_pr",), ("perception_pr",)],
        "manifold_geometry": [("hull_volume_ratio",), ("centroid_distance",)],
        "topological_rsa": [("rdm_correlation",), ("rdm_p_value",)],
        "reality_monitor": [("auc",), ("accuracy",)],
        "adversarial_reality": [("discriminator_accuracy",)],
        "reality_confusion": [("mean_confusion_score",)],
        "compositional": [("imagery_success_rate",), ("perception_success_rate",)],
        "predictive_coding": [("top_down_index",)],
        "uncertainty": [("mean_imagery_uncertainty",), ("mean_perception_uncertainty",)],
    }

    overall_comparisons = []
    for name in analyses:
        if name in weak_results and name in strong_results:
            paths = metric_registry.get(name, [])
            if paths:
                comp = compare_analysis_results(
                    weak_results[name], strong_results[name], paths,
                )
                results[name] = comp
                overall_comparisons.extend(comp.get("comparisons", []))

    # Global consistency
    if overall_comparisons:
        n_agree = sum(1 for c in overall_comparisons if c["sign_agreement"])
        all_weak = [c["weak_value"] for c in overall_comparisons]
        all_strong = [c["strong_value"] for c in overall_comparisons]

        if len(all_weak) >= 3:
            rho, p_val = scipy_stats.spearmanr(all_weak, all_strong)
        else:
            rho, p_val = float("nan"), float("nan")

        results["overall"] = {
            "n_total_metrics": len(overall_comparisons),
            "sign_agreement_rate": n_agree / len(overall_comparisons),
            "global_rank_correlation": float(rho) if not np.isnan(rho) else None,
            "global_rank_p": float(p_val) if not np.isnan(p_val) else None,
            "weak_model": bundle_weak.model_capacity or "unknown",
            "strong_model": bundle_strong.model_capacity or "unknown",
        }

    if output_dir:
        import json
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Save only JSON-serializable parts
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                serializable[k] = v
        with open(out / "cross_capacity_consistency.json", "w") as f:
            json.dump(serializable, f, indent=2, default=str)

    return results


def _get_default_analyses() -> Dict[str, Callable]:
    """Import and return the standard analysis functions."""
    funcs = {}

    try:
        from fmri2img.analysis.dimensionality import analyze_dimensionality_gap
        funcs["dimensionality"] = lambda b: analyze_dimensionality_gap(
            b.perception, b.imagery
        )
    except ImportError:
        pass

    try:
        from fmri2img.analysis.manifold_geometry import analyze_manifold_geometry
        funcs["manifold_geometry"] = lambda b: analyze_manifold_geometry(
            b.perception, b.imagery
        )
    except ImportError:
        pass

    try:
        from fmri2img.analysis.topological_rsa import analyze_topological_rsa
        funcs["topological_rsa"] = lambda b: analyze_topological_rsa(
            b.perception, b.imagery, b.perception_targets, b.imagery_targets
        )
    except ImportError:
        pass

    try:
        from fmri2img.analysis.reality_monitor import analyze_reality_monitor
        funcs["reality_monitor"] = lambda b: analyze_reality_monitor(
            b.perception, b.imagery
        )
    except ImportError:
        pass

    try:
        from fmri2img.analysis.adversarial_reality import analyze_adversarial_reality
        funcs["adversarial_reality"] = lambda b: analyze_adversarial_reality(
            b.perception, b.imagery
        )
    except ImportError:
        pass

    try:
        from fmri2img.analysis.reality_confusion import analyze_reality_confusion
        funcs["reality_confusion"] = lambda b: analyze_reality_confusion(
            b.perception, b.imagery
        )
    except ImportError:
        pass

    try:
        from fmri2img.analysis.compositional_imagination import analyze_compositional
        funcs["compositional"] = lambda b: analyze_compositional(
            b.perception, b.imagery, b.perception_targets, b.imagery_targets
        )
    except ImportError:
        pass

    logger.info(f"Loaded {len(funcs)} analysis functions for cross-capacity comparison")
    return funcs
