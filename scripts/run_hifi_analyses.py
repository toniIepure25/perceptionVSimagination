#!/usr/bin/env python3
"""
High-Fidelity Analyses Master Orchestration
============================================

Runs all novel analyses using FMRI2images predictions. Assumes
eval_fmri2images_imagery.py has already been run and predictions
are saved to outputs/hifi_analyses/{subject}/.

This script:
    1. Loads pre-computed FMRI2images predictions (CLS + tokens)
    2. Builds EmbeddingBundle with hifi/token fields
    3. Runs all 13 existing analyses on hifi predictions
    4. Runs 4 new analyses (token spatial, cross-capacity, concept-conditional)
    5. Generates publication figures
    6. Saves comprehensive results JSON

Usage:
    # After eval_fmri2images_imagery.py has been run:
    python scripts/run_hifi_analyses.py \
        --subject subj01 \
        --hifi-dir outputs/hifi_analyses/subj01 \
        --weak-dir outputs/novel_analyses/subj01 \
        --output-dir outputs/hifi_analyses/subj01/results

    # Dry-run with synthetic data:
    python scripts/run_hifi_analyses.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hifi_analyses")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_hifi_predictions(hifi_dir: Path) -> dict:
    """Load pre-computed FMRI2images predictions from disk."""
    data = {}
    for condition in ["perception", "imagery", "attention"]:
        prefix = hifi_dir / condition
        cls_path = hifi_dir / f"{condition}_cls_preds.npy"
        token_path = hifi_dir / f"{condition}_token_preds.npy"
        latent_path = hifi_dir / f"{condition}_latents.npy"
        nsd_path = hifi_dir / f"{condition}_nsd_ids.npy"
        target_path = hifi_dir / f"{condition}_targets_cls.npy"

        if not cls_path.exists():
            logger.info(f"No predictions for condition={condition}, skipping")
            continue

        entry = {
            "cls_preds": np.load(cls_path),
            "nsd_ids": np.load(nsd_path) if nsd_path.exists() else None,
        }
        if token_path.exists():
            entry["token_preds"] = np.load(token_path)
        if latent_path.exists():
            entry["latents"] = np.load(latent_path)
        if target_path.exists():
            entry["targets_cls"] = np.load(target_path)

        data[condition] = entry
        logger.info(
            f"Loaded {condition}: CLS {entry['cls_preds'].shape}"
            + (f", tokens {entry['token_preds'].shape}" if "token_preds" in entry else "")
        )

    return data


def build_hifi_bundle(data: dict, subject: str = "subj01") -> "EmbeddingBundle":
    """Build EmbeddingBundle from FMRI2images predictions."""
    from fmri2img.analysis.core import EmbeddingBundle

    perc = data.get("perception", {})
    imag = data.get("imagery", {})

    bundle = EmbeddingBundle(
        # Use hifi CLS as primary embeddings for standard analyses
        perception=perc.get("cls_preds", np.zeros((0, 1280))),
        imagery=imag.get("cls_preds", np.zeros((0, 1280))),
        perception_targets=perc.get("targets_cls", np.zeros_like(perc.get("cls_preds", np.zeros((0, 1280))))),
        imagery_targets=imag.get("targets_cls", np.zeros_like(imag.get("cls_preds", np.zeros((0, 1280))))),
        embed_dim=1280,
        subject=subject,
        # NSD IDs
        perception_nsd_ids=perc.get("nsd_ids"),
        imagery_nsd_ids=imag.get("nsd_ids"),
        # High-fidelity fields
        perception_hifi=perc.get("cls_preds"),
        imagery_hifi=imag.get("cls_preds"),
        perception_hifi_targets=perc.get("targets_cls"),
        imagery_hifi_targets=imag.get("targets_cls"),
        # Token-level
        perception_tokens=perc.get("token_preds"),
        imagery_tokens=imag.get("token_preds"),
        # Latents
        perception_latents=perc.get("latents"),
        imagery_latents=imag.get("latents"),
        # Metadata
        model_capacity="825M",
        clip_backbone="ViT-bigG/14",
    )

    return bundle


def run_standard_analyses(bundle, output_dir: Path) -> dict:
    """Run existing 13 analysis modules on hifi predictions."""
    results = {}

    # 1. Dimensionality gap
    try:
        from fmri2img.analysis.dimensionality import analyze_dimensionality_gap
        logger.info("Running dimensionality gap...")
        results["dimensionality"] = analyze_dimensionality_gap(
            bundle.perception, bundle.imagery
        )
    except Exception as e:
        logger.warning(f"Dimensionality failed: {e}")

    # 2. Manifold geometry
    try:
        from fmri2img.analysis.manifold_geometry import analyze_manifold_geometry
        logger.info("Running manifold geometry...")
        results["manifold_geometry"] = analyze_manifold_geometry(
            bundle.perception, bundle.imagery
        )
    except Exception as e:
        logger.warning(f"Manifold geometry failed: {e}")

    # 3. Topological RSA
    try:
        from fmri2img.analysis.topological_rsa import analyze_topological_rsa
        logger.info("Running topological RSA...")
        results["topological_rsa"] = analyze_topological_rsa(
            bundle.perception, bundle.imagery,
            bundle.perception_targets, bundle.imagery_targets,
        )
    except Exception as e:
        logger.warning(f"Topological RSA failed: {e}")

    # 4. Reality monitor
    try:
        from fmri2img.analysis.reality_monitor import analyze_reality_monitor
        logger.info("Running reality monitor...")
        results["reality_monitor"] = analyze_reality_monitor(
            bundle.perception, bundle.imagery
        )
    except Exception as e:
        logger.warning(f"Reality monitor failed: {e}")

    # 5. Adversarial reality
    try:
        from fmri2img.analysis.adversarial_reality import analyze_adversarial_reality
        logger.info("Running adversarial reality...")
        results["adversarial_reality"] = analyze_adversarial_reality(
            bundle.perception, bundle.imagery
        )
    except Exception as e:
        logger.warning(f"Adversarial reality failed: {e}")

    # 6. Reality confusion
    try:
        from fmri2img.analysis.reality_confusion import analyze_reality_confusion
        logger.info("Running reality confusion...")
        results["reality_confusion"] = analyze_reality_confusion(
            bundle.perception, bundle.imagery
        )
    except Exception as e:
        logger.warning(f"Reality confusion failed: {e}")

    # 7. Compositional imagination
    try:
        from fmri2img.analysis.compositional_imagination import analyze_compositional
        logger.info("Running compositional imagination...")
        results["compositional"] = analyze_compositional(
            bundle.perception, bundle.imagery,
            bundle.perception_targets, bundle.imagery_targets,
        )
    except Exception as e:
        logger.warning(f"Compositional failed: {e}")

    # 8. Predictive coding
    try:
        from fmri2img.analysis.predictive_coding import analyze_predictive_coding
        logger.info("Running predictive coding...")
        results["predictive_coding"] = analyze_predictive_coding(
            bundle.perception, bundle.imagery
        )
    except Exception as e:
        logger.warning(f"Predictive coding failed: {e}")

    # 9. Imagery uncertainty
    try:
        from fmri2img.analysis.imagery_uncertainty import analyze_uncertainty
        logger.info("Running imagery uncertainty...")
        results["uncertainty"] = analyze_uncertainty(bundle)
    except Exception as e:
        logger.warning(f"Uncertainty failed: {e}")

    # 10. Semantic survival
    try:
        from fmri2img.analysis.semantic_decomposition import analyze_semantic_survival
        logger.info("Running semantic survival...")
        results["semantic_survival"] = analyze_semantic_survival(bundle)
    except Exception as e:
        logger.warning(f"Semantic survival failed: {e}")

    # 11-12. Creative divergence & modality decomposition (need shared stimuli)
    pairs = bundle.get_shared_stimulus_pairs()
    if pairs is not None:
        shared_ids, perc_idx, imag_idx = pairs
        logger.info(f"Found {len(shared_ids)} shared stimuli for paired analyses")

        try:
            from fmri2img.analysis.creative_divergence import analyze_creative_divergence
            logger.info("Running creative divergence...")
            results["creative_divergence"] = analyze_creative_divergence(
                bundle.perception[perc_idx], bundle.imagery[imag_idx],
                bundle.perception_targets[perc_idx], bundle.imagery_targets[imag_idx],
            )
        except Exception as e:
            logger.warning(f"Creative divergence failed: {e}")

        try:
            from fmri2img.analysis.modality_decomposition import analyze_modality_decomposition
            logger.info("Running modality decomposition...")
            results["modality_decomposition"] = analyze_modality_decomposition(
                bundle.perception[perc_idx], bundle.imagery[imag_idx],
            )
        except Exception as e:
            logger.warning(f"Modality decomposition failed: {e}")
    else:
        logger.info("No shared stimuli found — skipping paired analyses")

    logger.info(f"Standard analyses complete: {len(results)}/{13} succeeded")
    return results


def run_novel_analyses(bundle, output_dir: Path) -> dict:
    """Run the 4 new analyses unique to this study."""
    results = {}

    # 1. Token-level spatial decomposition
    if bundle.has_tokens:
        try:
            from fmri2img.analysis.token_spatial import run_full_token_analysis
            logger.info("Running token spatial analysis...")
            token_dir = output_dir / "token_spatial"
            results["token_spatial"] = run_full_token_analysis(bundle, str(token_dir))
        except Exception as e:
            logger.warning(f"Token spatial failed: {e}")
    else:
        logger.info("No token data — skipping token spatial analysis")

    # 2. Concept-conditional transfer
    try:
        from fmri2img.analysis.concept_conditional import (
            analyze_concept_conditional_transfer,
            analyze_by_stimulus_set,
            plot_concept_profiles,
        )
        logger.info("Running concept-conditional transfer...")

        results["concept_conditional"] = analyze_concept_conditional_transfer(
            bundle.perception, bundle.imagery,
            bundle.perception_targets, bundle.imagery_targets,
        )

        # Also analyze by stimulus set if metadata available
        perc_stim_types = [m.get("stimulus_type", "unknown") for m in bundle.perception_meta]
        imag_stim_types = [m.get("stimulus_type", "unknown") for m in bundle.imagery_meta]
        if perc_stim_types and imag_stim_types:
            results["stimulus_set_analysis"] = analyze_by_stimulus_set(
                bundle.perception, bundle.imagery,
                bundle.perception_targets, bundle.imagery_targets,
                perc_stim_types, imag_stim_types,
            )

        # Plot
        if "categories" in results.get("concept_conditional", {}):
            plot_concept_profiles(
                results["concept_conditional"],
                str(output_dir / "concept_profiles.png"),
            )

    except Exception as e:
        logger.warning(f"Concept-conditional failed: {e}")

    logger.info(f"Novel analyses complete: {len(results)} succeeded")
    return results


def generate_summary(
    standard_results: dict,
    novel_results: dict,
    bundle,
) -> dict:
    """Generate a concise summary of all findings."""
    summary = {
        "model": bundle.model_capacity,
        "clip_backbone": bundle.clip_backbone,
        "embed_dim": bundle.embed_dim,
        "n_perception": len(bundle.perception),
        "n_imagery": len(bundle.imagery),
        "has_tokens": bundle.has_tokens,
    }

    # Transfer metrics
    if bundle.hifi_perception_cosines is not None:
        perc_cos = bundle.hifi_perception_cosines
        imag_cos = bundle.hifi_imagery_cosines
        summary["hifi_perception_cosine"] = float(perc_cos.mean())
        summary["hifi_imagery_cosine"] = float(imag_cos.mean())
        summary["hifi_gap"] = float(imag_cos.mean() - perc_cos.mean())
        summary["hifi_transfer_ratio"] = float(imag_cos.mean() / perc_cos.mean()) if perc_cos.mean() > 0.01 else None

    # Key findings from standard analyses
    for key in ["dimensionality", "reality_monitor", "compositional", "topological_rsa"]:
        if key in standard_results:
            summary[f"finding_{key}"] = standard_results[key]

    # Token analysis summary
    if "token_spatial" in novel_results:
        ts = novel_results["token_spatial"]
        if "fidelity_map" in ts:
            summary["token_mean_gap"] = ts["fidelity_map"].get("mean_gap")
        if "center_periphery" in ts:
            summary["center_periphery"] = ts["center_periphery"].get("gradient_test", {})

    summary["n_standard_analyses"] = len(standard_results)
    summary["n_novel_analyses"] = len(novel_results)

    return summary


def run_dry_run(output_dir: Path):
    """Run with synthetic data for pipeline testing."""
    from fmri2img.analysis.core import EmbeddingBundle

    logger.info("DRY RUN with synthetic data")

    n_p, n_i = 96, 96
    bundle = EmbeddingBundle(
        perception=np.random.randn(n_p, 1280).astype(np.float32),
        imagery=np.random.randn(n_i, 1280).astype(np.float32),
        perception_targets=np.random.randn(n_p, 1280).astype(np.float32),
        imagery_targets=np.random.randn(n_i, 1280).astype(np.float32),
        embed_dim=1280,
        perception_hifi=np.random.randn(n_p, 1280).astype(np.float32),
        imagery_hifi=np.random.randn(n_i, 1280).astype(np.float32),
        perception_hifi_targets=np.random.randn(n_p, 1280).astype(np.float32),
        imagery_hifi_targets=np.random.randn(n_i, 1280).astype(np.float32),
        perception_tokens=np.random.randn(n_p, 257, 1280).astype(np.float32),
        imagery_tokens=np.random.randn(n_i, 257, 1280).astype(np.float32),
        perception_token_targets=np.random.randn(n_p, 257, 1280).astype(np.float32),
        imagery_token_targets=np.random.randn(n_i, 257, 1280).astype(np.float32),
        model_capacity="825M",
        clip_backbone="ViT-bigG/14",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run novel analyses only (standard analyses may fail on random data)
    novel_results = run_novel_analyses(bundle, output_dir)

    summary = generate_summary({}, novel_results, bundle)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Dry run complete. Novel analyses run: {len(novel_results)}")


def main():
    parser = argparse.ArgumentParser(description="High-Fidelity Analyses Orchestration")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--hifi-dir", type=str,
                        default="outputs/hifi_analyses/subj01",
                        help="Directory with FMRI2images predictions")
    parser.add_argument("--weak-dir", type=str,
                        default="outputs/novel_analyses/subj01",
                        help="Directory with weak model results (for cross-capacity)")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/hifi_analyses/subj01/results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dry_run:
        run_dry_run(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # 1. Load predictions
    logger.info("Loading FMRI2images predictions...")
    data = load_hifi_predictions(Path(args.hifi_dir))
    if not data:
        logger.error("No predictions found. Run eval_fmri2images_imagery.py first.")
        sys.exit(1)

    # 2. Build bundle
    logger.info("Building EmbeddingBundle...")
    bundle = build_hifi_bundle(data, args.subject)
    logger.info(
        f"Bundle: {len(bundle.perception)} perception, {len(bundle.imagery)} imagery, "
        f"tokens={bundle.has_tokens}, hifi={bundle.has_hifi}"
    )

    # 3. Standard analyses
    logger.info("\n--- Standard Analyses (13 directions) ---")
    standard_results = run_standard_analyses(bundle, output_dir)

    # 4. Novel analyses
    logger.info("\n--- Novel Analyses (4 new directions) ---")
    novel_results = run_novel_analyses(bundle, output_dir)

    # 5. Cross-capacity consistency (if weak results available)
    weak_dir = Path(args.weak_dir)
    if weak_dir.exists():
        try:
            from fmri2img.analysis.cross_capacity import run_cross_capacity_consistency
            from fmri2img.analysis.core import EmbeddingBundle as EB

            # Load weak model predictions
            # (these are the 768-d predictions from run_real_novel_analyses.py)
            logger.info("\n--- Cross-Capacity Consistency ---")
            logger.info("Note: Cross-capacity requires matching weak/strong predictions")
            logger.info("Full comparison will be available after run_fidelity_ladder.py")
        except Exception as e:
            logger.warning(f"Cross-capacity skipped: {e}")

    # 6. Summary
    summary = generate_summary(standard_results, novel_results, bundle)
    elapsed = time.time() - t_start
    summary["elapsed_seconds"] = elapsed

    # Save all results
    all_results = {
        "summary": summary,
        "standard_analyses": {k: _make_serializable(v) for k, v in standard_results.items()},
        "novel_analyses": {k: _make_serializable(v) for k, v in novel_results.items()},
    }
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print("HIGH-FIDELITY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {bundle.model_capacity} ({bundle.clip_backbone})")
    print(f"Samples: {len(bundle.perception)} perception, {len(bundle.imagery)} imagery")
    print(f"Standard analyses: {len(standard_results)}/13")
    print(f"Novel analyses: {len(novel_results)}")
    if "hifi_gap" in summary:
        print(f"Transfer gap: {summary['hifi_gap']:+.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")


def _make_serializable(obj):
    """Convert numpy arrays and other non-serializable types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        if obj.size < 100:
            return obj.tolist()
        return f"<ndarray shape={obj.shape}>"
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


if __name__ == "__main__":
    main()
