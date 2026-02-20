#!/usr/bin/env python3
"""
Run All Novel Perception-vs-Imagery Analyses
==============================================

Orchestrates all five research directions:
1. Dimensionality Gap — imagery as compressed manifold
2. Uncertainty as Vividness — MC Dropout confidence proxy
3. Semantic Survival — what information survives imagery
4. Topological RSA — topology of perception vs imagery
5. Cross-Subject Imagery — individual imagery fingerprints

Supports dry-run mode with synthetic data for pipeline validation.

Usage:
    # Dry run with synthetic data (no checkpoint or data needed)
    python scripts/run_novel_analyses.py --dry-run --output-dir outputs/novel_analyses

    # Full run with real data
    python scripts/run_novel_analyses.py \\
        --checkpoint checkpoints/two_stage/subj01/best.pt \\
        --perception-index cache/indices/nsd_index/subj01.parquet \\
        --imagery-index cache/indices/imagery/subj01.parquet \\
        --output-dir outputs/novel_analyses/subj01

    # Select specific analyses
    python scripts/run_novel_analyses.py --dry-run \\
        --analyses dimensionality semantic uncertainty \\
        --output-dir outputs/novel_analyses
"""

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
logger = logging.getLogger("novel_analyses")

ALL_ANALYSES = [
    "dimensionality", "uncertainty", "semantic", "topology", "cross_subject", "dissociation",
    "reality_monitor", "confusion_mapping", "adversarial_reality", "hierarchical_reality",
]


def run_dimensionality(bundle, output_dir):
    from fmri2img.analysis.dimensionality import analyze_dimensionality_gap

    results = analyze_dimensionality_gap(bundle)
    out = output_dir / "dimensionality_gap.json"
    _save_json(results, out)
    return results


def run_uncertainty(bundle, output_dir, model=None, device="cpu"):
    from fmri2img.analysis.imagery_uncertainty import analyze_imagery_uncertainty

    results = analyze_imagery_uncertainty(bundle, model=model, device=device)
    out = output_dir / "uncertainty_vividness.json"
    _save_json(results, out)
    return results


def run_semantic(bundle, output_dir, device="cpu"):
    from fmri2img.analysis.semantic_decomposition import analyze_semantic_survival

    results = analyze_semantic_survival(bundle, device=device)
    out = output_dir / "semantic_survival.json"
    _save_json(results, out)
    return results


def run_topology(bundle, output_dir):
    from fmri2img.analysis.topological_rsa import analyze_topological_signatures

    results = analyze_topological_signatures(bundle)
    out = output_dir / "topological_rsa.json"
    _save_json(results, out)
    return results


def run_cross_subject(bundles, output_dir, adapter_checkpoints=None):
    from fmri2img.analysis.cross_subject import analyze_cross_subject

    results = analyze_cross_subject(bundles, adapter_checkpoints=adapter_checkpoints)
    out = output_dir / "cross_subject.json"
    _save_json(results, out)
    return results


def run_dissociation(output_dir, model=None, perception_dataset=None, imagery_dataset=None, device="cpu", max_samples=None, is_dry_run=False):
    from fmri2img.analysis.semantic_structural_dissociation import analyze_semantic_structural_dissociation

    results = analyze_semantic_structural_dissociation(
        model=model,
        perception_dataset=perception_dataset,
        imagery_dataset=imagery_dataset,
        device=device,
        max_samples=max_samples,
        is_dry_run=is_dry_run
    )
    out = output_dir / "semantic_structural_dissociation.json"
    _save_json(results, out)
    return results


def run_reality_monitor(bundle, output_dir):
    from fmri2img.analysis.reality_monitor import analyze_reality_monitor

    results = analyze_reality_monitor(bundle)
    out = output_dir / "reality_monitor.json"
    _save_json(results, out)
    return results


def run_confusion_mapping(bundle, output_dir):
    from fmri2img.analysis.reality_confusion import analyze_reality_confusion

    results = analyze_reality_confusion(bundle)
    out = output_dir / "reality_confusion.json"
    _save_json(results, out)
    return results


def run_adversarial_reality(bundle, output_dir, device="cpu", n_epochs=100):
    from fmri2img.analysis.adversarial_reality import analyze_adversarial_reality

    results = analyze_adversarial_reality(bundle, n_epochs=n_epochs, device=device)
    out = output_dir / "adversarial_reality.json"
    _save_json(results, out)
    return results


def run_hierarchical_reality(bundle, output_dir, model=None, perception_dataset=None,
                             imagery_dataset=None, device="cpu", max_samples=None):
    from fmri2img.analysis.hierarchical_reality import analyze_hierarchical_reality

    results = analyze_hierarchical_reality(
        bundle, model=model,
        perception_dataset=perception_dataset,
        imagery_dataset=imagery_dataset,
        device=device, max_samples=max_samples,
    )
    out = output_dir / "hierarchical_reality.json"
    _save_json(results, out)
    return results


def _save_json(data, path):
    """Save results, converting numpy types for JSON serialization."""

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(data), f, indent=2)
    logger.info(f"Saved results to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run novel perception-vs-imagery analyses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=ALL_ANALYSES + ["all"],
        default=["all"],
        help="Which analyses to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data")

    # Data arguments (not needed for dry-run)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-type", type=str, default="two_stage")
    parser.add_argument("--perception-index", type=str, default=None)
    parser.add_argument("--imagery-index", type=str, default=None)
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--cache-root", type=str, default="cache")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=None)

    # Cross-subject arguments
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["subj01"],
        help="Subjects for cross-subject analysis",
    )

    # Adversarial analysis parameters
    parser.add_argument("--adversarial-epochs", type=int, default=100,
                        help="Training epochs for adversarial reality probing")

    # Synthetic data parameters (for dry-run)
    parser.add_argument("--n-perception", type=int, default=500)
    parser.add_argument("--n-imagery", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=512)

    args = parser.parse_args()

    if "all" in args.analyses:
        analyses = ALL_ANALYSES
    else:
        analyses = args.analyses

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("NOVEL PERCEPTION-VS-IMAGERY ANALYSES")
    logger.info("=" * 80)
    logger.info(f"Analyses: {analyses}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("")

    t0 = time.time()

    # Load or generate embeddings
    if args.dry_run:
        from fmri2img.analysis.core import generate_synthetic_embeddings

        bundle = generate_synthetic_embeddings(
            n_perception=args.n_perception,
            n_imagery=args.n_imagery,
            embed_dim=args.embed_dim,
        )
        model = None
    else:
        if not args.checkpoint or not args.perception_index or not args.imagery_index:
            logger.error(
                "Real-data mode requires --checkpoint, --perception-index, "
                "and --imagery-index. Use --dry-run for synthetic data."
            )
            sys.exit(1)

        from fmri2img.analysis.core import collect_embeddings, load_model_for_analysis
        from fmri2img.data.nsd_imagery import NSDImageryDataset

        model = load_model_for_analysis(
            args.checkpoint, args.model_type, args.device
        )

        perc_ds = NSDImageryDataset(
            index_path=args.perception_index,
            subject=args.subject,
            condition="perception",
            cache_root=args.cache_root,
        )
        imag_ds = NSDImageryDataset(
            index_path=args.imagery_index,
            subject=args.subject,
            condition="imagery",
            cache_root=args.cache_root,
        )

        bundle = collect_embeddings(
            model, perc_ds, imag_ds, device=args.device, max_samples=args.max_samples
        )

    # Save bundle summary
    summary = {
        "perception_samples": bundle.perception.shape[0],
        "imagery_samples": bundle.imagery.shape[0],
        "embed_dim": bundle.embed_dim,
        "perception_cosine_mean": float(np.mean(bundle.perception_cosines)),
        "imagery_cosine_mean": float(np.mean(bundle.imagery_cosines)),
        "transfer_ratio": float(
            np.mean(bundle.imagery_cosines) / max(np.mean(bundle.perception_cosines), 1e-8)
        ),
        "dry_run": args.dry_run,
        "analyses": analyses,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(summary, output_dir / "summary.json")

    logger.info(f"Baseline — Perception cosine: {summary['perception_cosine_mean']:.4f}")
    logger.info(f"Baseline — Imagery cosine:    {summary['imagery_cosine_mean']:.4f}")
    logger.info(f"Transfer ratio: {summary['transfer_ratio']:.4f}")
    logger.info("")

    # Run selected analyses
    all_results = {"summary": summary}

    if "dimensionality" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 1: Dimensionality Gap")
        logger.info("-" * 60)
        all_results["dimensionality"] = run_dimensionality(bundle, output_dir)
        logger.info("")

    if "uncertainty" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 2: Uncertainty as Vividness")
        logger.info("-" * 60)
        all_results["uncertainty"] = run_uncertainty(
            bundle, output_dir, model=model, device=args.device
        )
        logger.info("")

    if "semantic" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 3: Semantic Survival")
        logger.info("-" * 60)
        all_results["semantic"] = run_semantic(bundle, output_dir, device=args.device)
        logger.info("")

    if "topology" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 4: Topological RSA")
        logger.info("-" * 60)
        all_results["topology"] = run_topology(bundle, output_dir)
        logger.info("")

    if "cross_subject" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 5: Cross-Subject Imagery")
        logger.info("-" * 60)
        if args.dry_run:
            from fmri2img.analysis.core import generate_synthetic_embeddings

            bundles = {}
            for i, subj in enumerate(args.subjects):
                bundles[subj] = generate_synthetic_embeddings(
                    n_perception=args.n_perception,
                    n_imagery=args.n_imagery,
                    embed_dim=args.embed_dim,
                    seed=42 + i,
                    imagery_dim_fraction=0.5 + 0.1 * i,
                )
            all_results["cross_subject"] = run_cross_subject(bundles, output_dir)
        else:
            logger.warning("Cross-subject analysis requires multiple bundles; "
                           "skipping in single-subject mode. Use per-subject runs "
                           "and then combine.")
        logger.info("")

    if "dissociation" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 6: Semantic-Structural Dissociation")
        logger.info("-" * 60)
        if args.dry_run:
            all_results["dissociation"] = run_dissociation(output_dir, is_dry_run=True)
        else:
            all_results["dissociation"] = run_dissociation(
                output_dir,
                model=model,
                perception_dataset=perc_ds,
                imagery_dataset=imag_ds,
                device=args.device,
                max_samples=args.max_samples,
                is_dry_run=False
            )
        logger.info("")

    if "reality_monitor" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 7: Computational Reality Monitor")
        logger.info("-" * 60)
        all_results["reality_monitor"] = run_reality_monitor(bundle, output_dir)
        logger.info("")

    if "confusion_mapping" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 8: Reality Confusion Mapping")
        logger.info("-" * 60)
        all_results["confusion_mapping"] = run_confusion_mapping(bundle, output_dir)
        logger.info("")

    if "adversarial_reality" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 9: Adversarial Reality Probing")
        logger.info("-" * 60)
        all_results["adversarial_reality"] = run_adversarial_reality(
            bundle, output_dir, device=args.device,
            n_epochs=getattr(args, "adversarial_epochs", 100),
        )
        logger.info("")

    if "hierarchical_reality" in analyses:
        logger.info("-" * 60)
        logger.info("DIRECTION 10: Hierarchical Reality Gradient")
        logger.info("-" * 60)
        if args.dry_run:
            bundle_ml = generate_synthetic_embeddings(
                n_perception=args.n_perception,
                n_imagery=args.n_imagery,
                embed_dim=args.embed_dim,
                include_multilayer=True,
            )
            all_results["hierarchical_reality"] = run_hierarchical_reality(
                bundle_ml, output_dir, device=args.device,
            )
        else:
            all_results["hierarchical_reality"] = run_hierarchical_reality(
                bundle, output_dir, model=model,
                perception_dataset=perc_ds,
                imagery_dataset=imag_ds,
                device=args.device,
                max_samples=args.max_samples,
            )
        logger.info("")

    # Save combined results
    _save_json(all_results, output_dir / "all_results.json")

    elapsed = time.time() - t0
    logger.info("=" * 80)
    logger.info(f"ALL ANALYSES COMPLETE in {elapsed:.1f}s")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")

    # Generate figures
    logger.info("\nGenerating figures...")
    try:
        from scripts.make_novel_figures import generate_all_figures

        generate_all_figures(output_dir, output_dir / "figures")
    except Exception as e:
        logger.info(f"Figure generation: running as subprocess...")
        import subprocess

        subprocess.run(
            [sys.executable, "scripts/make_novel_figures.py",
             "--results-dir", str(output_dir),
             "--output-dir", str(output_dir / "figures")],
            check=False,
        )


if __name__ == "__main__":
    main()
