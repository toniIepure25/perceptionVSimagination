#!/usr/bin/env python3
"""
Run Novel Analyses on Real NSD-Imagery Data

Builds EmbeddingBundle directly from perception and imagery data using
the Ridge model (which works natively without torch wrapper), and then
runs all 15 novel analysis directions.

Usage:
    python scripts/run_real_novel_analyses.py \
        --subject subj01 \
        --output-dir outputs/novel_analyses/subj01
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("real_novel_analyses")

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_embedding_bundle(
    subject: str,
    imagery_index: str,
    perception_index_dir: str,
    checkpoint_path: str,
    preproc_dir: str,
    data_root: str,
    clip_cache_dir: str,
    device: str = "cuda",
    max_perception: int = 1000,
    max_imagery: int = None,
):
    """
    Build an EmbeddingBundle from real data.
    
    Uses Ridge model for predictions and CLIP for targets.
    """
    from fmri2img.analysis.core import EmbeddingBundle
    from fmri2img.models.ridge import RidgeEncoder
    from fmri2img.data.preprocess import NSDPreprocessor
    from fmri2img.data.nsd_imagery import NSDImageryDataset

    # Load model
    logger.info(f"Loading Ridge model from {checkpoint_path}")
    encoder = RidgeEncoder.load(checkpoint_path)
    logger.info(f"Ridge loaded: alpha={encoder.alpha:.1f}, {encoder.coef_.shape}")

    # Load preprocessor
    logger.info(f"Loading preprocessor from {preproc_dir}")
    preprocessor = NSDPreprocessor(subject=subject, out_dir="__tmp__")
    preprocessor.set_out_dir(preproc_dir)
    preprocessor.load_artifacts()
    logger.info(f"Preprocessor: {preprocessor.mask_.sum()} voxels → {preprocessor.pca_info_.get('k_eff', '?')}D")

    # Load CLIP for targets
    import clip
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    clip_model.eval()

    def get_clip_embedding(image=None, text=None):
        """Get CLIP embedding for image or text."""
        with torch.no_grad():
            if image is not None:
                img_tensor = clip_preprocess(image).unsqueeze(0).to(device)
                emb = clip_model.encode_image(img_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.cpu().numpy()[0]
            elif text is not None and text.strip():
                text_token = clip.tokenize([text]).to(device)
                emb = clip_model.encode_text(text_token)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.cpu().numpy()[0]
        return np.zeros(768, dtype=np.float32)

    # ========== IMAGERY DATA ==========
    logger.info("Loading imagery data...")
    imag_ds = NSDImageryDataset(
        index_path=imagery_index,
        subject=subject,
        condition="imagery",
        split_filter=None,  # all imagery
        cache_root="cache",
        data_root=data_root,
        preprocessor=preprocessor,
        shuffle=False,
    )
    
    imag_voxels, imag_targets, imag_meta, imag_nsd_ids = [], [], [], []
    for i, sample in enumerate(tqdm(imag_ds, desc="Imagery", total=len(imag_ds))):
        if max_imagery and i >= max_imagery:
            break
        imag_voxels.append(sample['voxels'])
        
        # Get CLIP target
        target = get_clip_embedding(sample.get('target_image'), sample.get('target_text'))
        imag_targets.append(target)
        
        nsd_id = sample.get('nsd_id')
        imag_nsd_ids.append(nsd_id)
        imag_meta.append({
            'trial_id': sample.get('trial_id'),
            'stimulus_type': sample.get('stimulus_type', 'unknown'),
            'condition': 'imagery',
            'nsd_id': nsd_id,
        })

    imag_X = np.vstack(imag_voxels).astype(np.float32)
    imag_target_arr = np.array(imag_targets)
    logger.info(f"Imagery: {imag_X.shape}")

    # Get imagery predictions
    imag_preds = encoder.predict(imag_X)
    logger.info(f"Imagery predictions: {imag_preds.shape}")

    # ========== PERCEPTION DATA (from imagery index, perception trials only) ==========
    logger.info("Loading perception data...")
    perc_ds = NSDImageryDataset(
        index_path=imagery_index,
        subject=subject,
        condition="perception",
        split_filter=None,
        cache_root="cache",
        data_root=data_root,
        preprocessor=preprocessor,
        shuffle=False,
    )

    perc_voxels, perc_targets, perc_meta, perc_nsd_ids = [], [], [], []
    for i, sample in enumerate(tqdm(perc_ds, desc="Perception", total=len(perc_ds))):
        if max_perception and i >= max_perception:
            break
        perc_voxels.append(sample['voxels'])
        
        target = get_clip_embedding(sample.get('target_image'), sample.get('target_text'))
        perc_targets.append(target)
        
        nsd_id = sample.get('nsd_id')
        perc_nsd_ids.append(nsd_id)
        perc_meta.append({
            'trial_id': sample.get('trial_id'),
            'stimulus_type': sample.get('stimulus_type', 'unknown'),
            'condition': 'perception',
            'nsd_id': nsd_id,
        })

    perc_X = np.vstack(perc_voxels).astype(np.float32)
    perc_target_arr = np.array(perc_targets)
    logger.info(f"Perception: {perc_X.shape}")

    # Get perception predictions
    perc_preds = encoder.predict(perc_X)
    logger.info(f"Perception predictions: {perc_preds.shape}")

    # Build nsd_id arrays
    p_ids = np.array(perc_nsd_ids)
    i_ids = np.array(imag_nsd_ids)
    has_p_ids = not np.all(p_ids == None)  # noqa
    has_i_ids = not np.all(i_ids == None)  # noqa

    bundle = EmbeddingBundle(
        perception=perc_preds,
        imagery=imag_preds,
        perception_targets=perc_target_arr,
        imagery_targets=imag_target_arr,
        embed_dim=perc_preds.shape[1],
        subject=subject,
        perception_meta=perc_meta,
        imagery_meta=imag_meta,
        perception_nsd_ids=p_ids if has_p_ids else None,
        imagery_nsd_ids=i_ids if has_i_ids else None,
    )

    return bundle


def main():
    parser = argparse.ArgumentParser(description="Run novel analyses on real NSD-Imagery data")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--imagery-index", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--preproc-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--clip-cache-dir", type=str, default="cache/clip_embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-perception", type=int, default=None)
    parser.add_argument("--max-imagery", type=int, default=None)
    parser.add_argument("--analyses", nargs="+", default=["all"])

    args = parser.parse_args()

    # Auto-fill paths
    subj = args.subject
    if args.imagery_index is None:
        args.imagery_index = f"cache/indices/imagery/{subj}.parquet"
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/ridge_baseline/{subj}/ridge.pkl"
    if args.preproc_dir is None:
        args.preproc_dir = f"cache/preproc/subject={subj}/{subj}"
    if args.data_root is None:
        args.data_root = "/home/jovyan/work/data/nsd/nsdimagery"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("REAL-DATA NOVEL ANALYSES")
    logger.info("=" * 80)
    logger.info(f"Subject: {subj}")
    logger.info(f"Index: {args.imagery_index}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")

    t0 = time.time()

    # Build bundle
    bundle = build_embedding_bundle(
        subject=subj,
        imagery_index=args.imagery_index,
        perception_index_dir=f"data/indices/nsd_index/subject={subj}/",
        checkpoint_path=args.checkpoint,
        preproc_dir=args.preproc_dir,
        data_root=args.data_root,
        clip_cache_dir=args.clip_cache_dir,
        device=args.device,
        max_perception=args.max_perception,
        max_imagery=args.max_imagery,
    )

    # Summary
    perc_cos = bundle.perception_cosines
    imag_cos = bundle.imagery_cosines
    summary = {
        "subject": subj,
        "perception_samples": bundle.perception.shape[0],
        "imagery_samples": bundle.imagery.shape[0],
        "embed_dim": bundle.embed_dim,
        "perception_cosine_mean": float(perc_cos.mean()),
        "perception_cosine_std": float(perc_cos.std()),
        "imagery_cosine_mean": float(imag_cos.mean()),
        "imagery_cosine_std": float(imag_cos.std()),
        "transfer_ratio": float(imag_cos.mean() / max(perc_cos.mean(), 1e-8)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_json(summary, output_dir / "summary.json")

    logger.info(f"Perception cosine: {summary['perception_cosine_mean']:.4f} ± {summary['perception_cosine_std']:.4f}")
    logger.info(f"Imagery cosine:    {summary['imagery_cosine_mean']:.4f} ± {summary['imagery_cosine_std']:.4f}")
    logger.info(f"Transfer ratio:    {summary['transfer_ratio']:.4f}")

    # Determine analyses
    ALL = ["dimensionality", "uncertainty", "semantic", "topology",
           "reality_monitor", "confusion_mapping", "adversarial_reality",
           "manifold_geometry", "modality_decomposition", "creative_divergence",
           "dissociation", "compositional", "predictive_coding"]
    
    if "all" in args.analyses:
        analyses = ALL
    else:
        analyses = args.analyses

    all_results = {"summary": summary}

    # Import runners from the main script
    from scripts.run_novel_analyses import (
        run_dimensionality, run_uncertainty, run_semantic, run_topology,
        run_reality_monitor, run_confusion_mapping, run_adversarial_reality,
        run_manifold_geometry, run_modality_decomposition, run_creative_divergence,
        run_dissociation, run_compositional, run_predictive_coding,
    )

    analysis_runners = {
        "dimensionality": lambda: run_dimensionality(bundle, output_dir),
        "uncertainty": lambda: run_uncertainty(bundle, output_dir, device=args.device),
        "semantic": lambda: run_semantic(bundle, output_dir, device=args.device),
        "topology": lambda: run_topology(bundle, output_dir),
        "reality_monitor": lambda: run_reality_monitor(bundle, output_dir),
        "confusion_mapping": lambda: run_confusion_mapping(bundle, output_dir),
        "adversarial_reality": lambda: run_adversarial_reality(bundle, output_dir, device=args.device),
        "manifold_geometry": lambda: run_manifold_geometry(bundle, output_dir),
        "modality_decomposition": lambda: run_modality_decomposition(bundle, output_dir, device=args.device),
        "creative_divergence": lambda: run_creative_divergence(bundle, output_dir, device=args.device),
        "dissociation": lambda: run_dissociation(output_dir, is_dry_run=True),  # needs special handling
        "compositional": lambda: run_compositional(bundle, output_dir, device=args.device),
        "predictive_coding": lambda: run_predictive_coding(bundle, output_dir),
    }

    for name in analyses:
        if name in analysis_runners:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {name}")
            logger.info(f"{'='*60}")
            try:
                all_results[name] = analysis_runners[name]()
                logger.info(f"✓ {name} complete")
            except Exception as e:
                logger.error(f"✗ {name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[name] = {"error": str(e)}

    _save_json(all_results, output_dir / "all_results.json")

    elapsed = time.time() - t0
    logger.info(f"\n{'='*80}")
    logger.info(f"ALL ANALYSES COMPLETE in {elapsed:.1f}s")
    logger.info(f"Results in: {output_dir}")


def _save_json(data, path):
    """Save results with numpy type conversion."""
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(data), f, indent=2)
    logger.info(f"Saved: {path}")


if __name__ == "__main__":
    main()
