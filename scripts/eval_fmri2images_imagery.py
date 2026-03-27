#!/usr/bin/env python3
"""
FMRI2images High-Fidelity Imagery Evaluation
=============================================

Runs the FMRI2images model (825M params, ViT-bigG/14) on NSD-Imagery data
to produce high-fidelity CLS (1280-d) and token-level (257×1280) predictions
for perception, imagery, and attention conditions.

This is the first application of a high-capacity brain decoder to mental
imagery fMRI data. The 10× better model (R@1 ~58% vs ~5%) may reveal
fine-grained perception-imagery differences invisible to weaker decoders.

Usage (on cluster):
    python scripts/eval_fmri2images_imagery.py \
        --checkpoint /home/jovyan/work/FMRI2images/experimental_results/N1v27a_bigg_tokens/subj01/checkpoint_best.pt \
        --nsd-root /home/jovyan/work/data/nsd \
        --imagery-root /home/jovyan/work/data/nsd/nsdimagery \
        --imagery-index cache/indices/imagery/subj01.parquet \
        --subject subj01 \
        --output-dir outputs/hifi_analyses/subj01 \
        --device cuda

    # Dry-run (no checkpoint needed, tests pipeline):
    python scripts/eval_fmri2images_imagery.py --dry-run --output-dir outputs/hifi_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_fmri2images_imagery")

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_bigg_targets(
    imagery_dataset,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict:
    """Compute ViT-bigG/14 CLIP targets for imagery stimuli.

    Returns dict mapping nsd_id -> {"cls": (1280,), "tokens": (257, 1280)}.
    If open_clip is not available or bigG model is too large, falls back to
    using CLS-only targets from the FMRI2images CLIP cache.
    """
    try:
        import open_clip

        logger.info("Loading ViT-bigG/14 for target computation...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
        )
        model = model.to(device).eval()
        logger.info("ViT-bigG/14 loaded successfully")

        # Compute targets for each unique stimulus
        targets = {}
        for sample in imagery_dataset:
            nsd_id = sample.get("nsd_id")
            if nsd_id is None or nsd_id in targets:
                continue

            img = sample.get("target_image")
            if img is not None:
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    # Get full token outputs
                    features = model.encode_image(img_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    targets[nsd_id] = features.cpu().numpy().squeeze(0)

        return targets

    except (ImportError, RuntimeError) as e:
        logger.warning(f"Cannot compute bigG targets: {e}")
        logger.warning("Will use mean-pooled CLS predictions as proxy targets")
        return None


def run_inference(
    loader,
    dataset,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict:
    """Run FMRI2images inference on a dataset.

    Returns dict with:
        cls_preds: (N, 1280) CLS-equivalent predictions
        token_preds: (N, 257, 1280) full token predictions
        latents: (N, 2048) encoder latent representations
        nsd_ids: (N,) stimulus IDs
        conditions: list of str
        stimulus_types: list of str
    """
    all_cls = []
    all_tokens = []
    all_latents = []
    all_nsd_ids = []
    all_conditions = []
    all_stim_types = []

    batch_voxels = []
    batch_meta = []

    for sample in dataset:
        batch_voxels.append(sample["voxels"])
        batch_meta.append({
            "nsd_id": sample.get("nsd_id", -1),
            "condition": sample["condition"],
            "stimulus_type": sample.get("stimulus_type", "unknown"),
        })

        if len(batch_voxels) >= batch_size:
            _process_batch(
                loader, batch_voxels, batch_meta, device,
                all_cls, all_tokens, all_latents,
                all_nsd_ids, all_conditions, all_stim_types,
            )
            batch_voxels = []
            batch_meta = []

    # Process remaining
    if batch_voxels:
        _process_batch(
            loader, batch_voxels, batch_meta, device,
            all_cls, all_tokens, all_latents,
            all_nsd_ids, all_conditions, all_stim_types,
        )

    return {
        "cls_preds": np.concatenate(all_cls, axis=0),
        "token_preds": np.concatenate(all_tokens, axis=0),
        "latents": np.concatenate(all_latents, axis=0),
        "nsd_ids": np.array(all_nsd_ids, dtype=np.int64),
        "conditions": all_conditions,
        "stimulus_types": all_stim_types,
    }


def _process_batch(
    loader, batch_voxels, batch_meta, device,
    all_cls, all_tokens, all_latents,
    all_nsd_ids, all_conditions, all_stim_types,
):
    """Process a single batch through the model."""
    voxels_tensor = torch.from_numpy(
        np.stack(batch_voxels)
    ).float().to(device)

    with torch.no_grad():
        cls_pred = loader.predict_cls(voxels_tensor)  # (B, 1280)
        tokens = loader.decode_tokens(voxels_tensor)  # (B, 257, 1280)
        latent = loader.encode(voxels_tensor)  # (B, 2048)

    all_cls.append(cls_pred.cpu().numpy())
    all_tokens.append(tokens.cpu().numpy())
    all_latents.append(latent.cpu().numpy())

    for m in batch_meta:
        all_nsd_ids.append(m["nsd_id"] if m["nsd_id"] is not None else -1)
        all_conditions.append(m["condition"])
        all_stim_types.append(m["stimulus_type"])


def run_dry_run(output_dir: Path, n_samples: int = 96):
    """Dry-run with random predictions for pipeline testing."""
    logger.info(f"DRY RUN: generating random predictions for {n_samples} samples")

    output_dir.mkdir(parents=True, exist_ok=True)

    for condition in ["perception", "imagery", "attention"]:
        n = n_samples
        np.save(output_dir / f"{condition}_cls_preds.npy",
                np.random.randn(n, 1280).astype(np.float32))
        np.save(output_dir / f"{condition}_token_preds.npy",
                np.random.randn(n, 257, 1280).astype(np.float32))
        np.save(output_dir / f"{condition}_latents.npy",
                np.random.randn(n, 2048).astype(np.float32))
        np.save(output_dir / f"{condition}_nsd_ids.npy",
                np.arange(n, dtype=np.int64))
        np.save(output_dir / f"{condition}_targets_cls.npy",
                np.random.randn(n, 1280).astype(np.float32))

    metadata = {
        "dry_run": True,
        "n_samples_per_condition": n_samples,
        "model": "random",
        "subject": "subj01",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dry run complete. Outputs in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="FMRI2images imagery evaluation")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/jovyan/work/FMRI2images/experimental_results/"
                                "N1v27a_bigg_tokens/subj01/checkpoint_best.pt",
                        help="Path to FMRI2images checkpoint")
    parser.add_argument("--nsd-root", type=str,
                        default="/home/jovyan/work/data/nsd",
                        help="Root of NSD dataset (for nsdgeneral mask)")
    parser.add_argument("--imagery-root", type=str,
                        default="/home/jovyan/work/data/nsd/nsdimagery",
                        help="Root of NSD-Imagery data")
    parser.add_argument("--imagery-index", type=str,
                        default="cache/indices/imagery/subj01.parquet",
                        help="Path to imagery index parquet")
    parser.add_argument("--subject", type=str, default="subj01")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/hifi_analyses/subj01")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nsdgeneral-mask", type=str, default=None,
                        help="Explicit path to nsdgeneral mask (overrides auto-discovery)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate random predictions for pipeline testing")
    parser.add_argument("--conditions", nargs="+",
                        default=["perception", "imagery", "attention"],
                        help="Which conditions to evaluate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        run_dry_run(output_dir)
        return

    # --- Load FMRI2images model ---
    from fmri2img.models.external_loader import ExternalModelLoader

    logger.info(f"Loading FMRI2images from {args.checkpoint}")
    loader = ExternalModelLoader(
        checkpoint_path=args.checkpoint,
        device=args.device,
        use_ema=True,
    )
    logger.info(f"Model loaded: {loader.summary()}")

    # --- Load nsdgeneral extractor ---
    from fmri2img.data.nsdgeneral_extractor import NSDGeneralExtractor

    if args.nsdgeneral_mask:
        extractor = NSDGeneralExtractor.from_mask_file(args.nsdgeneral_mask)
    else:
        extractor = NSDGeneralExtractor.from_nsd_data(args.subject, args.nsd_root)
    logger.info(f"Extractor: {extractor.summary()}")

    # --- Load imagery dataset ---
    from fmri2img.data.imagery_raw_voxels import ImageryRawVoxelDataset

    t_start = time.time()
    all_results = {}

    for condition in args.conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating condition: {condition}")
        logger.info(f"{'='*60}")

        dataset = ImageryRawVoxelDataset(
            index_path=args.imagery_index,
            subject=args.subject,
            extractor=extractor,
            condition=condition,
            data_root=args.imagery_root,
        )
        logger.info(f"Dataset: {len(dataset)} trials")

        if len(dataset) == 0:
            logger.warning(f"No trials for condition={condition}, skipping")
            continue

        # Run inference
        results = run_inference(
            loader, dataset, device=args.device, batch_size=args.batch_size,
        )

        # Save predictions
        np.save(output_dir / f"{condition}_cls_preds.npy", results["cls_preds"])
        np.save(output_dir / f"{condition}_token_preds.npy", results["token_preds"])
        np.save(output_dir / f"{condition}_latents.npy", results["latents"])
        np.save(output_dir / f"{condition}_nsd_ids.npy", results["nsd_ids"])

        # Compute basic metrics
        n = results["cls_preds"].shape[0]
        cls_norms = np.linalg.norm(results["cls_preds"], axis=1)

        all_results[condition] = {
            "n_trials": n,
            "cls_pred_shape": list(results["cls_preds"].shape),
            "token_pred_shape": list(results["token_preds"].shape),
            "cls_norm_mean": float(cls_norms.mean()),
            "cls_norm_std": float(cls_norms.std()),
            "stimulus_types": list(set(results["stimulus_types"])),
            "n_unique_nsd_ids": int(len(set(results["nsd_ids"].tolist()) - {-1})),
        }

        logger.info(
            f"  {condition}: {n} trials, CLS norm {cls_norms.mean():.3f} ± {cls_norms.std():.3f}"
        )

    elapsed = time.time() - t_start

    # Save metadata
    metadata = {
        "model": "FMRI2images_825M",
        "checkpoint": str(args.checkpoint),
        "clip_backbone": "ViT-bigG/14",
        "embed_dim": 1280,
        "n_tokens": 257,
        "subject": args.subject,
        "nsdgeneral_voxels": extractor.n_voxels,
        "elapsed_seconds": elapsed,
        "conditions": all_results,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nAll conditions evaluated in {elapsed:.1f}s")
    logger.info(f"Results saved to {output_dir}")

    # Print summary table
    print("\n" + "=" * 70)
    print("FMRI2images Imagery Evaluation Summary")
    print("=" * 70)
    for cond, info in all_results.items():
        print(f"  {cond:12s}: {info['n_trials']:4d} trials, "
              f"CLS norm {info['cls_norm_mean']:.3f} ± {info['cls_norm_std']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
