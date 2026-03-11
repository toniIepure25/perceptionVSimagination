#!/usr/bin/env python3
"""
Evaluate all trained models on test split.
==========================================

Loads pre-extracted features, predicts with each model checkpoint,
computes retrieval metrics, and produces a comparison table.

Usage:
    python scripts/eval_all_models.py \
        --features-dir outputs/features \
        --clip-cache outputs/clip_cache/clip.parquet \
        --checkpoints-root checkpoints \
        --output-dir outputs/eval/test_split
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_split(features_dir, clip_cache_path, index_root, subject, seed=42):
    """Load features and split into train/val/test (same split as training)."""
    from fmri2img.data.clip_cache import CLIPCache
    from fmri2img.data.nsd_index_reader import read_subject_index

    fd = Path(features_dir)
    X = np.load(fd / "X.npy")
    nsd_ids = np.load(fd / "nsd_ids.npy")

    clip_cache = CLIPCache(clip_cache_path).load()
    Y_list, valid_mask = [], []
    for nid in nsd_ids:
        emb = clip_cache.get([int(nid)])
        if int(nid) in emb:
            vec = emb[int(nid)]
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-3):
                vec = vec / norm
            Y_list.append(vec)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    X = X[valid_mask]
    nsd_ids = nsd_ids[valid_mask]
    Y = np.stack(Y_list).astype(np.float32)

    # Center features (removes PCA mean bias from soft-reliability weighting)
    feature_mean = X.mean(axis=0)
    if np.linalg.norm(feature_mean) > 0.01:
        logger.info(f"  Centering features (mean norm={np.linalg.norm(feature_mean):.4f})")
        X = X - feature_mean

    # Same split as training
    n = len(X)
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    test_idx = perm[n_train + n_val:]

    return X[test_idx], Y[test_idx], nsd_ids[test_idx]


def predict_with_model(model_type, checkpoint_path, X_test, device="cuda"):
    """Load model and predict on test features."""
    import torch

    if model_type == "ridge":
        from fmri2img.models.ridge import RidgeEncoder
        model = RidgeEncoder.load(checkpoint_path)
        Y_pred = model.predict(X_test, normalize=True)
        return Y_pred

    elif model_type == "mlp":
        from fmri2img.models.mlp import load_mlp
        model, meta = load_mlp(str(checkpoint_path), map_location=device)
        model.eval().to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                batch = torch.from_numpy(X_test[i:i+512]).float().to(device)
                pred = model(batch).cpu().numpy()
                preds.append(pred)
        Y_pred = np.vstack(preds)
        Y_pred = Y_pred / np.linalg.norm(Y_pred, axis=1, keepdims=True)
        return Y_pred

    elif model_type in ("two_stage", "multilayer"):
        from fmri2img.models.encoders import (
            load_two_stage_encoder, load_multilayer_two_stage_encoder
        )
        if model_type == "multilayer":
            model, meta = load_multilayer_two_stage_encoder(str(checkpoint_path), map_location=device)
        else:
            model, meta = load_two_stage_encoder(str(checkpoint_path), map_location=device)
        model.eval().to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                batch = torch.from_numpy(X_test[i:i+512]).float().to(device)
                out = model(batch)
                if isinstance(out, dict):
                    out = out['final']
                preds.append(out.cpu().numpy())
        Y_pred = np.vstack(preds)
        Y_pred = Y_pred / np.linalg.norm(Y_pred, axis=1, keepdims=True)
        return Y_pred

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_metrics(Y_pred, Y_test):
    """Compute full retrieval + cosine metrics."""
    from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics

    # Also compute mean pairwise cosine
    cos_sim = (Y_pred * Y_test).sum(axis=1)
    cosine_mean = float(cos_sim.mean())
    cosine_std = float(cos_sim.std())

    gt_indices = np.arange(len(Y_test))
    ret = retrieval_at_k(Y_pred, Y_test, gt_indices, ks=(1, 5, 10, 20, 50))
    rank = compute_ranking_metrics(Y_pred, Y_test, gt_indices)

    return {
        "cosine_mean": cosine_mean,
        "cosine_std": cosine_std,
        **ret, **rank,
        "n_test": len(Y_test),
    }


def discover_checkpoints(checkpoints_root, subject):
    """Auto-discover all model checkpoints, return list of (name, type, path)."""
    root = Path(checkpoints_root)
    models = []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        subj_dir = d / subject
        if not subj_dir.exists():
            continue

        config_name = d.name

        # Ridge
        for ckpt in subj_dir.glob("ridge*.pkl"):
            models.append((config_name, "ridge", ckpt))
        # MLP
        for ckpt in subj_dir.glob("mlp*.pt"):
            models.append((config_name, "mlp", ckpt))
        # TwoStage
        for ckpt in subj_dir.glob("two_stage*.pt"):
            models.append((config_name, "two_stage", ckpt))
        # Multilayer
        for ckpt in subj_dir.glob("multilayer*.pt"):
            models.append((config_name, "multilayer", ckpt))

    return models


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on test split")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--features-dirs", nargs="+", required=True,
                        help="Feature directories (e.g., outputs/features/baseline/subj01 outputs/features/novel/subj01)")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet")
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs/eval/test_split")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data for each feature config
    test_data = {}
    for fd in args.features_dirs:
        fd_path = Path(fd)
        config = fd_path.parent.name  # "baseline" or "novel"
        logger.info(f"Loading test data for config={config} from {fd}")
        X_test, Y_test, nsd_ids = load_and_split(fd, args.clip_cache, args.index_root, args.subject)
        test_data[config] = (X_test, Y_test, nsd_ids)
        logger.info(f"  {config}: {len(X_test)} test samples, X={X_test.shape}, Y={Y_test.shape}")

    # Discover all checkpoints
    models = discover_checkpoints(args.checkpoints_root, args.subject)
    logger.info(f"\nFound {len(models)} model checkpoints:")
    for name, mtype, path in models:
        logger.info(f"  {name:30s} ({mtype:12s}) → {path}")

    # Evaluate each model
    all_results = []
    for config_name, model_type, ckpt_path in models:
        # Determine which feature config to use
        if "novel" in config_name:
            feat_config = "novel"
        elif "baseline" in config_name:
            feat_config = "baseline"
        else:
            # Use first available
            feat_config = list(test_data.keys())[0]

        if feat_config not in test_data:
            logger.warning(f"  Skipping {config_name}: no test data for {feat_config}")
            continue

        X_test, Y_test, nsd_ids = test_data[feat_config]

        logger.info(f"\nEvaluating: {config_name} ({model_type}) on {feat_config} features...")
        try:
            Y_pred = predict_with_model(model_type, ckpt_path, X_test, args.device)
            metrics = compute_metrics(Y_pred, Y_test)
            metrics["config_name"] = config_name
            metrics["model_type"] = model_type
            metrics["feature_config"] = feat_config
            metrics["checkpoint"] = str(ckpt_path)
            all_results.append(metrics)

            logger.info(f"  cosine={metrics['cosine_mean']:.4f}, "
                        f"R@1={metrics.get('R@1', 0):.4f}, R@5={metrics.get('R@5', 0):.4f}, "
                        f"R@10={metrics.get('R@10', 0):.4f}, "
                        f"median_rank={metrics.get('median_rank', -1):.0f}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

    if not all_results:
        logger.error("No models evaluated successfully!")
        return 1

    # Save detailed results
    results_path = output_dir / args.subject / "all_models_eval.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nDetailed results: {results_path}")

    # Print comparison table
    print("\n" + "=" * 110)
    print(f"{'Config':<30s} {'Model':<12s} {'Features':<10s} {'Cosine':>8s} "
          f"{'R@1':>8s} {'R@5':>8s} {'R@10':>8s} {'MedRank':>8s} {'MRR':>8s}")
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: -x.get('R@1', 0)):
        print(f"{r['config_name']:<30s} {r['model_type']:<12s} {r['feature_config']:<10s} "
              f"{r['cosine_mean']:>8.4f} "
              f"{r.get('R@1', 0):>8.4f} {r.get('R@5', 0):>8.4f} "
              f"{r.get('R@10', 0):>8.4f} "
              f"{r.get('median_rank', -1):>8.0f} {r.get('MRR', 0):>8.4f}")
    print("=" * 110)

    # Save markdown table
    md_path = output_dir / args.subject / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write(f"# Model Comparison — {args.subject} Test Split\n\n")
        f.write(f"| Config | Model | Features | Cosine | R@1 | R@5 | R@10 | Median Rank | MRR |\n")
        f.write(f"|--------|-------|----------|--------|-----|-----|------|-------------|-----|\n")
        for r in sorted(all_results, key=lambda x: -x.get('R@1', 0)):
            f.write(f"| {r['config_name']} | {r['model_type']} | {r['feature_config']} "
                    f"| {r['cosine_mean']:.4f} "
                    f"| {r.get('R@1', 0):.4f} | {r.get('R@5', 0):.4f} "
                    f"| {r.get('R@10', 0):.4f} "
                    f"| {r.get('median_rank', -1):.0f} | {r.get('MRR', 0):.4f} |\n")
        f.write(f"\n_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")

    logger.info(f"Markdown table: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
