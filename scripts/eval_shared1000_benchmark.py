#!/usr/bin/env python3
"""
NSD Shared-1000 Benchmark Evaluation.
=====================================

Standard NSD benchmark: 1000 images seen by all subjects with 3 fMRI reps.
Loads NIfTI data for exact shared-1000 images, averages across 3 reps,
applies preprocessing, predicts with each model, computes retrieval metrics.

Includes statistical comparisons (paired permutation test, bootstrap CI,
Cohen's d, Holm-Bonferroni correction).

Usage:
    python scripts/eval_shared1000_benchmark.py \
        --subject subj01 \
        --preproc-dir outputs/preproc/baseline/subj01 \
        --clip-cache outputs/clip_cache/clip.parquet \
        --checkpoints ridge=checkpoints/ridge_baseline/subj01/ridge.pkl \
                      mlp=checkpoints/mlp_baseline/subj01/mlp.pt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_shared1000_metadata(stim_info_path, subject):
    """Load shared-1000 metadata and get trial indices + nsd_ids."""
    df = pd.read_csv(stim_info_path)
    shared = df[df["shared1000"] == True].copy()
    logger.info(f"Shared-1000: {len(shared)} images")

    subj_num = int(subject.replace("subj", "").replace("0", ""))

    # Get all 3 repetition trial indices
    rep_cols = [f"subject{subj_num}_rep{r}" for r in range(3)]
    available = [c for c in rep_cols if c in shared.columns]

    if len(available) < 3:
        logger.warning(f"Only found {len(available)} rep columns: {available}")

    trials = shared[available].values  # (1000, 3) trial indices
    nsd_ids = shared["nsdId"].values  # (1000,)

    # Filter out NaN/invalid trials
    valid = ~np.isnan(trials).any(axis=1)
    trials = trials[valid].astype(int)
    nsd_ids = nsd_ids[valid]

    logger.info(f"Valid shared-1000 images: {len(nsd_ids)} with {trials.shape[1]} reps")
    return trials, nsd_ids


def extract_shared1000_features(trials, preproc_dir, beta_root, subject):
    """Extract fMRI features for shared-1000 images using preprocessing pipeline.

    Groups by session file, loads each NIfTI once, extracts relevant volumes,
    applies preprocessing, averages across 3 reps.

    Returns:
        X: (n_images, n_features) averaged features
    """
    preproc = Path(preproc_dir)

    # Load preprocessing artifacts
    pca_components = np.load(preproc / "pca_components.npy")
    pca_mean = np.load(preproc / "pca_mean.npy")
    scaler_mean = np.load(preproc / "scaler_mean.npy")
    scaler_std = np.load(preproc / "scaler_std.npy")

    # Weights (hard binary or soft continuous)
    if (preproc / "reliability_weights.npy").exists():
        weights = np.load(preproc / "reliability_weights.npy")
        logger.info(f"Using reliability weights: {(weights > 0).sum()} non-zero voxels")
    elif (preproc / "soft_weights.npy").exists():
        weights = np.load(preproc / "soft_weights.npy")
        logger.info(f"Using soft weights: {(weights > 0).sum()} non-zero voxels")
    elif (preproc / "reliability_mask.npy").exists():
        weights = np.load(preproc / "reliability_mask.npy").astype(np.float32)
        logger.info(f"Using reliability mask: {weights.sum():.0f} selected voxels")
    else:
        raise FileNotFoundError(f"No weight file found in {preproc}. Expected reliability_weights.npy")

    mask = weights > 0
    n_features = pca_components.shape[0]
    n_images, n_reps = trials.shape

    logger.info(f"Preprocessing: {mask.sum()} voxels → PCA {n_features} components")

    # Map trial index → (session_file, volume_index)
    # NSD: 40 sessions × 750 trials, 0-indexed
    # trial 0-749 = session 1, trial 750-1499 = session 2, etc.
    beta_path = Path(beta_root)

    # Group all trials by session
    session_map = {}  # session_num → list of (image_idx, rep_idx, vol_idx)
    for img_idx in range(n_images):
        for rep_idx in range(n_reps):
            trial = trials[img_idx, rep_idx]
            session = trial // 750 + 1  # 1-indexed
            vol_idx = trial % 750
            if session not in session_map:
                session_map[session] = []
            session_map[session].append((img_idx, rep_idx, vol_idx))

    logger.info(f"Trials span {len(session_map)} sessions")

    # Allocate output: n_images × n_reps × n_features
    features_all = np.zeros((n_images, n_reps, n_features), dtype=np.float32)

    import nibabel as nib

    for sess_num in sorted(session_map.keys()):
        sess_file = beta_path / f"betas_session{sess_num:02d}.nii.gz"
        if not sess_file.exists():
            logger.warning(f"Missing session file: {sess_file}")
            continue

        items = session_map[sess_num]
        logger.info(f"  Session {sess_num:02d}: loading {len(items)} volumes from {sess_file.name}")

        img = nib.load(str(sess_file))
        data_4d = img.get_fdata(dtype=np.float32)  # (X, Y, Z, T)

        for img_idx, rep_idx, vol_idx in items:
            vol = data_4d[..., vol_idx]  # (X, Y, Z)
            flat = vol.flatten()

            # Apply mask/weights
            v = flat[mask]
            if weights[mask].max() <= 1.0 and weights[mask].min() >= 0.0:
                # Soft weights
                v = v * weights[mask]

            # Z-score
            std = scaler_std[scaler_std > 0]  # avoid div by zero
            v_z = np.zeros_like(scaler_mean)
            valid = scaler_std > 0
            v_z[valid] = (v[valid] - scaler_mean[valid]) / scaler_std[valid]

            # PCA
            feat = (v_z - pca_mean) @ pca_components.T
            features_all[img_idx, rep_idx] = feat

        del data_4d  # free memory

    # Average across reps
    X = features_all.mean(axis=1)  # (n_images, n_features)
    logger.info(f"Shared-1000 features: {X.shape} (averaged {n_reps} reps)")

    return X


def predict_with_model(model_type, checkpoint_path, X, device="cuda"):
    """Load model and predict CLIP embeddings."""
    import torch

    if model_type == "ridge":
        from fmri2img.models.ridge import RidgeEncoder
        model = RidgeEncoder.load(checkpoint_path)
        Y_pred = model.predict(X, normalize=True)
        return Y_pred

    elif model_type == "mlp":
        from fmri2img.models.mlp import load_mlp
        model, meta = load_mlp(str(checkpoint_path), map_location=device)
        model.eval().to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.from_numpy(X[i:i+256]).float().to(device)
                preds.append(model(batch).cpu().numpy())
        Y_pred = np.vstack(preds)
        return Y_pred / np.linalg.norm(Y_pred, axis=1, keepdims=True)

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
        import torch
        with torch.no_grad():
            for i in range(0, len(X), 256):
                batch = torch.from_numpy(X[i:i+256]).float().to(device)
                out = model(batch)
                if isinstance(out, dict):
                    out = out['final']
                preds.append(out.cpu().numpy())
        Y_pred = np.vstack(preds)
        return Y_pred / np.linalg.norm(Y_pred, axis=1, keepdims=True)

    else:
        raise ValueError(f"Unknown: {model_type}")


def compute_retrieval_metrics(Y_pred, Y_gt, ks=(1, 5, 10, 20, 50)):
    """Compute R@K, mean/median rank, MRR."""
    # Cosine similarity matrix
    sim = Y_pred @ Y_gt.T  # (N, N)

    ranks_desc = np.argsort(-sim, axis=1)
    true_indices = np.arange(len(Y_pred))

    true_ranks = np.zeros(len(Y_pred), dtype=int)
    for i in range(len(Y_pred)):
        true_ranks[i] = np.where(ranks_desc[i] == true_indices[i])[0][0]

    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = float((true_ranks < k).mean() * 100)

    metrics["mean_rank"] = float(true_ranks.mean())
    metrics["median_rank"] = float(np.median(true_ranks))
    metrics["MRR"] = float((1.0 / (true_ranks + 1)).mean())

    # Per-sample cosine similarity (diagonal)
    diag_cos = np.diag(sim)
    metrics["cosine_mean"] = float(diag_cos.mean())
    metrics["cosine_std"] = float(diag_cos.std())

    return metrics, true_ranks, diag_cos


def statistical_comparison(results_dict, metric_key="true_ranks"):
    """Paired permutation tests between all model pairs."""
    model_names = list(results_dict.keys())
    comparisons = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            ranks_a = results_dict[name_a][metric_key]
            ranks_b = results_dict[name_b][metric_key]

            # Paired permutation test (lower rank = better)
            observed_diff = ranks_a.mean() - ranks_b.mean()
            n_perm = 10000
            count = 0
            for _ in range(n_perm):
                swap = np.random.choice([True, False], size=len(ranks_a))
                perm_a = np.where(swap, ranks_b, ranks_a)
                perm_b = np.where(swap, ranks_a, ranks_b)
                if abs(perm_a.mean() - perm_b.mean()) >= abs(observed_diff):
                    count += 1
            p_value = count / n_perm

            # Cohen's d
            diff = ranks_a - ranks_b
            d = diff.mean() / max(diff.std(), 1e-8)

            # Bootstrap 95% CI for mean rank difference
            n_boot = 2000
            boot_diffs = []
            for _ in range(n_boot):
                idx = np.random.choice(len(diff), size=len(diff), replace=True)
                boot_diffs.append(diff[idx].mean())
            ci_lower = np.percentile(boot_diffs, 2.5)
            ci_upper = np.percentile(boot_diffs, 97.5)

            comparisons.append({
                "model_a": name_a,
                "model_b": name_b,
                "mean_rank_a": float(ranks_a.mean()),
                "mean_rank_b": float(ranks_b.mean()),
                "diff": float(observed_diff),
                "p_value": float(p_value),
                "cohens_d": float(d),
                "ci_95_lower": float(ci_lower),
                "ci_95_upper": float(ci_upper),
            })

    # Holm-Bonferroni correction
    if comparisons:
        p_values = [c["p_value"] for c in comparisons]
        sorted_idx = np.argsort(p_values)
        n_tests = len(p_values)
        for rank, idx in enumerate(sorted_idx):
            adjusted = min(p_values[idx] * (n_tests - rank), 1.0)
            comparisons[idx]["p_adjusted"] = float(adjusted)

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="NSD Shared-1000 Benchmark Evaluation")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--stim-info", default="cache/nsd_stim_info_merged.csv")
    parser.add_argument("--preproc-dirs", nargs="+", required=True,
                        help="Preproc dirs (e.g., outputs/preproc/baseline/subj01 outputs/preproc/novel/subj01)")
    parser.add_argument("--beta-root", required=True,
                        help="Path to NSD betas directory (contains betas_session01.nii.gz ...)")
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="model_name=type:path pairs (e.g., ridge_baseline=ridge:checkpoints/ridge_baseline/subj01/ridge.pkl)")
    parser.add_argument("--output-dir", default="outputs/eval/shared1000")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.subject
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse checkpoint arguments
    model_specs = {}
    for spec in args.checkpoints:
        name, rest = spec.split("=", 1)
        mtype, path = rest.split(":", 1)
        model_specs[name] = {"type": mtype, "path": path}
    logger.info(f"Models to evaluate: {list(model_specs.keys())}")

    # Load shared-1000 metadata
    trials, nsd_ids = load_shared1000_metadata(args.stim_info, args.subject)

    # Load CLIP targets for shared-1000
    from fmri2img.data.clip_cache import CLIPCache
    clip_cache = CLIPCache(args.clip_cache).load()
    Y_gt_list = []
    valid = []
    for nid in nsd_ids:
        emb = clip_cache.get([int(nid)])
        if int(nid) in emb:
            vec = emb[int(nid)]
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-3):
                vec = vec / norm
            Y_gt_list.append(vec)
            valid.append(True)
        else:
            valid.append(False)

    valid = np.array(valid)
    Y_gt = np.stack(Y_gt_list).astype(np.float32)
    trials = trials[valid]
    nsd_ids = nsd_ids[valid]
    logger.info(f"CLIP targets: {Y_gt.shape} (dim={Y_gt.shape[1]})")

    # Extract features for each preprocessing config
    features_by_config = {}
    for pd_path in args.preproc_dirs:
        pd_path = Path(pd_path)
        config = pd_path.parent.name  # "baseline" or "novel"
        logger.info(f"\nExtracting shared-1000 features [{config}]...")
        X = extract_shared1000_features(trials, pd_path, args.beta_root, args.subject)
        features_by_config[config] = X

    # Evaluate each model
    all_results = {}
    results_for_table = []

    for model_name, spec in model_specs.items():
        mtype = spec["type"]
        mpath = spec["path"]

        # Determine which features to use
        if "novel" in model_name:
            feat_config = "novel"
        elif "baseline" in model_name:
            feat_config = "baseline"
        else:
            feat_config = list(features_by_config.keys())[0]

        if feat_config not in features_by_config:
            logger.warning(f"Skipping {model_name}: no features for {feat_config}")
            continue

        X = features_by_config[feat_config]
        logger.info(f"\nEvaluating: {model_name} ({mtype}) on {feat_config} features...")

        try:
            Y_pred = predict_with_model(mtype, mpath, X, args.device)
            metrics, true_ranks, diag_cos = compute_retrieval_metrics(Y_pred, Y_gt)

            all_results[model_name] = {
                "metrics": metrics,
                "true_ranks": true_ranks,
                "diag_cos": diag_cos,
                "feature_config": feat_config,
                "model_type": mtype,
            }
            results_for_table.append({
                "model_name": model_name,
                "model_type": mtype,
                "feature_config": feat_config,
                **metrics,
            })

            logger.info(f"  R@1={metrics['R@1']:.2f}%, R@5={metrics['R@5']:.2f}%, "
                        f"R@10={metrics['R@10']:.2f}%, "
                        f"median_rank={metrics['median_rank']:.0f}, "
                        f"cosine={metrics['cosine_mean']:.4f}")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        logger.error("No models evaluated!")
        return 1

    # Statistical comparisons
    logger.info("\n" + "=" * 80)
    logger.info("Statistical Comparisons (paired permutation test on ranks)")
    logger.info("=" * 80)

    comparisons = statistical_comparison(all_results, "true_ranks")
    for c in comparisons:
        sig = "***" if c["p_adjusted"] < 0.001 else "**" if c["p_adjusted"] < 0.01 else "*" if c["p_adjusted"] < 0.05 else "ns"
        logger.info(f"  {c['model_a']} vs {c['model_b']}: "
                    f"Δrank={c['diff']:+.1f}, p={c['p_value']:.4f} (adj={c['p_adjusted']:.4f}), "
                    f"d={c['cohens_d']:.3f}, 95%CI=[{c['ci_95_lower']:.1f}, {c['ci_95_upper']:.1f}] {sig}")

    # Save results
    # 1. Per-model JSON
    for name, res in all_results.items():
        m = res["metrics"].copy()
        m["model_name"] = name
        m["model_type"] = res["model_type"]
        m["feature_config"] = res["feature_config"]
        with open(output_dir / f"{name}_metrics.json", "w") as f:
            json.dump(m, f, indent=2, default=float)

    # 2. Comparison JSON
    with open(output_dir / "statistical_comparisons.json", "w") as f:
        json.dump(comparisons, f, indent=2, default=float)

    # 3. Summary table JSON
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(results_for_table, f, indent=2, default=float)

    # 4. Per-sample CSV for each model
    for name, res in all_results.items():
        csv_data = pd.DataFrame({
            "nsd_id": nsd_ids,
            "true_rank": res["true_ranks"],
            "cosine_similarity": res["diag_cos"],
        })
        csv_data.to_csv(output_dir / f"{name}_per_sample.csv", index=False)

    # 5. Summary markdown
    md_path = output_dir / "shared1000_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# NSD Shared-1000 Benchmark — {args.subject}\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**N images**: {len(nsd_ids)} (3-rep averaged)\n\n")

        f.write("## Retrieval Metrics\n\n")
        f.write("| Model | Type | Features | R@1 | R@5 | R@10 | R@20 | Med.Rank | MRR | Cosine |\n")
        f.write("|-------|------|----------|-----|-----|------|------|----------|-----|--------|\n")
        for r in sorted(results_for_table, key=lambda x: -x.get("R@1", 0)):
            f.write(f"| {r['model_name']} | {r['model_type']} | {r['feature_config']} "
                    f"| {r.get('R@1', 0):.2f} | {r.get('R@5', 0):.2f} | {r.get('R@10', 0):.2f} "
                    f"| {r.get('R@20', 0):.2f} | {r.get('median_rank', -1):.0f} "
                    f"| {r.get('MRR', 0):.4f} | {r.get('cosine_mean', 0):.4f} |\n")

        if comparisons:
            f.write("\n## Statistical Comparisons\n\n")
            f.write("| Model A | Model B | ΔRank | p-value | p-adjusted | Cohen's d | 95% CI | Sig |\n")
            f.write("|---------|---------|-------|---------|------------|-----------|--------|-----|\n")
            for c in comparisons:
                sig = "***" if c["p_adjusted"] < 0.001 else "**" if c["p_adjusted"] < 0.01 else "*" if c["p_adjusted"] < 0.05 else "ns"
                f.write(f"| {c['model_a']} | {c['model_b']} "
                        f"| {c['diff']:+.1f} | {c['p_value']:.4f} | {c['p_adjusted']:.4f} "
                        f"| {c['cohens_d']:.3f} | [{c['ci_95_lower']:.1f}, {c['ci_95_upper']:.1f}] | {sig} |\n")

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(f"Summary: {md_path}")

    # Print final table
    print("\n" + "=" * 120)
    print(f"{'Model':<30s} {'Type':<12s} {'Features':<10s} "
          f"{'R@1':>7s} {'R@5':>7s} {'R@10':>7s} {'MedRank':>8s} {'MRR':>8s} {'Cosine':>8s}")
    print("-" * 120)
    for r in sorted(results_for_table, key=lambda x: -x.get("R@1", 0)):
        print(f"{r['model_name']:<30s} {r['model_type']:<12s} {r['feature_config']:<10s} "
              f"{r.get('R@1', 0):>7.2f} {r.get('R@5', 0):>7.2f} {r.get('R@10', 0):>7.2f} "
              f"{r.get('median_rank', -1):>8.0f} {r.get('MRR', 0):>8.4f} {r.get('cosine_mean', 0):>8.4f}")
    print("=" * 120)

    return 0


if __name__ == "__main__":
    sys.exit(main())
