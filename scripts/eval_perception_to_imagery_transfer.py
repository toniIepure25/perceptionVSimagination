#!/usr/bin/env python3
"""
Evaluate Perception-to-Imagery Transfer

Evaluates perception-trained models on imagery test data to quantify cross-domain
generalization. Supports within-domain (perception→perception) and cross-domain
(perception→imagery) evaluation.

Usage:
    # Evaluate on imagery test set (cross-domain)
    python scripts/eval_perception_to_imagery_transfer.py \\
        --index cache/indices/imagery/subj01.parquet \\
        --checkpoint checkpoints/two_stage/subj01/best.pt \\
        --mode imagery \\
        --output-dir outputs/reports/imagery/
    
    # Dry run (test without loading real checkpoint)
    python scripts/eval_perception_to_imagery_transfer.py \\
        --index cache/indices/imagery/subj01.parquet \\
        --checkpoint dummy.pt \\
        --mode imagery \\
        --output-dir outputs/reports/imagery/ \\
        --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_checkpoint_and_model(
    checkpoint_path: str,
    model_type: str,
    device: str,
    dry_run: bool = False,
    adapter_checkpoint: Optional[str] = None,
    adapter_type: Optional[str] = None
):
    """
    Load model checkpoint with optional adapter.
    
    Returns:
        (model_wrapper, metadata)
    """
    if dry_run:
        logger.info("[DRY RUN] Skipping checkpoint loading, using random predictions")
        
        class DummyModel:
            def predict(self, X: np.ndarray) -> np.ndarray:
                # Return random embeddings
                return np.random.randn(len(X), 512).astype(np.float32)
        
        return DummyModel(), {'model_type': 'dry_run', 'embedding_dim': 512}
    
    logger.info(f"Loading {model_type} model from {checkpoint_path}")
    
    # Import model loaders
    from fmri2img.models.ridge import RidgeEncoder
    
    if model_type == "ridge":
        encoder = RidgeEncoder.load(checkpoint_path)
        logger.info(f"✅ Loaded Ridge encoder (alpha={encoder.alpha:.1f})")
        
        class EncoderWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                return self.model.predict(X)
        
        return EncoderWrapper(encoder), {'model_type': 'ridge'}
    
    elif model_type == "mlp":
        from fmri2img.models.mlp import load_mlp
        model, meta = load_mlp(checkpoint_path, map_location=device)
        model = model.to(device)
        model.eval()
        logger.info(f"✅ Loaded MLP encoder")
        
        class MLPWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float().to(self.device)
                    pred = self.model(X_tensor)
                    return pred.cpu().numpy()
        
        return MLPWrapper(model, device), meta
    
    elif model_type == "two_stage":
        from fmri2img.models.encoders import load_two_stage_encoder
        model, meta = load_two_stage_encoder(checkpoint_path, map_location=device)
        model = model.to(device)
        model.eval()
        logger.info(f"✅ Loaded TwoStage encoder")
        
        class TwoStageWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float().to(self.device)
                    pred = self.model(X_tensor)
                    return pred.cpu().numpy()
        
        return TwoStageWrapper(model, device), meta
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: ridge, mlp, two_stage")


def load_adapter_model(
    base_model,
    adapter_checkpoint: str,
    adapter_type: Optional[str],
    device: str
):
    """Load adapter and wrap with base model."""
    from fmri2img.models.adapters import load_adapter, AdaptedModel
    
    logger.info(f"Loading adapter from {adapter_checkpoint}")
    adapter, adapter_meta = load_adapter(
        adapter_checkpoint,
        adapter_type=adapter_type,
        embed_dim=512,
        map_location=device
    )
    adapter = adapter.to(device)
    adapter.eval()
    
    # For ridge models, unwrap the base model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'ridge_model'):
        # It's a RidgeWrapper, need to handle differently
        logger.info("Wrapping Ridge model with adapter")
        
        class RidgeAdapterWrapper:
            def __init__(self, ridge_model, adapter, device):
                self.ridge_model = ridge_model
                self.adapter = adapter
                self.device = device
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                # Ridge prediction (numpy)
                base_pred = self.ridge_model.predict(X)
                # Convert to tensor
                base_tensor = torch.from_numpy(base_pred).float().to(self.device)
                # Adapter forward
                with torch.no_grad():
                    adapted = self.adapter(base_tensor)
                return adapted.cpu().numpy()
        
        return RidgeAdapterWrapper(base_model.model, adapter, device), adapter_meta
    
    # For neural models, wrap directly
    elif hasattr(base_model, 'model'):
        # It's MLPWrapper or TwoStageWrapper
        adapted_model = AdaptedModel(base_model.model, adapter)
        adapted_model.to(device)
        adapted_model.eval()
        
        class AdaptedWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            
            def predict(self, X: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    X_tensor = torch.from_numpy(X).float().to(self.device)
                    pred = self.model(X_tensor)
                    return pred.cpu().numpy()
        
        return AdaptedWrapper(adapted_model, device), adapter_meta
    
    else:
        raise ValueError("Unsupported base model structure for adapter wrapping")


def compute_clip_embeddings(images: List, texts: List, device: str, cache_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CLIP embeddings for images and texts.
    
    Args:
        images: List of PIL Images
        texts: List of text strings
        device: Device for CLIP model
        cache_dir: Optional cache directory for embeddings
    
    Returns:
        (image_embeddings, text_embeddings) as numpy arrays
    """
    try:
        import clip
    except ImportError:
        raise ImportError("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    image_embs = []
    text_embs = []
    
    # Process images
    if images:
        logger.info(f"Computing CLIP embeddings for {len(images)} images...")
        with torch.no_grad():
            for img in tqdm(images, desc="CLIP images"):
                if img is not None:
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    emb = model.encode_image(img_tensor)
                    emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize
                    image_embs.append(emb.cpu().numpy()[0])
                else:
                    image_embs.append(np.zeros(512, dtype=np.float32))
    
    # Process texts
    if texts:
        logger.info(f"Computing CLIP embeddings for {len(texts)} texts...")
        with torch.no_grad():
            for text in tqdm(texts, desc="CLIP texts"):
                if text is not None and text.strip():
                    text_token = clip.tokenize([text]).to(device)
                    emb = model.encode_text(text_token)
                    emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize
                    text_embs.append(emb.cpu().numpy()[0])
                else:
                    text_embs.append(np.zeros(512, dtype=np.float32))
    
    return (np.array(image_embs) if image_embs else None, 
            np.array(text_embs) if text_embs else None)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted embeddings (N, D)
        targets: Target embeddings (N, D)
        k_values: K values for retrieval@K metric
    
    Returns:
        Dictionary of metrics
    """
    # Normalize
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    target_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity (per sample)
    cosine_sims = np.sum(pred_norm * target_norm, axis=1)
    
    # Retrieval@K (in-batch)
    similarity_matrix = pred_norm @ target_norm.T  # (N, N)
    
    retrieval_metrics = {}
    for k in k_values:
        # Get top-k indices for each prediction
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]
        # Check if ground truth (diagonal) is in top-k
        correct = np.any(top_k_indices == np.arange(len(predictions))[:, None], axis=1)
        retrieval_metrics[f'retrieval@{k}'] = correct.mean()
    
    metrics = {
        'clip_cosine_mean': float(cosine_sims.mean()),
        'clip_cosine_std': float(cosine_sims.std()),
        'clip_cosine_median': float(np.median(cosine_sims)),
        **retrieval_metrics
    }
    
    return metrics, cosine_sims


def evaluate_on_dataset(
    model,
    dataset,
    device: str,
    batch_size: int = 32,
    k_values: List[int] = [1, 5, 10],
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run evaluation on a dataset.
    
    Returns:
        (overall_metrics, per_trial_df)
    """
    logger.info(f"Evaluating on {len(dataset)} samples...")
    
    # Collect data
    voxels_list = []
    images_list = []
    texts_list = []
    trial_info = []
    
    for sample in tqdm(dataset, desc="Loading dataset"):
        voxels_list.append(sample['voxels'])
        images_list.append(sample.get('target_image'))
        texts_list.append(sample.get('target_text'))
        trial_info.append({
            'trial_id': sample['trial_id'],
            'stimulus_type': sample['stimulus_type'],
            'condition': sample['condition'],
        })
    
    # Stack voxels
    X = np.vstack(voxels_list).astype(np.float32)
    logger.info(f"Input shape: {X.shape}")
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model.predict(X)
    logger.info(f"Predictions shape: {predictions.shape}")
    
    # Compute target embeddings
    logger.info("Computing target embeddings...")
    img_targets, text_targets = compute_clip_embeddings(images_list, texts_list, device)
    
    # Determine which targets to use
    targets = []
    for i, info in enumerate(trial_info):
        if images_list[i] is not None and img_targets is not None:
            targets.append(img_targets[i])
        elif texts_list[i] is not None and text_targets is not None:
            targets.append(text_targets[i])
        else:
            # No valid target, use zero vector
            targets.append(np.zeros(predictions.shape[1], dtype=np.float32))
    
    targets = np.array(targets)
    
    # Compute metrics
    logger.info("Computing metrics...")
    overall_metrics, per_sample_scores = compute_metrics(predictions, targets, k_values)
    
    # Build per-trial dataframe
    per_trial_data = []
    for i, info in enumerate(trial_info):
        per_trial_data.append({
            'trial_id': info['trial_id'],
            'stimulus_type': info['stimulus_type'],
            'condition': info['condition'],
            'clip_cosine': per_sample_scores[i],
            'has_image': images_list[i] is not None,
            'has_text': texts_list[i] is not None,
        })
    
    per_trial_df = pd.DataFrame(per_trial_data)
    
    # Compute per-stimulus-type metrics
    stimulus_type_metrics = {}
    for stype in per_trial_df['stimulus_type'].unique():
        mask = per_trial_df['stimulus_type'] == stype
        scores = per_trial_df.loc[mask, 'clip_cosine'].values
        stimulus_type_metrics[stype] = {
            'count': int(mask.sum()),
            'clip_cosine_mean': float(scores.mean()),
            'clip_cosine_std': float(scores.std()),
        }
    
    overall_metrics['by_stimulus_type'] = stimulus_type_metrics
    
    return overall_metrics, per_trial_df


def write_report(output_dir: Path, metrics: Dict, per_trial_df: pd.DataFrame, args):
    """Write evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write metrics JSON
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"✓ Metrics saved to {metrics_file}")
    
    # Write per-trial CSV
    per_trial_file = output_dir / "per_trial.csv"
    per_trial_df.to_csv(per_trial_file, index=False)
    logger.info(f"✓ Per-trial results saved to {per_trial_file}")
    
    # Write README summary
    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write("# Perception-to-Imagery Transfer Evaluation\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Command\n\n")
        f.write("```bash\n")
        f.write(f"python {' '.join(sys.argv)}\n")
        f.write("```\n\n")
        f.write("## Overall Metrics\n\n")
        f.write(f"- CLIP Cosine (mean): {metrics['clip_cosine_mean']:.4f}\n")
        f.write(f"- CLIP Cosine (std): {metrics['clip_cosine_std']:.4f}\n")
        f.write(f"- CLIP Cosine (median): {metrics['clip_cosine_median']:.4f}\n")
        for k, v in metrics.items():
            if k.startswith('retrieval@'):
                f.write(f"- {k}: {v:.4f}\n")
        f.write("\n## By Stimulus Type\n\n")
        for stype, stype_metrics in metrics.get('by_stimulus_type', {}).items():
            f.write(f"### {stype}\n")
            f.write(f"- Count: {stype_metrics['count']}\n")
            f.write(f"- CLIP Cosine (mean): {stype_metrics['clip_cosine_mean']:.4f}\n")
            f.write(f"- CLIP Cosine (std): {stype_metrics['clip_cosine_std']:.4f}\n")
            f.write("\n")
    
    logger.info(f"✓ Report summary saved to {readme_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate perception-to-imagery transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument('--index', type=str, required=True, help='Path to imagery index parquet file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt file)')
    parser.add_argument('--mode', type=str, choices=['imagery', 'perception', 'both'], required=True)
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--model-type', type=str, choices=['ridge', 'mlp', 'two_stage', 'auto'], default='auto')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cache-root', type=str, default='cache')
    parser.add_argument('--retrieval-k', type=int, nargs='+', default=[1, 5, 10])
    parser.add_argument('--dry-run', action='store_true', help='Test pipeline without loading real checkpoint')
    parser.add_argument('--verbose', action='store_true')
    
    # Adapter arguments
    parser.add_argument('--adapter-checkpoint', type=str, default=None, help='Path to adapter checkpoint (optional)')
    parser.add_argument('--adapter-type', type=str, choices=['linear', 'mlp', 'auto'], default='auto',
                        help='Adapter type (auto-detected from checkpoint if not specified)')
    
    args = parser.parse_args()
    
    # Validate inputs
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"ERROR: Index not found: {index_path}", file=sys.stderr)
        sys.exit(1)
    
    checkpoint_path = Path(args.checkpoint)
    if not args.dry_run and not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)
    
    # Auto-detect model type from checkpoint name
    if args.model_type == 'auto':
        if 'ridge' in str(checkpoint_path).lower():
            args.model_type = 'ridge'
        elif 'mlp' in str(checkpoint_path).lower():
            args.model_type = 'mlp'
        elif 'two_stage' in str(checkpoint_path).lower():
            args.model_type = 'two_stage'
        else:
            args.model_type = 'two_stage'  # Default
        logger.info(f"Auto-detected model type: {args.model_type}")
    
    logger.info("=" * 80)
    logger.info("PERCEPTION-TO-IMAGERY TRANSFER EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Index: {index_path}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Dry run: {args.dry_run}")
    if args.adapter_checkpoint:
        logger.info(f"Adapter: {args.adapter_checkpoint}")
    logger.info("")
    
    # Load model
    model, meta = load_checkpoint_and_model(
        str(checkpoint_path), 
        args.model_type, 
        args.device,
        dry_run=args.dry_run,
        adapter_checkpoint=args.adapter_checkpoint,
        adapter_type=args.adapter_type if args.adapter_type != 'auto' else None
    )
    
    # Load adapter if specified
    if args.adapter_checkpoint and not args.dry_run:
        adapter_path = Path(args.adapter_checkpoint)
        if not adapter_path.exists():
            print(f"ERROR: Adapter checkpoint not found: {adapter_path}", file=sys.stderr)
            sys.exit(1)
        
        model, adapter_meta = load_adapter_model(
            model,
            str(adapter_path),
            args.adapter_type if args.adapter_type != 'auto' else None,
            args.device
        )
        meta['adapter'] = adapter_meta
        logger.info(f"✓ Loaded adapter (type={adapter_meta.get('adapter_type', 'unknown')})")
    
    # Load dataset
    logger.info(f"Loading dataset from {index_path}...")
    from fmri2img.data.nsd_imagery import NSDImageryDataset
    
    # Determine subject from index
    df_peek = pd.read_parquet(index_path)
    subject = df_peek['subject'].iloc[0]
    logger.info(f"Subject: {subject}")
    
    dataset = NSDImageryDataset(
        index_path=str(index_path),
        subject=subject,
        condition=args.mode if args.mode != 'both' else None,
        split_filter=args.split if args.split != 'all' else None,
        cache_root=args.cache_root,
        shuffle=False,
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Run evaluation
    metrics, per_trial_df = evaluate_on_dataset(
        model=model,
        dataset=dataset,
        device=args.device,
        batch_size=args.batch_size,
        k_values=args.retrieval_k,
    )
    
    # Write report
    output_dir = Path(args.output_dir)
    write_report(output_dir, metrics, per_trial_df, args)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"CLIP Cosine: {metrics['clip_cosine_mean']:.4f} ± {metrics['clip_cosine_std']:.4f}")
    logger.info("")


if __name__ == "__main__":
    main()
