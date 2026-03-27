#!/usr/bin/env python3
"""
Train Imagery Adapter
=====================

Train a lightweight adapter on top of a frozen perception encoder to improve
cross-domain transfer from visual perception to mental imagery.

Strategy:
- Load pre-trained perception checkpoint (ridge/mlp/two_stage)
- Freeze all base model parameters
- Train only adapter parameters on imagery training set
- Validate on imagery validation set
- Save best adapter checkpoint based on validation performance

Usage:
    python scripts/train_imagery_adapter.py \\
        --index cache/indices/imagery/subj01.parquet \\
        --checkpoint checkpoints/two_stage/subj01/best.pt \\
        --model-type two_stage \\
        --adapter mlp \\
        --output-dir outputs/adapters/subj01 \\
        --epochs 50 \\
        --lr 1e-3
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_base_model(checkpoint_path: str, model_type: str, device: str):
    """Load frozen base perception model."""
    logger.info(f"Loading {model_type} base model from {checkpoint_path}")
    
    if model_type == "ridge":
        from fmri2img.models.ridge import RidgeEncoder
        encoder = RidgeEncoder.load(checkpoint_path)
        
        # Wrap in torch module for consistency
        class RidgeWrapper(nn.Module):
            def __init__(self, ridge_model):
                super().__init__()
                self.ridge_model = ridge_model
            
            def forward(self, x):
                # Convert to numpy, predict, convert back
                x_np = x.cpu().numpy()
                pred = self.ridge_model.predict(x_np)
                return torch.from_numpy(pred).to(x.device)
        
        model = RidgeWrapper(encoder).to(device)
    
    elif model_type == "mlp":
        from fmri2img.models.mlp import load_mlp
        model, meta = load_mlp(checkpoint_path, map_location=device)
        model = model.to(device)
    
    elif model_type == "two_stage":
        from fmri2img.models.encoders import load_two_stage_encoder
        model, meta = load_two_stage_encoder(checkpoint_path, map_location=device)
        model = model.to(device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info(f"✓ Loaded base model (frozen)")
    return model


def compute_clip_targets(
    dataset,
    device: str,
    cache_path: Optional[Path] = None,
    force_recompute: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute CLIP embeddings for target images/texts.
    
    Returns:
        (targets, valid_indices) - targets is (N, 512), valid_indices maps to dataset
    """
    if cache_path is not None and cache_path.exists() and not force_recompute:
        logger.info(f"Loading cached CLIP targets from {cache_path}")
        cached = torch.load(cache_path)
        return cached['targets'].to(device), cached['valid_indices']
    
    logger.info("Computing CLIP targets from dataset...")
    
    try:
        import clip
    except ImportError:
        raise ImportError("CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
    
    # Load CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    targets = []
    valid_indices = []
    
    for idx, sample in enumerate(tqdm(dataset, desc="Computing CLIP targets")):
        target_embed = None
        
        # Try image first
        if sample.get('target_image') is not None:
            img = sample['target_image']
            with torch.no_grad():
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                embed = clip_model.encode_image(img_tensor)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                target_embed = embed[0]
        
        # Fall back to text
        elif sample.get('target_text') is not None and sample['target_text'].strip():
            text = sample['target_text']
            with torch.no_grad():
                text_token = clip.tokenize([text]).to(device)
                embed = clip_model.encode_text(text_token)
                embed = embed / embed.norm(dim=-1, keepdim=True)
                target_embed = embed[0]
        
        if target_embed is not None:
            targets.append(target_embed.cpu())
            valid_indices.append(idx)
    
    targets = torch.stack(targets).to(device)
    
    logger.info(f"✓ Computed {len(targets)} CLIP targets ({len(valid_indices)}/{len(dataset)} samples)")
    
    # Cache results
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'targets': targets.cpu(),
            'valid_indices': valid_indices
        }, cache_path)
        logger.info(f"✓ Cached CLIP targets to {cache_path}")
    
    return targets, valid_indices


def create_filtered_dataset(dataset, valid_indices):
    """Create filtered dataset with only valid samples."""
    class FilteredDataset:
        def __init__(self, dataset, valid_indices):
            self.dataset = dataset
            self.valid_indices = set(valid_indices)
            self.samples = []
            
            # Pre-load valid samples
            for idx, sample in enumerate(dataset):
                if idx in self.valid_indices:
                    self.samples.append(sample)
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    return FilteredDataset(dataset, valid_indices)


def train_epoch(
    model: nn.Module,
    adapter: nn.Module,
    voxels: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: str,
    batch_size: int = 32,
    use_condition: bool = False
) -> Dict:
    """Train for one epoch."""
    adapter.train()
    model.eval()
    
    n_samples = len(voxels)
    indices = torch.randperm(n_samples)
    
    total_loss = 0.0
    n_batches = 0
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        
        x_batch = voxels[batch_indices].to(device)
        y_batch = targets[batch_indices].to(device)
        
        # Forward: base model (frozen) -> adapter (trainable)
        with torch.no_grad():
            base_embed = model(x_batch)
        
        # Adapter forward
        if use_condition:
            # All imagery samples -> condition_idx = 1
            condition_idx = torch.ones(len(x_batch), dtype=torch.long, device=device)
            pred_embed = adapter(base_embed, condition_idx=condition_idx)
        else:
            pred_embed = adapter(base_embed)
        
        # Compute loss
        loss = loss_fn(pred_embed, y_batch)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'n_batches': n_batches
    }


def validate_epoch(
    model: nn.Module,
    adapter: nn.Module,
    voxels: torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
    device: str,
    batch_size: int = 32,
    use_condition: bool = False
) -> Dict:
    """Validate for one epoch."""
    adapter.eval()
    model.eval()
    
    n_samples = len(voxels)
    total_loss = 0.0
    cosine_sims = []
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            x_batch = voxels[i:i+batch_size].to(device)
            y_batch = targets[i:i+batch_size].to(device)
            
            # Forward
            base_embed = model(x_batch)
            
            if use_condition:
                condition_idx = torch.ones(len(x_batch), dtype=torch.long, device=device)
                pred_embed = adapter(base_embed, condition_idx=condition_idx)
            else:
                pred_embed = adapter(base_embed)
            
            # Compute loss
            loss = loss_fn(pred_embed, y_batch)
            total_loss += loss.item() * len(x_batch)
            
            # Compute cosine similarity
            cos_sim = torch.sum(pred_embed * y_batch, dim=-1)
            cosine_sims.extend(cos_sim.cpu().numpy())
    
    return {
        'loss': total_loss / n_samples if n_samples > 0 else 0.0,
        'cosine_mean': float(np.mean(cosine_sims)),
        'cosine_std': float(np.std(cosine_sims)),
        'cosine_median': float(np.median(cosine_sims))
    }


def create_loss_function(loss_type: str, device: str):
    """Create loss function."""
    if loss_type == "cosine":
        # Cosine embedding loss (1 - cosine similarity)
        def cosine_loss(pred, target):
            return 1.0 - torch.sum(pred * target, dim=-1).mean()
        return cosine_loss
    
    elif loss_type == "mse":
        return nn.MSELoss()
    
    elif loss_type == "hybrid":
        # Weighted combination of cosine + MSE
        mse_fn = nn.MSELoss()
        def hybrid_loss(pred, target):
            cosine = 1.0 - torch.sum(pred * target, dim=-1).mean()
            mse = mse_fn(pred, target)
            return 0.5 * cosine + 0.5 * mse
        return hybrid_loss
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Train imagery adapter on frozen perception encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--index', type=str, required=True, help='Path to imagery index parquet')
    parser.add_argument('--split-train', type=str, default='train', help='Training split')
    parser.add_argument('--split-val', type=str, default='val', help='Validation split')
    parser.add_argument('--cache-root', type=str, default='cache', help='Cache directory')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True, help='Base model checkpoint')
    parser.add_argument('--model-type', type=str, choices=['ridge', 'mlp', 'two_stage'], required=True)
    parser.add_argument('--adapter', type=str, choices=['linear', 'mlp'], required=True)
    parser.add_argument('--condition-token', action='store_true', help='Use condition embeddings')
    
    # Training arguments
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--loss', type=str, choices=['cosine', 'mse', 'hybrid'], default='cosine')
    parser.add_argument('--early-stop-patience', type=int, default=10, help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--clip-cache-dir', type=str, default=None, help='CLIP target cache directory')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    # Setup file logging
    fh = logging.FileHandler(output_dir / 'logs' / 'training.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    logger.info("=" * 80)
    logger.info("IMAGERY ADAPTER TRAINING")
    logger.info("=" * 80)
    logger.info(f"Index: {args.index}")
    logger.info(f"Base checkpoint: {args.checkpoint}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Adapter: {args.adapter}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    
    # Load base model
    base_model = load_base_model(args.checkpoint, args.model_type, args.device)
    
    # Load datasets
    logger.info("Loading datasets...")
    from fmri2img.data.nsd_imagery import NSDImageryDataset
    
    # Infer subject from index
    df_peek = pd.read_parquet(args.index)
    subject = df_peek['subject'].iloc[0]
    
    train_dataset = NSDImageryDataset(
        index_path=args.index,
        subject=subject,
        condition='imagery',
        split_filter=args.split_train,
        cache_root=args.cache_root,
        shuffle=False
    )
    
    val_dataset = NSDImageryDataset(
        index_path=args.index,
        subject=subject,
        condition='imagery',
        split_filter=args.split_val,
        cache_root=args.cache_root,
        shuffle=False
    )
    
    logger.info(f"✓ Train: {len(train_dataset)} samples")
    logger.info(f"✓ Val: {len(val_dataset)} samples")
    
    # Compute CLIP targets
    cache_dir = Path(args.clip_cache_dir) if args.clip_cache_dir else output_dir / 'clip_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    train_targets, train_valid_idx = compute_clip_targets(
        train_dataset,
        args.device,
        cache_path=cache_dir / f'train_{args.split_train}.pt'
    )
    
    val_targets, val_valid_idx = compute_clip_targets(
        val_dataset,
        args.device,
        cache_path=cache_dir / f'val_{args.split_val}.pt'
    )
    
    # Collect voxel data
    logger.info("Loading voxel data...")
    train_voxels = []
    for idx in tqdm(train_valid_idx, desc="Loading train voxels"):
        sample = list(train_dataset)[idx]
        train_voxels.append(torch.from_numpy(sample['voxels']).float())
    train_voxels = torch.stack(train_voxels)
    
    val_voxels = []
    for idx in tqdm(val_valid_idx, desc="Loading val voxels"):
        sample = list(val_dataset)[idx]
        val_voxels.append(torch.from_numpy(sample['voxels']).float())
    val_voxels = torch.stack(val_voxels)
    
    logger.info(f"✓ Train voxels: {train_voxels.shape}")
    logger.info(f"✓ Val voxels: {val_voxels.shape}")
    
    # Create adapter
    logger.info(f"Creating {args.adapter} adapter...")
    from fmri2img.models.adapters import create_adapter
    
    adapter = create_adapter(
        adapter_type=args.adapter,
        embed_dim=512,
        use_condition=args.condition_token,
        condition_mode='add'
    ).to(args.device)
    
    n_params = sum(p.numel() for p in adapter.parameters())
    logger.info(f"✓ Adapter parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create loss function
    loss_fn = create_loss_function(args.loss, args.device)
    
    # Training loop
    logger.info("")
    logger.info("Starting training...")
    best_val_cosine = -1.0
    patience_counter = 0
    
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            base_model, adapter, train_voxels, train_targets,
            optimizer, loss_fn, args.device, args.batch_size,
            use_condition=args.condition_token
        )
        
        # Validate
        val_metrics = validate_epoch(
            base_model, adapter, val_voxels, val_targets,
            loss_fn, args.device, args.batch_size,
            use_condition=args.condition_token
        )
        
        epoch_time = time.time() - epoch_start
        
        # Log
        logger.info(
            f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val cosine: {val_metrics['cosine_mean']:.4f}"
        )
        
        # Save metrics
        train_metrics['epoch'] = epoch
        val_metrics['epoch'] = epoch
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Save best model
        if val_metrics['cosine_mean'] > best_val_cosine:
            best_val_cosine = val_metrics['cosine_mean']
            patience_counter = 0
            
            from fmri2img.models.adapters import save_adapter
            save_adapter(
                adapter,
                str(output_dir / 'checkpoints' / 'adapter_best.pt'),
                meta={
                    'adapter_type': args.adapter,
                    'use_condition': args.condition_token,
                    'embed_dim': 512,
                    'epoch': epoch,
                    'val_cosine': best_val_cosine,
                    'train_args': vars(args)
                }
            )
            logger.info(f"  ✓ Saved best model (val_cosine={best_val_cosine:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Save last model
    from fmri2img.models.adapters import save_adapter
    save_adapter(
        adapter,
        str(output_dir / 'checkpoints' / 'adapter_last.pt'),
        meta={
            'adapter_type': args.adapter,
            'use_condition': args.condition_token,
            'embed_dim': 512,
            'epoch': epoch,
            'train_args': vars(args)
        }
    )
    
    # Save training history
    with open(output_dir / 'metrics_train.json', 'w') as f:
        json.dump(train_metrics_history, f, indent=2)
    
    with open(output_dir / 'metrics_val.json', 'w') as f:
        json.dump(val_metrics_history, f, indent=2)
    
    # Save config
    with open(output_dir / 'config_resolved.yaml', 'w') as f:
        import yaml
        yaml.dump(vars(args), f, default_flow_style=False)
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Best val cosine: {best_val_cosine:.4f}")
    logger.info(f"Train samples: {len(train_valid_idx)}")
    logger.info(f"Val samples: {len(val_valid_idx)}")
    logger.info(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    logger.info("")


if __name__ == "__main__":
    main()
