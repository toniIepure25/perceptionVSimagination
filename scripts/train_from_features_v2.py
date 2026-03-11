#!/usr/bin/env python3
"""
Train all models from pre-extracted features (v2).
===================================================

Extended with:
- Symmetric InfoNCE (CLIP-style, bidirectional)
- Ablation config tagging (--config-name)
- Multi-layer TwoStage training (--multilayer)
- Learnable layer weights
- Multi-layer InfoNCE

Usage:
    # MLP with ablation tag
    python scripts/train_from_features_v2.py \
        --features-dir outputs/features/novel/subj01 \
        --model mlp --config-name novel_cosine \
        --cosine-weight 1.0 --mse-weight 0.0 --infonce-weight 0.0

    # Multi-layer TwoStage
    python scripts/train_from_features_v2.py \
        --features-dir outputs/features/baseline/subj01 \
        --model multilayer \
        --multilayer-cache outputs/clip_cache/clip_multilayer.parquet \
        --config-name multilayer_baseline \
        --learnable-weights --multilayer-infonce
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data loading
# ============================================================================

def load_features_and_targets(features_dir, clip_cache_path, index_root, subject, limit=None):
    """Load pre-extracted features and match with CLIP embeddings."""
    from fmri2img.data.clip_cache import CLIPCache
    from fmri2img.data.nsd_index_reader import read_subject_index

    fd = Path(features_dir)
    X = np.load(fd / "X.npy")
    nsd_ids = np.load(fd / "nsd_ids.npy")

    logger.info(f"Loaded features: X={X.shape}, nsd_ids={nsd_ids.shape}")

    # Load CLIP cache
    clip_cache = CLIPCache(clip_cache_path).load()
    stats = clip_cache.stats()
    logger.info(f"CLIP cache: {stats['cache_size']} embeddings")

    # Match CLIP embeddings
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

    logger.info(f"Matched: X={X.shape}, Y={Y.shape} ({valid_mask.sum()}/{len(valid_mask)} valid)")

    # Center features (critical for novel soft-reliability weights where
    # PCA mean mismatch creates a constant bias)
    feature_mean = X.mean(axis=0)
    feature_mean_norm = np.linalg.norm(feature_mean)
    if feature_mean_norm > 0.01:
        logger.warning(f"Feature centering: mean norm={feature_mean_norm:.4f} (>0.01 threshold). "
                       f"Centering to remove PCA mean bias.")
        X = X - feature_mean
    else:
        logger.info(f"Features already centered (mean norm={feature_mean_norm:.6f})")

    df = read_subject_index(index_root, subject)
    if limit:
        df = df.head(limit)

    return X, Y, nsd_ids, df


def load_multilayer_targets(nsd_ids, multilayer_cache_path):
    """Load multi-layer CLIP targets for given nsd_ids.

    Uses projected representations from the actual CLIP cache:
    - layer_12_proj → layer_12 (768-D): Mid-level semantic features
    - layer_18_proj → layer_18 (768-D): Late semantic features
    - final (768-D): Final CLIP embedding (ViT-L/14)

    Returns:
        Y_dict: {layer_name: np.ndarray (N, D)} for layer_12/layer_18/final (all 768-D)
        valid_mask: boolean mask of which nsd_ids had cache hits
    """
    df = pd.read_parquet(multilayer_cache_path)
    nsd_col = 'nsd_id' if 'nsd_id' in df.columns else 'nsdId'
    df = df.set_index(nsd_col)

    # Map cache column names to model layer names
    # Use projected versions (768-dim) for consistency
    column_map = {}
    if 'layer_12_proj' in df.columns:
        column_map['layer_12'] = 'layer_12_proj'
    elif 'layer_12' in df.columns:
        column_map['layer_12'] = 'layer_12'
    
    if 'layer_18_proj' in df.columns:
        column_map['layer_18'] = 'layer_18_proj'
    elif 'layer_18' in df.columns:
        column_map['layer_18'] = 'layer_18'
    
    if 'final' in df.columns:
        column_map['final'] = 'final'
    elif 'fused' in df.columns:
        column_map['final'] = 'fused'
    
    layers = list(column_map.keys())
    logger.info(f"Multilayer: using layers {layers} from columns {column_map}")
    
    Y_dict = {l: [] for l in layers}
    valid_mask = []

    for nid in nsd_ids:
        nid_int = int(nid)
        if nid_int in df.index:
            valid_mask.append(True)
            for layer_name, col_name in column_map.items():
                vec = np.array(df.loc[nid_int, col_name], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    vec = vec / norm
                Y_dict[layer_name].append(vec)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    for l in layers:
        Y_dict[l] = np.stack(Y_dict[l]).astype(np.float32) if Y_dict[l] else np.empty((0, 768))

    logger.info(f"Multilayer cache: {valid_mask.sum()}/{len(valid_mask)} matched, "
                f"dims: {', '.join(f'{l}={Y_dict[l].shape[1]}' for l in layers)}")
    return Y_dict, valid_mask


def split_by_index(X, Y, nsd_ids, df, seed=42):
    """Split data 80/10/10 using deterministic permutation."""
    n = len(X)
    np.random.seed(seed)
    perm = np.random.permutation(n)

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return (
        X[train_idx], Y[train_idx], nsd_ids[train_idx],
        X[val_idx], Y[val_idx], nsd_ids[val_idx],
        X[test_idx], Y[test_idx], nsd_ids[test_idx],
    )


def split_multilayer(X, Y_dict, nsd_ids, seed=42):
    """Split multi-layer targets consistently with X."""
    n = len(X)
    np.random.seed(seed)
    perm = np.random.permutation(n)

    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    def subset(idx):
        return X[idx], {l: Y_dict[l][idx] for l in Y_dict}, nsd_ids[idx]

    return subset(train_idx), subset(val_idx), subset(test_idx)


# ============================================================================
# Evaluation helper
# ============================================================================

def evaluate_predictions_and_retrieval(Y_test, Y_pred):
    """Compute cosine + retrieval metrics."""
    from fmri2img.models.ridge import evaluate_predictions
    from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics

    test_metrics = evaluate_predictions(Y_test, Y_pred, normalize=True)
    gt_indices = np.arange(len(Y_test))
    ret = retrieval_at_k(Y_pred, Y_test, gt_indices, ks=(1, 5, 10))
    rank = compute_ranking_metrics(Y_pred, Y_test, gt_indices)

    all_metrics = {**test_metrics, **ret, **rank}
    logger.info(f"TEST: cosine={test_metrics['cosine']:.4f}, "
                f"R@1={ret.get('R@1', 0):.4f}, R@5={ret.get('R@5', 0):.4f}, "
                f"R@10={ret.get('R@10', 0):.4f}, "
                f"median_rank={rank.get('median_rank', -1):.0f}")
    return all_metrics


# ============================================================================
# Ridge
# ============================================================================

def train_ridge(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                alpha_grid, checkpoint_dir, subject, report_dir, config_name=""):
    """Train Ridge with alpha selection on validation set."""
    from fmri2img.models.ridge import RidgeEncoder, evaluate_predictions
    from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics

    logger.info(f"Training Ridge: X={X_train.shape}, alpha_grid={alpha_grid}")

    best_alpha, best_cosine = None, -np.inf
    for alpha in alpha_grid:
        model = RidgeEncoder(alpha=alpha)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_val, normalize=True)
        metrics = evaluate_predictions(Y_val, Y_pred, normalize=True)
        logger.info(f"  alpha={alpha:8.3f}: cosine={metrics['cosine']:.4f}")
        if metrics['cosine'] > best_cosine:
            best_cosine = metrics['cosine']
            best_alpha = alpha

    logger.info(f"Best alpha={best_alpha:.3f} (val cosine={best_cosine:.4f})")

    # Retrain on train+val
    X_tv = np.vstack([X_train, X_val])
    Y_tv = np.vstack([Y_train, Y_val])
    final = RidgeEncoder(alpha=best_alpha)
    final.fit(X_tv, Y_tv)

    Y_pred = final.predict(X_test, normalize=True)
    all_metrics = evaluate_predictions_and_retrieval(Y_test, Y_pred)

    ckpt = Path(checkpoint_dir) / subject / "ridge.pkl"
    final.save(ckpt)
    logger.info(f"Saved: {ckpt}")

    report = {
        "model": "Ridge", "config_name": config_name, "subject": subject,
        "best_alpha": best_alpha,
        "data": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "test_metrics": all_metrics,
    }
    _write_report(report, report_dir, subject, "ridge_eval.json")
    return all_metrics


# ============================================================================
# MLP
# ============================================================================

def train_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test,
              hidden, dropout, lr, wd, epochs, patience, batch_size,
              cosine_w, mse_w, infonce_w, temperature,
              checkpoint_dir, subject, report_dir, config_name="", device="cuda"):
    """Train MLP encoder with symmetric InfoNCE + early stopping."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from fmri2img.models.mlp import MLPEncoder, save_mlp
    from fmri2img.models.train_utils import torch_seed_all
    # Use symmetric InfoNCE from training.losses (now fixed)
    from fmri2img.training.losses import mse_loss, cosine_loss, info_nce_loss

    torch_seed_all(42)

    input_dim, output_dim = X_train.shape[1], Y_train.shape[1]
    logger.info(f"Training MLP [{config_name}]: {input_dim}D → {hidden}D → {output_dim}D")
    logger.info(f"  Loss weights: cosine={cosine_w}, mse={mse_w}, infonce={infonce_w}, temp={temperature}")

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = MLPEncoder(input_dim=input_dim, hidden=hidden, dropout=dropout, output_dim=output_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_cosine, best_epoch, patience_counter = -np.inf, 0, 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)

            loss = torch.tensor(0.0, device=device)
            if cosine_w > 0:
                loss = loss + cosine_w * cosine_loss(pred, yb)
            if mse_w > 0:
                loss = loss + mse_w * mse_loss(pred, yb)
            if infonce_w > 0:
                loss = loss + infonce_w * info_nce_loss(pred, yb, temperature)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate
        model.eval()
        val_pred = _predict_all(model, val_loader, device)
        val_true = np.vstack([yb.numpy() for _, yb in val_loader])
        from fmri2img.models.ridge import evaluate_predictions
        val_cosine = evaluate_predictions(val_true, val_pred, normalize=True)['cosine']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss/max(n_batches,1):.4f}, val_cosine={val_cosine:.4f}")

        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    logger.info(f"  Best epoch: {best_epoch}, val_cosine: {best_val_cosine:.4f}")

    # Retrain on train+val
    logger.info(f"  Retraining on train+val for {best_epoch} epochs...")
    X_tv = np.vstack([X_train, X_val])
    Y_tv = np.vstack([Y_train, Y_val])
    tv_ds = TensorDataset(torch.from_numpy(X_tv).float(), torch.from_numpy(Y_tv).float())
    tv_loader = DataLoader(tv_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    final_model = MLPEncoder(input_dim=input_dim, hidden=hidden, dropout=dropout, output_dim=output_dim).to(device)
    final_opt = AdamW(final_model.parameters(), lr=lr, weight_decay=wd)
    final_sched = CosineAnnealingLR(final_opt, T_max=best_epoch)

    for ep in range(best_epoch):
        final_model.train()
        for xb, yb in tv_loader:
            xb, yb = xb.to(device), yb.to(device)
            final_opt.zero_grad()
            pred = final_model(xb)
            loss = torch.tensor(0.0, device=device)
            if cosine_w > 0:
                loss = loss + cosine_w * cosine_loss(pred, yb)
            if mse_w > 0:
                loss = loss + mse_w * mse_loss(pred, yb)
            if infonce_w > 0:
                loss = loss + infonce_w * info_nce_loss(pred, yb, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_opt.step()
        final_sched.step()

    # Test evaluation
    Y_pred = _predict_all(final_model, test_loader, device)
    all_metrics = evaluate_predictions_and_retrieval(Y_test, Y_pred)

    ckpt = Path(checkpoint_dir) / subject / "mlp.pt"
    meta = {
        "input_dim": input_dim, "hidden": hidden, "dropout": dropout,
        "output_dim": output_dim, "best_epoch": best_epoch,
        "best_val_cosine": float(best_val_cosine),
        "lr": lr, "weight_decay": wd, "config_name": config_name,
        "cosine_w": cosine_w, "mse_w": mse_w, "infonce_w": infonce_w,
        "temperature": temperature, "subject": subject,
    }
    from fmri2img.models.mlp import save_mlp
    save_mlp(final_model, str(ckpt), meta)
    logger.info(f"  Saved: {ckpt}")

    report = {
        "model": "MLP", "config_name": config_name, "subject": subject,
        "hyperparameters": meta,
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "data": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "test_metrics": all_metrics,
    }
    _write_report(report, report_dir, subject, "mlp_eval.json")
    return all_metrics


# ============================================================================
# TwoStage (single-layer)
# ============================================================================

def train_two_stage(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                    latent_dim, n_blocks, head_hidden,
                    mse_w, cosine_w, infonce_w, temperature,
                    lr, wd, epochs, patience, batch_size,
                    checkpoint_dir, subject, report_dir, config_name="", device="cuda"):
    """Train two-stage encoder with symmetric InfoNCE."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from fmri2img.models.encoders import TwoStageEncoder, save_two_stage_encoder
    from fmri2img.training.losses import MultiLoss
    from fmri2img.models.train_utils import torch_seed_all

    torch_seed_all(42)

    input_dim, output_dim = X_train.shape[1], Y_train.shape[1]
    logger.info(f"Training TwoStage [{config_name}]: {input_dim}D → {latent_dim}D ({n_blocks} blocks) → {output_dim}D")
    logger.info(f"  Loss: mse={mse_w}, cosine={cosine_w}, infonce={infonce_w}, temp={temperature}")

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = TwoStageEncoder(
        input_dim=input_dim, latent_dim=latent_dim, n_blocks=n_blocks,
        head_type="mlp", head_hidden_dim=head_hidden, dropout=0.3, output_dim=output_dim,
    ).to(device)

    criterion = MultiLoss(
        mse_weight=mse_w, cosine_weight=cosine_w,
        info_nce_weight=infonce_w, temperature=temperature,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Parameters: {n_params:,}")

    best_val_cosine, best_epoch, patience_counter, best_state = -np.inf, 0, 0, None

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss, _ = criterion(pred, yb, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        val_pred = _predict_all(model, val_loader, device)
        val_true = np.vstack([yb.numpy() for _, yb in val_loader])
        from fmri2img.models.ridge import evaluate_predictions
        val_cosine = evaluate_predictions(val_true, val_pred, normalize=True)['cosine']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss/max(n_batches,1):.4f}, val_cosine={val_cosine:.4f}")

        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    logger.info(f"  Best epoch: {best_epoch}, val_cosine: {best_val_cosine:.4f}")

    # Retrain on train+val
    logger.info(f"  Retraining on train+val for {best_epoch} epochs...")
    X_tv = np.vstack([X_train, X_val])
    Y_tv = np.vstack([Y_train, Y_val])
    tv_ds = TensorDataset(torch.from_numpy(X_tv).float(), torch.from_numpy(Y_tv).float())
    tv_loader = DataLoader(tv_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    final_model = TwoStageEncoder(
        input_dim=input_dim, latent_dim=latent_dim, n_blocks=n_blocks,
        head_type="mlp", head_hidden_dim=head_hidden, dropout=0.3, output_dim=output_dim,
    ).to(device)
    final_opt = AdamW(final_model.parameters(), lr=lr, weight_decay=wd)
    final_sched = CosineAnnealingLR(final_opt, T_max=best_epoch)

    for ep in range(best_epoch):
        final_model.train()
        for xb, yb in tv_loader:
            xb, yb = xb.to(device), yb.to(device)
            final_opt.zero_grad()
            pred = final_model(xb)
            loss, _ = criterion(pred, yb, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            final_opt.step()
        final_sched.step()

    Y_pred = _predict_all(final_model, test_loader, device)
    all_metrics = evaluate_predictions_and_retrieval(Y_test, Y_pred)

    ckpt = Path(checkpoint_dir) / subject / "two_stage_best.pt"
    meta = {
        "input_dim": input_dim, "latent_dim": latent_dim,
        "n_blocks": n_blocks, "head_hidden": head_hidden,
        "head_hidden_dim": head_hidden, "head_type": "mlp",
        "output_dim": output_dim, "dropout": 0.3,
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "n_params": n_params, "config_name": config_name, "subject": subject,
    }
    save_two_stage_encoder(final_model, str(ckpt), meta)
    logger.info(f"  Saved: {ckpt}")

    report = {
        "model": "TwoStage", "config_name": config_name, "subject": subject,
        "hyperparameters": {
            "latent_dim": latent_dim, "n_blocks": n_blocks, "head_hidden": head_hidden,
            "lr": lr, "wd": wd, "mse_w": mse_w, "cosine_w": cosine_w,
            "infonce_w": infonce_w, "temperature": temperature,
        },
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "data": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "test_metrics": all_metrics,
    }
    _write_report(report, report_dir, subject, "two_stage_eval.json")
    return all_metrics


# ============================================================================
# Multi-layer TwoStage
# ============================================================================

class MultiLayerTensorDataset:
    """Dataset that returns (x, y_dict) where y_dict has per-layer arrays."""

    def __init__(self, X, Y_dict):
        import torch
        self.X = torch.from_numpy(X).float()
        self.Y_dict = {l: torch.from_numpy(Y_dict[l]).float() for l in Y_dict}
        self.n = len(X)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.X[idx]
        y = {l: self.Y_dict[l][idx] for l in self.Y_dict}
        return x, y


def multilayer_collate_fn(batch):
    """Collate function for MultiLayerTensorDataset."""
    import torch
    xs, ys = zip(*batch)
    x_batch = torch.stack(xs)
    layers = ys[0].keys()
    y_batch = {l: torch.stack([y[l] for y in ys]) for l in layers}
    return x_batch, y_batch


def train_multilayer(X_train, Y_train_dict, X_val, Y_val_dict, X_test, Y_test_final,
                     latent_dim, n_blocks, head_hidden,
                     cosine_w, mse_w, infonce_w, temperature,
                     lr, wd, epochs, patience, batch_size,
                     learnable_weights, multilayer_infonce, infonce_combination,
                     checkpoint_dir, subject, report_dir, config_name="", device="cuda"):
    """Train MultiLayerTwoStageEncoder with per-layer supervision."""
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from fmri2img.models.encoders import MultiLayerTwoStageEncoder, save_two_stage_encoder
    from fmri2img.training.losses import MultiLayerLoss
    from fmri2img.models.train_utils import torch_seed_all

    torch_seed_all(42)

    input_dim = X_train.shape[1]
    layer_names = list(Y_train_dict.keys())
    logger.info(f"Training MultiLayer TwoStage [{config_name}]: {input_dim}D → {latent_dim}D ({n_blocks} blocks)")
    logger.info(f"  layers: {layer_names}")
    logger.info(f"  shared_head_backbone=True, head_hidden={head_hidden}")
    logger.info(f"  learnable_weights={learnable_weights}, multilayer_infonce={multilayer_infonce}")

    train_ds = MultiLayerTensorDataset(X_train, Y_train_dict)
    val_ds = MultiLayerTensorDataset(X_val, Y_val_dict)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=multilayer_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, collate_fn=multilayer_collate_fn)

    # Build layer_dims from actual data
    layer_dims = {l: Y_train_dict[l].shape[1] for l in layer_names}

    model = MultiLayerTwoStageEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_blocks=n_blocks,
        dropout=0.3,
        head_type="mlp",
        head_hidden_dim=head_hidden,
        enabled_layers=layer_names,
        layer_dims=layer_dims,
        shared_head_backbone=True,
    ).to(device)

    # Uniform initial weights across layers
    n_layers = len(layer_names)
    layer_weights = {l: 1.0 / n_layers for l in layer_names}
    criterion = MultiLayerLoss(
        layer_weights=layer_weights,
        layer_names=layer_names,
        use_mse=mse_w > 0,
        mse_weight=mse_w,
        use_learnable_weights=learnable_weights,
        use_multilayer_infonce=multilayer_infonce,
        infonce_weight=infonce_w if multilayer_infonce else 0.0,
        infonce_temperature=temperature,
        infonce_combination=infonce_combination,
    )

    # Combine model + criterion params (learnable weights are in criterion)
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = AdamW(all_params, lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    n_loss_params = sum(p.numel() for p in criterion.parameters())
    logger.info(f"  Model params: {n_params:,}, Loss params: {n_loss_params}")

    best_val_cosine, best_epoch, patience_counter, best_state = -np.inf, 0, 0, None

    for epoch in range(epochs):
        model.train()
        criterion.train()
        epoch_loss, n_batches = 0.0, 0

        for xb, yb_dict in train_loader:
            xb = xb.to(device)
            yb_dict = {l: yb_dict[l].to(device) for l in yb_dict}

            optimizer.zero_grad()
            pred_dict = model(xb)

            if multilayer_infonce:
                loss, components = criterion(pred_dict, yb_dict, model=model, return_components=True)
            else:
                loss, components = criterion(pred_dict, yb_dict, return_components=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate using 'final' layer cosine
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb_dict in val_loader:
                xb = xb.to(device)
                pred_dict = model(xb)
                val_preds.append(pred_dict['final'].cpu().numpy())
                val_trues.append(yb_dict['final'].numpy())

        val_pred = np.vstack(val_preds)
        val_true = np.vstack(val_trues)
        from fmri2img.models.ridge import evaluate_predictions
        val_cosine = evaluate_predictions(val_true, val_pred, normalize=True)['cosine']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            eff_w = criterion.get_effective_weights() if learnable_weights else layer_weights
            w_str = ", ".join(f"{l}={eff_w[l]:.3f}" for l in layer_names)
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss/max(n_batches,1):.4f}, "
                        f"val_cosine(final)={val_cosine:.4f}, weights=[{w_str}]")

        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    logger.info(f"  Best epoch: {best_epoch}, val_cosine(final): {best_val_cosine:.4f}")

    # Retrain on train+val
    logger.info(f"  Retraining on train+val for {best_epoch} epochs...")
    X_tv = np.vstack([X_train, X_val])
    Y_tv_dict = {l: np.vstack([Y_train_dict[l], Y_val_dict[l]]) for l in Y_train_dict}
    tv_ds = MultiLayerTensorDataset(X_tv, Y_tv_dict)
    tv_loader = DataLoader(tv_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, collate_fn=multilayer_collate_fn)

    final_model = MultiLayerTwoStageEncoder(
        input_dim=input_dim, latent_dim=latent_dim, n_blocks=n_blocks,
        dropout=0.3, head_type="mlp", head_hidden_dim=head_hidden,
        enabled_layers=layer_names,
        layer_dims=layer_dims,
        shared_head_backbone=True,
    ).to(device)

    final_criterion = MultiLayerLoss(
        layer_weights=layer_weights, layer_names=layer_names,
        use_mse=mse_w > 0, mse_weight=mse_w,
        use_learnable_weights=learnable_weights,
        use_multilayer_infonce=multilayer_infonce,
        infonce_weight=infonce_w if multilayer_infonce else 0.0,
        infonce_temperature=temperature, infonce_combination=infonce_combination,
    )
    all_final_params = list(final_model.parameters()) + list(final_criterion.parameters())
    final_opt = AdamW(all_final_params, lr=lr, weight_decay=wd)
    final_sched = CosineAnnealingLR(final_opt, T_max=best_epoch)

    for ep in range(best_epoch):
        final_model.train()
        final_criterion.train()
        for xb, yb_dict in tv_loader:
            xb = xb.to(device)
            yb_dict = {l: yb_dict[l].to(device) for l in yb_dict}
            final_opt.zero_grad()
            pred_dict = final_model(xb)
            if multilayer_infonce:
                loss, _ = final_criterion(pred_dict, yb_dict, model=final_model, return_components=True)
            else:
                loss, _ = final_criterion(pred_dict, yb_dict, return_components=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_final_params, 1.0)
            final_opt.step()
        final_sched.step()

    # Test evaluation using 'final' layer vs single-layer CLIP targets
    final_model.eval()
    test_ds_simple = MultiLayerTensorDataset(X_test, {'final': Y_test_final})
    test_loader = DataLoader(test_ds_simple, batch_size=batch_size * 2, shuffle=False,
                             num_workers=0, collate_fn=multilayer_collate_fn)

    all_pred = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred_dict = final_model(xb.to(device))
            all_pred.append(pred_dict['final'].cpu().numpy())

    Y_pred = np.vstack(all_pred)
    all_metrics = evaluate_predictions_and_retrieval(Y_test_final, Y_pred)

    # Save
    ckpt = Path(checkpoint_dir) / subject / "multilayer_best.pt"
    eff_w = final_criterion.get_effective_weights() if learnable_weights else layer_weights
    meta = {
        "input_dim": input_dim, "latent_dim": latent_dim,
        "n_blocks": n_blocks, "head_hidden_dim": head_hidden,
        "head_type": "mlp", "dropout": 0.3,
        "shared_head_backbone": True,
        "enabled_layers": ['layer_4', 'layer_8', 'layer_12', 'final'],
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "n_params": n_params, "config_name": config_name, "subject": subject,
        "final_layer_weights": {k: float(v) for k, v in eff_w.items()},
    }
    save_two_stage_encoder(final_model, str(ckpt), meta)
    logger.info(f"  Saved: {ckpt}")

    report = {
        "model": "MultiLayerTwoStage", "config_name": config_name, "subject": subject,
        "hyperparameters": {
            "latent_dim": latent_dim, "n_blocks": n_blocks, "head_hidden": head_hidden,
            "lr": lr, "wd": wd, "learnable_weights": learnable_weights,
            "multilayer_infonce": multilayer_infonce,
        },
        "final_layer_weights": meta["final_layer_weights"],
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "data": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "test_metrics": all_metrics,
    }
    _write_report(report, report_dir, subject, "multilayer_eval.json")
    return all_metrics


# ============================================================================
# Utilities
# ============================================================================

def _predict_all(model, loader, device):
    """Run model inference on all batches, return stacked numpy."""
    import torch
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, *_ in loader:
            pred = model(xb.to(device))
            if isinstance(pred, dict):
                pred = pred['final']
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def _write_report(report, report_dir, subject, filename):
    """Write JSON report."""
    rp = Path(report_dir) / subject / filename
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info(f"  Report: {rp}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train models from pre-extracted features (v2)")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet")
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--model", required=True,
                        choices=["ridge", "mlp", "two_stage", "multilayer"])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--config-name", default="default",
                        help="Tag for ablation tracking")
    parser.add_argument("--device", default="cuda")

    # Ridge
    parser.add_argument("--alpha-grid", default="0.1,1,3,10,30,100,300,1000")

    # MLP / TwoStage
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--infonce-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)

    # TwoStage-specific
    parser.add_argument("--latent-dim", type=int, default=768)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--head-hidden", type=int, default=512)

    # Multilayer-specific
    parser.add_argument("--multilayer-cache", default="outputs/clip_cache/clip_multilayer.parquet",
                        help="Path to multilayer CLIP cache (parquet)")
    parser.add_argument("--learnable-weights", action="store_true",
                        help="Learn layer weights via gradient descent")
    parser.add_argument("--multilayer-infonce", action="store_true",
                        help="Add InfoNCE on combined multi-layer representation")
    parser.add_argument("--infonce-combination", default="weighted_pool",
                        choices=["weighted_pool", "concat_project", "average"])

    args = parser.parse_args()

    # Load features + single-layer CLIP targets
    X, Y, nsd_ids, df = load_features_and_targets(
        args.features_dir, args.clip_cache, args.index_root, args.subject
    )

    t0 = time.time()

    if args.model == "multilayer":
        # Load multilayer targets
        Y_dict, ml_mask = load_multilayer_targets(nsd_ids, args.multilayer_cache)

        # Filter to only samples with multilayer cache
        X = X[ml_mask]
        Y = Y[ml_mask]
        nsd_ids = nsd_ids[ml_mask]
        for l in Y_dict:
            Y_dict[l] = Y_dict[l]  # already filtered in load_multilayer_targets

        (X_train, Y_train_dict, _), (X_val, Y_val_dict, _), (X_test, Y_test_dict, _) = \
            split_multilayer(X, Y_dict, nsd_ids)

        # For test evaluation, use single-layer 'final' targets from main CLIP cache
        # (to be comparable with single-layer models)
        _, Y_test_final_single, _ = split_by_index(
            X, Y, nsd_ids, df
        )[6:9]  # test split
        # Actually, let's use the 'final' from the multilayer dict for simplicity
        Y_test_final = Y_test_dict['final']

        logger.info(f"MultiLayer splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        train_multilayer(
            X_train, Y_train_dict, X_val, Y_val_dict, X_test, Y_test_final,
            args.latent_dim, args.n_blocks, args.head_hidden,
            args.cosine_weight, args.mse_weight, args.infonce_weight, args.temperature,
            args.lr, args.wd, args.epochs, args.patience, args.batch_size,
            args.learnable_weights, args.multilayer_infonce, args.infonce_combination,
            args.checkpoint_dir, args.subject, args.report_dir, args.config_name, args.device,
        )

    else:
        # Standard single-layer models
        (X_train, Y_train, _, X_val, Y_val, _, X_test, Y_test, _) = split_by_index(
            X, Y, nsd_ids, df
        )
        logger.info(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        if args.model == "ridge":
            alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
            train_ridge(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                        alpha_grid, args.checkpoint_dir, args.subject,
                        args.report_dir, args.config_name)

        elif args.model == "mlp":
            train_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                      args.hidden, args.dropout, args.lr, args.wd,
                      args.epochs, args.patience, args.batch_size,
                      args.cosine_weight, args.mse_weight, args.infonce_weight,
                      args.temperature,
                      args.checkpoint_dir, args.subject,
                      args.report_dir, args.config_name, args.device)

        elif args.model == "two_stage":
            train_two_stage(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                            args.latent_dim, args.n_blocks, args.head_hidden,
                            args.mse_weight, args.cosine_weight, args.infonce_weight,
                            args.temperature,
                            args.lr, args.wd, args.epochs, args.patience, args.batch_size,
                            args.checkpoint_dir, args.subject,
                            args.report_dir, args.config_name, args.device)

    elapsed = time.time() - t0
    logger.info(f"Training completed in {elapsed / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
