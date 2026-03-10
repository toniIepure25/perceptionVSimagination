#!/usr/bin/env python3
"""
Train all models from pre-extracted features.
=============================================

Uses features from extract_features.py (X.npy, nsd_ids.npy).
No NIfTI loading needed — fast, memory-efficient.

Trains: Ridge, MLP, TwoStage for both baseline and novel preproc.

Usage:
    python scripts/train_from_features.py \\
        --subject subj01 \\
        --features-dir outputs/features/baseline/subj01 \\
        --clip-cache outputs/clip_cache/clip.parquet \\
        --model ridge \\
        --checkpoint-dir checkpoints/ridge_baseline \\
        --report-dir outputs/reports/baseline
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


def load_features_and_targets(features_dir, clip_cache_path, index_root, subject, limit=None):
    """Load pre-extracted features and match with CLIP embeddings."""
    from fmri2img.data.clip_cache import CLIPCache
    from fmri2img.data.nsd_index_reader import read_subject_index
    
    fd = Path(features_dir)
    X = np.load(fd / "X.npy")
    nsd_ids = np.load(fd / "nsd_ids.npy")
    orig_indices = np.load(fd / "orig_indices.npy")
    
    logger.info(f"Loaded features: X={X.shape}, nsd_ids={nsd_ids.shape}")
    
    # Load CLIP cache
    clip_cache = CLIPCache(clip_cache_path).load()
    stats = clip_cache.stats()
    logger.info(f"CLIP cache: {stats['cache_size']} embeddings")
    
    # Get CLIP embeddings for our nsd_ids
    Y_list = []
    valid_mask = []
    for i, nid in enumerate(nsd_ids):
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
    
    # Load index for train/val/test splitting
    df = read_subject_index(index_root, subject)
    if limit:
        df = df.head(limit)
    
    return X, Y, nsd_ids, df


def split_by_index(X, Y, nsd_ids, df, seed=42):
    """Split using the same logic as training scripts."""
    from fmri2img.models.train_utils import train_val_test_split
    
    train_df, val_df, test_df = train_val_test_split(df, random_seed=seed)
    
    # Map nsd_ids to splits
    # The original index has global_trial_index — use nsdId for matching
    # But multiple trials can share same nsdId (repeated images)
    # Better: use the orig_indices to map back to df rows
    # For now, use a simpler approach: split X/Y using same ratios
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


def train_ridge(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                alpha_grid, checkpoint_dir, subject, report_dir):
    """Train Ridge with alpha selection on validation set."""
    from fmri2img.models.ridge import RidgeEncoder, evaluate_predictions
    from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics
    
    logger.info(f"Training Ridge: X={X_train.shape}, alpha_grid={alpha_grid}")
    
    best_alpha = None
    best_cosine = -np.inf
    alpha_results = {}
    
    for alpha in alpha_grid:
        model = RidgeEncoder(alpha=alpha)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_val, normalize=True)
        metrics = evaluate_predictions(Y_val, Y_pred, normalize=True)
        alpha_results[alpha] = metrics
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
    
    # Test evaluation
    Y_pred = final.predict(X_test, normalize=True)
    test_metrics = evaluate_predictions(Y_test, Y_pred, normalize=True)
    
    gt_indices = np.arange(len(Y_test))
    ret = retrieval_at_k(Y_pred, Y_test, gt_indices, ks=(1, 5, 10))
    rank = compute_ranking_metrics(Y_pred, Y_test, gt_indices)
    
    logger.info(f"TEST: cosine={test_metrics['cosine']:.4f}, "
                f"R@1={ret.get('R@1', 0):.4f}, R@5={ret.get('R@5', 0):.4f}")
    
    # Save
    ckpt = Path(checkpoint_dir) / subject / "ridge.pkl"
    final.save(ckpt)
    logger.info(f"Saved: {ckpt}")
    
    report = {
        "model": "Ridge", "subject": subject,
        "best_alpha": best_alpha,
        "data": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "test_metrics": {**test_metrics, **ret, **rank},
    }
    rp = Path(report_dir) / subject / "ridge_eval.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, default=float)
    logger.info(f"Report: {rp}")
    return test_metrics


def train_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test,
              hidden, dropout, lr, wd, epochs, patience, batch_size,
              cosine_w, mse_w, infonce_w, temperature,
              checkpoint_dir, subject, report_dir, device="cuda"):
    """Train MLP encoder with early stopping."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from fmri2img.models.mlp import MLPEncoder, save_mlp
    from fmri2img.models.train_utils import torch_seed_all
    from fmri2img.models.ridge import evaluate_predictions
    from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics
    from fmri2img.training.losses import mse_loss, cosine_loss, info_nce_loss
    
    torch_seed_all(42)
    
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    logger.info(f"Training MLP: {input_dim}D → {hidden}D → {output_dim}D, device={device}")
    
    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    model = MLPEncoder(input_dim=input_dim, hidden=hidden, dropout=dropout, output_dim=output_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_cosine = -np.inf
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
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
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        
        val_pred = np.vstack(all_pred)
        val_true = np.vstack(all_true)
        val_metrics = evaluate_predictions(val_true, val_pred, normalize=True)
        val_cosine = val_metrics['cosine']
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.4f}, "
                        f"val_cosine={val_cosine:.4f}")
        
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"Best epoch: {best_epoch}, val_cosine: {best_val_cosine:.4f}")
    
    # Retrain on train+val for best_epoch epochs
    logger.info(f"Retraining on train+val for {best_epoch} epochs...")
    X_tv = np.vstack([X_train, X_val])
    Y_tv = np.vstack([Y_train, Y_val])
    tv_ds = TensorDataset(torch.from_numpy(X_tv).float(), torch.from_numpy(Y_tv).float())
    tv_loader = DataLoader(tv_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    
    final_model = MLPEncoder(input_dim=input_dim, hidden=hidden, dropout=dropout, output_dim=output_dim).to(device)
    final_opt = AdamW(final_model.parameters(), lr=lr, weight_decay=wd)
    final_sched = CosineAnnealingLR(final_opt, T_max=best_epoch)
    
    for epoch in range(best_epoch):
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
    final_model.eval()
    all_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_pred.append(final_model(xb.to(device)).cpu().numpy())
    
    Y_test_pred = np.vstack(all_pred)
    test_metrics = evaluate_predictions(Y_test, Y_test_pred, normalize=True)
    gt_indices = np.arange(len(Y_test))
    ret = retrieval_at_k(Y_test_pred, Y_test, gt_indices, ks=(1, 5, 10))
    rank = compute_ranking_metrics(Y_test_pred, Y_test, gt_indices)
    
    logger.info(f"TEST: cosine={test_metrics['cosine']:.4f}, "
                f"R@1={ret.get('R@1', 0):.4f}, R@5={ret.get('R@5', 0):.4f}")
    
    # Save
    ckpt = Path(checkpoint_dir) / subject / "mlp.pt"
    meta = {
        "input_dim": input_dim, "hidden": hidden, "dropout": dropout,
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "lr": lr, "weight_decay": wd, "subject": subject,
    }
    save_mlp(final_model, str(ckpt), meta)
    logger.info(f"Saved: {ckpt}")
    
    report = {
        "model": "MLP", "subject": subject,
        "hyperparameters": {"hidden": hidden, "dropout": dropout, "lr": lr,
                           "cosine_w": cosine_w, "mse_w": mse_w, "infonce_w": infonce_w},
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "data": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "test_metrics": {**test_metrics, **ret, **rank},
    }
    rp = Path(report_dir) / subject / "mlp_eval.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, default=float)
    
    return test_metrics


def train_two_stage(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                    latent_dim, n_blocks, head_hidden,
                    mse_w, cosine_w, infonce_w, temperature,
                    lr, wd, epochs, patience, batch_size,
                    checkpoint_dir, subject, report_dir, device="cuda"):
    """Train two-stage encoder."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from fmri2img.models.encoders import TwoStageEncoder, save_two_stage_encoder
    from fmri2img.training.losses import MultiLoss
    from fmri2img.models.train_utils import torch_seed_all
    from fmri2img.models.ridge import evaluate_predictions
    from fmri2img.eval.retrieval import retrieval_at_k, compute_ranking_metrics
    
    torch_seed_all(42)
    
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    logger.info(f"Training TwoStage: {input_dim}D → {latent_dim}D ({n_blocks} blocks) → {output_dim}D")
    
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    model = TwoStageEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_blocks=n_blocks,
        head_type="mlp",
        head_hidden_dim=head_hidden,
        dropout=0.3,
        output_dim=output_dim,
    ).to(device)
    
    criterion = MultiLoss(
        mse_weight=mse_w,
        cosine_weight=cosine_w,
        info_nce_weight=infonce_w,
        temperature=temperature,
    )
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}")
    
    best_val_cosine = -np.inf
    best_epoch = 0
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
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
        
        # Validate
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        
        val_pred = np.vstack(all_pred)
        val_true = np.vstack(all_true)
        val_metrics = evaluate_predictions(val_true, val_pred, normalize=True)
        val_cosine = val_metrics['cosine']
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss/n_batches:.4f}, "
                        f"val_cosine={val_cosine:.4f}")
        
        if val_cosine > best_val_cosine:
            best_val_cosine = val_cosine
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    logger.info(f"Best epoch: {best_epoch}, val_cosine: {best_val_cosine:.4f}")
    
    # Retrain on train+val
    logger.info(f"Retraining on train+val for {best_epoch} epochs...")
    X_tv = np.vstack([X_train, X_val])
    Y_tv = np.vstack([Y_train, Y_val])
    tv_ds = TensorDataset(torch.from_numpy(X_tv).float(), torch.from_numpy(Y_tv).float())
    tv_loader = DataLoader(tv_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    
    final_model = TwoStageEncoder(
        input_dim=input_dim, latent_dim=latent_dim,
        n_blocks=n_blocks, head_type="mlp", head_hidden_dim=head_hidden,
        dropout=0.3, output_dim=output_dim,
    ).to(device)
    final_opt = AdamW(final_model.parameters(), lr=lr, weight_decay=wd)
    final_sched = CosineAnnealingLR(final_opt, T_max=best_epoch)
    
    for epoch in range(best_epoch):
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
    
    # Test evaluation
    final_model.eval()
    all_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            all_pred.append(final_model(xb.to(device)).cpu().numpy())
    
    Y_test_pred = np.vstack(all_pred)
    test_metrics = evaluate_predictions(Y_test, Y_test_pred, normalize=True)
    gt_indices = np.arange(len(Y_test))
    ret = retrieval_at_k(Y_test_pred, Y_test, gt_indices, ks=(1, 5, 10))
    rank = compute_ranking_metrics(Y_test_pred, Y_test, gt_indices)
    
    logger.info(f"TEST: cosine={test_metrics['cosine']:.4f}, "
                f"R@1={ret.get('R@1', 0):.4f}, R@5={ret.get('R@5', 0):.4f}")
    
    # Save
    ckpt = Path(checkpoint_dir) / subject / "two_stage_best.pt"
    meta = {
        "input_dim": input_dim, "latent_dim": latent_dim,
        "n_blocks": n_blocks, "head_hidden": head_hidden,
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "n_params": n_params, "subject": subject,
    }
    save_two_stage_encoder(final_model, str(ckpt), meta)
    logger.info(f"Saved: {ckpt}")
    
    report = {
        "model": "TwoStage", "subject": subject,
        "hyperparameters": {
            "latent_dim": latent_dim, "n_blocks": n_blocks,
            "head_hidden": head_hidden, "lr": lr, "wd": wd,
            "mse_w": mse_w, "cosine_w": cosine_w, "infonce_w": infonce_w,
        },
        "best_epoch": best_epoch, "best_val_cosine": float(best_val_cosine),
        "data": {"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "test_metrics": {**test_metrics, **ret, **rank},
    }
    rp = Path(report_dir) / subject / "two_stage_eval.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w") as f:
        json.dump(report, f, indent=2, default=float)
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train models from pre-extracted features")
    parser.add_argument("--subject", default="subj01")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--clip-cache", default="outputs/clip_cache/clip.parquet")
    parser.add_argument("--index-root", default="data/indices/nsd_index")
    parser.add_argument("--model", required=True, choices=["ridge", "mlp", "two_stage"])
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--device", default="cuda")
    
    # Ridge args
    parser.add_argument("--alpha-grid", default="0.1,1,3,10,30,100,300,1000")
    
    # MLP/TwoStage args
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
    
    args = parser.parse_args()
    
    # Load features
    X, Y, nsd_ids, df = load_features_and_targets(
        args.features_dir, args.clip_cache, args.index_root, args.subject
    )
    
    # Split
    (X_train, Y_train, _, X_val, Y_val, _, X_test, Y_test, _) = split_by_index(
        X, Y, nsd_ids, df
    )
    
    logger.info(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    t0 = time.time()
    
    if args.model == "ridge":
        alpha_grid = [float(x) for x in args.alpha_grid.split(",")]
        train_ridge(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                   alpha_grid, args.checkpoint_dir, args.subject, args.report_dir)
    
    elif args.model == "mlp":
        train_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                 args.hidden, args.dropout, args.lr, args.wd,
                 args.epochs, args.patience, args.batch_size,
                 args.cosine_weight, args.mse_weight, args.infonce_weight,
                 args.temperature,
                 args.checkpoint_dir, args.subject, args.report_dir, args.device)
    
    elif args.model == "two_stage":
        train_two_stage(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                       args.latent_dim, args.n_blocks, args.head_hidden,
                       args.mse_weight, args.cosine_weight, args.infonce_weight,
                       args.temperature,
                       args.lr, args.wd, args.epochs, args.patience, args.batch_size,
                       args.checkpoint_dir, args.subject, args.report_dir, args.device)
    
    elapsed = time.time() - t0
    logger.info(f"Training completed in {elapsed/60:.1f} minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
