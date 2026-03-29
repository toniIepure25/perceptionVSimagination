from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from fmri2img.models.interfaces import DecoderBatch, DecoderOutputs
from fmri2img.models.latent import orthogonality_penalty

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalLossWeights:
    content_cosine: float = 0.4
    content_mse: float = 0.3
    content_infonce: float = 0.3
    paired_shared: float = 0.2
    orthogonality: float = 0.05
    domain: float = 0.1
    vividness: float = 0.1
    confidence: float = 0.05
    reconstruction: float = 0.1
    temperature: float = 0.07


def _info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if pred.size(0) < 2:
        return pred.new_tensor(0.0)
    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target, dim=-1)
    logits = pred_norm @ target_norm.T / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _paired_shared_loss(z_shared: torch.Tensor, pair_ids: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    total = z_shared.new_tensor(0.0)
    count = 0
    for pair_id in pair_ids.unique():
        mask = pair_ids == pair_id
        if mask.sum() < 2:
            continue
        pair_shared = z_shared[mask]
        pair_conditions = condition[mask]
        perception = pair_shared[pair_conditions == 0]
        imagery = pair_shared[pair_conditions == 1]
        if len(perception) == 0 or len(imagery) == 0:
            continue
        perception_mean = perception.mean(dim=0, keepdim=True)
        imagery_mean = imagery.mean(dim=0, keepdim=True)
        total = total + (1.0 - F.cosine_similarity(perception_mean, imagery_mean, dim=-1)).mean()
        count += 1
    if count == 0:
        return total
    return total / count


def compute_canonical_loss(
    batch: DecoderBatch,
    outputs: DecoderOutputs,
    weights: CanonicalLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_cosine = (1.0 - F.cosine_similarity(outputs.content_pred, batch.targets.clip_target_768, dim=-1)).mean()
    loss_mse = F.mse_loss(outputs.content_pred, batch.targets.clip_target_768)
    loss_infonce = _info_nce(outputs.content_pred, batch.targets.clip_target_768, temperature=weights.temperature)
    loss_paired = _paired_shared_loss(outputs.z_shared, batch.pair_ids, batch.condition)
    loss_ortho = orthogonality_penalty(
        outputs.z_shared,
        outputs.z_perception_private,
        outputs.z_imagery_private,
    )
    total = (
        weights.content_cosine * loss_cosine
        + weights.content_mse * loss_mse
        + weights.content_infonce * loss_infonce
        + weights.paired_shared * loss_paired
        + weights.orthogonality * loss_ortho
    )
    loss_domain = outputs.content_pred.new_tensor(0.0)
    if outputs.domain_logits is not None and batch.targets.domain_label is not None:
        loss_domain = F.cross_entropy(outputs.domain_logits, batch.targets.domain_label)
        total = total + weights.domain * loss_domain
    loss_vividness = outputs.content_pred.new_tensor(0.0)
    if outputs.vividness_pred is not None and batch.targets.vividness is not None:
        vivid_mask = torch.isfinite(batch.targets.vividness)
        if vivid_mask.any():
            loss_vividness = F.mse_loss(outputs.vividness_pred[vivid_mask], batch.targets.vividness[vivid_mask])
            total = total + weights.vividness * loss_vividness
    loss_confidence = outputs.content_pred.new_tensor(0.0)
    if outputs.confidence_pred is not None and batch.targets.confidence is not None:
        confidence_mask = torch.isfinite(batch.targets.confidence)
        if confidence_mask.any():
            loss_confidence = F.mse_loss(outputs.confidence_pred[confidence_mask], batch.targets.confidence[confidence_mask])
            total = total + weights.confidence * loss_confidence
    loss_reconstruction = outputs.content_pred.new_tensor(0.0)
    if outputs.reconstructed_visual is not None and outputs.visual_target is not None:
        loss_reconstruction = F.mse_loss(outputs.reconstructed_visual, outputs.visual_target)
        total = total + weights.reconstruction * loss_reconstruction
    metrics = {
        "loss": float(total.detach().cpu()),
        "content_cosine": float(loss_cosine.detach().cpu()),
        "content_mse": float(loss_mse.detach().cpu()),
        "content_infonce": float(loss_infonce.detach().cpu()),
        "paired_shared": float(loss_paired.detach().cpu()),
        "orthogonality": float(loss_ortho.detach().cpu()),
        "domain": float(loss_domain.detach().cpu()),
        "vividness": float(loss_vividness.detach().cpu()),
        "confidence": float(loss_confidence.detach().cpu()),
        "reconstruction": float(loss_reconstruction.detach().cpu()),
    }
    return total, metrics


class SharedPrivateTrainer:
    """Minimal research-grade trainer for the canonical shared/private model."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_weights: CanonicalLossWeights | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_weights = loss_weights or CanonicalLossWeights()
        self.device = device
        self.model.to(device)

    def run_epoch(self, loader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            batch = batch.to_device(self.device)
            outputs = self.model(batch)
            loss, metrics = compute_canonical_loss(batch, outputs, self.loss_weights)
            if training:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1
        if n_batches == 0:
            return {"loss": float("nan")}
        return {key: value / n_batches for key, value in totals.items()}

    def fit(self, train_loader, val_loader, epochs: int, output_dir: str | Path, config_snapshot: dict) -> dict[str, float]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best_metric = float("-inf")
        best_summary: dict[str, float] = {}
        history = []
        for epoch in range(1, epochs + 1):
            train_metrics = self.run_epoch(train_loader, training=True)
            val_metrics = self.run_epoch(val_loader, training=False)
            val_content_cosine = 1.0 - val_metrics.get("content_cosine", float("nan"))
            summary = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss", float("nan")),
                "val_loss": val_metrics.get("loss", float("nan")),
                "val_content_cosine": val_content_cosine,
            }
            history.append(summary)
            logger.info(
                "Epoch %s/%s train_loss=%.4f val_loss=%.4f",
                epoch,
                epochs,
                summary["train_loss"],
                summary["val_loss"],
            )
            if summary["val_content_cosine"] > best_metric:
                best_metric = summary["val_content_cosine"]
                best_summary = summary
                checkpoint_path = output_dir / "best_decoder.pt"
                torch.save(
                    {
                        "state_dict": self.model.state_dict(),
                        "config": config_snapshot,
                        "best_summary": best_summary,
                    },
                    checkpoint_path,
                )
        with open(output_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)
        return best_summary


def load_canonical_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: str = "cpu",
    device: str | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    checkpoint_config = checkpoint.get("config", {}).get("model", {})
    if hasattr(model, "config"):
        comparisons = {
            "target_dim": getattr(model.config, "target_dim", None),
            "branch_embedding_dim": getattr(model.config, "branch_embedding_dim", None),
            "shared_dim": getattr(model.config, "shared_dim", None),
            "private_dim": getattr(model.config, "private_dim", None),
            "use_domain_head": getattr(model.config, "use_domain_head", None),
            "use_vividness_head": getattr(model.config, "use_vividness_head", None),
        }
        mismatches = []
        for key, model_value in comparisons.items():
            if key in checkpoint_config and checkpoint_config[key] != model_value:
                mismatches.append(f"{key}: checkpoint={checkpoint_config[key]} model={model_value}")
        if mismatches:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is incompatible with the instantiated canonical model: "
                + "; ".join(mismatches)
            )
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model.to(device)
    return checkpoint


def inspect_canonical_checkpoint(checkpoint_path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(checkpoint_path, map_location=map_location)
