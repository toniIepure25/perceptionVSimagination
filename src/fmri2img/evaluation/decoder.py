from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from fmri2img.models.ridge import evaluate_predictions


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def collect_predictions(model, loader, device: str = "cpu") -> dict[str, Any]:
    model.eval()
    preds = []
    targets = []
    conditions = []
    pair_ids = []
    nsd_ids = []
    uncertainties = []
    domain_logits = []
    vividness_preds = []
    vividness_targets = []
    confidence_preds = []
    confidence_targets = []
    branch_norms = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to_device(device)
            outputs = model(batch)
            preds.append(outputs.content_pred.cpu().numpy())
            targets.append(batch.targets.clip_target_768.cpu().numpy())
            conditions.append(batch.condition.cpu().numpy())
            pair_ids.append(batch.pair_ids.cpu().numpy())
            nsd_ids.append(batch.nsd_ids.cpu().numpy())
            if outputs.uncertainty is not None:
                uncertainties.append(outputs.uncertainty.cpu().numpy())
            if outputs.domain_logits is not None:
                domain_logits.append(outputs.domain_logits.cpu().numpy())
            if outputs.vividness_pred is not None:
                vividness_preds.append(outputs.vividness_pred.cpu().numpy())
                if batch.targets.vividness is not None:
                    vividness_targets.append(batch.targets.vividness.cpu().numpy())
                else:
                    vividness_targets.append(np.full(len(batch.nsd_ids), np.nan, dtype=np.float32))
            if outputs.confidence_pred is not None:
                confidence_preds.append(outputs.confidence_pred.cpu().numpy())
                if batch.targets.confidence is not None:
                    confidence_targets.append(batch.targets.confidence.cpu().numpy())
                else:
                    confidence_targets.append(np.full(len(batch.nsd_ids), np.nan, dtype=np.float32))
            branch_norms.extend(
                [
                    {
                        "nsd_id": int(nsd_id),
                        "condition": "perception" if int(cond) == 0 else "imagery",
                        **{
                            f"{name}_norm": float(torch.linalg.norm(value[i]).cpu())
                            for name, value in outputs.branch_embeddings.items()
                        },
                    }
                    for i, (nsd_id, cond) in enumerate(zip(batch.nsd_ids, batch.condition))
                ]
            )
    if not preds:
        raise ValueError("Canonical evaluation loader produced no batches.")
    pred = np.vstack(preds)
    target = np.vstack(targets)
    condition = np.concatenate(conditions)
    pair_id = np.concatenate(pair_ids)
    nsd_id = np.concatenate(nsd_ids)
    uncertainty = np.concatenate(uncertainties) if uncertainties else None
    return {
        "pred": pred,
        "target": target,
        "condition": condition,
        "pair_id": pair_id,
        "nsd_id": nsd_id,
        "uncertainty": uncertainty,
        "domain_logits": np.vstack(domain_logits) if domain_logits else None,
        "vividness_pred": np.concatenate(vividness_preds) if vividness_preds else None,
        "vividness_target": np.concatenate(vividness_targets) if vividness_targets else None,
        "confidence_pred": np.concatenate(confidence_preds) if confidence_preds else None,
        "confidence_target": np.concatenate(confidence_targets) if confidence_targets else None,
        "branch_norms": pd.DataFrame(branch_norms),
    }


def compute_decoder_metrics(bundle: dict[str, Any]) -> dict[str, Any]:
    pred = bundle["pred"]
    target = bundle["target"]
    base_metrics = evaluate_predictions(target, pred, normalize=True)
    cosine_scores = np.sum(
        pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        * target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8),
        axis=1,
    )
    df = pd.DataFrame(
        {
            "nsd_id": bundle["nsd_id"],
            "pair_id": bundle["pair_id"],
            "condition": np.where(bundle["condition"] == 0, "perception", "imagery"),
            "cosine": cosine_scores,
        }
    )
    by_condition = (
        df.groupby("condition")["cosine"].agg(["mean", "std", "count"]).reset_index().to_dict(orient="records")
    )
    pair_metrics = compute_pair_metrics(df)
    metrics = {
        "target_space": "vit_l14_image_768",
        "overall": base_metrics,
        "by_condition": by_condition,
        "pair_metrics": pair_metrics,
    }
    if bundle["domain_logits"] is not None:
        domain_pred = np.argmax(bundle["domain_logits"], axis=1)
        metrics["domain_accuracy"] = float(np.mean(domain_pred == bundle["condition"]))
    if bundle["vividness_pred"] is not None and bundle["vividness_target"] is not None:
        vivid_mask = np.isfinite(bundle["vividness_target"])
        if vivid_mask.any():
            metrics["vividness_mse"] = float(np.mean((bundle["vividness_pred"][vivid_mask] - bundle["vividness_target"][vivid_mask]) ** 2))
    if bundle["confidence_pred"] is not None and bundle["confidence_target"] is not None:
        confidence_mask = np.isfinite(bundle["confidence_target"])
        if confidence_mask.any():
            metrics["confidence_mse"] = float(np.mean((bundle["confidence_pred"][confidence_mask] - bundle["confidence_target"][confidence_mask]) ** 2))
    if bundle["uncertainty"] is not None:
        metrics["uncertainty_mean"] = float(np.mean(bundle["uncertainty"]))
    return metrics


def compute_pair_metrics(df: pd.DataFrame) -> dict[str, Any]:
    grouped = df.groupby(["pair_id", "condition"])["cosine"].mean().unstack(fill_value=np.nan)
    grouped = grouped.dropna(subset=["perception", "imagery"], how="any")
    if len(grouped) == 0:
        return {"n_pairs": 0}
    gap = grouped["imagery"] - grouped["perception"]
    return {
        "n_pairs": int(len(grouped)),
        "mean_gap_imagery_minus_perception": float(gap.mean()),
        "median_gap_imagery_minus_perception": float(gap.median()),
    }


def compute_roi_summary(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    branch_norms = bundle["branch_norms"]
    if branch_norms.empty:
        return []
    columns = [col for col in branch_norms.columns if col.endswith("_norm")]
    summary = branch_norms.groupby("condition")[columns].mean().reset_index()
    return summary.to_dict(orient="records")


def write_evaluation_bundle(output_dir: str | Path, metrics: dict[str, Any], roi_summary: list[dict[str, Any]]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    with open(output_dir / "roi_summary.json", "w") as f:
        json.dump(_json_safe(roi_summary), f, indent=2)
