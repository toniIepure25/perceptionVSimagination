from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fmri2img.data.canonical import normalize_decoder_index
from fmri2img.evaluation.decoder import _json_safe, compute_pair_metrics
from fmri2img.models.ridge import RidgeEncoder, evaluate_predictions
from fmri2img.targets import LatentTargetSpec, LatentTargetStore
from fmri2img.workflows.common import load_workflow_config, validate_canonical_workflow_config


def _resolve_mixed_index_path(config: dict[str, Any]) -> Path:
    dataset_cfg = config["dataset"]
    if dataset_cfg.get("mixed_index"):
        return Path(dataset_cfg["mixed_index"])
    if dataset_cfg.get("mixed_output_index"):
        return Path(dataset_cfg["mixed_output_index"])
    raise ValueError(
        "Legacy ridge comparison requires a prepared mixed index. "
        "Set dataset.mixed_index or dataset.mixed_output_index in the canonical config."
    )


def _load_features(df: pd.DataFrame, group_order: list[str]) -> np.ndarray:
    if "roi_values_json" in df.columns and df["roi_values_json"].notna().all():
        rows = [np.asarray(json.loads(raw), dtype=np.float32).reshape(-1) for raw in df["roi_values_json"]]
    elif "roi_features_json" in df.columns and df["roi_features_json"].notna().all():
        rows = []
        for raw in df["roi_features_json"]:
            feature_map = json.loads(raw)
            ordered = group_order or sorted(feature_map)
            pieces = [np.asarray(feature_map[name], dtype=np.float32).reshape(-1) for name in ordered if name in feature_map]
            if not pieces:
                raise ValueError("roi_features_json is present but contains no usable feature vectors.")
            rows.append(np.concatenate(pieces, axis=0))
    else:
        raise ValueError(
            "Legacy ridge comparison requires per-row roi_values_json or roi_features_json in the prepared mixed index."
        )

    dims = {row.shape[0] for row in rows}
    if len(dims) != 1:
        raise ValueError(f"ROI feature dimension mismatch across rows: found dimensions {sorted(dims)}")
    return np.vstack(rows)


def _load_targets(df: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    target_spec = LatentTargetSpec(
        name=config["targets"].get("name", "vit_l14_image_768"),
        dimension=int(config["targets"].get("dimension", 768)),
        embedding_column=config["targets"].get("embedding_column"),
    )
    store = LatentTargetStore(
        cache_path=config["targets"]["cache_path"],
        spec=target_spec,
        id_column=config["targets"].get("id_column"),
    )
    return np.vstack([store.get(int(nsd_id)) for nsd_id in df["nsdId"].tolist()])


def _metrics_by_condition(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.groupby("condition")["cosine"].agg(["mean", "std", "count"]).reset_index().to_dict(orient="records")


def _score_rows(df: pd.DataFrame, predictions: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
    pred_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    target_norm = targets / (np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8)
    cosine = np.sum(pred_norm * target_norm, axis=1)
    return pd.DataFrame(
        {
            "subject": df["subject"].tolist(),
            "nsd_id": df["nsdId"].astype(int).tolist(),
            "pair_id": df["pair_id"].astype(int).tolist(),
            "condition": df["condition"].tolist(),
            "split": df["split"].tolist(),
            "cosine": cosine.tolist(),
        }
    )


def _default_output_dir(config: dict[str, Any]) -> Path:
    experiment_name = str(config.get("experiment", {}).get("name", "canonical_baseline")).strip() or "canonical_baseline"
    if experiment_name.endswith("_bootstrap"):
        experiment_name = experiment_name[: -len("_bootstrap")]
    return Path("outputs/canonical/baselines") / f"{experiment_name}_ridge_legacy"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a simple legacy Ridge baseline on the canonical prepared mixed index."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--alpha-grid", default="0.01,0.1,1.0,10.0,100.0")
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    validate_canonical_workflow_config(config)

    mixed_index_path = _resolve_mixed_index_path(config)
    df = normalize_decoder_index(pd.read_parquet(mixed_index_path))
    split_counts = {name: int((df["split"] == name).sum()) for name in ("train", "val", "test")}
    if any(count <= 0 for count in split_counts.values()):
        raise ValueError(
            "Legacy ridge comparison requires non-empty train/val/test splits in the prepared mixed index. "
            f"Found {split_counts}."
        )

    group_order = list(config.get("roi", {}).get("groups", {}).keys())
    X = _load_features(df, group_order=group_order)
    Y = _load_targets(df, config)

    split_mask = {name: df["split"] == name for name in ("train", "val", "test")}
    X_train, Y_train = X[split_mask["train"]], Y[split_mask["train"]]
    X_val, Y_val = X[split_mask["val"]], Y[split_mask["val"]]
    X_test, Y_test = X[split_mask["test"]], Y[split_mask["test"]]

    alpha_grid = [float(item) for item in args.alpha_grid.split(",") if item.strip()]
    selection_history = []
    best_alpha = None
    best_val_metrics = None
    best_val_score = None

    for alpha in alpha_grid:
        model = RidgeEncoder(alpha=alpha)
        model.fit(X_train, Y_train)
        val_pred = model.predict(X_val)
        val_metrics = evaluate_predictions(Y_val, val_pred, normalize=True)
        selection_history.append({"alpha": alpha, **val_metrics})
        score = float(val_metrics["cosine"])
        if best_val_score is None or score > best_val_score:
            best_alpha = alpha
            best_val_score = score
            best_val_metrics = val_metrics

    if best_alpha is None or best_val_metrics is None:
        raise RuntimeError("Failed to select a ridge alpha from the provided grid.")

    final_model = RidgeEncoder(alpha=best_alpha)
    final_model.fit(np.vstack([X_train, X_val]), np.vstack([Y_train, Y_val]))
    test_pred = final_model.predict(X_test)
    test_metrics = evaluate_predictions(Y_test, test_pred, normalize=True)
    test_scores = _score_rows(df.loc[split_mask["test"]].reset_index(drop=True), test_pred, Y_test)

    metrics = {
        "baseline_name": "legacy_ridge_on_roi_values",
        "config_path": str(Path(args.config)),
        "mixed_index_path": str(mixed_index_path),
        "feature_space": {
            "type": "roi_values_json" if "roi_values_json" in df.columns and df["roi_values_json"].notna().all() else "roi_features_json",
            "dimension": int(X.shape[1]),
        },
        "split_counts": split_counts,
        "alpha_grid": alpha_grid,
        "selected_alpha": float(best_alpha),
        "validation": best_val_metrics,
        "test": {
            "overall": test_metrics,
            "by_condition": _metrics_by_condition(test_scores),
            "pair_metrics": compute_pair_metrics(test_scores),
            "domain_accuracy": None,
        },
        "selection_history": selection_history,
    }

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(output_dir / "ridge.pkl")
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(_json_safe(metrics), handle, indent=2)
    with open(output_dir / "test_scores.json", "w") as handle:
        json.dump(_json_safe(test_scores.to_dict(orient="records")), handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
