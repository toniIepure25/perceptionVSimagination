from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fmri2img.data.canonical import (
    CanonicalDecoderDataset,
    PairedConditionBatchSampler,
    build_mixed_condition_index,
    decoder_collate_fn,
)
from fmri2img.models.canonical_decoder import SharedPrivateMultitaskDecoder
from fmri2img.models.interfaces import DecoderConfig
from fmri2img.preprocessing import describe_preprocessing
from fmri2img.roi import (
    DEFAULT_ROI_GROUPS,
    ROIGroupResolver,
    ROIGroupSpec,
    project_group_features,
    summarize_roi_groups,
)
from fmri2img.targets import LatentTargetSpec, LatentTargetStore
from fmri2img.utils.config_loader import ConfigDict, load_config

logger = logging.getLogger(__name__)


def load_workflow_config(config_path: str, overrides: list[str] | None = None) -> ConfigDict:
    override_dict: dict[str, Any] = {}
    for override in overrides or []:
        key, value = override.split("=", 1)
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        override_dict[key] = parsed_value
    return load_config(config_path, overrides=override_dict)


def resolve_runtime_device(requested_device: str | None) -> str:
    requested = str(requested_device or "cpu").strip().lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        logger.warning("Requested runtime device '%s' is unavailable; falling back to cpu.", requested)
        return "cpu"
    if requested == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        logger.warning("Requested runtime device 'mps' is unavailable; falling back to cpu.")
        return "cpu"
    return requested


def validate_canonical_workflow_config(config: ConfigDict) -> None:
    required_sections = ("dataset", "roi", "targets", "model", "training", "evaluation", "analysis", "export")
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise KeyError(f"Canonical workflow config is missing sections: {missing_sections}")

    target_dim = int(config["targets"].get("dimension", 768))
    if target_dim != 768:
        raise ValueError(
            f"Canonical shared/private workflows require 768-D ViT-L/14 targets, got dimension={target_dim}."
        )

    dataset_cfg = config["dataset"]
    input_paths = []
    if dataset_cfg.get("mixed_index"):
        input_paths.append(("dataset.mixed_index", Path(dataset_cfg["mixed_index"])))
    else:
        for key in ("perception_index", "imagery_index"):
            if not dataset_cfg.get(key):
                raise KeyError(
                    f"Canonical workflow config requires dataset.{key} when dataset.mixed_index is not provided."
                )
            input_paths.append((f"dataset.{key}", Path(dataset_cfg[key])))

    input_paths.append(("targets.cache_path", Path(config["targets"]["cache_path"])))

    missing_paths = [f"{name}={path}" for name, path in input_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Canonical workflow prerequisites are missing:\n- "
            + "\n- ".join(missing_paths)
            + "\nUse the checked-in smoke fixture or materialize the required indices/target cache first."
        )

    roi_cfg = config["roi"]
    if not roi_cfg.get("groups"):
        raise ValueError("Canonical ROI config must define roi.groups.")


def _create_roi_resolver(config: ConfigDict):
    roi_config = config.get("roi", {})
    group_spec = ROIGroupSpec(
        groups=roi_config.get("groups", DEFAULT_ROI_GROUPS),
        missing_policy=roi_config.get("missing_policy", "error"),
        fallback_policy=roi_config.get("fallback_policy", "error"),
    )

    config_roi_names = roi_config.get("roi_names", [])

    def _row_roi_names(row: pd.Series) -> list[str]:
        if "roi_names_json" in row and pd.notna(row.get("roi_names_json")):
            raw = json.loads(row["roi_names_json"])
            if not isinstance(raw, list):
                raise ValueError("roi_names_json must decode to a list of ROI names.")
            return [str(value) for value in raw]
        return list(config_roi_names)

    def resolve_fn(fmri, row):
        if "roi_features_json" in row and pd.notna(row.get("roi_features_json")):
            raw = json.loads(row["roi_features_json"])
            return {key: pd.Series(value, dtype="float32").to_numpy() for key, value in raw.items()}
        if "roi_values_json" in row and pd.notna(row.get("roi_values_json")):
            roi_values = pd.Series(json.loads(row["roi_values_json"])).astype("float32").to_numpy()
            roi_names = _row_roi_names(row)
            return project_group_features(
                roi_values=roi_values,
                roi_names=roi_names,
                spec=group_spec,
                fallback_vector=fmri,
            )
        if group_spec.fallback_policy == "full_feature_vector":
            return {name: fmri.astype("float32") for name in group_spec.groups}
        raise ValueError(
            "Canonical ROI grouping requires either roi_features_json/roi_values_json "
            "or fallback_policy=full_feature_vector for smoke tests."
        )

    resolved = {}
    if config_roi_names:
        resolved = summarize_roi_groups(ROIGroupResolver(group_spec).resolve(config_roi_names))
    return resolve_fn, group_spec, resolved


def _infer_roi_input_dims(dataset: CanonicalDecoderDataset, config: ConfigDict, roi_group_spec: ROIGroupSpec) -> dict[str, int]:
    if len(dataset) == 0:
        raise ValueError("Canonical dataset split is empty; cannot infer ROI branch dimensions.")

    row = dataset.df.iloc[0]
    if "roi_features_json" in row and pd.notna(row.get("roi_features_json")):
        raw = json.loads(row["roi_features_json"])
        return {name: len(values) for name, values in raw.items()}

    if "roi_values_json" in row and pd.notna(row.get("roi_values_json")):
        if "roi_names_json" in row and pd.notna(row.get("roi_names_json")):
            roi_names = json.loads(row["roi_names_json"])
        else:
            roi_names = config.get("roi", {}).get("roi_names", [])
        if not roi_names:
            raise ValueError(
                "roi_values_json is present but no ROI names are available. "
                "Provide per-row roi_names_json or configure roi.roi_names."
            )
        resolved = ROIGroupResolver(roi_group_spec).resolve(roi_names)
        return {name: group.input_dim for name, group in resolved.items()}

    if roi_group_spec.fallback_policy == "full_feature_vector":
        fmri_dim = config.get("dataset.fmri_dim", None)
        if fmri_dim is not None:
            return {name: int(fmri_dim) for name in roi_group_spec.groups}

    # Fall back to a real sample only when no schema-level route exists.
    sample = dataset[0]
    return {name: len(value) for name, value in sample["roi_features"].items()}


def build_datasets(config: ConfigDict):
    validate_canonical_workflow_config(config)
    dataset_cfg = config["dataset"]
    dataset_source: Path | pd.DataFrame
    if dataset_cfg.get("mixed_index"):
        mixed_index = Path(dataset_cfg["mixed_index"])
        dataset_source = mixed_index
    else:
        prepared_mixed = dataset_cfg.get("mixed_output_index")
        if prepared_mixed and Path(prepared_mixed).exists():
            dataset_source = Path(prepared_mixed)
        else:
            df = build_mixed_condition_index(
                perception_index=dataset_cfg["perception_index"],
                imagery_index=dataset_cfg["imagery_index"],
                output_path=dataset_cfg.get("mixed_output_index"),
                subject=dataset_cfg.get("subject"),
                perception_conditions=dataset_cfg.get("perception_conditions", ["perception"]),
                imagery_conditions=dataset_cfg.get("imagery_conditions", ["imagery"]),
            )
            dataset_source = Path(prepared_mixed) if prepared_mixed else df
    target_spec = LatentTargetSpec(
        name=config["targets"].get("name", "vit_l14_image_768"),
        dimension=int(config["targets"].get("dimension", 768)),
        embedding_column=config["targets"].get("embedding_column"),
    )
    target_store = LatentTargetStore(
        cache_path=config["targets"]["cache_path"],
        spec=target_spec,
        id_column=config["targets"].get("id_column"),
    )
    roi_resolver, roi_group_spec, roi_summary = _create_roi_resolver(config)
    train_ds = CanonicalDecoderDataset(dataset_source, target_store=target_store, roi_feature_resolver=roi_resolver, split="train")
    val_ds = CanonicalDecoderDataset(dataset_source, target_store=target_store, roi_feature_resolver=roi_resolver, split="val")
    test_ds = CanonicalDecoderDataset(dataset_source, target_store=target_store, roi_feature_resolver=roi_resolver, split="test")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError(
            "Canonical dataset splits are incomplete. "
            f"Found train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}."
        )
    return train_ds, val_ds, test_ds, roi_group_spec, roi_summary, target_store.describe()


def build_loaders(config: ConfigDict, train_ds, val_ds, test_ds):
    batch_size = int(config["training"].get("batch_size", 8))
    if train_ds.capabilities.has_pairing:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=PairedConditionBatchSampler(train_ds, batch_size=batch_size, seed=int(config["training"].get("seed", 0))),
            collate_fn=decoder_collate_fn,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=decoder_collate_fn)
    eval_batch_size = int(config["evaluation"].get("batch_size", batch_size))
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=decoder_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=decoder_collate_fn)
    return train_loader, val_loader, test_loader


def instantiate_model_from_dataset(config: ConfigDict, dataset: CanonicalDecoderDataset):
    roi_group_spec = ROIGroupSpec(
        groups=config["roi"].get("groups", DEFAULT_ROI_GROUPS),
        missing_policy=config["roi"].get("missing_policy", "error"),
        fallback_policy=config["roi"].get("fallback_policy", "error"),
    )
    roi_input_dims = _infer_roi_input_dims(dataset, config, roi_group_spec)
    model_cfg = config["model"]
    disentanglement_mode = model_cfg.get("disentanglement_mode", "shared_private")
    use_vividness_head = bool(model_cfg.get("use_vividness_head", True))
    if use_vividness_head and not (dataset.capabilities.has_vividness or dataset.capabilities.has_confidence):
        logger.warning(
            "Canonical vividness/confidence head requested, but the dataset has no vividness or confidence labels. "
            "Disabling the head for this run."
        )
        use_vividness_head = False
    use_domain_head = bool(model_cfg.get("use_domain_head", True))
    if disentanglement_mode == "shared_only" and use_domain_head:
        logger.warning(
            "Canonical shared-only ablation requested. Disabling the domain head because private latents are inactive."
        )
        use_domain_head = False
    decoder_config = DecoderConfig(
        target_dim=int(config["targets"].get("dimension", 768)),
        branch_embedding_dim=int(model_cfg.get("branch_embedding_dim", 128)),
        shared_dim=int(model_cfg.get("shared_dim", 128)),
        private_dim=int(model_cfg.get("private_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        disentanglement_mode=disentanglement_mode,
        use_domain_head=use_domain_head,
        use_vividness_head=use_vividness_head,
        vividness_mode=model_cfg.get("vividness_mode", "evidential"),
    )
    return SharedPrivateMultitaskDecoder(roi_input_dims=roi_input_dims, config=decoder_config)


def checkpoint_artifact_spec(
    config: ConfigDict,
    checkpoint_path: str,
    target_spec: dict,
    roi_summary: dict,
    *,
    effective_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    effective_config = effective_config or {}
    effective_model_cfg = effective_config.get("model", config["model"])
    experiment_cfg = config.get("experiment", {})
    animus_cfg = config.get("animus", {})
    domain_enabled = bool(effective_model_cfg.get("use_domain_head", True))
    vividness_enabled = bool(effective_model_cfg.get("use_vividness_head", True))
    source_interface_status = animus_cfg.get(
        "source_interface_status",
        "active" if domain_enabled else "scaffolded",
    )
    confidence_interface_status = animus_cfg.get(
        "confidence_interface_status",
        "active" if vividness_enabled else "scaffolded",
    )
    return {
        "artifact_version": "1.0",
        "target_spec": target_spec,
        "preprocessing_spec": describe_preprocessing(None),
        "roi_spec": {
            "groups": config["roi"].get("groups", {}),
            "resolved": roi_summary,
        },
        "checkpoint_path": str(checkpoint_path),
        "metadata": {
            "project": "fmri2img",
            "workflow": "shared_private_decoder",
            "compatibility_version": "animus-decoder-v1",
            "experiment": {
                "name": experiment_cfg.get("name"),
                "description": experiment_cfg.get("description"),
                "benchmark_role": experiment_cfg.get("benchmark_role"),
                "evidence_tier": experiment_cfg.get("evidence_tier"),
            },
            "animus": {
                "subproject": animus_cfg.get("subproject"),
                "decoder_role": animus_cfg.get("decoder_role"),
                "stability_tier": animus_cfg.get("stability_tier"),
                "intended_use": animus_cfg.get("intended_use"),
                "interfaces": {
                    "content": {"enabled": True, "status": "active"},
                    "source": {"enabled": domain_enabled, "status": source_interface_status},
                    "confidence": {"enabled": vividness_enabled, "status": confidence_interface_status},
                },
            },
            "dataset_capabilities": effective_config.get("dataset_capabilities", {}),
            "heads": {
                "content": {"target_dim": int(config["targets"].get("dimension", 768))},
                "disentanglement": {"mode": effective_model_cfg.get("disentanglement_mode", "shared_private")},
                "domain": {"enabled": domain_enabled},
                "vividness": {"enabled": vividness_enabled},
            },
        },
    }
