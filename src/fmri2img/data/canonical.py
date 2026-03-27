from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, Dataset

from fmri2img.models.interfaces import DecoderBatch, DecoderTargets
from fmri2img.roi import ROIGroupSpec
from fmri2img.targets import LatentTargetStore

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - exercised in lean runtime environments
    nib = None


def _require_nibabel(context: str) -> None:
    if nib is None:
        raise RuntimeError(
            f"nibabel is required to {context}, but it is not installed in the current environment. "
            "Install nibabel or use materialized numpy/ROI features for canonical workflows."
        )


@dataclass(frozen=True)
class DatasetCapabilities:
    has_pairing: bool
    has_vividness: bool
    has_confidence: bool
    paired_group_count: int = 0


def _normalize_condition(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"perception", "stimulus", "percept"}:
        return "perception"
    if text in {"imagery", "imagination"}:
        return "imagery"
    return text


def _canonicalize_split(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"train", "training"}:
        return "train"
    if text in {"val", "valid", "validation", "dev"}:
        return "val"
    if text in {"test", "testing"}:
        return "test"
    return None


def _infer_pair_id(row: pd.Series) -> int:
    if pd.notna(row.get("pair_id")):
        return int(row["pair_id"])
    if pd.notna(row.get("nsdId")):
        return int(row["nsdId"])
    if pd.notna(row.get("nsd_id")):
        return int(row["nsd_id"])
    return int(row.get("trial_id", row.name))


def _count_paired_groups(df: pd.DataFrame) -> int:
    if "pair_id" not in df.columns or "condition" not in df.columns:
        return 0
    grouped = df.groupby("pair_id")["condition"].agg(lambda values: set(values))
    return int(sum({"perception", "imagery"}.issubset(values) for values in grouped))


def _assign_pair_splits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pair_id" not in out.columns:
        out["pair_id"] = out.apply(_infer_pair_id, axis=1)

    out["split"] = out.get("split", pd.Series([None] * len(out))).map(_canonicalize_split)

    # First propagate any known split labels across all rows sharing a pair id.
    for pair_id, group in out.groupby("pair_id"):
        known_splits = [value for value in group["split"].dropna().unique().tolist() if value is not None]
        if not known_splits:
            continue
        if len(known_splits) > 1:
            raise ValueError(
                f"Canonical pair_id={pair_id} spans multiple splits {known_splits}. "
                "Pairing must not cross train/val/test boundaries."
            )
        out.loc[group.index, "split"] = known_splits[0]

    unresolved_pair_ids = [pair_id for pair_id, group in out.groupby("pair_id") if group["split"].isna().all()]
    if unresolved_pair_ids:
        rng = random.Random(0)
        rng.shuffle(unresolved_pair_ids)
        n = len(unresolved_pair_ids)
        n_train = max(1, int(round(n * 0.8))) if n >= 3 else max(1, n - 2)
        n_val = max(1, int(round(n * 0.1))) if n >= 3 else max(0, n - n_train - 1)
        n_test = n - n_train - n_val
        if n >= 3 and n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val = max(0, n_val - 1)
        split_ids = {
            "train": set(unresolved_pair_ids[:n_train]),
            "val": set(unresolved_pair_ids[n_train:n_train + n_val]),
            "test": set(unresolved_pair_ids[n_train + n_val:]),
        }
        for split_name, ids in split_ids.items():
            if ids:
                out.loc[out["pair_id"].isin(ids), "split"] = split_name

    remaining = out["split"].isna()
    if remaining.any():
        out.loc[remaining, "split"] = "train"

    return out


def normalize_decoder_index(df: pd.DataFrame, default_condition: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if "condition" not in out.columns:
        out["condition"] = default_condition or "perception"
    out["condition"] = out["condition"].map(_normalize_condition)
    invalid_conditions = sorted(set(out["condition"]) - {"perception", "imagery"})
    if invalid_conditions:
        raise ValueError(
            "Canonical decoder indices only support 'perception' and 'imagery' conditions; "
            f"found {invalid_conditions}"
        )
    if "nsd_id" not in out.columns and "nsdId" in out.columns:
        out["nsd_id"] = out["nsdId"]
    if "nsdId" not in out.columns and "nsd_id" in out.columns:
        out["nsdId"] = out["nsd_id"]
    if "nsdId" not in out.columns:
        raise ValueError(
            "Canonical decoder index requires an 'nsdId' or 'nsd_id' column for content targets and pairing."
        )
    if out["nsdId"].isna().any():
        bad_rows = out.index[out["nsdId"].isna()].tolist()[:5]
        raise ValueError(
            f"Canonical decoder index contains missing nsdId values at rows {bad_rows}. "
            "Pairing and target lookup require a valid nsdId for every sample."
        )
    out["nsdId"] = out["nsdId"].astype(int)
    out["nsd_id"] = out["nsdId"]
    if "pair_id" not in out.columns:
        out["pair_id"] = out.apply(_infer_pair_id, axis=1)
    out["pair_id"] = out["pair_id"].astype(int)
    if "subject" not in out.columns:
        out["subject"] = "unknown"
    out = _assign_pair_splits(out)
    return out.reset_index(drop=True)


def build_mixed_condition_index(
    perception_index: str | Path,
    imagery_index: str | Path,
    output_path: str | Path | None = None,
    subject: Optional[str] = None,
) -> pd.DataFrame:
    perception_df = normalize_decoder_index(pd.read_parquet(perception_index), default_condition="perception")
    imagery_df = normalize_decoder_index(pd.read_parquet(imagery_index), default_condition="imagery")
    if subject is not None:
        perception_df = perception_df[perception_df["subject"] == subject]
        imagery_df = imagery_df[imagery_df["subject"] == subject]
    mixed = pd.concat([perception_df, imagery_df], ignore_index=True)
    mixed["pair_id"] = mixed.apply(_infer_pair_id, axis=1)
    # Recompute splits jointly on the combined paired dataset. Per-condition
    # source indices may have inferred train/val/test partitions independently,
    # which can legitimately disagree for the same nsdId/pair_id.
    if "split" in mixed.columns:
        mixed["split"] = None
    mixed = _assign_pair_splits(mixed)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mixed.to_parquet(output_path, index=False)
    return mixed.reset_index(drop=True)


class CanonicalDecoderDataset(Dataset):
    """Canonical mixed-condition dataset used by the shared/private workflows."""

    def __init__(
        self,
        index: str | Path | pd.DataFrame,
        target_store: Optional[LatentTargetStore] = None,
        roi_feature_resolver: Optional[Callable[[np.ndarray, pd.Series], dict[str, np.ndarray]]] = None,
        split: Optional[str] = None,
    ):
        if isinstance(index, (str, Path)):
            self.index_path = Path(index).resolve()
            df = pd.read_parquet(self.index_path)
        else:
            self.index_path = None
            df = index.copy()
        self.df = normalize_decoder_index(df)
        if split is not None:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)
        self.target_store = target_store
        self.roi_feature_resolver = roi_feature_resolver
        self.capabilities = DatasetCapabilities(
            has_pairing=_count_paired_groups(self.df) > 0,
            has_vividness="vividness" in self.df.columns and self.df["vividness"].notna().any(),
            has_confidence="confidence" in self.df.columns and self.df["confidence"].notna().any(),
            paired_group_count=_count_paired_groups(self.df),
        )

    def _resolve_fmri_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path)
        if path.is_absolute() or self.index_path is None:
            return path
        return (self.index_path.parent / path).resolve()

    def __len__(self) -> int:
        return len(self.df)

    def _load_fmri(self, row: pd.Series) -> np.ndarray:
        if pd.notna(row.get("fmri_path")):
            path = self._resolve_fmri_path(row["fmri_path"])
            if path.suffix == ".npy":
                return np.load(path).astype(np.float32)
            if path.suffix == ".gz" or path.suffix == ".nii":
                _require_nibabel(f"load local NIfTI data from {path}")
                img = nib.load(path)
                data = img.get_fdata().astype(np.float32)
                beta_index = int(row.get("beta_index", 0))
                if data.ndim == 4:
                    return data[..., beta_index].astype(np.float32).reshape(-1)
                return data.reshape(-1)
        if pd.notna(row.get("beta_path")):
            from fmri2img.io.s3 import NIfTILoader

            try:
                loader = NIfTILoader()
                img = loader.load(row["beta_path"])
                data = img.get_fdata().astype(np.float32)
                beta_index = int(row.get("beta_index", 0))
                if data.ndim == 4:
                    return data[..., beta_index].reshape(-1)
                return data.reshape(-1)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load canonical fMRI sample from beta_path. "
                    "The current row points to a remote or unavailable NIfTI. "
                    "For canonical local workflows, provide materialized fmri_path arrays or ensure "
                    "S3-backed NSD access is available."
                ) from exc
        raise ValueError("Row does not contain a supported fMRI path.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        fmri = self._load_fmri(row)
        if self.target_store is None:
            raise ValueError("CanonicalDecoderDataset requires a LatentTargetStore.")
        target = self.target_store.get(int(row["nsdId"]))
        roi_features = None
        if self.roi_feature_resolver is not None:
            roi_features = self.roi_feature_resolver(fmri, row)
        if roi_features is None:
            raise ValueError(
                "Canonical decoder sample did not produce ROI features. "
                "Provide roi_features_json/roi_values_json or configure fallback_policy=full_feature_vector."
            )
        sample = {
            "subject": row["subject"],
            "condition": row["condition"],
            "nsd_id": int(row["nsdId"]),
            "pair_id": int(row["pair_id"]),
            "split": row["split"],
            "fmri": fmri.astype(np.float32),
            "clip_target_768": target.astype(np.float32),
            "roi_features": roi_features,
            "vividness": None if not pd.notna(row.get("vividness")) else float(row.get("vividness")),
            "confidence": None if not pd.notna(row.get("confidence")) else float(row.get("confidence")),
            "metadata": {
                "trial_id": row.get("trial_id"),
                "subject": row["subject"],
                "condition": row["condition"],
            },
        }
        return sample


def decoder_collate_fn(samples: Sequence[dict[str, Any]]) -> DecoderBatch:
    fmri = torch.from_numpy(np.stack([sample["fmri"] for sample in samples])).float()
    roi_features = {
        key: torch.from_numpy(np.stack([sample["roi_features"][key] for sample in samples])).float()
        for key in samples[0]["roi_features"]
    }
    condition = torch.tensor([0 if sample["condition"] == "perception" else 1 for sample in samples], dtype=torch.long)
    nsd_ids = torch.tensor([sample["nsd_id"] for sample in samples], dtype=torch.long)
    pair_ids = torch.tensor([sample["pair_id"] for sample in samples], dtype=torch.long)
    domain = condition.clone()
    vividness_values = [sample["vividness"] for sample in samples]
    confidence_values = [sample["confidence"] for sample in samples]
    vividness = None
    if any(value is not None for value in vividness_values):
        vividness = torch.tensor(
            [float("nan") if value is None else float(value) for value in vividness_values],
            dtype=torch.float32,
        )
    confidence = None
    if any(value is not None for value in confidence_values):
        confidence = torch.tensor(
            [float("nan") if value is None else float(value) for value in confidence_values],
            dtype=torch.float32,
        )
    targets = DecoderTargets(
        clip_target_768=torch.from_numpy(np.stack([sample["clip_target_768"] for sample in samples])).float(),
        domain_label=domain,
        vividness=vividness,
        confidence=confidence,
    )
    return DecoderBatch(
        fmri=fmri,
        roi_features=roi_features,
        condition=condition,
        nsd_ids=nsd_ids,
        pair_ids=pair_ids,
        targets=targets,
        metadata=[sample["metadata"] for sample in samples],
    )


class PairedConditionBatchSampler(BatchSampler):
    """
    Batch sampler that preferentially emits paired perception/imagery samples.

    This keeps the MVP paired-loss path active without requiring a separate
    pair-only dataset format.
    """

    def __init__(self, dataset: CanonicalDecoderDataset, batch_size: int, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        groups: dict[int, list[int]] = {}
        for idx, row in dataset.df.iterrows():
            groups.setdefault(int(row["pair_id"]), []).append(idx)
        self.groups = []
        for indices in groups.values():
            conditions = {_normalize_condition(dataset.df.iloc[idx]["condition"]) for idx in indices}
            if "perception" in conditions and "imagery" in conditions:
                self.groups.append(indices)
        if not self.groups:
            raise ValueError(
                "PairedConditionBatchSampler requires at least one pair_id containing both perception and imagery samples."
            )

    def __iter__(self):
        rng = random.Random(self.seed)
        groups = self.groups.copy()
        rng.shuffle(groups)
        batch: list[int] = []
        for group in groups:
            perception = [idx for idx in group if self.dataset.df.iloc[idx]["condition"] == "perception"]
            imagery = [idx for idx in group if self.dataset.df.iloc[idx]["condition"] == "imagery"]
            if not perception or not imagery:
                continue
            batch.extend([rng.choice(perception), rng.choice(imagery)])
            if len(batch) >= self.batch_size:
                yield batch[: self.batch_size]
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        return len(self.groups)
