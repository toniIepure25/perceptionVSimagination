from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from fmri2img.io.nsd_images import load_nsd_images
from fmri2img.io.s3 import get_s3_filesystem

from .specs import LatentTargetSpec, LatentTargetStore

logger = logging.getLogger(__name__)


def _resolve_id_column(df: pd.DataFrame, id_column: Optional[str] = None) -> str:
    if id_column is not None:
        if id_column not in df.columns:
            raise KeyError(f"Configured id column '{id_column}' was not found in the target cache input.")
        return id_column
    for candidate in ("nsdId", "nsd_id", "pair_id"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Could not infer an identifier column from the target cache input.")


def _resolve_embedding_column(df: pd.DataFrame, spec: LatentTargetSpec) -> str:
    for candidate in spec.column_candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Could not find a supported embedding column in the target cache input. "
        f"Tried: {', '.join(spec.column_candidates)}"
    )


def canonicalize_target_cache(
    *,
    input_path: str | Path,
    output_path: str | Path,
    spec: LatentTargetSpec | None = None,
    id_column: Optional[str] = None,
) -> dict[str, Any]:
    spec = spec or LatentTargetSpec()
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Target cache input not found: {input_path}")

    df = pd.read_parquet(input_path)
    resolved_id = _resolve_id_column(df, id_column=id_column)
    resolved_embedding = _resolve_embedding_column(df, spec)
    if df[resolved_id].isna().any():
        raise ValueError(f"Target cache input {input_path} contains missing {resolved_id} values.")
    if df[resolved_id].duplicated().any():
        duplicates = df.loc[df[resolved_id].duplicated(), resolved_id].astype(int).tolist()[:10]
        raise ValueError(f"Target cache input {input_path} contains duplicate ids: {duplicates}")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        vector = np.asarray(row[resolved_embedding], dtype=np.float32).reshape(-1)
        if vector.shape[0] != spec.dimension:
            raise ValueError(
                f"Target cache input {input_path} is not canonical 768-D ViT-L/14 data. "
                f"Found vector dim {vector.shape[0]} for id {int(row[resolved_id])}."
            )
        if spec.normalized:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        rows.append({"nsdId": int(row[resolved_id]), "clip_target_768": vector.tolist()})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return LatentTargetStore(output_path, spec=spec, id_column="nsdId").describe()


def build_target_cache_from_index(
    *,
    index_path: str | Path,
    output_path: str | Path,
    model_id: str = "openai/clip-vit-large-patch14",
    batch_size: int = 128,
    inference_batch_size: int = 32,
    device: str = "cpu",
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    from fmri2img.data.build_target_clip_cache_robust import compute_embeddings_batch, load_clip_encoder

    spec = LatentTargetSpec()
    index_path = Path(index_path)
    output_path = Path(output_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index for target preparation not found: {index_path}")

    df = pd.read_parquet(index_path)
    if "nsdId" not in df.columns and "nsd_id" in df.columns:
        df["nsdId"] = df["nsd_id"]
    if "nsdId" not in df.columns:
        raise ValueError(f"Index {index_path} does not contain nsdId for target preparation.")

    nsd_ids = [int(value) for value in pd.Series(df["nsdId"]).dropna().drop_duplicates().tolist()]
    if limit is not None:
        nsd_ids = nsd_ids[:limit]

    existing: dict[int, np.ndarray] = {}
    if resume and output_path.exists():
        store = LatentTargetStore(output_path, spec=spec, id_column="nsdId")
        store.load()
        for nsd_id in store._df.index.tolist():  # noqa: SLF001 - local helper to resume quickly
            existing[int(nsd_id)] = store.get(int(nsd_id))

    pending = [nsd_id for nsd_id in nsd_ids if nsd_id not in existing]
    if not pending:
        return LatentTargetStore(output_path, spec=spec, id_column="nsdId").describe()

    clip_model, processor, target_dim = load_clip_encoder(model_id, device)
    if int(target_dim) != spec.dimension:
        raise ValueError(
            f"Canonical target preparation requires 768-D ViT-L/14 embeddings, but model '{model_id}' "
            f"yields {target_dim} dimensions."
        )
    s3_fs = get_s3_filesystem()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_embeddings = existing.copy()
    for start in range(0, len(pending), batch_size):
        batch_ids = pending[start : start + batch_size]
        images = load_nsd_images(batch_ids, s3_fs=s3_fs, prefer="hdf5")
        if not images:
            logger.warning("No NSD images were loaded for target batch starting at index %s", start)
            continue
        embeddings = compute_embeddings_batch(images, clip_model, processor, device, inference_batch_size)
        all_embeddings.update({int(key): value.astype(np.float32) for key, value in embeddings.items()})
        pd.DataFrame(
            [{"nsdId": int(nsd_id), "clip_target_768": emb.tolist()} for nsd_id, emb in sorted(all_embeddings.items())]
        ).to_parquet(output_path, index=False)

    if not output_path.exists():
        raise RuntimeError(
            f"Target preparation did not produce {output_path}. "
            "Check CLIP model availability and NSD stimulus access."
        )
    return LatentTargetStore(output_path, spec=spec, id_column="nsdId").describe()

