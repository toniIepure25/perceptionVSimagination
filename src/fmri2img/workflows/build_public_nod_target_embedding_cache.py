from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fmri2img.data.build_target_clip_cache_robust import compute_embeddings_batch, load_clip_encoder
from fmri2img.workflows._venv_guard import ensure_project_venv
from fmri2img.workflows.prepare_public_nod_target_embedding_cache import (
    DEFAULT_OUTPUT as DEFAULT_MANIFEST,
    EMBEDDING_COLUMN,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_ID,
    EXPECTED_SELECTION_ROWS,
    EXPECTED_UNIQUE_TARGETS,
)


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / DEFAULT_MANIFEST


def _default_output_path() -> Path:
    return Path(__file__).resolve().parents[3] / "cache/indices/public_nod/imagenet_run10_target_embedding_cache.parquet"


def _load_images_from_manifest(manifest: pd.DataFrame, repo_root: Path) -> dict[int, Image.Image]:
    images: dict[int, Image.Image] = {}
    for pair_id, relpath in zip(manifest["pair_id"].tolist(), manifest["stimulus_path"].tolist()):
        image_path = repo_root / relpath
        with Image.open(image_path) as image:
            images[int(pair_id)] = image.convert("RGB")
    return images


def build_public_nod_target_embedding_cache(
    manifest_path: Path,
    *,
    model_id: str = EMBEDDING_MODEL_ID,
    device: str | None = None,
    inference_batch_size: int = 32,
) -> tuple[pd.DataFrame, dict]:
    manifest_path = manifest_path.resolve()
    manifest = pd.read_parquet(manifest_path).sort_values(["subject", "session", "run", "trial_index"]).reset_index(drop=True)
    if len(manifest) != EXPECTED_SELECTION_ROWS:
        raise ValueError(
            f"NOD target-embedding cache build requires the fixed {EXPECTED_SELECTION_ROWS}-row manifest, "
            f"but {manifest_path} exposes {len(manifest)} rows."
        )
    if int(manifest["target_identifier"].nunique()) != EXPECTED_UNIQUE_TARGETS:
        raise ValueError(
            "NOD target-embedding cache build requires one unique target identifier per manifest row in the fixed slice."
        )
    if not bool(manifest["stimulus_payload_resolved"].all()):
        unresolved = int((~manifest["stimulus_payload_resolved"]).sum())
        raise ValueError(
            f"NOD target-embedding cache build requires all fixed-slice stimulus JPEGs to be resolved, "
            f"but {manifest_path} still has {unresolved} unresolved payloads."
        )

    repo_root = manifest_path.parents[3]
    manifest = manifest.copy()
    manifest["pair_id"] = np.arange(1, len(manifest) + 1, dtype=np.int64)

    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, processor, target_dim = load_clip_encoder(model_id, runtime_device)
    if int(target_dim) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Canonical NOD target-embedding cache requires {EMBEDDING_DIMENSION}-D embeddings, but {model_id} yields {target_dim}."
        )
    images = _load_images_from_manifest(manifest, repo_root)
    embeddings = compute_embeddings_batch(images, clip_model, processor, runtime_device, batch_size=inference_batch_size)

    rows = []
    for row in manifest.to_dict(orient="records"):
        pair_id = int(row["pair_id"])
        vector = np.asarray(embeddings[pair_id], dtype=np.float32).reshape(-1)
        if vector.shape[0] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Canonical NOD target-embedding cache expected {EMBEDDING_DIMENSION}-D vectors, got {vector.shape[0]} "
                f"for pair_id={pair_id}."
            )
        rows.append(
            {
                "pair_id": pair_id,
                "target_identifier": row["target_identifier"],
                "stimulus_path": row["stimulus_path"],
                "embedding_model_id": model_id,
                "embedding_dimension": EMBEDDING_DIMENSION,
                EMBEDDING_COLUMN: vector.tolist(),
            }
        )

    cache = pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)
    report = {
        "source_manifest": str(manifest_path),
        "target_selection_rows": int(len(manifest)),
        "unique_target_identifiers": int(manifest["target_identifier"].nunique()),
        "embeddings_materialized": int(len(cache)),
        "embedding_model_id": model_id,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "embedding_column": EMBEDDING_COLUMN,
        "id_column": "pair_id",
        "state": {
            "target_embedding_ready": True,
            "downstream_prep_ready": True,
            "training_ready": False,
        },
        "still_missing_before_training": [
            "ROI materialization contract aligned to the NOD derivatives",
            "shared-only training/eval config that points to the NOD adapter, target-selection artifact, and target cache",
            "dataset-side join contract from the fixed NOD slice into the canonical shared-only trainer",
        ],
    }
    return cache, report


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.build_public_nod_target_embedding_cache")
    parser = argparse.ArgumentParser(
        description="Build the real canonical 768-D target-embedding cache for the fixed NOD target-selection slice."
    )
    parser.add_argument("--manifest", type=Path, default=_default_manifest_path())
    parser.add_argument("--output", type=Path, default=_default_output_path())
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--model-id", default=EMBEDDING_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    report_path = args.report.resolve() if args.report is not None else output_path.with_suffix(".report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache, report = build_public_nod_target_embedding_cache(
        manifest_path,
        model_id=args.model_id,
        device=args.device,
        inference_batch_size=int(args.inference_batch_size),
    )
    cache.to_parquet(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Prepared NOD target-embedding cache: {output_path}")
    print(f"Rows: {len(cache)}")
    print(f"Unique target identifiers: {report['unique_target_identifiers']}")
    print(f"Embeddings materialized: {report['embeddings_materialized']}")
    print(f"Target-embedding ready: {report['state']['target_embedding_ready']}")
    print(f"Downstream prep ready: {report['state']['downstream_prep_ready']}")
    print(f"Training ready: {report['state']['training_ready']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
