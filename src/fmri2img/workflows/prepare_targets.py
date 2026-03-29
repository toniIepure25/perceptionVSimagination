from __future__ import annotations

import argparse
from pathlib import Path

from fmri2img.targets import LatentTargetSpec, build_target_cache_from_index, canonicalize_target_cache
from fmri2img.workflows.common import load_workflow_config, resolve_runtime_device
from fmri2img.workflows.prep_common import get_preparation_section


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the canonical 768-D ViT-L/14 target cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    runtime_device = resolve_runtime_device(config["training"].get("device", "cpu"))
    prep_cfg = get_preparation_section(config, "targets")
    spec = LatentTargetSpec(
        name=config["targets"].get("name", "vit_l14_image_768"),
        dimension=int(config["targets"].get("dimension", 768)),
        embedding_column=config["targets"].get("embedding_column"),
    )
    output_path = Path(config["targets"]["cache_path"])

    if prep_cfg.get("input_cache"):
        summary = canonicalize_target_cache(
            input_path=prep_cfg["input_cache"],
            output_path=output_path,
            spec=spec,
            id_column=config["targets"].get("id_column"),
        )
    else:
        index_path = prep_cfg.get("index_path")
        if index_path is None:
            dataset_cfg = config["dataset"]
            if dataset_cfg.get("mixed_output_index") and Path(dataset_cfg["mixed_output_index"]).exists():
                index_path = dataset_cfg["mixed_output_index"]
            elif dataset_cfg.get("mixed_index") and Path(dataset_cfg["mixed_index"]).exists():
                index_path = dataset_cfg["mixed_index"]
            else:
                index_path = dataset_cfg.get("perception_index")
        if index_path is None:
            raise KeyError(
                "Canonical target preparation requires preparation.targets.input_cache or an index path."
            )
        if output_path.exists() and not args.rebuild:
            summary = canonicalize_target_cache(
                input_path=output_path,
                output_path=output_path,
                spec=spec,
                id_column=config["targets"].get("id_column"),
            )
        else:
            summary = build_target_cache_from_index(
                index_path=index_path,
                output_path=output_path,
                model_id=prep_cfg.get("model_id", "openai/clip-vit-large-patch14"),
                batch_size=int(prep_cfg.get("batch_size", 128)),
                inference_batch_size=int(prep_cfg.get("inference_batch_size", 32)),
                device=resolve_runtime_device(prep_cfg.get("device", runtime_device)),
                limit=prep_cfg.get("limit"),
                resume=bool(prep_cfg.get("resume", True)) and not args.rebuild,
            )

    print(f"Prepared target cache: {output_path}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
