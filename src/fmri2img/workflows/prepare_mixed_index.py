from __future__ import annotations

import argparse
from pathlib import Path

from fmri2img.data.canonical import build_mixed_condition_index
from fmri2img.workflows.common import load_workflow_config
from fmri2img.workflows.prep_common import config_subject


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical mixed perception/imagery index.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    dataset_cfg = config["dataset"]
    output_path = dataset_cfg.get("mixed_output_index") or dataset_cfg.get("mixed_index")
    if not output_path:
        raise KeyError("Canonical mixed-index preparation requires dataset.mixed_output_index or dataset.mixed_index.")

    mixed = build_mixed_condition_index(
        perception_index=dataset_cfg["perception_index"],
        imagery_index=dataset_cfg["imagery_index"],
        output_path=output_path,
        subject=config_subject(config),
        perception_conditions=dataset_cfg.get("perception_conditions", ["perception"]),
        imagery_conditions=dataset_cfg.get("imagery_conditions", ["imagery"]),
    )

    output = Path(output_path)
    print(f"Prepared mixed index: {output}")
    print(f"Rows: {len(mixed)}")
    print(f"Conditions: {mixed['condition'].value_counts().to_dict()}")
    print(f"Shared pairs: {mixed.groupby('pair_id')['condition'].nunique().ge(2).sum()}")
    print(f"Splits: {mixed['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
