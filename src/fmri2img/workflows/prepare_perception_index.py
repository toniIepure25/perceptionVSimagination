from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fmri2img.data.build_full_index import build_full_index
from fmri2img.data.canonical import normalize_decoder_index
from fmri2img.workflows.common import load_workflow_config
from fmri2img.workflows.prep_common import config_subject


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or normalize the canonical perception index.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--rebuild-from-s3", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    subject = config_subject(config)
    output_template = config["dataset"].get("perception_index") or config.get("preparation.overlap.perception_index_template")
    if not output_template:
        raise KeyError(
            "Canonical perception preparation requires dataset.perception_index or "
            "preparation.overlap.perception_index_template."
        )
    output_path = Path(str(output_template).format(subject=subject))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.rebuild_from_s3 or not output_path.exists():
        df = build_full_index(subject=subject, output_path=output_path, max_sessions=args.max_sessions)
    else:
        df = pd.read_parquet(output_path)

    allowed_conditions = config["dataset"].get("perception_conditions", ["perception"])
    prepared = normalize_decoder_index(
        df,
        default_condition="perception",
        allowed_conditions=allowed_conditions,
    )
    prepared = prepared[prepared["subject"] == subject].reset_index(drop=True)
    prepared.to_parquet(output_path, index=False)

    print(f"Prepared perception index: {output_path}")
    print(f"Rows: {len(prepared)}")
    print(f"Unique nsdId: {prepared['nsdId'].nunique()}")
    print(f"Splits: {prepared['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
