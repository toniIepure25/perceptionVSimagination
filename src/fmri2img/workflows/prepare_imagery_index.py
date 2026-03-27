from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fmri2img.data.canonical import normalize_decoder_index
from fmri2img.data.nsd_imagery import build_nsd_imagery_index
from fmri2img.workflows.common import load_workflow_config
from fmri2img.workflows.prep_common import config_subject, get_preparation_section


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical imagery index used by shared/private decoding.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_workflow_config(args.config, args.override)
    subject = config_subject(config)
    prep_cfg = get_preparation_section(config, "imagery")
    data_root = prep_cfg.get("data_root") or config.get("dataset.imagery_data_root")
    if not data_root:
        raise KeyError(
            "Canonical imagery preparation requires preparation.imagery.data_root or dataset.imagery_data_root."
        )

    output_path = Path(config["dataset"]["imagery_index"])
    cache_root = Path(prep_cfg.get("cache_root", "cache"))
    stimulus_root = prep_cfg.get("stimulus_root")
    result_path = build_nsd_imagery_index(
        data_root=Path(data_root),
        subject=subject,
        cache_root=cache_root,
        output_path=output_path,
        stimulus_root=None if stimulus_root is None else Path(stimulus_root),
        dry_run=args.dry_run,
        verbose=True,
    )
    if args.dry_run:
        return 0

    df = pd.read_parquet(result_path)
    prepared = normalize_decoder_index(df, default_condition="imagery")
    prepared.to_parquet(result_path, index=False)

    print(f"Prepared imagery index: {result_path}")
    print(f"Rows: {len(prepared)}")
    print(f"Unique nsdId: {prepared['nsdId'].nunique()}")
    print(f"Paired groups: {prepared['pair_id'].nunique()}")
    print(f"Splits: {prepared['split'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
