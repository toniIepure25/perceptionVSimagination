from __future__ import annotations

import argparse
import json
from pathlib import Path


def _catalog_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "public_datasets" / "catalog.json"


def _load_catalog() -> list[dict]:
    with open(_catalog_path()) as f:
        return json.load(f)["datasets"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the repo's ranked public dataset options.")
    parser.add_argument(
        "--role",
        choices=["A", "B", "C"],
        default=None,
        help="Filter to a specific role: A threshold expansion, B Animus strengthening, C future paper paths.",
    )
    args = parser.parse_args(argv)

    datasets = _load_catalog()
    filtered = [item for item in datasets if args.role is None or args.role in item["role"].split("/")]
    filtered.sort(key=lambda item: (item["role"], item["rank_within_role"], item["label"]))

    for item in filtered:
        print(f"{item['id']}: {item['label']}")
        print(f"  role: {item['role']}")
        print(f"  dataset_id: {item['dataset_id']}")
        print(f"  classification: {item['classification']}")
        print(f"  lane_fit: {item['lane_fit']}")
        print(f"  primary_use: {item['primary_use']}")
        print(f"  status: {item['status']}")
        print(f"  notes: {item['notes']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
