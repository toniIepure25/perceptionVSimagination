from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _format_summary(card: dict) -> str:
    experiment = card.get("experiment", {})
    animus = card.get("animus", {})
    target = card.get("target", {})
    condition_semantics = card.get("condition_semantics", {})
    interfaces = card.get("interfaces", {})
    artifacts = card.get("artifacts", {})

    lines = [
        "Animus Export Summary",
        f"experiment: {experiment.get('name')}",
        f"benchmark_role: {experiment.get('benchmark_role')}",
        f"evidence_tier: {experiment.get('evidence_tier')}",
        f"decoder_role: {animus.get('decoder_role')}",
        f"stability_tier: {animus.get('stability_tier')}",
        f"target: {target.get('name')} ({target.get('dimension')}-D)",
        f"present_conditions: {condition_semantics.get('present_conditions')}",
        f"missing_conditions: {condition_semantics.get('missing_conditions')}",
        f"paired_metrics_available: {condition_semantics.get('paired_metrics_available')}",
        f"content_interface: {interfaces.get('content', {}).get('status')}",
        f"source_interface: {interfaces.get('source', {}).get('status')}",
        f"confidence_interface: {interfaces.get('confidence', {}).get('status')}",
        f"checkpoint: {artifacts.get('checkpoint')}",
        f"config_snapshot: {artifacts.get('config_snapshot')}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a compact Animus export bundle summary.")
    parser.add_argument("bundle", help="Path to an Animus export bundle directory.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Also verify that the expected bundle files are present.",
    )
    args = parser.parse_args(argv)

    bundle_dir = Path(args.bundle)
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Animus export bundle does not exist: {bundle_dir}")

    decoder_card_path = bundle_dir / "decoder_card.json"
    manifest_path = bundle_dir / "manifest.json"
    if not decoder_card_path.exists():
        raise FileNotFoundError(f"Missing decoder card: {decoder_card_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    card = _load_json(decoder_card_path)
    print(_format_summary(card))

    if args.validate:
        artifacts = card.get("artifacts", {})
        expected = {
            "manifest": manifest_path,
            "decoder_card_json": decoder_card_path,
            "decoder_card_md": bundle_dir / "decoder_card.md",
        }
        checkpoint_name = artifacts.get("checkpoint")
        if checkpoint_name:
            expected["checkpoint"] = bundle_dir / checkpoint_name
        config_snapshot_name = artifacts.get("config_snapshot")
        if config_snapshot_name:
            expected["config_snapshot"] = bundle_dir / config_snapshot_name

        missing = [f"{name}={path}" for name, path in expected.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Animus export bundle is missing expected files:\n- " + "\n- ".join(missing))
        print("\nBundle validation: OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
