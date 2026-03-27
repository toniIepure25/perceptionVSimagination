from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fmri2img.utils.config_loader import ConfigDict


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - defensive
        np = None

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def write_report(path: str | Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(json_safe(payload), handle, indent=2)


def config_subject(config: ConfigDict) -> str:
    return str(config.get("dataset.subject", "subj01"))


def get_preparation_section(config: ConfigDict, key: str) -> ConfigDict:
    value = config.get(f"preparation.{key}", {})
    if isinstance(value, ConfigDict):
        return value
    if isinstance(value, dict):
        return ConfigDict(value)
    return ConfigDict()

