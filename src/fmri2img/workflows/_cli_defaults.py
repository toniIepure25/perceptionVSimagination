from __future__ import annotations

import sys


DEFAULT_ANIMUS_CORE_CONFIG = "configs/canonical/animus_core_decoder.yaml"


def with_default_config(argv: list[str] | None, default_config: str) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    for index, arg in enumerate(args):
        if arg == "--config":
            if index + 1 >= len(args):
                raise ValueError("--config was provided without a value.")
            return args
        if arg.startswith("--config="):
            return args
    return ["--config", default_config, *args]
