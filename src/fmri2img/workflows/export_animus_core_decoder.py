from __future__ import annotations

from fmri2img.workflows._cli_defaults import DEFAULT_ANIMUS_CORE_CONFIG, with_default_config
from fmri2img.workflows.export_for_animus import main as export_main


def main(argv: list[str] | None = None) -> int:
    return export_main(with_default_config(argv, DEFAULT_ANIMUS_CORE_CONFIG))


if __name__ == "__main__":
    raise SystemExit(main())
