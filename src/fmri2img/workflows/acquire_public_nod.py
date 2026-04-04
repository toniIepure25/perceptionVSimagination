from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fmri2img.workflows._venv_guard import ensure_project_venv


def main(argv: list[str] | None = None) -> int:
    ensure_project_venv("fmri2img.workflows.acquire_public_nod")
    args = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "download_nod_metadata.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Canonical ds004496 acquisition wrapper could not find {script_path}."
        )

    result = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
