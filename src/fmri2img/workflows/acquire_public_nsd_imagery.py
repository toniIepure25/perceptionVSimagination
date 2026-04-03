from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "download_nsd_imagery.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Canonical imagery acquisition wrapper could not find {script_path}."
        )

    result = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
