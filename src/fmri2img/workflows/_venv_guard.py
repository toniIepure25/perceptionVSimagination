from __future__ import annotations

from pathlib import Path
import sys


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_venv_bin_dir(repo_root: Path | None = None) -> Path:
    root = project_root() if repo_root is None else Path(repo_root)
    return root / ".venv" / "bin"


def is_running_in_project_venv(
    executable: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> bool:
    root = project_root() if repo_root is None else Path(repo_root)
    exe_path = Path(sys.executable if executable is None else executable).expanduser()
    if not exe_path.is_absolute():
        exe_path = (Path.cwd() / exe_path).absolute()
    return exe_path.parent == project_venv_bin_dir(root)


def ensure_project_venv(
    module_name: str,
    executable: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> None:
    if is_running_in_project_venv(executable=executable, repo_root=repo_root):
        return
    root = project_root() if repo_root is None else Path(repo_root)
    current_executable = sys.executable if executable is None else str(executable)
    expected_python = project_venv_bin_dir(root) / "python"
    raise SystemExit(
        f"{module_name} must be run from the project .venv.\n"
        f"Current interpreter: {current_executable}\n"
        "Use one of:\n"
        "- source .venv/bin/activate\n"
        f"- ./.venv/bin/python -m {module_name}\n"
        f"Expected interpreter location: {expected_python}"
    )
