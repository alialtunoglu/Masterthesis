"""Resolve the experiment workspace separately from import/package search paths."""
import os
from pathlib import Path


def get_project_root() -> Path:
    configured = os.environ.get("MASTERTHESIS_PROJECT_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Project root does not exist: {root}")
        return root
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file():
        return checkout
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "configs").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Experiment workspace not found; set MASTERTHESIS_PROJECT_ROOT to the project directory.")
