"""
Path resolution helpers shared across the project.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """
    Walk upward from `start` until a pyproject.toml is found, indicating the repo root.
    Falls back to the provided start path if nothing is found.
    """

    cur = start.resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return cur


def resolve_path(root: Path, p: str | Path) -> Path:
    """
    Resolve a path relative to `root` if not already absolute.
    """

    p = Path(p)
    if p.is_absolute():
        return p
    return root / p

