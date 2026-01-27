from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Minimal YAML loader.

    Raises:
        FileNotFoundError: if the path does not exist.
        ImportError: if PyYAML is missing (with install hint).
        yaml.YAMLError: if the YAML content is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML config not found: {p}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyYAML is required to load YAML configs. Install via `pip install pyyaml`."
        ) from exc

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


__all__ = ["load_yaml"]
