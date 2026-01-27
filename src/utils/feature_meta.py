from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_feature_meta_from_metadata(metadata_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load total_num_embeddings and feature_meta from metadata.json.
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found at {path}")

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Failed to parse metadata.json at {path}") from exc

    total_num_embeddings = metadata.get("total_num_embeddings")
    feature_meta_flags = metadata.get("feature_meta")
    if total_num_embeddings is None or feature_meta_flags is None:
        raise KeyError("metadata.json must contain both 'total_num_embeddings' and 'feature_meta'.")
    return total_num_embeddings, feature_meta_flags


def build_model_feature_meta(metadata_path: Path, embedding_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Merge feature flags with embedding sizes/modes for model consumption.
    """
    total_num_embeddings, feature_meta_flags = load_feature_meta_from_metadata(metadata_path)

    default_dim = embedding_cfg.get("default_embedding_dim")
    if default_dim is None:
        raise ValueError("embedding.default_embedding_dim is required in the config.")
    overrides = embedding_cfg.get("embedding_dim_overrides") or {}
    mode = embedding_cfg.get("mode", "sum")

    # Sanity: two-way presence check to avoid silent feature drops.
    missing_counts = set(feature_meta_flags) - set(total_num_embeddings)
    if missing_counts:
        raise AssertionError(
            f"total_num_embeddings missing bases: {sorted(missing_counts)}"
        )
    missing_flags = set(total_num_embeddings) - set(feature_meta_flags)
    if missing_flags:
        raise AssertionError(
            f"feature_meta missing bases: {sorted(missing_flags)}"
        )

    merged: Dict[str, Dict[str, Any]] = {}
    for base, meta in feature_meta_flags.items():
        num_embeddings = int(total_num_embeddings[base])
        emb_dim = int(overrides.get(base, default_dim))
        merged[base] = {
            **meta,
            "num_embeddings": num_embeddings,
            "embedding_dim": emb_dim,
            "mode": mode,
        }
    return merged


__all__ = ["load_feature_meta_from_metadata", "build_model_feature_meta"]
