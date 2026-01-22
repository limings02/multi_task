"""
Pipeline orchestration for building Ali-CCP canonical artifacts.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from src.core.logging import get_logger
from src.core.paths import find_repo_root, resolve_path
from src.data.aliccp_reader import (
    TokenStats,
    build_common_features_sqlite,
    build_manifest,
    write_samples,
    write_tokens,
)

logger = get_logger(__name__)


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _maybe_override(config: Dict, overrides: Optional[Dict]) -> Dict:
    if not overrides:
        return config
    merged = json.loads(json.dumps(config))  # deep copy without bringing in pydantic/dataclasses
    for key, value in overrides.items():
        # supports dotted keys like raw.skeleton_path
        if value is None:
            continue
        target = merged
        *parents, leaf = key.split(".")
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = value
    return merged


def load_aliccp_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    config = _load_yaml(config_path)
    env_skeleton = os.getenv("ALICCP_SKELETON_PATH")
    env_common = os.getenv("ALICCP_COMMON_FEATURES_PATH")
    if env_skeleton:
        config.setdefault("raw", {})["skeleton_path"] = env_skeleton
    if env_common:
        config.setdefault("raw", {})["common_features_path"] = env_common
    return _maybe_override(config, overrides)


def _parquet_num_rows(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    except ImportError:
        logger.warning(
            "pyarrow not available, falling back to pandas for parquet metadata on %s (slower).",
            path,
        )
        import pandas as pd

        return pd.read_parquet(path, columns=[]).shape[0]


def _load_tokens_from_manifest(manifest_path: Path) -> Optional[TokenStats]:
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TokenStats(
            tokens_total_rows=int(data.get("tokens_total_rows", 0)),
            parse_error_tokens=data.get("parse_error_tokens"),
            tokens_parts=data.get("tokens_parts", []),
            join_hit_ratio=data.get("join_hit_ratio"),
            samples_rows=int(data.get("samples_rows", 0)),
            samples_unique_entity_id=data.get("samples_unique_entity_id"),
        )
    except Exception as exc:
        logger.warning("Failed to reuse tokens stats from manifest %s: %s", manifest_path, exc)
        return None


def _scan_tokens_dir(tokens_dir: Path, samples_rows: int) -> TokenStats:
    parts: List[Dict[str, object]] = []
    total_rows = 0
    for part in sorted(tokens_dir.glob("*.parquet")):
        rows = _parquet_num_rows(part)
        total_rows += rows
        stem = part.stem
        row_min = None
        row_max = None
        if "part_" in stem:
            try:
                row_min, row_max = map(int, stem.split("part_", 1)[1].split("_"))
            except Exception:
                row_min, row_max = None, None
        parts.append(
            {
                "path": str(part),
                "row_id_min": row_min,
                "row_id_max": row_max,
                "token_rows": rows,
            }
        )
    logger.info(
        "Scanned existing tokens_dir %s: %d parts, %d total token rows (join_hit_ratio unknown).",
        tokens_dir,
        len(parts),
        total_rows,
    )
    return TokenStats(
        tokens_total_rows=total_rows,
        parse_error_tokens=None,
        tokens_parts=parts,
        join_hit_ratio=None,
        samples_rows=samples_rows,
        samples_unique_entity_id=None,
    )


def build_canonical_aliccp(
    config_path: str, overrides: Optional[Dict] = None, seed: int = 20260121
) -> Dict:
    """
    Entry point used by CLI/tests to run the canonical preprocessing.
    """

    random.seed(seed)
    np.random.seed(seed)

    config_path = str(Path(config_path).resolve())
    repo_root = find_repo_root(Path(config_path).parent)
    config = load_aliccp_config(config_path, overrides)
    raw_cfg = config.get("raw", {})
    canonical_cfg = config.get("canonical", {})
    params = config.get("params", {})

    out_dir = resolve_path(
        repo_root, canonical_cfg.get("out_dir", Path("data/interim/aliccp_canonical"))
    )
    sqlite_path = canonical_cfg.get("sqlite_path")
    samples_path = canonical_cfg.get("samples_path")
    tokens_dir = canonical_cfg.get("tokens_dir")

    if sqlite_path is None:
        sqlite_path = Path(out_dir) / "common_features.sqlite"
    else:
        sqlite_path = resolve_path(repo_root, sqlite_path)
    if samples_path is None:
        samples_path = Path(out_dir) / "samples_train.parquet"
    else:
        samples_path = resolve_path(repo_root, samples_path)
    if tokens_dir is None:
        tokens_dir = Path(out_dir) / "tokens_train"
    else:
        tokens_dir = resolve_path(repo_root, tokens_dir)

    manifest_path = Path(out_dir) / "manifest.json"

    skeleton_path = raw_cfg.get("skeleton_path")
    common_features_path = raw_cfg.get("common_features_path")
    if not skeleton_path or not common_features_path:
        raise ValueError("skeleton_path and common_features_path must be provided in config.")

    skeleton_path = resolve_path(repo_root, skeleton_path)
    common_features_path = resolve_path(repo_root, common_features_path)

    nrows = params.get("nrows")
    chunksize_sk = params.get("chunksize_sk", 100_000)
    chunksize_cf = params.get("chunksize_cf", 50_000)
    buffer_max_tokens = params.get("buffer_max_tokens", 3_000_000)
    rebuild_sqlite = bool(params.get("rebuild_sqlite", False))
    overwrite = bool(params.get("overwrite", False))

    logger.info(
        "Canonical config: repo_root=%s skeleton=%s common=%s out_dir=%s sqlite=%s samples=%s tokens_dir=%s",
        repo_root,
        skeleton_path,
        common_features_path,
        out_dir,
        sqlite_path,
        samples_path,
        tokens_dir,
    )
    logger.info(
        "Params: nrows=%s chunksize_sk=%s chunksize_cf=%s buffer_max_tokens=%s seed=%s rebuild_sqlite=%s overwrite=%s",
        nrows,
        chunksize_sk,
        chunksize_cf,
        buffer_max_tokens,
        seed,
        rebuild_sqlite,
        overwrite,
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sqlite_rows = build_common_features_sqlite(
        str(common_features_path),
        str(sqlite_path),
        chunksize_cf=chunksize_cf,
        overwrite=rebuild_sqlite,
    )
    logger.info("SQLite rows available: %d", sqlite_rows)

    samples_rows = write_samples(
        str(skeleton_path),
        str(samples_path),
        nrows=nrows,
        chunksize_sk=chunksize_sk,
        overwrite=overwrite,
    )

    tokens_dir_path = Path(tokens_dir)
    existing_parts = list(tokens_dir_path.glob("*.parquet")) if tokens_dir_path.exists() else []
    token_stats: Optional[TokenStats] = None

    if existing_parts and not overwrite:
        logger.info(
            "Detected existing tokens parts in %s (count=%d); attempting reuse because overwrite=False.",
            tokens_dir_path,
            len(existing_parts),
        )
        token_stats = _load_tokens_from_manifest(manifest_path)
        if token_stats:
            logger.info(
                "Reused tokens stats from existing manifest (tokens_total_rows=%s).",
                token_stats.tokens_total_rows,
            )
        else:
            token_stats = _scan_tokens_dir(tokens_dir_path, samples_rows=samples_rows)
        token_stats.samples_rows = samples_rows
    else:
        token_stats = write_tokens(
            str(skeleton_path),
            str(sqlite_path),
            str(tokens_dir),
            nrows=nrows,
            chunksize_sk=chunksize_sk,
            buffer_max_tokens=buffer_max_tokens,
            overwrite=overwrite,
        )

    if samples_rows != token_stats.samples_rows:
        logger.warning(
            "Row count mismatch between samples (%d) and tokens (%d). Using tokens count.",
            samples_rows,
            token_stats.samples_rows,
        )

    sorted_parts = sorted(
        token_stats.tokens_parts, key=lambda p: (p.get("row_id_min") or 0) if isinstance(p, dict) else 0
    )

    build_manifest(
        str(manifest_path),
        raw_paths={
            "skeleton_path": str(skeleton_path),
            "common_features_path": str(common_features_path),
        },
        output_dir=str(out_dir),
        samples_rows=token_stats.samples_rows,
        samples_unique_entity_id=token_stats.samples_unique_entity_id,
        join_hit_ratio=token_stats.join_hit_ratio,
        tokens_total_rows=token_stats.tokens_total_rows,
        tokens_parts=sorted_parts,
        parse_error_tokens=token_stats.parse_error_tokens,
        config_dump={
            "config_path": config_path,
            "nrows": nrows,
            "chunksize_sk": chunksize_sk,
            "chunksize_cf": chunksize_cf,
            "buffer_max_tokens": buffer_max_tokens,
            "rebuild_sqlite": rebuild_sqlite,
            "overwrite": overwrite,
            "seed": seed,
        },
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
