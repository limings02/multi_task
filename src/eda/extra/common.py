from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.core.logging import get_logger
from src.core.paths import resolve_path
from src.data.canonical import load_aliccp_config


logger = get_logger(__name__)

MULTIHOT_NNZ_THRESHOLD = 1.5


@dataclass
class TokenDirs:
    train: Path
    valid: Path
    source: str


def _path_posix(p: Path) -> str:
    return str(p.as_posix())


def write_table(table: pa.Table, path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        logger.info("Skip writing %s (exists, overwrite=False)", path)
        return
    pq.write_table(table, path)


def write_json(path: Path, data: Dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        logger.info("Skip writing %s (exists, overwrite=False)", path)
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def current_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, shell=False)
        return out.decode().strip()
    except Exception:
        return None


def load_config(config_path: Path) -> Dict:
    return load_aliccp_config(str(config_path))


def locate_tokens_dirs(
    repo_root: Path,
    config: Dict,
    debug_sample: bool = False,
) -> TokenDirs:
    """
    Locate train/valid token parquet directories.
    Priority:
      1) debug_sample -> data/raw/fieldwise_100k.parquet (train=valid)
      2) tokens_split.out_dir/train_tokens + valid_tokens
      3) data/**/train_tokens + valid_tokens
      4) data/**/tokens_train + tokens_valid directories
    """
    root = repo_root
    if debug_sample:
        sample = root / "data" / "raw" / "fieldwise_100k.parquet"
        if not sample.exists():
            raise FileNotFoundError("debug-sample requested but data/raw/fieldwise_100k.parquet not found")
        logger.info("Using debug sample tokens from %s", sample)
        return TokenDirs(train=sample, valid=sample, source="debug_sample")

    ts_cfg = config.get("tokens_split", {})
    out_dir = ts_cfg.get("out_dir")
    if out_dir:
        base = resolve_path(root, out_dir)
        train_dir = base / "train_tokens"
        valid_dir = base / "valid_tokens"
        if train_dir.exists() and valid_dir.exists():
            logger.info("Using tokens from tokens_split.out_dir=%s", base)
            return TokenDirs(train=train_dir, valid=valid_dir, source="tokens_split.out_dir")

    # Glob search for common patterns
    train_candidates = list(root.glob("data/**/train_tokens"))
    valid_candidates = list(root.glob("data/**/valid_tokens"))
    if train_candidates and valid_candidates:
        logger.warning("Config tokens not found; discovered train_tokens candidates=%s", train_candidates)
        logger.warning("Config tokens not found; discovered valid_tokens candidates=%s", valid_candidates)
        train_dir = sorted(train_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        valid_dir = sorted(valid_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.warning("Selected latest train_tokens=%s valid_tokens=%s", train_dir, valid_dir)
        return TokenDirs(train=train_dir, valid=valid_dir, source="glob_train_valid_tokens")

    train_candidates = list(root.glob("data/**/tokens_train"))
    valid_candidates = list(root.glob("data/**/tokens_valid"))
    if train_candidates and valid_candidates:
        logger.warning("Config tokens not found; discovered tokens_train candidates=%s", train_candidates)
        logger.warning("Config tokens not found; discovered tokens_valid candidates=%s", valid_candidates)
        train_dir = sorted(train_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        valid_dir = sorted(valid_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        logger.warning("Selected latest tokens_train=%s tokens_valid=%s", train_dir, valid_dir)
        return TokenDirs(train=train_dir, valid=valid_dir, source="glob_tokens_train_valid")

    raise FileNotFoundError("Could not locate tokens_train/valid directories. Please check config or data paths.")


def summarize_metadata(
    config_path: Path,
    in_stats: Path,
    token_dirs: TokenDirs,
    sample_rows_train: Optional[int],
    sample_rows_valid: Optional[int],
    repo_root: Path,
    debug_sample: bool,
) -> Dict:
    return {
        "git_commit": current_git_commit(repo_root),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": _path_posix(config_path),
        "in_stats": _path_posix(in_stats),
        "tokens_train": _path_posix(token_dirs.train),
        "tokens_valid": _path_posix(token_dirs.valid),
        "debug_sample": debug_sample,
        "train_rows": sample_rows_train,
        "valid_rows": sample_rows_valid,
    }
