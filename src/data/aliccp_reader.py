"""
Canonical preprocessing utilities for Ali-CCP (scheme C: samples + tokens tables).

The code intentionally keeps parsing streaming-friendly to avoid loading gigantic
CSV files into memory. Feature strings are tokenized lazily via `iter_tokens`.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.logging import get_logger

logger = get_logger(__name__)

TOKEN_SEP = "\x01"
FIELD_SEP = "\x02"
VALUE_SEP = "\x03"


def require_pyarrow():
    """Import pyarrow with a clear error message if missing."""

    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in runtime usage
        raise RuntimeError(
            "pyarrow is required for parquet output. Install via `pip install pyarrow`."
        ) from exc


class TokenIterator:
    def __init__(self, feat_str: Optional[str]):
        self.feat_str = feat_str
        self.error_count: int = 0

    def __iter__(self) -> Generator[Tuple[str, str, float], None, None]:
        if self.feat_str is None:
            return
        if self.feat_str == "":
            return
        try:
            if pd.isna(self.feat_str):
                return
        except Exception:
            return
        for raw_token in str(self.feat_str).split(TOKEN_SEP):
            if not raw_token:
                continue
            if FIELD_SEP not in raw_token:
                self.error_count += 1
                continue
            field, rest = raw_token.split(FIELD_SEP, 1)
            if VALUE_SEP not in rest:
                self.error_count += 1
                continue
            fid, val_str = rest.split(VALUE_SEP, 1)
            try:
                val = float(val_str)
            except ValueError:
                self.error_count += 1
                continue
            yield field, fid, val


def iter_tokens(feat_str: Optional[str]) -> TokenIterator:
    """
    Yield parsed tokens from a feature string.

    Token format: field \\x02 fid \\x03 value, separated by \\x01. Parsing errors
    are counted on the iterator (via .error_count) instead of throwing to keep
    the pipeline running.
    """

    return TokenIterator(feat_str)


def fetch_common_feat_strs(
    conn: sqlite3.Connection, entity_ids: Sequence[str], batch_size: int = 900
) -> Dict[str, str]:
    """
    Batch-fetch feat_str for the provided entity_ids.

    SQLite parameter limits are respected by chunking the IN clause.
    """

    if not entity_ids:
        return {}

    uniq_ids = list(dict.fromkeys(entity_ids))  # preserve order, drop dupes
    cur = conn.cursor()
    results: Dict[str, str] = {}
    for start in range(0, len(uniq_ids), batch_size):
        batch = uniq_ids[start : start + batch_size]
        placeholders = ",".join(["?"] * len(batch))
        query = f"SELECT entity_id, feat_str FROM common_feat WHERE entity_id IN ({placeholders})"
        cur.execute(query, batch)
        for eid, feat_str in cur.fetchall():
            results[eid] = feat_str
    return results


def _read_csv_with_fallback(path: str, **kwargs):
    """
    Use pandas' default C engine with NA inference disabled; fallback to python engine on parse errors.
    """

    read_kwargs = {
        **kwargs,
        "na_filter": False,
        "keep_default_na": False,
    }
    try:
        return pd.read_csv(path, **read_kwargs)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.warning("C engine failed for %s, falling back to python engine: %s", path, exc)
        return pd.read_csv(path, engine="python", **read_kwargs)


def build_common_features_sqlite(
    cf_path: str, db_path: str, chunksize_cf: int = 50_000, overwrite: bool = False
) -> int:
    """
    Build (or rebuild) the SQLite index for common features.

    The table is a reusable lookup keyed by entity_id to avoid reloading the
    massive common_features CSV repeatedly.
    """

    if not os.path.exists(cf_path):
        raise FileNotFoundError(f"Common features CSV not found: {cf_path}")

    if os.path.exists(db_path) and not overwrite:
        logger.info("Reusing existing SQLite DB at %s", db_path)
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM common_feat").fetchone()[0]
        return int(count)

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    if overwrite and os.path.exists(db_path):
        os.remove(db_path)

    total_rows = 0
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS common_feat (entity_id TEXT PRIMARY KEY, feat_str TEXT)"
        )
        reader = _read_csv_with_fallback(
            cf_path,
            header=None,
            names=["entity_id", "nnz", "feat_str"],
            dtype={"entity_id": str, "nnz": "Int64", "feat_str": str},
            chunksize=chunksize_cf,
        )
        for idx, chunk in enumerate(reader):
            records = [(row.entity_id, row.feat_str) for row in chunk.itertuples(index=False)]
            conn.executemany(
                "INSERT OR REPLACE INTO common_feat(entity_id, feat_str) VALUES (?, ?)", records
            )
            conn.commit()
            total_rows += len(records)
            logger.info(
                "Inserted chunk %d with %d rows into SQLite (running total=%d)",
                idx,
                len(records),
                total_rows,
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_common_feat_eid ON common_feat(entity_id)"
        )
        conn.commit()
    logger.info("Finished building SQLite DB %s with %d rows", db_path, total_rows)
    return total_rows


@dataclass
class TokenStats:
    tokens_total_rows: int
    parse_error_tokens: Optional[int]
    tokens_parts: List[Dict[str, object]]
    join_hit_ratio: Optional[float]
    samples_rows: int
    samples_unique_entity_id: Optional[int]


def _ensure_tokens_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path):
        existing_parts = [p for p in os.listdir(path) if p.endswith(".parquet")]
        if existing_parts and not overwrite:
            logger.info(
                "tokens_dir %s already has %d parquet files; overwrite=False so reusing existing parts.",
                path,
                len(existing_parts),
            )
            return
        if overwrite:
            logger.info(
                "Clearing %d existing parquet parts in tokens_dir %s before rebuild.",
                len(existing_parts),
                path,
            )
            for fname in existing_parts:
                os.remove(os.path.join(path, fname))
    else:
        os.makedirs(path, exist_ok=True)


def write_samples(
    sk_path: str,
    out_parquet: str,
    nrows: Optional[int] = None,
    chunksize_sk: int = 100_000,
    overwrite: bool = False,
) -> int:
    """
    Convert the skeleton CSV into the canonical samples parquet.

    Only the lightweight columns are preserved; the sparse feature string is
    parsed later when building the tokens table.
    """

    if not os.path.exists(sk_path):
        raise FileNotFoundError(f"Skeleton CSV not found: {sk_path}")

    require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    if os.path.exists(out_parquet) and not overwrite:
        logger.info("Reusing existing samples parquet at %s", out_parquet)
        return pq.ParquetFile(out_parquet).metadata.num_rows

    out_dir = os.path.dirname(out_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    row_id = 0
    schema = pa.schema(
        [
            ("row_id", pa.int64()),
            ("y1", pa.int8()),
            ("y2", pa.int8()),
            ("entity_id", pa.string()),
            ("c4", pa.int32()),
            ("c0", pa.int64()),
        ]
    )

    writer = None
    try:
        reader = _read_csv_with_fallback(
            sk_path,
            header=None,
            names=["c0", "y1", "y2", "entity_id", "c4", "sk_feat_str"],
            dtype={
                "c0": "int64",
                "y1": "int8",
                "y2": "int8",
                "entity_id": str,
                "c4": "int32",
                "sk_feat_str": str,
            },
            chunksize=chunksize_sk,
            nrows=nrows,
        )
        for chunk_idx, chunk in enumerate(reader):
            if nrows:
                chunk = chunk.iloc[: nrows - row_id].copy()
            else:
                chunk = chunk.copy()
            chunk["row_id"] = np.arange(row_id, row_id + len(chunk), dtype=np.int64)
            row_id += len(chunk)

            df_out = chunk[["row_id", "y1", "y2", "entity_id", "c4", "c0"]].copy()
            table = pa.Table.from_pandas(df_out, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, schema=schema, compression="snappy")
            writer.write_table(table)
            logger.info(
                "Samples chunk %d written (%d rows, total row_id=%d)",
                chunk_idx,
                len(chunk),
                row_id,
            )
            if nrows and row_id >= nrows:
                break
    finally:
        if writer:
            writer.close()
    logger.info("Finished writing samples to %s (%d rows)", out_parquet, row_id)
    return row_id


def write_tokens(
    sk_path: str,
    db_path: str,
    out_dir_tokens: str,
    nrows: Optional[int] = None,
    chunksize_sk: int = 100_000,
    buffer_max_tokens: int = 3_000_000,
    overwrite: bool = False,
) -> TokenStats:
    """
    Build the long tokens table with skeleton/common sources split via `src`.

    Rows are streamed from the skeleton CSV; for each chunk the corresponding
    common features are fetched in bulk from SQLite to avoid per-row lookups.
    A buffered writer flushes to multiple parquet part files to keep memory
    predictable and enable re-entrant runs.
    """

    if not os.path.exists(sk_path):
        raise FileNotFoundError(f"Skeleton CSV not found: {sk_path}")
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"SQLite DB not found: {db_path}. Build it first via build_common_features_sqlite()."
        )

    require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    _ensure_tokens_dir(out_dir_tokens, overwrite)

    tokens_schema = pa.schema(
        [
            ("row_id", pa.int64()),
            ("src", pa.int8()),
            ("field", pa.string()),
            ("fid", pa.string()),
            ("val", pa.float32()),
        ]
    )

    token_buffer = {"row_id": [], "src": [], "field": [], "fid": [], "val": []}

    def flush_buffer(part_meta: List[Dict[str, object]]) -> None:
        if not token_buffer["row_id"]:
            return
        row_min = int(min(token_buffer["row_id"]))
        row_max = int(max(token_buffer["row_id"]))
        table = pa.Table.from_pydict(
            {
                "row_id": np.array(token_buffer["row_id"], dtype=np.int64),
                "src": np.array(token_buffer["src"], dtype=np.int8),
                "field": token_buffer["field"],
                "fid": token_buffer["fid"],
                "val": np.array(token_buffer["val"], dtype=np.float32),
            },
            schema=tokens_schema,
        )
        part_path = os.path.join(out_dir_tokens, f"tokens_train.part_{row_min}_{row_max}.parquet")
        pq.write_table(table, part_path, compression="snappy")
        part_meta.append(
            {
                "path": part_path,
                "row_id_min": row_min,
                "row_id_max": row_max,
                "token_rows": len(token_buffer["row_id"]),
            }
        )
        logger.info(
            "Flushed token buffer -> %s (%d rows, row_id [%d, %d])",
            part_path,
            len(token_buffer["row_id"]),
            row_min,
            row_max,
        )
        for key in token_buffer:
            token_buffer[key].clear()

    tokens_total_rows = 0
    parse_error_tokens = 0
    part_meta: List[Dict[str, object]] = []
    row_id_cursor = 0
    join_hits = 0
    rows_seen = 0
    unique_tracker: set[str] = set()
    unique_tracker_limit = 5_000_000
    unique_tracking_disabled = False

    with sqlite3.connect(db_path) as conn:
        reader = _read_csv_with_fallback(
            sk_path,
            header=None,
            names=["c0", "y1", "y2", "entity_id", "c4", "sk_feat_str"],
            dtype={
                "c0": "int64",
                "y1": "int8",
                "y2": "int8",
                "entity_id": str,
                "c4": "int32",
                "sk_feat_str": str,
            },
            chunksize=chunksize_sk,
            nrows=nrows,
        )
        for chunk_idx, chunk in enumerate(reader):
            if nrows:
                chunk = chunk.iloc[: nrows - rows_seen].copy()
            else:
                chunk = chunk.copy()
            chunk["row_id"] = np.arange(row_id_cursor, row_id_cursor + len(chunk), dtype=np.int64)
            row_id_cursor += len(chunk)
            rows_seen += len(chunk)

            eid_list = chunk["entity_id"].dropna().tolist()
            if not unique_tracking_disabled:
                unique_tracker.update(eid_list)
                if len(unique_tracker) > unique_tracker_limit:
                    logger.warning(
                        "Unique entity tracking disabled after exceeding limit (%d); samples_unique_entity_id will be None in manifest.",
                        unique_tracker_limit,
                    )
                    unique_tracker.clear()
                    unique_tracking_disabled = True

            cf_map = fetch_common_feat_strs(conn, eid_list)
            cf_keys = set(cf_map.keys())
            join_hits += chunk["entity_id"].isin(cf_keys).sum()

            for row in chunk.itertuples(index=False):
                row_id = int(row.row_id)

                sk_tokens = iter_tokens(row.sk_feat_str)
                for field, fid, val in sk_tokens:
                    token_buffer["row_id"].append(row_id)
                    token_buffer["src"].append(0)
                    token_buffer["field"].append(field)
                    token_buffer["fid"].append(fid)
                    token_buffer["val"].append(np.float32(val))
                    tokens_total_rows += 1
                    if len(token_buffer["row_id"]) >= buffer_max_tokens:
                        flush_buffer(part_meta)
                parse_error_tokens += sk_tokens.error_count

                cf_feat = cf_map.get(row.entity_id)
                if cf_feat is None:
                    continue
                cf_tokens = iter_tokens(cf_feat)
                for field, fid, val in cf_tokens:
                    token_buffer["row_id"].append(row_id)
                    token_buffer["src"].append(1)
                    token_buffer["field"].append(field)
                    token_buffer["fid"].append(fid)
                    token_buffer["val"].append(np.float32(val))
                    tokens_total_rows += 1
                    if len(token_buffer["row_id"]) >= buffer_max_tokens:
                        flush_buffer(part_meta)
                parse_error_tokens += cf_tokens.error_count

            logger.info(
                "Tokens chunk %d processed rows [%d, %d), buffer=%d, tokens_total=%d",
                chunk_idx,
                row_id_cursor - len(chunk),
                row_id_cursor,
                len(token_buffer["row_id"]),
                tokens_total_rows,
            )
            if nrows and rows_seen >= nrows:
                break

    flush_buffer(part_meta)

    if not unique_tracking_disabled:
        samples_unique_entity_id: Optional[int] = len(unique_tracker)
    else:
        samples_unique_entity_id = None

    join_hit_ratio = float(join_hits) / rows_seen if rows_seen else 0.0

    samples_unique_val = int(samples_unique_entity_id) if samples_unique_entity_id is not None else None

    return TokenStats(
        tokens_total_rows=tokens_total_rows,
        parse_error_tokens=parse_error_tokens,
        tokens_parts=part_meta,
        join_hit_ratio=join_hit_ratio,
        samples_rows=rows_seen,
        samples_unique_entity_id=samples_unique_val,
    )


def build_manifest(
    manifest_path: str,
    *,
    raw_paths: Dict[str, str],
    output_dir: str,
    samples_rows: int,
    samples_unique_entity_id: Optional[int],
    join_hit_ratio: Optional[float],
    tokens_total_rows: int,
    tokens_parts: List[Dict[str, object]],
    parse_error_tokens: Optional[int],
    config_dump: Dict[str, object],
) -> None:
    """
    Persist a manifest that captures reproducibility-critical metadata.
    """

    versions = {
        "python": os.sys.version.split()[0],
    }
    try:
        import pandas as pd  # noqa: F401

        versions["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import pyarrow  # noqa: F401

        versions["pyarrow"] = pyarrow.__version__
    except Exception:
        versions["pyarrow"] = None
    try:
        with sqlite3.connect(":memory:") as conn:
            versions["sqlite"] = conn.execute("select sqlite_version()").fetchone()[0]
    except Exception:
        versions["sqlite"] = None

    try:
        import subprocess

        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        commit = None

    manifest = {
        "raw_paths": raw_paths,
        "output_dir": output_dir,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": commit,
        "samples_rows": samples_rows,
        "samples_unique_entity_id": samples_unique_entity_id,
        "join_hit_ratio": join_hit_ratio,
        "tokens_total_rows": tokens_total_rows,
        "tokens_parts": tokens_parts,
        "parse_error_tokens": parse_error_tokens,
        "versions": versions,
        "config_dump": config_dump,
    }
    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info("Manifest written to %s", manifest_path)
