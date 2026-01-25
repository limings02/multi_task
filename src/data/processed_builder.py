"""
Split -> Processed builder.

Industrialised pipeline that:
- Reads split samples + token shards.
- Builds vocab/freq heads using **EDA topK** first, falls back to DuckDB topN.
- Applies token_select (freq/value/auto/auto_mix) without ever loading full-token
  counters into Python memory.
- Emits multi-hot columns that respect FeatureSpec.use_value strictly:
    use_value=True  -> {field}_idx / {field}_val / {field}_off
    use_value=False -> {field}_idx / {field}_off   (no *_val to avoid silent bias)
- Writes sharded parquet datasets (train/valid/metadata.json/_SUCCESS).

Default IO (aligned with user request):
    featuremap : configs/dataset/featuremap.yaml
    split_dir  : data/splits/aliccp_entity_hash_v1/
    processed  : data/processed/aliccp_entity_hash_v1/

Example (PowerShell):
    python -m src.data.processed_builder `
        --config configs/dataset/featuremap.yaml `
        --split_dir data/splits/aliccp_entity_hash_v1 `
        --out data/processed/aliccp_entity_hash_v1 `
        --batch_size 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:  # DuckDB is optional but recommended for fallback topN.
    import duckdb  # type: ignore
except ImportError:  # pragma: no cover - exercised in runtime
    duckdb = None

from src.core.logging import get_logger
from src.data.featuremap_parser import (
    FeatureKey,
    FeatureMap,
    FeatureSpec,
    TokenPolicy,
    featuremap_hash,
    load_featuremap,
)
from src.data.token_select import select_tokens

logger = get_logger(__name__)

# ------------- constants -------------
DEFAULT_FEATUREMAP_PATH = "configs/dataset/featuremap.yaml"
DEFAULT_SPLIT_DIR = "data/splits/aliccp_entity_hash_v1"
DEFAULT_PROCESSED_DIR = "data/processed/aliccp_entity_hash_v1"
DEFAULT_EDA_STATS = "data/stats/eda_v1"

VALUE_AGG_FN = "max"
CHUNK_ROWS = 200_000
VAL_SAMPLE_LIMIT = 50_000
_MISSING_SRC_WARNED: set[str] = set()


# ------------- hashing utilities -------------
def _hash_token(token: str, bucket: int, seed: int, field: str = "", src: int = 0) -> int:
    """
    Deterministic field-aware hash -> bucket index.
    Always uses blake2b; (seed, src, field, token) baked into message to prevent cross-field collision.
    Returns value in [0, bucket-1].
    """
    h = hashlib.blake2b(f"{seed}:{src}:{field}:{token}".encode("utf-8"), digest_size=16)
    return int.from_bytes(h.digest(), "big") % bucket


def _aggregate_token_values(tok_pairs: List[Tuple[str, float]], agg_fn: str = VALUE_AGG_FN) -> Dict[str, float]:
    """
    Aggregate duplicate tokens for one row/field.
    agg_fn:
      - max (default): avoids value amplification, keeps strongest signal.
      - sum: additive weights.
    Complexity: O(n) over tokens in the row.
    """
    out: Dict[str, float] = {}
    for tok, val in tok_pairs:
        if agg_fn == "max":
            out[tok] = max(out.get(tok, val), val)
        elif agg_fn == "sum":
            out[tok] = out.get(tok, 0.0) + val
        else:  # overwrite
            out[tok] = val
    return out


def _log_missing_src_once(part_name: str) -> None:
    if part_name not in _MISSING_SRC_WARNED:
        logger.warning("tokens part %s missing src column; defaulting src=0", part_name)
        _MISSING_SRC_WARNED.add(part_name)


# ------------- token shard helpers -------------
def _load_manifest_ranges(tokens_dir: Path, split_dir: Path) -> Dict[str, Tuple[int, int]]:
    """
    Read tokens_split_manifest.json.
    Priority: split root > tokens_dir (if both exist).
    """
    candidates = [
        split_dir / "tokens_split_manifest.json",
        tokens_dir / "tokens_split_manifest.json",
    ]
    for manifest in candidates:
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception as e:  # pragma: no cover
                logger.warning("Failed to read %s: %s", manifest, e)
                return {}
            ranges = {}
            for part in data.get("parts", []):
                fname = part.get("file")
                lo, hi = part.get("row_id_min"), part.get("row_id_max")
                if fname is not None and lo is not None and hi is not None:
                    ranges[fname] = (int(lo), int(hi))
            return ranges
    return {}


def _parse_part_interval(path: Path, manifest_ranges: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Determine row_id interval for a token shard.
    Priority: manifest -> filename regex -> (0, inf) fallback.
    """
    if path.name in manifest_ranges:
        return manifest_ranges[path.name]
    m = re.search(r"\.part_(\d+)_([0-9]+)\.parquet$", path.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    logger.warning("Cannot parse row range for %s; falling back to full scan (slower).", path.name)
    return 0, 2**63 - 1


# ------------- freq/vocab builders -------------
def _freq_from_eda(eda_dir: Path, feature_map: FeatureMap):
    """
    Use precomputed EDA topK (preferred, zero extra scan).
    Expected schema: field, src, fid, cnt, topk_coverage (columns beyond these are ignored).
    """
    path = eda_dir / "field_topk_train.parquet"
    if not path.exists():
        return None
    tbl = pq.read_table(path, columns=["field", "src", "fid", "cnt"])
    df = tbl.to_pandas()
    bucket: Dict[Tuple[int, str], List[Tuple[str, int]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        key = (int(row.src), str(row.field))
        bucket[key].append((str(row.fid), int(row.cnt)))

    freq_rank: Dict[FeatureKey, Dict[str, int]] = {}
    vocab_table: Dict[FeatureKey, Dict[str, int]] = {}
    for spec in feature_map.features:
        key = (spec.src, spec.field)
        items = bucket.get((spec.src, str(spec.field)), [])
        if not items:
            continue
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        freq_rank[key] = {tok: i for i, (tok, _) in enumerate(items)}
        if spec.encoding in {"vocab", "hybrid"}:
            top_n = spec.vocab_num_tokens or len(items)
            vocab_table[key] = {tok: i for i, (tok, _) in enumerate(items[:top_n])}
    logger.info("Loaded freq/vocab from EDA topK at %s", path)
    return freq_rank, vocab_table


def _freq_with_duckdb(train_tokens_dir: Path, feature_map: FeatureMap, max_topn: int):
    """
    Fallback: run aggregation inside DuckDB to avoid Python Counters.
    Complexity: O(total tokens) streamed inside DuckDB; memory bounded by topN.
    """
    if duckdb is None:
        raise RuntimeError(
            "duckdb is not installed and EDA topK is unavailable. "
            "Install duckdb (`pip install duckdb`) or run EDA to produce data/stats/eda_v1/field_topk_train.parquet."
        )
    glob = str(train_tokens_dir / "*.parquet").replace("\\", "/")
    sql = f"""
        WITH agg AS (
            SELECT
                COALESCE(src, 0) AS src,
                CAST(field AS VARCHAR) AS field,
                CAST(fid AS VARCHAR) AS fid,
                COUNT(*) AS cnt
            FROM read_parquet('{glob}')
            GROUP BY src, field, fid
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY src, field ORDER BY cnt DESC, fid) AS r
            FROM agg
        )
        SELECT src, field, fid, cnt
        FROM ranked
        WHERE r <= {max_topn}
    """
    df = duckdb.query(sql).to_df()
    bucket: Dict[Tuple[int, str], List[Tuple[str, int]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        bucket[(int(row.src), str(row.field))].append((str(row.fid), int(row.cnt)))

    freq_rank: Dict[FeatureKey, Dict[str, int]] = {}
    vocab_table: Dict[FeatureKey, Dict[str, int]] = {}
    for spec in feature_map.features:
        key = (spec.src, spec.field)
        items = bucket.get((spec.src, str(spec.field)), [])
        if not items:
            continue
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        freq_rank[key] = {tok: i for i, (tok, _) in enumerate(items)}
        if spec.encoding in {"vocab", "hybrid"}:
            top_n = spec.vocab_num_tokens or len(items)
            vocab_table[key] = {tok: i for i, (tok, _) in enumerate(items[:top_n])}
    logger.info("Loaded freq/vocab via DuckDB topN (%d) from %s", max_topn, train_tokens_dir)
    return freq_rank, vocab_table


def build_freq_and_vocab(
    train_tokens_dir: Path,
    feature_map: FeatureMap,
    eda_stats_dir: Path,
) -> Tuple[Dict[FeatureKey, Dict[str, int]], Dict[FeatureKey, Dict[str, int]]]:
    """
    Build frequency ranks and vocab tables (topN only).
    Priority: EDA topK parquet -> DuckDB group-by limit N -> explicit error.
    """
    freq_vocab = _freq_from_eda(eda_stats_dir, feature_map)
    if freq_vocab is not None:
        return freq_vocab

    max_topn = 0
    for spec in feature_map.features:
        max_topn = max(
            max_topn,
            spec.vocab_num_tokens or 0,
            spec.max_len or 0,
        )
    max_topn = max(max_topn, 1)
    if duckdb is not None:
        return _freq_with_duckdb(train_tokens_dir, feature_map, max_topn)
    raise RuntimeError(
        "Cannot build freq/vocab: EDA topK missing and duckdb not installed. "
        "Run EDA first or install duckdb."
    )


# ------------- token mapping -------------
def map_token_to_index(
    tok: str, spec: FeatureSpec, policy: TokenPolicy, vocab_table: Dict[str, int] | None
) -> Tuple[int, bool]:
    """
    Map raw token string to embedding row id following token_policy.
    Returns (idx, is_vocab_oov) where is_vocab_oov is True only for vocab/hybrid-head OOV.
    """
    if spec.encoding == "hash":
        # P0-1: field-aware hash; P1-1: no base_offset, returns [0, bucket-1]
        idx = policy.hash_base_offset + _hash_token(tok, spec.hash_bucket, policy.hash_seed, spec.field, spec.src)
        return idx, False  # hash has no vocab OOV concept
    if vocab_table is None:
        return policy.vocab_oov_id, True
    rank = vocab_table.get(tok)
    if rank is None:
        return policy.vocab_oov_id, True
    # bugfix: schemaA head uses per-feature special_base_offset, not global policy offset
    head_idx = spec.special_base_offset + rank
    assert 0 <= head_idx, f"{spec.src}/{spec.field} head_idx negative: {head_idx}"
    # bugfix: guard head index overflow for clearer failure
    assert head_idx < spec.vocab_num_embeddings, f"{spec.src}/{spec.field} head_idx overflow: {head_idx} >= {spec.vocab_num_embeddings}"
    if spec.encoding == "vocab":
        return head_idx, False
    return head_idx, False  # hybrid head


def map_hybrid_tail(tok: str, spec: FeatureSpec, policy: TokenPolicy) -> int:
    """P1-1: Hybrid tail is compact: idx = vocab_num_embeddings + hash_value (no base_offset gap)."""
    hash_val = _hash_token(tok, spec.hash_bucket, policy.hash_seed, spec.field, spec.src)
    return spec.vocab_num_embeddings + hash_val


# ------------- token collection -------------
def _collect_tokens_for_batch(
    part_index: List[Tuple[int, int, Path, bool]],
    row_ids: List[int],
    token_dir: Path,
) -> Tuple[Dict[Tuple[int, int, str], List[Tuple[str, float]]], Dict[Tuple[int, int, str], int], Dict[Tuple[int, str], int]]:
    """
    Collect tokens for given row_ids using DuckDB glob query (single pass over all parts).
    
    Memory optimization: Uses DuckDB's internal memory management and streaming.
    
    Returns:
      - mapping (row_id, src, field) -> list[(fid, val)]
      - nan_inf_counter: (src, field) -> count of NaN/Inf values
      - total_tokens_counter: (src, field) -> total token count (P1-4)
    """
    out: Dict[Tuple[int, int, str], List[Tuple[str, float]]] = defaultdict(list)
    nan_inf_counter: Dict[Tuple[int, int, str], int] = defaultdict(int)
    total_tokens_counter: Dict[Tuple[int, str], int] = defaultdict(int)
    if not row_ids:
        return out, nan_inf_counter, total_tokens_counter

    if duckdb is None:
        raise RuntimeError(
            "P0-5: duckdb is required for full-scale token processing. "
            "Install with: pip install duckdb"
        )

    min_id, max_id = min(row_ids), max(row_ids)
    
    # Check if any part has 'src' column
    has_src = any(hs for _, _, _, hs in part_index)
    n_parts = len(part_index)
    
    logger.info("  Collecting tokens: %d row_ids (range %d-%d) via glob over %d parts", 
                len(row_ids), min_id, max_id, n_parts)

    # Use glob pattern - single query over all parts
    glob_pattern = str(token_dir / "*.parquet").replace("\\", "/")
    
    # Memory-safe DuckDB configuration
    con = duckdb.connect()
    # Limit DuckDB memory to ~4GB to avoid OOM (adjust based on your system)
    con.execute("SET memory_limit = '4GB'")
    # Use temp directory for spilling if needed
    con.execute("SET temp_directory = 'data/.duckdb_tmp'")
    # Enable parallel processing
    con.execute("SET threads TO 4")  # Conservative for memory
    
    con.register("row_ids_tbl", pa.table({"row_id": pa.array(row_ids, type=pa.int64())}))

    src_col = "COALESCE(t.src, 0)" if has_src else "0"
    # Key optimization: row_id BETWEEN filter is pushed down by DuckDB's optimizer
    # before the JOIN, dramatically reducing scanned data
    sql = f"""
        WITH filtered AS (
            SELECT
                t.row_id,
                CAST({src_col} AS INTEGER) AS src,
                CAST(t.field AS VARCHAR) AS field,
                CAST(t.fid AS VARCHAR) AS fid,
                CAST(t.val AS DOUBLE) AS val
            FROM read_parquet('{glob_pattern}') t
            WHERE t.row_id BETWEEN {min_id} AND {max_id}
        ),
        joined AS (
            SELECT f.*
            FROM filtered f
            JOIN row_ids_tbl r ON f.row_id = r.row_id
        )
        SELECT
            row_id,
            src,
            field,
            list(fid) AS fids,
            list(val) AS vals,
            COUNT(*) AS cnt,
            SUM(CASE WHEN NOT isfinite(val) THEN 1 ELSE 0 END) AS nan_inf_cnt
        FROM joined
        GROUP BY row_id, src, field
    """
    
    logger.info("  Running DuckDB glob query...")
    import time
    t0 = time.perf_counter()
    
    try:
        agg_tbl = con.execute(sql).fetch_arrow_table()
        elapsed = time.perf_counter() - t0
        logger.info("  Query returned %d groups in %.1fs", agg_tbl.num_rows, elapsed)
    except Exception as e:
        logger.error("DuckDB glob query failed: %s", e)
        con.close()
        raise

    # Fast vectorized extraction - convert entire columns to Python lists once
    t1 = time.perf_counter()
    total_groups = agg_tbl.num_rows
    
    # Batch convert to Python (much faster than per-row access)
    all_rids = agg_tbl["row_id"].to_pylist()
    all_srcs = agg_tbl["src"].to_pylist()
    all_fields = agg_tbl["field"].to_pylist()
    all_fids = agg_tbl["fids"].to_pylist()
    all_vals = agg_tbl["vals"].to_pylist()
    all_cnts = agg_tbl["cnt"].to_pylist()
    all_nan_infs = agg_tbl["nan_inf_cnt"].to_pylist()
    
    logger.info("  Column extraction: %.1fs", time.perf_counter() - t1)
    
    # Process all groups (single pass, optimized)
    t2 = time.perf_counter()
    for i in range(total_groups):
        rid = all_rids[i]
        src = all_srcs[i]
        field = all_fields[i]
        fids = all_fids[i]
        vals = all_vals[i]
        cnt = all_cnts[i]
        nan_inf = all_nan_infs[i]

        # P1-4: track total tokens
        total_tokens_counter[(src, field)] += cnt
        if nan_inf > 0:
            nan_inf_counter[(src, field)] += nan_inf

        # Build (fid, val) pairs - optimized: avoid repeated tuple unpacking
        key = (rid, src, field)
        pairs = out[key]
        for j in range(len(fids)):
            fid = fids[j]
            val = vals[j]
            if val is None or not math.isfinite(val):
                val = 0.0
            pairs.append((fid, val))
    
    logger.info("  Group processing: %.1fs for %d groups", time.perf_counter() - t2, total_groups)

    con.close()
    return out, nan_inf_counter, total_tokens_counter


def _to_offsets(lengths: List[int]) -> List[int]:
    offs = []
    acc = 0
    for l in lengths:
        offs.append(acc)
        acc += l
    return offs


# ------------- per-batch processing -------------
def _process_batch(
    batch: pa.RecordBatch,
    part_index: List[Tuple[int, int, Path, bool]],
    token_dir: Path,
    feature_map: FeatureMap,
    freq_rank: Dict[FeatureKey, Dict[str, int]],
    vocab_table: Dict[FeatureKey, Dict[str, int]],
    strategy_log: Dict[str, Dict[str, float]],
) -> Tuple[pa.Table, Dict[str, Dict[str, float]]]:
    """
    Convert one sample batch to feature columns.
    Returns Arrow table and per-field stats for this batch.
    """
    rows = batch.num_rows
    row_ids = batch["row_id"].to_pylist()
    tokens, nan_inf_counts, total_tokens_counts = _collect_tokens_for_batch(part_index, row_ids, token_dir)
    feat_by_key = feature_map.feature_by_key()

    columns: Dict[str, List] = {}
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # labels
    y_ctr = batch["y1"].to_numpy().astype("int8")
    y_cvr = batch["y2"].to_numpy().astype("int8")
    columns["y_ctr"] = y_ctr
    columns["y_cvr"] = y_cvr
    columns["y_ctcvr"] = (y_ctr * y_cvr).astype("int8")
    columns["click_mask"] = y_ctr
    columns["row_id"] = row_ids
    columns["entity_id"] = batch["entity_id"].to_pylist()

    for key, spec in feat_by_key.items():
        field = spec.field
        col_prefix = f"f{spec.src}_{field}"
        freq_map = freq_rank.get(key, {})
        vocab_map = vocab_table.get(key)
        total_slots = spec.total_num_embeddings()

        if spec.is_multi_hot:
            idx_lists: List[List[int]] = []
            val_lists: List[List[float]] = []
            len_lists: List[int] = []
            missing_count = 0
            oov_count = 0
            tail_fallback_count = 0  # P1-3: track hybrid tail fallback
            max_idx_seen = 0
            token_rows = 0
            strategy_counts: Dict[str, int] = defaultdict(int)
            retained_sum = 0.0
            retained_rows = 0

            for rid in row_ids:
                tok_pairs = tokens.get((rid, spec.src, field), [])
                if not tok_pairs:
                    missing_id = (
                        feature_map.token_policy.vocab_missing_id
                        if spec.is_vocab_head()
                        else feature_map.token_policy.hash_missing_id
                    )
                    idx_lists.append([missing_id])
                    if spec.use_value:
                        val_lists.append([1.0])
                    len_lists.append(1)
                    missing_count += 1
                    max_idx_seen = max(max_idx_seen, missing_id)
                    continue

                token_rows += 1
                value_map = _aggregate_token_values(tok_pairs, VALUE_AGG_FN)
                chosen_tokens, chosen_vals, strategy_used, retained_frac = select_tokens(
                    spec, list(value_map.items()), freq_map
                )
                if strategy_used is not None:
                    strategy_counts[strategy_used] += 1
                retained_sum += retained_frac
                retained_rows += 1
                if not chosen_tokens:
                    missing_id = (
                        feature_map.token_policy.vocab_missing_id
                        if spec.is_vocab_head()
                        else feature_map.token_policy.hash_missing_id
                    )
                    idx_lists.append([missing_id])
                    if spec.use_value:
                        val_lists.append([1.0])
                    len_lists.append(1)
                    missing_count += 1
                    max_idx_seen = max(max_idx_seen, missing_id)
                    continue

                mapped_idx: List[int] = []
                mapped_vals: List[float] = []
                for t_idx, tok in enumerate(chosen_tokens):
                    is_vocab_oov = False
                    is_tail_fallback = False
                    if spec.encoding == "hybrid" and spec.tail_encoding == "hash":
                        if vocab_map is not None and tok in vocab_map:
                            idx, is_vocab_oov = map_token_to_index(tok, spec, feature_map.token_policy, vocab_map)
                        else:
                            idx = map_hybrid_tail(tok, spec, feature_map.token_policy)
                            is_tail_fallback = True  # P1-3
                    else:
                        idx, is_vocab_oov = map_token_to_index(tok, spec, feature_map.token_policy, vocab_map)
                    # P0-2: only count OOV for vocab/hybrid-head, not for hash
                    if is_vocab_oov:
                        oov_count += 1
                    if is_tail_fallback:
                        tail_fallback_count += 1
                    if idx >= total_slots:
                        raise AssertionError(f"{col_prefix} idx {idx} exceeds total embeddings {total_slots}")
                    mapped_idx.append(int(idx))
                    if spec.use_value:
                        mapped_vals.append(float(chosen_vals[t_idx]))
                    max_idx_seen = max(max_idx_seen, idx)

                idx_lists.append(mapped_idx)
                if spec.use_value:
                    val_lists.append(mapped_vals)
                len_lists.append(len(mapped_idx))

            columns[f"{col_prefix}_idx"] = pa.array(idx_lists, type=pa.list_(pa.int64()))
            # P1-5: removed *_off column (semantic mismatch with list<list> idx)
            if spec.use_value:
                columns[f"{col_prefix}_val"] = pa.array(val_lists, type=pa.list_(pa.float32()))
            stats[col_prefix]["rows"] += rows
            stats[col_prefix]["token_rows"] += token_rows
            stats[col_prefix]["missing"] += missing_count
            stats[col_prefix]["oov"] += oov_count
            stats[col_prefix]["tail_fallback"] += tail_fallback_count  # P1-3
            # P1-4: track total_tokens for proper nan/inf rate
            stats[col_prefix]["total_tokens"] += total_tokens_counts.get((spec.src, field), 0)
            stats[col_prefix]["nan_inf_tokens"] += nan_inf_counts.get((spec.src, field), 0)
            stats[col_prefix]["len_sum"] += sum(len_lists)
            stats[col_prefix]["len_rows"] += len(len_lists)
            stats[col_prefix]["max_idx"] = max(stats[col_prefix].get("max_idx", 0), max_idx_seen)
            if strategy_counts:
                # bugfix: store per-strategy counts instead of raw names so aggregation is numeric
                for name, cnt in strategy_counts.items():
                    stats[col_prefix][f"strategy_cnt_{name}"] += cnt
            if retained_rows:
                stats[col_prefix]["retained_frac_sum"] = stats[col_prefix].get("retained_frac_sum", 0.0) + retained_sum
                stats[col_prefix]["retained_frac_n"] = stats[col_prefix].get("retained_frac_n", 0) + retained_rows
                stats[col_prefix]["retained_rows"] = stats[col_prefix].get("retained_rows", 0) + retained_rows
        else:
            idxs: List[int] = []
            vals_out: List[float] = []
            missing_count = 0
            oov_count = 0
            tail_fallback_count = 0  # P1-3
            max_idx_seen = 0
            token_rows = 0
            for rid in row_ids:
                tok_pairs = tokens.get((rid, spec.src, field), [])
                if not tok_pairs:
                    missing_id = (
                        feature_map.token_policy.vocab_missing_id
                        if spec.is_vocab_head()
                        else feature_map.token_policy.hash_missing_id
                    )
                    idxs.append(missing_id)
                    if spec.use_value:
                        vals_out.append(1.0)  # P0-3: consistent with multi-hot missing value
                    missing_count += 1
                    max_idx_seen = max(max_idx_seen, missing_id)
                    continue

                token_rows += 1
                value_map = _aggregate_token_values(tok_pairs, VALUE_AGG_FN)
                # pick highest value then lowest freq rank
                best_val = max(value_map.values())
                candidates = [t for t, v in value_map.items() if v == best_val]
                candidates.sort(key=lambda t: (freq_map.get(t, 1_000_000_000), str(t)))
                chosen_tok = candidates[0]
                is_vocab_oov = False
                is_tail_fallback = False
                if spec.encoding == "hybrid" and spec.tail_encoding == "hash":
                    if vocab_map is not None and chosen_tok in vocab_map:
                        idx, is_vocab_oov = map_token_to_index(chosen_tok, spec, feature_map.token_policy, vocab_map)
                    else:
                        idx = map_hybrid_tail(chosen_tok, spec, feature_map.token_policy)
                        is_tail_fallback = True  # P1-3
                else:
                    idx, is_vocab_oov = map_token_to_index(chosen_tok, spec, feature_map.token_policy, vocab_map)
                # P0-2: only count OOV for vocab/hybrid-head
                if is_vocab_oov:
                    oov_count += 1
                if is_tail_fallback:
                    tail_fallback_count += 1
                if idx >= total_slots:
                    raise AssertionError(f"{col_prefix} idx {idx} exceeds total embeddings {total_slots}")
                idxs.append(int(idx))
                if spec.use_value:
                    vals_out.append(float(best_val))
                max_idx_seen = max(max_idx_seen, idx)

            columns[f"{col_prefix}_idx"] = pa.array(idxs, type=pa.int64())
            if spec.use_value:
                columns[f"{col_prefix}_val"] = pa.array(vals_out, type=pa.float32())
            stats[col_prefix]["rows"] += rows
            stats[col_prefix]["token_rows"] += token_rows
            stats[col_prefix]["missing"] += missing_count
            stats[col_prefix]["oov"] += oov_count
            stats[col_prefix]["tail_fallback"] += tail_fallback_count  # P1-3
            stats[col_prefix]["total_tokens"] += total_tokens_counts.get((spec.src, field), 0)  # P1-4
            stats[col_prefix]["nan_inf_tokens"] += nan_inf_counts.get((spec.src, field), 0)  # P1-4
            stats[col_prefix]["max_idx"] = max(stats[col_prefix].get("max_idx", 0), max_idx_seen)

    table = pa.Table.from_pydict(columns)
    return table, stats


# ------------- split processing -------------
def _build_part_index(tokens_dir: Path, split_dir: Path) -> List[Tuple[int, int, Path, bool]]:
    manifest_ranges = _load_manifest_ranges(tokens_dir, split_dir)
    index: List[Tuple[int, int, Path, bool]] = []
    for p in sorted(tokens_dir.glob("*.parquet")):
        pf = pq.ParquetFile(p)
        has_src = "src" in pf.schema_arrow.names
        lo, hi = _parse_part_interval(p, manifest_ranges)
        index.append((lo, hi, p, has_src))
    return index


def _write_table_chunk(out_dir: Path, split: str, part_id: int, table: pa.Table) -> None:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    fname = split_dir / f"part-{part_id:05d}.parquet"
    pq.write_table(table, fname, compression="snappy")


def _process_split(
    samples_path: Path,
    tokens_dir: Path,
    split_name: str,
    feature_map: FeatureMap,
    freq_rank: Dict[FeatureKey, Dict[str, int]],
    vocab_table: Dict[FeatureKey, Dict[str, int]],
    out_root: Path,
    batch_size: int,
    split_dir: Path,
    strategy_log: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    pf = pq.ParquetFile(samples_path)
    total_rows = pf.metadata.num_rows if pf.metadata else 0
    part_index = _build_part_index(tokens_dir, split_dir)
    logger.info("[%s] Starting: %d samples, %d token parts, batch_size=%d", split_name, total_rows, len(part_index), batch_size)
    rows_written = 0
    part_id = 0
    aggregate_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for batch in pf.iter_batches(batch_size=batch_size):
        logger.info("[%s] Batch %d: rows %d-%d / %d (%.1f%%)", 
                    split_name, part_id, rows_written, rows_written + batch.num_rows, total_rows,
                    100 * (rows_written + batch.num_rows) / max(1, total_rows))
        table, stats = _process_batch(batch, part_index, tokens_dir, feature_map, freq_rank, vocab_table, strategy_log)
        _write_table_chunk(out_root, split_name, part_id, table)
        part_id += 1
        rows_written += batch.num_rows
        for k, v in stats.items():
            for sk, sv in v.items():
                aggregate_stats[k][sk] += sv

    expected_rows = pf.metadata.num_rows if pf.metadata else rows_written
    if rows_written != expected_rows:
        raise ValueError(f"Join mismatch for {split_name}: expected {expected_rows} rows, wrote {rows_written}")

    # Finalise per-field summaries (strategy, nan/inf, retained frac)
    for base, sdict in aggregate_stats.items():
        if base == "_global":
            continue
        token_rows = sdict.get("token_rows", sdict.get("rows", 0))
        # P1-4: proper nan/inf rate with total_tokens as denominator
        total_tokens = sdict.get("total_tokens", 0)
        nan_inf_tokens = sdict.get("nan_inf_tokens", 0)
        sdict["nan_inf_rate"] = nan_inf_tokens / max(1, total_tokens) if total_tokens > 0 else 0.0

        # P1-3: log tail_fallback for hybrid fields
        tail_fallback = sdict.get("tail_fallback", 0)
        if tail_fallback > 0:
            sdict["tail_fallback_rate"] = tail_fallback / max(1, total_tokens) if total_tokens > 0 else 0.0

        strategy_counts = {k: v for k, v in sdict.items() if k.startswith("strategy_cnt_")}
        top_strategy = None
        if strategy_counts:
            # bugfix: compute strategy share rather than storing raw name
            for name, cnt in strategy_counts.items():
                sdict[f"{name}_rate"] = cnt / max(1, token_rows)
            top_strategy = max(strategy_counts.items(), key=lambda kv: kv[1])[0].replace("strategy_cnt_", "")

        retained_den = int(sdict.get("retained_frac_n", sdict.get("retained_rows", 0)))
        if retained_den > 0:
            avg_retained = sdict.get("retained_frac_sum", 0.0) / max(1, retained_den)
            strategy_log.setdefault(base, {})
            strategy_log[base][split_name] = avg_retained
            if top_strategy:
                top_rate = strategy_counts.get(f"strategy_cnt_{top_strategy}", 0) / max(1, token_rows)
                logger.info("[%s] %s token_select=%s retained_token_frac=%.4f strategy_share=%.4f tail_fallback=%d", split_name, base, top_strategy, avg_retained, top_rate, tail_fallback)
            else:
                logger.info("[%s] %s retained_token_frac=%.4f tail_fallback=%d", split_name, base, avg_retained, tail_fallback)

    logger.info("[%s] wrote %d rows across %d parts -> %s", split_name, rows_written, part_id, out_root / split_name)
    aggregate_stats["_global"]["rows"] = rows_written
    return aggregate_stats


# ------------- metadata helpers -------------
def _git_commit() -> str | None:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path("."))
            .decode("utf-8")
            .strip()
        )
    except Exception:  # pragma: no cover
        return None


def _feature_meta(feature_map: FeatureMap) -> Dict[str, Dict[str, bool]]:
    out: Dict[str, Dict[str, bool]] = {}
    for spec in feature_map.features:
        base = f"f{spec.src}_{spec.field}"
        out[base] = {"is_multi_hot": bool(spec.is_multi_hot), "use_value": bool(spec.use_value)}
    return out


def _total_embeddings_map(feature_map: FeatureMap) -> Dict[str, int]:
    return {f"f{f.src}_{f.field}": f.total_num_embeddings() for f in feature_map.features}


# ------------- public API -------------
def build_processed_dataset(
    config_path: str | Path = DEFAULT_FEATUREMAP_PATH,
    split_dir: str | Path = DEFAULT_SPLIT_DIR,
    out_root: str | Path = DEFAULT_PROCESSED_DIR,
    batch_size: int = CHUNK_ROWS,
    eda_stats_dir: str | Path = DEFAULT_EDA_STATS,
) -> Dict[str, object]:
    split_dir = Path(split_dir)
    out_root = Path(out_root)
    eda_stats_dir = Path(eda_stats_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    feature_map = load_featuremap(config_path)
    freq_rank, vocab_table = build_freq_and_vocab(split_dir / "tokens" / "train_tokens", feature_map, eda_stats_dir)

    strategy_log: Dict[str, Dict[str, float]] = {}
    train_stats = _process_split(
        samples_path=split_dir / "samples_train.parquet",
        tokens_dir=split_dir / "tokens" / "train_tokens",
        split_name="train",
        feature_map=feature_map,
        freq_rank=freq_rank,
        vocab_table=vocab_table,
        out_root=out_root,
        batch_size=batch_size,
        split_dir=split_dir,
        strategy_log=strategy_log,
    )

    valid_stats = _process_split(
        samples_path=split_dir / "samples_valid.parquet",
        tokens_dir=split_dir / "tokens" / "valid_tokens",
        split_name="valid",
        feature_map=feature_map,
        freq_rank=freq_rank,
        vocab_table=vocab_table,  # valid never updates vocab
        out_root=out_root,
        batch_size=batch_size,
        split_dir=split_dir,
        strategy_log=strategy_log,
    )

    # metadata
    metadata = {
        "featuremap_hash": featuremap_hash(feature_map),
        "schema_version": feature_map.raw.get("schema_version"),
        "split_spec": json.load(open(split_dir / "split_spec.json", "r", encoding="utf-8")) if (split_dir / "split_spec.json").exists() else None,
        "split_stats": json.load(open(split_dir / "split_stats.json", "r", encoding="utf-8")) if (split_dir / "split_stats.json").exists() else None,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(),
        "rows": {
            "train": train_stats.get("_global", {}).get("rows", 0),
            "valid": valid_stats.get("_global", {}).get("rows", 0),
        },
        "field_stats": {"train": train_stats, "valid": valid_stats},
        "total_num_embeddings": _total_embeddings_map(feature_map),
        "feature_meta": _feature_meta(feature_map),
        "token_select_retained_frac": strategy_log,
    }
    with open(out_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # success marker
    (out_root / "_SUCCESS").write_text("", encoding="utf-8")
    logger.info("Processed dataset built at %s", out_root)
    return {"processed_root": str(out_root), "metadata": metadata}


# ------------- CLI -------------
def _parse_args():
    parser = argparse.ArgumentParser(description="Build processed dataset from split + tokens.")
    parser.add_argument("--config", dest="config_path", default=DEFAULT_FEATUREMAP_PATH, help="Path to featuremap.yaml")
    parser.add_argument("--split_dir", dest="split_dir", default=DEFAULT_SPLIT_DIR, help="Split directory root")
    parser.add_argument("--out", dest="out_root", default=DEFAULT_PROCESSED_DIR, help="Output processed root")
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=CHUNK_ROWS, help="Rows per output part")
    parser.add_argument("--eda_stats", dest="eda_stats_dir", default=DEFAULT_EDA_STATS, help="EDA stats directory")
    return parser.parse_args()


def main():  # pragma: no cover - small wrapper
    args = _parse_args()
    build_processed_dataset(
        config_path=args.config_path,
        split_dir=args.split_dir,
        out_root=args.out_root,
        batch_size=args.batch_size,
        eda_stats_dir=args.eda_stats_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_processed_dataset"]
