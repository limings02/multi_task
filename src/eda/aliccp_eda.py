from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.core.logging import get_logger
from src.core.paths import find_repo_root, resolve_path
from src.data.canonical import load_aliccp_config

try:  # Optional fast path
    import duckdb  # type: ignore

    DUCKDB_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when duckdb missing
    duckdb = None
    DUCKDB_AVAILABLE = False


ROW_NNZ_BINS = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

logger = get_logger(__name__)


def _select_backend(preference: str) -> str:
    pref = (preference or "auto").lower()
    if pref == "duckdb":
        if DUCKDB_AVAILABLE:
            return "duckdb"
        logger.warning("duckdb requested but not available; falling back to pyarrow backend.")
        return "pyarrow"
    if pref == "pyarrow":
        return "pyarrow"
    return "duckdb" if DUCKDB_AVAILABLE else "pyarrow"


def _path_posix(p: Path) -> str:
    return str(p.as_posix())


def _load_json_file(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        logger.info("Skip writing %s (exists, overwrite=False)", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_table(table: pa.Table, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        logger.info("Skip writing %s (exists, overwrite=False)", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _count_parquet_rows_in_dir(directory: Path) -> int:
    total = 0
    for part in sorted(directory.glob("*.parquet")):
        total += pq.ParquetFile(part).metadata.num_rows
    return total


def _build_bin_case_expr(bins: List[int]) -> str:
    parts = []
    for idx, left in enumerate(bins):
        if idx < len(bins) - 1:
            right = bins[idx + 1]
            parts.append(f"WHEN nnz >= {left} AND nnz < {right} THEN {idx}")
        else:
            parts.append(f"WHEN nnz >= {left} THEN {idx}")
    return "CASE " + " ".join(parts) + " END"


def _bin_index(value: int, bins: List[int]) -> int:
    for idx, left in enumerate(bins):
        if idx == len(bins) - 1:
            return idx
        right = bins[idx + 1]
        if left <= value < right:
            return idx
    return len(bins) - 1


def _entity_overlap_count(
    train_freq_path: Path,
    valid_freq_path: Path,
    backend: str,
    conn=None,
    train_entities: Optional[Set[str]] = None,
    valid_entities: Optional[Set[str]] = None,
) -> int:
    if backend == "duckdb" and conn is not None and train_freq_path.exists() and valid_freq_path.exists():
        sql = (
            "SELECT COUNT(*) FROM read_parquet('{train}') t "
            "INNER JOIN read_parquet('{valid}') v USING(entity_id)"
        ).format(train=_path_posix(train_freq_path), valid=_path_posix(valid_freq_path))
        return int(conn.execute(sql).fetchone()[0])
    if train_entities is None or valid_entities is None:
        logger.warning("Entity sets not available; treating overlap as 0.")
        return 0
    return len(train_entities & valid_entities)


def _merge_topk_rows(
    counts: Dict[Tuple[str, int, str], int],
    token_rows_map: Dict[Tuple[str, int], int],
    topk_n: int,
) -> pa.Table:
    grouped: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for (field, src, fid), cnt in counts.items():
        grouped.setdefault((field, src), []).append((fid, cnt))
    rows = {"field": [], "src": [], "fid": [], "cnt": [], "topk_coverage": []}
    for (field, src), lst in grouped.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        token_rows = token_rows_map.get((field, src), 0)
        for fid, cnt in lst[:topk_n]:
            rows["field"].append(field)
            rows["src"].append(np.int8(src))
            rows["fid"].append(fid)
            rows["cnt"].append(np.int64(cnt))
            rows["topk_coverage"].append(float(cnt) / token_rows if token_rows else 0.0)
    return pa.Table.from_pydict(rows)


def _samples_profile_duckdb(
    conn,
    samples_path: Path,
    entity_freq_path: Path,
) -> Dict[str, object]:
    spath = _path_posix(samples_path)
    freq_path = _path_posix(entity_freq_path)
    entity_freq_path.parent.mkdir(parents=True, exist_ok=True)

    conn.execute(
        f"COPY (SELECT entity_id, COUNT(*) AS cnt FROM read_parquet('{spath}') GROUP BY entity_id)"
        f" TO '{freq_path}' (FORMAT PARQUET)"
    )
    rows, y1_sum, y2_sum, funnel_bad = conn.execute(
        f"""
        SELECT
            COUNT(*) AS rows,
            SUM(y1) AS y1_sum,
            SUM(y2) AS y2_sum,
            SUM(CASE WHEN y2 = 1 AND y1 = 0 THEN 1 ELSE 0 END) AS funnel_bad
        FROM read_parquet('{spath}')
        """
    ).fetchone()

    unique_entity, total_rows, p50, p90, p99, top_sum = conn.execute(
        f"""
        WITH freq AS (SELECT * FROM read_parquet('{freq_path}')),
        totals AS (
            SELECT
                COUNT(*) AS unique_entity,
                SUM(cnt) AS total_rows,
                approx_quantile(cnt, 0.5) AS p50,
                approx_quantile(cnt, 0.9) AS p90,
                approx_quantile(cnt, 0.99) AS p99
            FROM freq
        ),
        ranked AS (
            SELECT
                cnt,
                ROW_NUMBER() OVER (ORDER BY cnt DESC) AS rn,
                (SELECT unique_entity FROM totals) AS total_entities
            FROM freq
        ),
        top1 AS (
            SELECT SUM(cnt) AS top_sum
            FROM ranked
            WHERE rn <= GREATEST(1, CAST(total_entities * 0.01 AS BIGINT))
        )
        SELECT
            unique_entity,
            total_rows,
            p50,
            p90,
            p99,
            top_sum
        FROM totals, top1
        """
    ).fetchone()
    top_coverage = (top_sum or 0) / total_rows if total_rows else 0.0
    ctr = float(y1_sum) / rows if rows else 0.0
    cvr = float(y2_sum) / rows if rows else 0.0
    return {
        "rows": int(rows),
        "unique_entity": int(unique_entity or 0),
        "ctr": ctr,
        "cvr": cvr,
        "funnel_bad": int(funnel_bad or 0),
        "entity_freq_path": entity_freq_path,
        "entity_freq_entities": None,
        "entity_freq_quantiles": {
            "p50": float(p50 or 0),
            "p90": float(p90 or 0),
            "p99": float(p99 or 0),
        },
        "top1pct_coverage": float(top_coverage),
    }


def _samples_profile_pyarrow(
    samples_path: Path,
    entity_freq_path: Path,
    batch_size: int,
    overwrite: bool,
) -> Dict[str, object]:
    pf = pq.ParquetFile(samples_path)
    counts: Dict[str, int] = {}
    rows = 0
    y1_sum = 0
    y2_sum = 0
    funnel_bad = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        tbl = pa.Table.from_batches([batch])
        rows += tbl.num_rows
        y1_sum += pc.sum(tbl.column("y1")).as_py()
        y2_sum += pc.sum(tbl.column("y2")).as_py()
        funnel_bad += pc.sum(
            pc.and_(
                pc.equal(tbl.column("y2"), 1),
                pc.equal(tbl.column("y1"), 0),
            )
        ).as_py()
        for eid in tbl.column("entity_id").to_pylist():
            counts[eid] = counts.get(eid, 0) + 1

    freq_rows = {"entity_id": [], "cnt": [], "coverage": []}
    for eid, cnt in counts.items():
        freq_rows["entity_id"].append(eid)
        freq_rows["cnt"].append(np.int64(cnt))
        freq_rows["coverage"].append(float(cnt) / rows if rows else 0.0)
    freq_table = pa.Table.from_pydict(freq_rows)
    _write_table(freq_table, entity_freq_path, overwrite=overwrite)

    freq_vals = np.array(list(counts.values()), dtype=np.int64)
    unique_entity = len(counts)
    if freq_vals.size:
        p50 = float(np.percentile(freq_vals, 50))
        p90 = float(np.percentile(freq_vals, 90))
        p99 = float(np.percentile(freq_vals, 99))
        topk = max(1, math.ceil(unique_entity * 0.01))
        top_coverage = float(np.sort(freq_vals)[::-1][:topk].sum()) / rows if rows else 0.0
    else:
        p50 = p90 = p99 = 0.0
        top_coverage = 0.0

    ctr = float(y1_sum) / rows if rows else 0.0
    cvr = float(y2_sum) / rows if rows else 0.0
    return {
        "rows": int(rows),
        "unique_entity": int(unique_entity),
        "ctr": ctr,
        "cvr": cvr,
        "funnel_bad": int(funnel_bad),
        "entity_freq_path": entity_freq_path,
        "entity_freq_entities": set(counts.keys()),
        "entity_freq_quantiles": {
            "p50": float(p50),
            "p90": float(p90),
            "p99": float(p99),
        },
        "top1pct_coverage": float(top_coverage),
    }


def compute_samples_profile(
    backend: str,
    samples_path: Path,
    entity_freq_path: Path,
    batch_size: int,
    overwrite: bool,
    conn=None,
) -> Dict[str, object]:
    logger.info("Profiling samples: %s backend=%s", samples_path, backend)
    if backend == "duckdb" and conn is not None:
        return _samples_profile_duckdb(conn, samples_path, entity_freq_path)
    return _samples_profile_pyarrow(samples_path, entity_freq_path, batch_size, overwrite)


def _row_nnz_hist_duckdb(
    conn,
    tokens_glob: str,
    bins: List[int],
    out_path: Path,
    overwrite: bool,
) -> Tuple[Dict[str, float], pa.Table]:
    conn.execute(
        f"CREATE OR REPLACE TEMP VIEW row_nnz AS "
        f"SELECT row_id, COUNT(*) AS nnz FROM read_parquet('{tokens_glob}') GROUP BY row_id"
    )
    total_rows, p50, p90, p99 = conn.execute(
        "SELECT COUNT(*) AS rows, approx_quantile(nnz, 0.5), approx_quantile(nnz, 0.9), approx_quantile(nnz, 0.99) FROM row_nnz"
    ).fetchone()
    case_expr = _build_bin_case_expr(bins)
    hist_rows = conn.execute(
        f"SELECT {case_expr} AS bin_idx, COUNT(*) AS cnt FROM row_nnz GROUP BY bin_idx"
    ).fetchall()

    counts = [0 for _ in bins]
    for idx, cnt in hist_rows:
        if idx is None:
            continue
        counts[int(idx)] = int(cnt)
    ratios = [float(c) / total_rows if total_rows else 0.0 for c in counts]
    bin_left = [float(x) for x in bins]
    bin_right = [float(bins[i + 1]) if i < len(bins) - 1 else float("inf") for i in range(len(bins))]
    table = pa.Table.from_pydict(
        {
            "bin_left": bin_left,
            "bin_right": bin_right,
            "count_rows": counts,
            "ratio_rows": ratios,
        }
    )
    _write_table(table, out_path, overwrite=overwrite)
    return (
        {"p50": float(p50 or 0), "p90": float(p90 or 0), "p99": float(p99 or 0)},
        table,
    )


def _row_nnz_hist_pyarrow(
    tokens_dir: Path,
    bins: List[int],
    batch_size: int,
    out_path: Path,
    overwrite: bool,
) -> Tuple[Dict[str, float], pa.Table]:
    counts = [0 for _ in bins]
    nnz_values: List[int] = []
    total_rows = 0
    for part in sorted(tokens_dir.glob("*.parquet")):
        pf = pq.ParquetFile(part)
        row_counts: Dict[int, int] = {}
        for batch in pf.iter_batches(batch_size=batch_size, columns=["row_id"]):
            arr = batch.column(0).to_numpy(zero_copy_only=False)
            for rid in arr:
                rid_int = int(rid)
                row_counts[rid_int] = row_counts.get(rid_int, 0) + 1
        for nnz in row_counts.values():
            idx = _bin_index(nnz, bins)
            counts[idx] += 1
            total_rows += 1
            nnz_values.append(nnz)

    ratios = [float(c) / total_rows if total_rows else 0.0 for c in counts]
    bin_left = [float(x) for x in bins]
    bin_right = [float(bins[i + 1]) if i < len(bins) - 1 else float("inf") for i in range(len(bins))]
    table = pa.Table.from_pydict(
        {
            "bin_left": bin_left,
            "bin_right": bin_right,
            "count_rows": counts,
            "ratio_rows": ratios,
        }
    )
    _write_table(table, out_path, overwrite=overwrite)
    if nnz_values:
        quantiles = {
            "p50": float(np.percentile(nnz_values, 50)),
            "p90": float(np.percentile(nnz_values, 90)),
            "p99": float(np.percentile(nnz_values, 99)),
        }
    else:
        quantiles = {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    return quantiles, table


def compute_row_nnz_hist(
    backend: str,
    tokens_dir: Path,
    bins: List[int],
    batch_size: int,
    out_path: Path,
    overwrite: bool,
    conn=None,
) -> Dict[str, float]:
    logger.info("Computing row nnz histogram for %s", tokens_dir)
    if backend == "duckdb" and conn is not None:
        quantiles, _ = _row_nnz_hist_duckdb(conn, _path_posix(tokens_dir / "*.parquet"), bins, out_path, overwrite)
        return quantiles
    quantiles, _ = _row_nnz_hist_pyarrow(tokens_dir, bins, batch_size, out_path, overwrite)
    return quantiles


def _field_stats_duckdb(
    conn,
    tokens_glob: str,
    n_samples: int,
    field_stats_path: Path,
    field_topk_path: Path,
    topk_n: int,
    overwrite: bool,
) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int], Set[str]]]:
    conn.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT row_id, src, field, fid, val FROM read_parquet('{tokens_glob}')
          ),
          agg AS (
            SELECT
              field,
              src,
              COUNT(*) AS token_rows,
              COUNT(DISTINCT row_id) AS row_coverage,
              approx_count_distinct(fid) AS fid_distinct_est,
              SUM(CASE WHEN val != 1 THEN 1 ELSE 0 END) AS val_non1_cnt,
              MIN(val) AS val_min,
              MAX(val) AS val_max,
              AVG(val) AS val_mean,
              -- approx_count_distinct to avoid massive temp spill from exact distinct on large struct_pack(row_id,fid)
              approx_count_distinct(hash(struct_pack(row_id, fid))) AS distinct_row_fid_est
            FROM base
            GROUP BY field, src
          )
          SELECT
            field,
            src,
            token_rows,
            row_coverage,
            CAST(row_coverage AS DOUBLE) / {n_samples} AS coverage_rate,
            token_rows / NULLIF(row_coverage, 0) AS avg_nnz_per_covered_row,
            fid_distinct_est,
            CAST(val_non1_cnt AS DOUBLE) / token_rows AS val_non1_ratio,
            val_min,
            val_max,
            val_mean,
            1 - (CAST(distinct_row_fid_est AS DOUBLE) / token_rows) AS dup_rate_est
          FROM agg
        ) TO '{_path_posix(field_stats_path)}' (FORMAT PARQUET)
        """
    )
    conn.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT row_id, src, field, fid FROM read_parquet('{tokens_glob}')
          ),
          totals AS (
            SELECT field, src, COUNT(*) AS token_rows FROM base GROUP BY field, src
          ),
          ranked AS (
            SELECT
              b.field,
              b.src,
              b.fid,
              COUNT(*) AS cnt,
              ROW_NUMBER() OVER (PARTITION BY b.field, b.src ORDER BY COUNT(*) DESC) AS rn
            FROM base b
            GROUP BY b.field, b.src, b.fid
          )
          SELECT
            r.field,
            r.src,
            r.fid,
            r.cnt,
            r.cnt / NULLIF(t.token_rows, 0) AS topk_coverage
          FROM ranked r
          JOIN totals t USING(field, src)
          WHERE rn <= {topk_n}
          ORDER BY field, src, rn
        ) TO '{_path_posix(field_topk_path)}' (FORMAT PARQUET)
        """
    )

    coverage_map: Dict[Tuple[str, int], int] = {}
    for field, src, row_cov in conn.execute(
        f"SELECT field, src, row_coverage FROM read_parquet('{_path_posix(field_stats_path)}')"
    ).fetchall():
        coverage_map[(field, int(src))] = int(row_cov)

    topk_sets: Dict[Tuple[str, int], Set[str]] = {}
    for field, src, fid in conn.execute(
        f"SELECT field, src, fid FROM read_parquet('{_path_posix(field_topk_path)}')"
    ).fetchall():
        topk_sets.setdefault((field, int(src)), set()).add(fid)
    return coverage_map, topk_sets


def _field_stats_pyarrow(
    tokens_dir: Path,
    n_samples: int,
    batch_size: int,
    field_stats_path: Path,
    field_topk_path: Path,
    topk_n: int,
    overwrite: bool,
) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int], Set[str]]]:
    stats: Dict[Tuple[str, int], Dict[str, object]] = {}
    fid_counts: Dict[Tuple[str, int, str], int] = {}
    for part in sorted(tokens_dir.glob("*.parquet")):
        pf = pq.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=batch_size):
            row_ids = batch.column("row_id").to_numpy(zero_copy_only=False)
            srcs = batch.column("src").to_numpy(zero_copy_only=False)
            fields = batch.column("field").to_pylist()
            fids = batch.column("fid").to_pylist()
            vals = batch.column("val").to_numpy(zero_copy_only=False)
            for i in range(len(row_ids)):
                key = (fields[i], int(srcs[i]))
                st = stats.setdefault(
                    key,
                    {
                        "token_rows": 0,
                        "row_ids": set(),
                        "fid_set": set(),
                        "row_fid_set": set(),
                        "val_non1": 0,
                        "val_min": None,
                        "val_max": None,
                        "val_sum": 0.0,
                    },
                )
                st["token_rows"] += 1
                st["row_ids"].add(int(row_ids[i]))
                st["fid_set"].add(fids[i])
                st["row_fid_set"].add((int(row_ids[i]), fids[i]))
                val = float(vals[i])
                st["val_sum"] += val
                st["val_min"] = val if st["val_min"] is None else min(st["val_min"], val)
                st["val_max"] = val if st["val_max"] is None else max(st["val_max"], val)
                if val != 1.0:
                    st["val_non1"] += 1
                fid_key = (fields[i], int(srcs[i]), fids[i])
                fid_counts[fid_key] = fid_counts.get(fid_key, 0) + 1

    rows = {
        "field": [],
        "src": [],
        "token_rows": [],
        "row_coverage": [],
        "coverage_rate": [],
        "avg_nnz_per_covered_row": [],
        "fid_distinct_est": [],
        "val_non1_ratio": [],
        "val_min": [],
        "val_max": [],
        "val_mean": [],
        "dup_rate": [],
    }
    coverage_map: Dict[Tuple[str, int], int] = {}
    token_rows_map: Dict[Tuple[str, int], int] = {}
    for (field, src), st in stats.items():
        token_rows = int(st["token_rows"])
        row_cov = len(st["row_ids"])
        coverage_rate = float(row_cov) / n_samples if n_samples else 0.0
        avg_nnz = float(token_rows) / row_cov if row_cov else 0.0
        fid_distinct = len(st["fid_set"])
        val_non1_ratio = float(st["val_non1"]) / token_rows if token_rows else 0.0
        val_mean = float(st["val_sum"]) / token_rows if token_rows else 0.0
        dup_rate = 1.0 - (len(st["row_fid_set"]) / token_rows) if token_rows else 0.0

        rows["field"].append(field)
        rows["src"].append(np.int8(src))
        rows["token_rows"].append(np.int64(token_rows))
        rows["row_coverage"].append(np.int64(row_cov))
        rows["coverage_rate"].append(coverage_rate)
        rows["avg_nnz_per_covered_row"].append(avg_nnz)
        rows["fid_distinct_est"].append(np.int64(fid_distinct))
        rows["val_non1_ratio"].append(val_non1_ratio)
        rows["val_min"].append(st["val_min"] if st["val_min"] is not None else 0.0)
        rows["val_max"].append(st["val_max"] if st["val_max"] is not None else 0.0)
        rows["val_mean"].append(val_mean)
        rows["dup_rate"].append(dup_rate)
        coverage_map[(field, src)] = row_cov
        token_rows_map[(field, src)] = token_rows

    table = pa.Table.from_pydict(rows)
    _write_table(table, field_stats_path, overwrite=overwrite)
    topk_table = _merge_topk_rows(fid_counts, token_rows_map, topk_n)
    _write_table(topk_table, field_topk_path, overwrite=overwrite)

    topk_sets: Dict[Tuple[str, int], Set[str]] = {}
    for field, src, fid in zip(
        topk_table.column("field").to_pylist(),
        topk_table.column("src").to_numpy(zero_copy_only=False),
        topk_table.column("fid").to_pylist(),
    ):
        topk_sets.setdefault((field, int(src)), set()).add(fid)
    return coverage_map, topk_sets


def compute_field_stats(
    backend: str,
    tokens_dir: Path,
    n_samples: int,
    batch_size: int,
    field_stats_path: Path,
    field_topk_path: Path,
    topk_n: int,
    overwrite: bool,
    conn=None,
) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int], Set[str]]]:
    logger.info("Computing field stats/topk from %s", tokens_dir)
    tokens_glob = _path_posix(tokens_dir / "*.parquet")
    if backend == "duckdb" and conn is not None:
        return _field_stats_duckdb(conn, tokens_glob, n_samples, field_stats_path, field_topk_path, topk_n, overwrite)
    return _field_stats_pyarrow(tokens_dir, n_samples, batch_size, field_stats_path, field_topk_path, topk_n, overwrite)


def _fid_lift_duckdb(
    conn,
    tokens_glob: str,
    samples_path: Path,
    topk_path: Path,
    out_path: Path,
    min_support: int,
    overall_ctr: float,
    overall_cvr: float,
    overwrite: bool,
) -> None:
    conn.execute(
        f"""
        COPY (
          WITH topk AS (SELECT field, src, fid FROM read_parquet('{_path_posix(topk_path)}')),
          base AS (
            SELECT DISTINCT
              t.row_id,
              t.field,
              t.src,
              t.fid,
              s.y1,
              s.y2
            FROM read_parquet('{tokens_glob}') t
            JOIN topk k ON t.field = k.field AND t.src = k.src AND t.fid = k.fid
            JOIN read_parquet('{_path_posix(samples_path)}') s ON t.row_id = s.row_id
          ),
          agg AS (
            SELECT
              field,
              src,
              fid,
              COUNT(*) AS cnt_rows,
              AVG(y1) AS ctr_fid,
              AVG(y2) AS cvr_fid
            FROM base
            GROUP BY field, src, fid
            HAVING cnt_rows >= {min_support}
          )
          SELECT
            field,
            src,
            fid,
            cnt_rows,
            ctr_fid,
            cvr_fid,
            CASE WHEN {overall_ctr} = 0 THEN 0 ELSE ctr_fid / {overall_ctr} END AS lift_ctr,
            CASE WHEN {overall_cvr} = 0 THEN 0 ELSE cvr_fid / {overall_cvr} END AS lift_cvr
          FROM agg
        ) TO '{_path_posix(out_path)}' (FORMAT PARQUET)
        """
    )


def _fid_lift_pyarrow(
    tokens_dir: Path,
    samples_path: Path,
    topk_path: Path,
    out_path: Path,
    batch_size: int,
    min_support: int,
    overall_ctr: float,
    overall_cvr: float,
    overwrite: bool,
) -> None:
    labels: Dict[int, Tuple[int, int]] = {}
    pf_samples = pq.ParquetFile(samples_path)
    for batch in pf_samples.iter_batches(batch_size=batch_size, columns=["row_id", "y1", "y2"]):
        row_ids = batch.column("row_id").to_numpy(zero_copy_only=False)
        y1s = batch.column("y1").to_numpy(zero_copy_only=False)
        y2s = batch.column("y2").to_numpy(zero_copy_only=False)
        for rid, y1, y2 in zip(row_ids, y1s, y2s):
            labels[int(rid)] = (int(y1), int(y2))

    topk_tbl = pq.read_table(topk_path)
    topk_keys = {
        (field, int(src), fid)
        for field, src, fid in zip(
            topk_tbl.column("field").to_pylist(),
            topk_tbl.column("src").to_numpy(zero_copy_only=False),
            topk_tbl.column("fid").to_pylist(),
        )
    }
    lift_state: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for part in sorted(tokens_dir.glob("*.parquet")):
        pf = pq.ParquetFile(part)
        for batch in pf.iter_batches(batch_size=batch_size):
            row_ids = batch.column("row_id").to_numpy(zero_copy_only=False)
            srcs = batch.column("src").to_numpy(zero_copy_only=False)
            fields = batch.column("field").to_pylist()
            fids = batch.column("fid").to_pylist()
            for rid, src, field, fid in zip(row_ids, srcs, fields, fids):
                key = (field, int(src), fid)
                if key not in topk_keys:
                    continue
                state = lift_state.setdefault(key, {"rows": set(), "sum_y1": 0.0, "sum_y2": 0.0})
                rid_int = int(rid)
                if rid_int in state["rows"]:
                    continue
                lbl = labels.get(rid_int)
                if lbl is None:
                    continue
                state["rows"].add(rid_int)
                state["sum_y1"] += lbl[0]
                state["sum_y2"] += lbl[1]

    rows = {
        "field": [],
        "src": [],
        "fid": [],
        "cnt_rows": [],
        "ctr_fid": [],
        "cvr_fid": [],
        "lift_ctr": [],
        "lift_cvr": [],
    }
    for (field, src, fid), st in lift_state.items():
        cnt_rows = len(st["rows"])
        if cnt_rows < min_support:
            continue
        ctr_fid = float(st["sum_y1"]) / cnt_rows if cnt_rows else 0.0
        cvr_fid = float(st["sum_y2"]) / cnt_rows if cnt_rows else 0.0
        rows["field"].append(field)
        rows["src"].append(np.int8(src))
        rows["fid"].append(fid)
        rows["cnt_rows"].append(np.int64(cnt_rows))
        rows["ctr_fid"].append(ctr_fid)
        rows["cvr_fid"].append(cvr_fid)
        rows["lift_ctr"].append(0.0 if overall_ctr == 0 else ctr_fid / overall_ctr)
        rows["lift_cvr"].append(0.0 if overall_cvr == 0 else cvr_fid / overall_cvr)

    table = pa.Table.from_pydict(rows)
    _write_table(table, out_path, overwrite=overwrite)


def compute_fid_lift(
    backend: str,
    tokens_dir: Path,
    samples_path: Path,
    topk_path: Path,
    out_path: Path,
    batch_size: int,
    min_support: int,
    overall_ctr: float,
    overall_cvr: float,
    overwrite: bool,
    conn=None,
) -> None:
    logger.info("Computing fid lift for %s", tokens_dir)
    tokens_glob = _path_posix(tokens_dir / "*.parquet")
    if backend == "duckdb" and conn is not None:
        _fid_lift_duckdb(
            conn,
            tokens_glob,
            samples_path,
            topk_path,
            out_path,
            min_support,
            overall_ctr,
            overall_cvr,
            overwrite,
        )
    else:
        _fid_lift_pyarrow(
            tokens_dir,
            samples_path,
            topk_path,
            out_path,
            batch_size,
            min_support,
            overall_ctr,
            overall_cvr,
            overwrite,
        )


def _topk_sets_from_parquet(path: Path, limit: int) -> Dict[Tuple[str, int], Set[str]]:
    if not path.exists():
        return {}
    tbl = pq.read_table(path, columns=["field", "src", "fid"])
    sets: Dict[Tuple[str, int], Set[str]] = {}
    for field, src, fid in zip(
        tbl.column("field").to_pylist(),
        tbl.column("src").to_numpy(zero_copy_only=False),
        tbl.column("fid").to_pylist(),
    ):
        key = (field, int(src))
        bucket = sets.setdefault(key, set())
        if len(bucket) < limit:
            bucket.add(fid)
    return sets


def _topk_sets_duckdb(conn, tokens_glob: str, topk_n: int) -> Dict[Tuple[str, int], Set[str]]:
    rows = conn.execute(
        f"""
        WITH base AS (
          SELECT field, src, fid, COUNT(*) AS cnt
          FROM read_parquet('{tokens_glob}')
          GROUP BY field, src, fid
        ),
        ranked AS (
          SELECT
            field,
            src,
            fid,
            ROW_NUMBER() OVER (PARTITION BY field, src ORDER BY cnt DESC) AS rn
          FROM base
        )
        SELECT field, src, fid FROM ranked WHERE rn <= {topk_n}
        """
    ).fetchall()
    sets: Dict[Tuple[str, int], Set[str]] = {}
    for field, src, fid in rows:
        sets.setdefault((field, int(src)), set()).add(fid)
    return sets


def compute_drift_summary(
    backend: str,
    train_tokens_dir: Path,
    valid_tokens_dir: Path,
    n_train: int,
    n_valid: int,
    field_stats_path: Path,
    field_topk_path: Path,
    topk_jaccard_n: int,
    out_path: Path,
    overwrite: bool,
    conn=None,
) -> Dict[str, object]:
    logger.info("Computing drift summary (train vs valid)")
    train_cov: Dict[Tuple[str, int], int] = {}
    train_topk = _topk_sets_from_parquet(field_topk_path, topk_jaccard_n)
    if field_stats_path.exists():
        tbl = pq.read_table(field_stats_path, columns=["field", "src", "row_coverage"])
        for field, src, cov in zip(
            tbl.column("field").to_pylist(),
            tbl.column("src").to_numpy(zero_copy_only=False),
            tbl.column("row_coverage").to_numpy(zero_copy_only=False),
        ):
            train_cov[(field, int(src))] = int(cov)

    valid_cov_map: Dict[Tuple[str, int], int] = {}
    if backend == "duckdb" and conn is not None:
        valid_cov_rows = conn.execute(
            f"SELECT field, src, COUNT(DISTINCT row_id) AS rcov FROM read_parquet('{_path_posix(valid_tokens_dir / '*.parquet')}') GROUP BY field, src"
        ).fetchall()
        for field, src, cov in valid_cov_rows:
            valid_cov_map[(field, int(src))] = int(cov)
        valid_topk_sets = _topk_sets_duckdb(conn, _path_posix(valid_tokens_dir / "*.parquet"), topk_jaccard_n)
    else:
        fid_counts: Dict[Tuple[str, int, str], int] = {}
        cov_sets: Dict[Tuple[str, int], Set[int]] = {}
        for part in sorted(valid_tokens_dir.glob("*.parquet")):
            pf = pq.ParquetFile(part)
            for batch in pf.iter_batches():
                row_ids = batch.column("row_id").to_numpy(zero_copy_only=False)
                srcs = batch.column("src").to_numpy(zero_copy_only=False)
                fields = batch.column("field").to_pylist()
                fids = batch.column("fid").to_pylist()
                for rid, src, field, fid in zip(row_ids, srcs, fields, fids):
                    key_field = (field, int(src))
                    cov_sets.setdefault(key_field, set()).add(int(rid))
                    fid_counts[(field, int(src), fid)] = fid_counts.get((field, int(src), fid), 0) + 1
        valid_cov_map = {k: len(v) for k, v in cov_sets.items()}
        valid_topk_table = _merge_topk_rows(fid_counts, {}, topk_jaccard_n)
        valid_topk_sets: Dict[Tuple[str, int], Set[str]] = {}
        for field, src, fid in zip(
            valid_topk_table.column("field").to_pylist(),
            valid_topk_table.column("src").to_numpy(zero_copy_only=False),
            valid_topk_table.column("fid").to_pylist(),
        ):
            valid_topk_sets.setdefault((field, int(src)), set()).add(fid)

    all_keys = set(train_cov.keys()) | set(valid_cov_map.keys()) | set(train_topk.keys()) | set(valid_topk_sets.keys())
    rows = {
        "field": [],
        "src": [],
        "coverage_train": [],
        "coverage_valid": [],
        "coverage_diff": [],
        "topk_jaccard_100": [],
    }
    for field, src in sorted(all_keys):
        key = (field, src)
        cov_train_rate = float(train_cov.get(key, 0)) / n_train if n_train else 0.0
        cov_valid_rate = float(valid_cov_map.get(key, 0)) / n_valid if n_valid else 0.0
        train_set = train_topk.get(key, set())
        valid_set = valid_topk_sets.get(key, set())
        union = len(train_set | valid_set)
        inter = len(train_set & valid_set)
        jaccard = 1.0 if union == 0 else float(inter) / union
        rows["field"].append(field)
        rows["src"].append(np.int8(src))
        rows["coverage_train"].append(cov_train_rate)
        rows["coverage_valid"].append(cov_valid_rate)
        rows["coverage_diff"].append(cov_train_rate - cov_valid_rate)
        rows["topk_jaccard_100"].append(jaccard)
    table = pa.Table.from_pydict(rows)
    _write_table(table, out_path, overwrite=overwrite)
    return {
        "rows": len(rows["field"]),
        "summary_table": table,
        "data": rows,
    }


def render_drift_report(
    report_path: Path,
    train_profile: Dict[str, object],
    valid_profile: Dict[str, object],
    nnz_quant_train: Dict[str, float],
    nnz_quant_valid: Dict[str, float],
    drift_rows: pa.Table,
    top_k: int = 20,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def _top_by_abs(field_name: str, limit: int) -> List[Tuple[str, int, float, float, float]]:
        rows = []
        for field, src, cov_train, cov_valid, diff in zip(
            drift_rows.column("field").to_pylist(),
            drift_rows.column("src").to_numpy(zero_copy_only=False),
            drift_rows.column("coverage_train").to_numpy(zero_copy_only=False),
            drift_rows.column("coverage_valid").to_numpy(zero_copy_only=False),
            drift_rows.column(field_name).to_numpy(zero_copy_only=False),
        ):
            rows.append((field, int(src), float(cov_train), float(cov_valid), float(diff)))
        rows.sort(key=lambda x: abs(x[4]), reverse=True)
        return rows[:limit]

    def _bottom_jaccard(limit: int) -> List[Tuple[str, int, float]]:
        rows = []
        for field, src, jac in zip(
            drift_rows.column("field").to_pylist(),
            drift_rows.column("src").to_numpy(zero_copy_only=False),
            drift_rows.column("topk_jaccard_100").to_numpy(zero_copy_only=False),
        ):
            rows.append((field, int(src), float(jac)))
        rows.sort(key=lambda x: x[2])
        return rows[:limit]

    coverage_rows = _top_by_abs("coverage_diff", top_k)
    jaccard_rows = _bottom_jaccard(top_k)

    lines = []
    lines.append("# Drift Report (Ali-CCP)")
    lines.append("")
    lines.append("## Overall")
    lines.append(
        f"- CTR: train={train_profile['ctr']:.4f} valid={valid_profile['ctr']:.4f}; "
        f"CVR: train={train_profile['cvr']:.4f} valid={valid_profile['cvr']:.4f}"
    )
    lines.append(
        f"- Row NNZ P99: train={nnz_quant_train.get('p99', 0):.2f} valid={nnz_quant_valid.get('p99', 0):.2f}"
    )
    lines.append("")
    lines.append(f"## Coverage gap (top {top_k})")
    lines.append("| field | src | coverage_train | coverage_valid | diff |")
    lines.append("| --- | --- | --- | --- | --- |")
    for field, src, cov_t, cov_v, diff in coverage_rows:
        lines.append(
            f"| {field} | {src} | {cov_t:.4f} | {cov_v:.4f} | {diff:+.4f} |"
        )
    lines.append("")
    lines.append(f"## TopK drift (lowest Jaccard, top {top_k})")
    lines.append("| field | src | topk_jaccard_100 |")
    lines.append("| --- | --- | --- |")
    for field, src, jac in jaccard_rows:
        lines.append(f"| {field} | {src} | {jac:.4f} |")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _sanitize_backend_config(config: Dict[str, object]) -> Dict[str, object]:
    eda_cfg = config.get("eda", {})
    eda_cfg.setdefault("split_dir", "data/splits/aliccp_entity_hash_v1")
    eda_cfg.setdefault("out_dir", "data/splits/aliccp_entity_hash_v1/eda")
    eda_cfg.setdefault("report_dir", "reports/aliccp_entity_hash_v1")
    eda_cfg.setdefault("overwrite", False)
    eda_cfg.setdefault("backend", "auto")
    eda_cfg.setdefault("batch_size", 500_000)
    eda_cfg.setdefault("topk_n", 200)
    eda_cfg.setdefault("min_support", 1000)
    eda_cfg.setdefault("topk_jaccard_n", 100)
    config["eda"] = eda_cfg
    return config


def run_eda_aliccp(config_path: str, overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    config_path = str(Path(config_path).resolve())
    repo_root = find_repo_root(Path(config_path).parent)
    config = load_aliccp_config(config_path, overrides)
    config = _sanitize_backend_config(config)
    eda_cfg = config.get("eda", {})
    canonical_cfg = config.get("canonical", {})

    split_dir = resolve_path(repo_root, eda_cfg.get("split_dir"))
    out_dir = resolve_path(repo_root, eda_cfg.get("out_dir"))
    report_dir = resolve_path(repo_root, eda_cfg.get("report_dir"))
    overwrite = bool(eda_cfg.get("overwrite", False))
    backend = _select_backend(eda_cfg.get("backend", "auto"))
    batch_size = int(eda_cfg.get("batch_size", 500_000))
    topk_n = int(eda_cfg.get("topk_n", 200))
    min_support = int(eda_cfg.get("min_support", 1000))
    topk_jaccard_n = int(eda_cfg.get("topk_jaccard_n", 100))

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_samples = split_dir / "samples_train.parquet"
    valid_samples = split_dir / "samples_valid.parquet"
    tokens_root = split_dir / "tokens"
    train_tokens_dir = tokens_root / "train_tokens"
    valid_tokens_dir = tokens_root / "valid_tokens"
    tokens_manifest_path = tokens_root / "tokens_split_manifest.json"
    split_spec_path = split_dir / "split_spec.json"
    split_stats_path = split_dir / "split_stats.json"
    canonical_manifest = resolve_path(
        repo_root, canonical_cfg.get("out_dir", "data/interim/aliccp_canonical")
    ) / "manifest.json"

    sanity_path = out_dir / "sanity.json"
    samples_stats_train_path = out_dir / "samples_stats_train.json"
    samples_stats_valid_path = out_dir / "samples_stats_valid.json"
    entity_freq_train_path = out_dir / "entity_freq_train.parquet"
    entity_freq_valid_path = out_dir / "entity_freq_valid.parquet"
    row_nnz_hist_train_path = out_dir / "row_nnz_hist_train.parquet"
    row_nnz_hist_valid_path = out_dir / "row_nnz_hist_valid.parquet"
    field_stats_path = out_dir / "field_stats_train.parquet"
    field_topk_path = out_dir / "field_topk_train.parquet"
    fid_lift_path = out_dir / "fid_lift_train.parquet"
    drift_summary_path = out_dir / "drift_summary.parquet"
    drift_report_path = report_dir / "drift_report.md"

    conn = duckdb.connect() if backend == "duckdb" else None
    try:
        train_profile = compute_samples_profile(
            backend,
            train_samples,
            entity_freq_train_path,
            batch_size,
            overwrite,
            conn=conn,
        )
        valid_profile = compute_samples_profile(
            backend,
            valid_samples,
            entity_freq_valid_path,
            batch_size,
            overwrite,
            conn=conn,
        )

        overlap_count = _entity_overlap_count(
            entity_freq_train_path,
            entity_freq_valid_path,
            backend,
            conn=conn,
            train_entities=train_profile.get("entity_freq_entities"),
            valid_entities=valid_profile.get("entity_freq_entities"),
        )
        if overlap_count:
            raise ValueError(f"entity_overlap_count must be 0; observed {overlap_count}")

        nnz_quant_train = compute_row_nnz_hist(
            backend,
            train_tokens_dir,
            ROW_NNZ_BINS,
            batch_size,
            row_nnz_hist_train_path,
            overwrite,
            conn=conn,
        )
        nnz_quant_valid = compute_row_nnz_hist(
            backend,
            valid_tokens_dir,
            ROW_NNZ_BINS,
            batch_size,
            row_nnz_hist_valid_path,
            overwrite,
            conn=conn,
        )

        compute_field_stats(
            backend,
            train_tokens_dir,
            train_profile["rows"],
            batch_size,
            field_stats_path,
            field_topk_path,
            topk_n,
            overwrite,
            conn=conn,
        )
        compute_fid_lift(
            backend,
            train_tokens_dir,
            train_samples,
            field_topk_path,
            fid_lift_path,
            batch_size,
            min_support,
            train_profile["ctr"],
            train_profile["cvr"],
            overwrite,
            conn=conn,
        )

        drift_info = compute_drift_summary(
            backend,
            train_tokens_dir,
            valid_tokens_dir,
            train_profile["rows"],
            valid_profile["rows"],
            field_stats_path,
            field_topk_path,
            topk_jaccard_n,
            drift_summary_path,
            overwrite,
            conn=conn,
        )
        drift_table = drift_info.get("summary_table")
        render_drift_report(
            drift_report_path,
            train_profile,
            valid_profile,
            nnz_quant_train,
            nnz_quant_valid,
            drift_table,
        )

        tokens_manifest = _load_json_file(tokens_manifest_path)
        split_spec = _load_json_file(split_spec_path)
        split_stats = _load_json_file(split_stats_path)
        canonical_manifest_data = _load_json_file(canonical_manifest)
        tokens_train_manifest = tokens_manifest.get("train_token_rows")
        tokens_valid_manifest = tokens_manifest.get("valid_token_rows")
        tokens_train_scan = _count_parquet_rows_in_dir(train_tokens_dir)
        tokens_valid_scan = _count_parquet_rows_in_dir(valid_tokens_dir)
        expected_total = canonical_manifest_data.get("tokens_total_rows") if canonical_manifest_data else None
        if expected_total is None:
            expected_total = tokens_manifest.get("expected_tokens_total_rows") or tokens_manifest.get("observed_tokens_total_rows")
        observed_total = tokens_train_scan + tokens_valid_scan
        tokens_rows_sum_check = {
            "expected_total": expected_total,
            "observed_total": observed_total,
            "diff": None if expected_total is None else observed_total - expected_total,
        }

        sanity = {
            "n_train_samples": train_profile["rows"],
            "n_valid_samples": valid_profile["rows"],
            "train_ctr": train_profile["ctr"],
            "valid_ctr": valid_profile["ctr"],
            "train_cvr": train_profile["cvr"],
            "valid_cvr": valid_profile["cvr"],
            "funnel_bad_train": train_profile["funnel_bad"],
            "funnel_bad_valid": valid_profile["funnel_bad"],
            "entity_overlap_count": overlap_count,
            "tokens_train_rows_manifest": tokens_train_manifest,
            "tokens_valid_rows_manifest": tokens_valid_manifest,
            "tokens_train_rows_scan": tokens_train_scan,
            "tokens_valid_rows_scan": tokens_valid_scan,
            "tokens_rows_sum_check": tokens_rows_sum_check,
            "split_spec": split_spec,
            "split_stats": split_stats,
        }
        _write_json(sanity_path, sanity, overwrite=overwrite)
        _write_json(
            samples_stats_train_path,
            {
                "rows": train_profile["rows"],
                "unique_entity": train_profile["unique_entity"],
                "ctr": train_profile["ctr"],
                "cvr": train_profile["cvr"],
                "funnel_bad": train_profile["funnel_bad"],
                "entity_freq_quantiles": train_profile["entity_freq_quantiles"],
                "top1pct_coverage": train_profile["top1pct_coverage"],
            },
            overwrite=overwrite,
        )
        _write_json(
            samples_stats_valid_path,
            {
                "rows": valid_profile["rows"],
                "unique_entity": valid_profile["unique_entity"],
                "ctr": valid_profile["ctr"],
                "cvr": valid_profile["cvr"],
                "funnel_bad": valid_profile["funnel_bad"],
                "entity_freq_quantiles": valid_profile["entity_freq_quantiles"],
                "top1pct_coverage": valid_profile["top1pct_coverage"],
            },
            overwrite=overwrite,
        )

        return {
            "backend": backend,
            "sanity": sanity_path,
            "samples_train": samples_stats_train_path,
            "samples_valid": samples_stats_valid_path,
            "entity_freq_train": entity_freq_train_path,
            "entity_freq_valid": entity_freq_valid_path,
            "row_nnz_train": row_nnz_hist_train_path,
            "row_nnz_valid": row_nnz_hist_valid_path,
            "field_stats_train": field_stats_path,
            "field_topk_train": field_topk_path,
            "fid_lift_train": fid_lift_path,
            "drift_summary": drift_summary_path,
            "drift_report": drift_report_path,
        }
    finally:
        if conn is not None:
            conn.close()
