from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

from src.core.logging import get_logger
from src.eda.extra.common import write_table, _path_posix


logger = get_logger(__name__)


HEAD_SIZES = [1_000, 5_000, 20_000, 100_000, 200_000]


def _ensure_train_topn(
    con: duckdb.DuckDBPyConnection,
    train_path_glob: str,
    out_path: Path,
    overwrite: bool,
) -> Path:
    if out_path.exists() and not overwrite:
        logger.info("Reuse cached train topN from %s", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Building train top frequencies (<=200k per field) from %s", train_path_glob)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE train_topn AS
        SELECT *
        FROM (
            SELECT
                field,
                src,
                fid,
                cnt,
                row_number() OVER (PARTITION BY field, src ORDER BY cnt DESC) AS rnk
            FROM (
                SELECT field, src, fid, COUNT(*) AS cnt
                FROM read_parquet('{train_path_glob}')
                GROUP BY field, src, fid
            )
            WHERE cnt > 0
        )
        WHERE rnk <= 200000
        """
    )
    con.execute(f"COPY train_topn TO '{_path_posix(out_path)}' (FORMAT PARQUET)")
    stats_df = con.execute(
        "SELECT field, src, COUNT(*) AS topk_rows FROM train_topn GROUP BY field, src ORDER BY field, src"
    ).fetch_df()
    total_rows = int(stats_df["topk_rows"].sum()) if not stats_df.empty else 0
    logger.info("train_topn written to %s; total rows=%s; per-field rows=%s", out_path, total_rows, stats_df.to_dict("records"))
    return out_path


def compute_oov_curves(
    con: duckdb.DuckDBPyConnection,
    train_tokens: Path,
    valid_tokens: Path,
    cache_topn_path: Path,
    out_curve_path: Path,
    out_summary_path: Path,
    overwrite: bool,
) -> Tuple[pa.Table, pa.Table]:
    train_glob = _path_posix(train_tokens / "*.parquet") if train_tokens.is_dir() else _path_posix(train_tokens)
    valid_glob = _path_posix(valid_tokens / "*.parquet") if valid_tokens.is_dir() else _path_posix(valid_tokens)

    topn_path = _ensure_train_topn(con, train_glob, cache_topn_path, overwrite)

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW valid_joined AS
        SELECT
            v.field,
            v.src,
            v.row_id,
            v.fid,
            COALESCE(t.rnk, 999999999) AS rnk
        FROM read_parquet('{valid_glob}') v
        LEFT JOIN read_parquet('{_path_posix(topn_path)}') t USING(field, src, fid)
        """
    )
    logger.info("valid_joined is TEMP VIEW (not materialized) from %s and %s", valid_glob, topn_path)

    token_agg = con.execute(
        f"""
        SELECT
            field,
            src,
            COUNT(*) AS total_tokens,
            {", ".join([f"SUM(CASE WHEN rnk>{n} THEN 1 ELSE 0 END) AS oov_tok_{n}" for n in HEAD_SIZES])}
        FROM valid_joined
        GROUP BY field, src
        """
    ).fetch_df()

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE valid_row_max AS
        SELECT field, src, row_id, MAX(rnk) AS max_rnk
        FROM valid_joined
        GROUP BY field, src, row_id
        """
    )
    row_max_rows = con.execute("SELECT COUNT(*) FROM valid_row_max").fetchone()[0]
    row_agg = con.execute(
        f"""
        SELECT
            field,
            src,
            COUNT(*) AS total_rows,
            {", ".join([f"SUM(CASE WHEN max_rnk>{n} THEN 1 ELSE 0 END) AS oov_rows_{n}" for n in HEAD_SIZES])}
        FROM valid_row_max
        GROUP BY field, src
        """
    ).fetch_df()
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE valid_fid_min AS
        SELECT field, src, fid, MIN(rnk) AS min_rnk
        FROM valid_joined
        GROUP BY field, src, fid
        """
    )
    fid_min_rows = con.execute("SELECT COUNT(*) FROM valid_fid_min").fetchone()[0]
    fid_agg = con.execute(
        f"""
        SELECT
            field,
            src,
            COUNT(*) AS total_distinct,
            {", ".join([f"SUM(CASE WHEN min_rnk>{n} THEN 1 ELSE 0 END) AS oov_dist_{n}" for n in HEAD_SIZES])}
        FROM valid_fid_min
        GROUP BY field, src
        """
    ).fetch_df()

    merged = token_agg.merge(row_agg, on=["field", "src"], how="left").merge(fid_agg, on=["field", "src"], how="left")
    logger.info(
        "Aggregated OOV metrics in one pass: token_agg_rows=%s row_agg_rows=%s fid_agg_rows=%s valid_row_max_rows=%s valid_fid_min_rows=%s",
        len(token_agg),
        len(row_agg),
        len(fid_agg),
        row_max_rows,
        fid_min_rows,
    )
    records = []
    for _, r in merged.iterrows():
        for n in HEAD_SIZES:
            total_tokens = r["total_tokens"]
            total_rows = r["total_rows"]
            total_distinct = r["total_distinct"]
            records.append(
                {
                    "field": r["field"],
                    "src": int(r["src"]),
                    "head_size_N": n,
                    "oov_token_rate_valid": float(r[f"oov_tok_{n}"]) / total_tokens if total_tokens else 0.0,
                    "oov_row_rate_valid": float(r[f"oov_rows_{n}"]) / total_rows if total_rows else 0.0,
                    "new_value_rate_valid_distinct": float(r[f"oov_dist_{n}"]) / total_distinct if total_distinct else 0.0,
                }
            )
    curves_df = pd.DataFrame(records)
    logger.info("oov_curve rows=%s (fields * %s heads)", len(curves_df), len(HEAD_SIZES))
    curve_table = pa.Table.from_pandas(curves_df)
    write_table(curve_table, out_curve_path, overwrite=overwrite)

    summary_records = []
    thresholds = {
        "row_rate_good": 0.02,
        "row_rate_ok": 0.05,
        "new_value_good": 0.05,
    }
    grouped = curves_df.groupby(["field", "src"])
    for (field, src), g in grouped:
        g_sorted = g.sort_values("head_size_N")
        cond_good = (g_sorted.oov_row_rate_valid <= thresholds["row_rate_good"]) & (
            g_sorted.new_value_rate_valid_distinct <= thresholds["new_value_good"]
        )
        cond_ok = (g_sorted.oov_row_rate_valid <= thresholds["row_rate_ok"])

        if cond_good.any():
            best_idx = cond_good.idxmax()
            encoding = "vocab" if g_sorted.loc[best_idx, "head_size_N"] <= 50_000 else "hybrid"
        elif cond_ok.any():
            best_idx = cond_ok.idxmax()
            encoding = "hybrid"
        else:
            best_idx = g_sorted["oov_row_rate_valid"].idxmin()
            encoding = "hash"
        recommended_head = int(g_sorted.loc[best_idx, "head_size_N"])
        summary_records.append(
            {
                "field": field,
                "src": int(src),
                "recommended_head_size": recommended_head if encoding != "hash" else None,
                "recommended_encoding_hint": encoding,
                "decision_thresholds_used": json.dumps(thresholds),
            }
        )
    summary_df = pd.DataFrame(summary_records)
    summary_table = pa.Table.from_pandas(summary_df)
    write_table(summary_table, out_summary_path, overwrite=overwrite)
    return curve_table, summary_table
