from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Set

import duckdb
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.logging import get_logger
from src.eda.extra.common import write_table, _path_posix, MULTIHOT_NNZ_THRESHOLD


logger = get_logger(__name__)


TRUNC_K = [10, 20, 30, 40, 60, 80, 120, 200]


def _build_lens_table(con: duckdb.DuckDBPyConnection, train_glob: str, multi_hot: Set[Tuple[str, int]]) -> None:
    if not multi_hot:
        con.execute("CREATE OR REPLACE TEMP TABLE token_lens AS SELECT NULL::BIGINT AS row_id, NULL::VARCHAR AS field, NULL::TINYINT AS src, NULL::BIGINT AS l WHERE FALSE")
        return
    filters = ",".join([f"('{f}', {s})" for f, s in sorted(multi_hot)])
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE token_lens AS
        SELECT v.row_id, v.field, v.src, COUNT(*) AS l
        FROM read_parquet('{train_glob}') v
        INNER JOIN (VALUES {filters}) AS t(field, src) ON v.field = t.field AND v.src = t.src
        GROUP BY v.row_id, v.field, v.src
        """
    )


def compute_length_metrics(
    con: duckdb.DuckDBPyConnection,
    train_tokens: Path,
    field_stats_df: pd.DataFrame,
    train_rows: int,
    out_quantile_path: Path,
    out_trunc_path: Path,
    overwrite: bool,
) -> Tuple[pa.Table, pa.Table]:
    train_glob = _path_posix(train_tokens / "*.parquet") if train_tokens.is_dir() else _path_posix(train_tokens)
    multi_hot = {
        (str(r["field"]), int(r["src"]))
        for _, r in field_stats_df[field_stats_df["avg_nnz_per_covered_row"] > MULTIHOT_NNZ_THRESHOLD].iterrows()
    }
    _build_lens_table(con, train_glob, multi_hot)
    lens_rows = con.execute("SELECT COUNT(*) FROM token_lens").fetchone()[0]
    logger.info("token_lens materialized as TEMP TABLE rows=%s fields=%s (multi-hot only)", lens_rows, len(multi_hot))

    quant_df = con.execute(
        f"""
        SELECT
            field,
            src,
            COUNT(*) AS covered_rows,
            CAST(COUNT(*) AS DOUBLE) / {train_rows} AS coverage_rate,
            AVG(l) AS mean_len,
            approx_quantile(l, 0.5) AS p50,
            approx_quantile(l, 0.9) AS p90,
            approx_quantile(l, 0.95) AS p95,
            approx_quantile(l, 0.99) AS p99
        FROM token_lens
        GROUP BY field, src
        """
    ).fetch_df()
    if quant_df.empty:
        quant_df = pd.DataFrame(
            columns=["field", "src", "covered_rows", "coverage_rate", "mean_len", "p50", "p90", "p95", "p99"]
        )
    quant_table = pa.Table.from_pandas(quant_df)
    write_table(quant_table, out_quantile_path, overwrite=overwrite)

    values_clause = ", ".join(f"({k})" for k in TRUNC_K)
    trunc_df = con.execute(
        f"""
        WITH ks(k) AS (VALUES {values_clause})
        SELECT
            field,
            src,
            k AS K,
            SUM(CASE WHEN l > k THEN k ELSE l END)::DOUBLE / NULLIF(SUM(l), 0) AS retained_token_frac,
            SUM(CASE WHEN l <= k THEN 1 ELSE 0 END)::DOUBLE / NULLIF(COUNT(*), 0) AS retained_row_full_frac
        FROM token_lens
        CROSS JOIN ks
        GROUP BY field, src, k
        """
    ).fetch_df()
    if trunc_df.empty:
        trunc_df = pd.DataFrame(columns=["field", "src", "K", "retained_token_frac", "retained_row_full_frac"])
    trunc_table = pa.Table.from_pandas(trunc_df)
    write_table(trunc_table, out_trunc_path, overwrite=overwrite)
    return quant_table, trunc_table


def plot_truncation_curves(
    trunc_df: pd.DataFrame,
    field_stats_df: pd.DataFrame,
    out_path: Path,
    top_n: int = 8,
) -> None:
    if trunc_df.empty:
        return
    stats = field_stats_df.copy()
    if "token_rows" in stats.columns:
        stats["token_share"] = stats["token_rows"] / stats["token_rows"].sum()
    else:
        stats["token_share"] = stats["coverage_rate"] * stats["avg_nnz_per_covered_row"]
    stats = stats.sort_values("token_share", ascending=False).head(top_n)
    fields = set(zip(stats["field"], stats["src"]))
    plot_df = trunc_df[trunc_df.apply(lambda r: (r["field"], r["src"]) in fields, axis=1)]
    if plot_df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=plot_df,
        x="K",
        y="retained_token_frac",
        hue=plot_df["field"].astype(str) + "_s" + plot_df["src"].astype(str),
        marker="o",
    )
    plt.xlabel("K (max_len candidate)")
    plt.ylabel("Retained token fraction")
    plt.title("Truncation loss curves (multi-hot top fields)")
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
