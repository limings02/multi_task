from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.logging import get_logger
from src.eda.extra.common import write_table, _path_posix


logger = get_logger(__name__)


def _quantiles(con: duckdb.DuckDBPyConnection, glob: str, field: str, src: int, qs: List[float]) -> List[float]:
    cols = ", ".join(f"approx_quantile(val, {q}) AS q{int(q*100)}" for q in qs)
    row = con.execute(
        f"""
        SELECT {cols}
        FROM read_parquet('{glob}')
        WHERE field='{field}' AND src={src} AND isfinite(val)
        """
    ).fetchone()
    vals = [row[i] for i in range(len(qs))]
    clean = [float(v) for v in vals if v is not None]
    return clean


def _bin_edges(train_edges: List[float]) -> List[float]:
    uniq = sorted({e for e in train_edges})
    return uniq


def _case_expr(edges: List[float]) -> str:
    if not edges:
        return "0"
    clauses = []
    for idx, edge in enumerate(edges):
        clauses.append(f"WHEN val < {edge} THEN {idx}")
    clauses.append(f"ELSE {len(edges)}")
    return "CASE " + " ".join(clauses) + " END"


def _histogram(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    field: str,
    src: int,
    edges: List[float],
) -> Tuple[List[float], int]:
    case_expr = _case_expr(edges)
    sql = f"""
    SELECT bin, COUNT(*) AS cnt
    FROM (
        SELECT {case_expr} AS bin
        FROM read_parquet('{glob}')
        WHERE field='{field}' AND src={src} AND isfinite(val)
    )
    GROUP BY bin
    """
    df = con.execute(sql).fetch_df()
    total = int(df["cnt"].sum()) if not df.empty else 0
    bins = [0.0] * (len(edges) + 1)
    for _, r in df.iterrows():
        bin_idx = int(r["bin"])
        if 0 <= bin_idx < len(bins):
            bins[bin_idx] = float(r["cnt"]) / total if total else 0.0
    return bins, total


def _profile(
    con: duckdb.DuckDBPyConnection,
    glob: str,
    field: str,
    src: int,
) -> Dict[str, float]:
    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN val = 0 THEN 1 ELSE 0 END) AS zeros,
            SUM(CASE WHEN val < 0 THEN 1 ELSE 0 END) AS negs,
            SUM(CASE WHEN isnan(val) THEN 1 ELSE 0 END) AS nans,
            SUM(CASE WHEN isinf(val) THEN 1 ELSE 0 END) AS infs,
            approx_quantile(val, 0.5) FILTER (WHERE isfinite(val)) AS p50,
            approx_quantile(val, 0.9) FILTER (WHERE isfinite(val)) AS p90,
            approx_quantile(val, 0.99) FILTER (WHERE isfinite(val)) AS p99,
            max(val) FILTER (WHERE isfinite(val)) AS vmax
        FROM read_parquet('{glob}')
        WHERE field='{field}' AND src={src}
        """
    ).fetchone()
    total = row[0] or 0
    def ratio(x): return float(x) / total if total else 0.0
    return {
        "total": total,
        "zero_ratio": ratio(row[1]),
        "neg_ratio": ratio(row[2]),
        "nan_ratio": ratio(row[3]),
        "inf_ratio": ratio(row[4]),
        "p50": row[5],
        "p90": row[6],
        "p99": row[7],
        "max": row[8],
    }


def compute_val_drift(
    con: duckdb.DuckDBPyConnection,
    fields: List[Tuple[str, int]],
    train_tokens: Path,
    valid_tokens: Path,
    out_profile_path: Path,
    out_psi_path: Path,
    overwrite: bool,
) -> Tuple[pa.Table, pa.Table]:
    train_glob = _path_posix(train_tokens / "*.parquet") if train_tokens.is_dir() else _path_posix(train_tokens)
    valid_glob = _path_posix(valid_tokens / "*.parquet") if valid_tokens.is_dir() else _path_posix(valid_tokens)
    profile_rows = []
    psi_rows = []
    for field, src in fields:
        edges = _bin_edges(_quantiles(con, train_glob, field, src, [i / 10 for i in range(1, 10)]))
        train_bins, train_total = _histogram(con, train_glob, field, src, edges)
        valid_bins, valid_total = _histogram(con, valid_glob, field, src, edges)
        eps = 1e-6
        psi = 0.0
        for p, q in zip(train_bins, valid_bins):
            p = max(p, eps)
            q = max(q, eps)
            psi += (p - q) * math.log(p / q)

        train_prof = _profile(con, train_glob, field, src)
        valid_prof = _profile(con, valid_glob, field, src)
        profile_rows.append(
            {
                "field": field,
                "src": src,
                "train_p50": train_prof["p50"],
                "train_p90": train_prof["p90"],
                "train_p99": train_prof["p99"],
                "train_max": train_prof["max"],
                "train_zero_ratio": train_prof["zero_ratio"],
                "train_neg_ratio": train_prof["neg_ratio"],
                "train_nan_ratio": train_prof["nan_ratio"],
                "train_inf_ratio": train_prof["inf_ratio"],
                "valid_p50": valid_prof["p50"],
                "valid_p90": valid_prof["p90"],
                "valid_p99": valid_prof["p99"],
                "valid_max": valid_prof["max"],
                "valid_zero_ratio": valid_prof["zero_ratio"],
                "valid_neg_ratio": valid_prof["neg_ratio"],
                "valid_nan_ratio": valid_prof["nan_ratio"],
                "valid_inf_ratio": valid_prof["inf_ratio"],
            }
        )
        psi_rows.append(
            {
                "field": field,
                "src": src,
                "psi": psi,
                "bins": json.dumps(edges),
                "train_bin_frac": json.dumps(train_bins),
                "valid_bin_frac": json.dumps(valid_bins),
            }
        )
    profile_table = pa.Table.from_pandas(pd.DataFrame(profile_rows))
    psi_table = pa.Table.from_pandas(pd.DataFrame(psi_rows))
    write_table(profile_table, out_profile_path, overwrite=overwrite)
    write_table(psi_table, out_psi_path, overwrite=overwrite)
    return profile_table, psi_table


def plot_val_psi(psi_df: pd.DataFrame, out_path: Path, top_n: int = 8) -> None:
    if psi_df.empty:
        return
    plot_df = psi_df.sort_values("psi", ascending=False).head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="psi", y=plot_df["field"].astype(str) + "_s" + plot_df["src"].astype(str))
    plt.xlabel("PSI")
    plt.ylabel("field|src")
    plt.title("Value PSI (train vs valid)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
