from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa

from src.core.logging import get_logger
from src.eda.extra.common import write_json, write_table


logger = get_logger(__name__)

BUCKET_CANDIDATES = [
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
    4_194_304,
]


def _collision_metrics(n: int, m: int) -> Tuple[float, float, float]:
    if m <= 0:
        return 0.0, 0.0, 0.0
    occupied = m * (1 - math.exp(-n / m))
    collision = max(0.0, n - occupied)
    rate = collision / n if n else 0.0
    return occupied, collision, rate


def compute_hash_collision(
    field_stats_df: pd.DataFrame,
    out_parquet: Path,
    out_json: Path,
    overwrite: bool,
) -> pa.Table:
    if "token_rows" in field_stats_df.columns and field_stats_df["token_rows"].sum() > 0:
        token_share = field_stats_df["token_rows"] / field_stats_df["token_rows"].sum()
    else:
        token_share = (field_stats_df["coverage_rate"] * field_stats_df["avg_nnz_per_covered_row"])
        token_share = token_share / token_share.sum()
    field_stats_df = field_stats_df.assign(token_share=token_share)

    records = []
    reco: Dict[str, Dict[str, object]] = {}
    for _, row in field_stats_df.iterrows():
        field = row["field"]
        src = int(row["src"])
        n = int(row.get("fid_distinct_est", 0) or 0)
        share = float(row.get("token_share", 0.0) or 0.0)
        if share >= 0.2:
            th_local, th_online = 0.02, 0.01
        elif share >= 0.05:
            th_local, th_online = 0.05, 0.02
        else:
            th_local, th_online = 0.10, 0.05
        for m in BUCKET_CANDIDATES:
            occupied, coll, rate = _collision_metrics(n, m)
            records.append(
                {
                    "field": field,
                    "src": src,
                    "fid_distinct_est": n,
                    "bucket": m,
                    "expected_occupied": occupied,
                    "expected_collision": coll,
                    "collision_rate": rate,
                }
            )
        field_key = f"{field}|{src}"
        rates = [(m, _collision_metrics(n, m)[2]) for m in BUCKET_CANDIDATES]
        local_bucket = next((m for m, r in rates if r < th_local), BUCKET_CANDIDATES[-1])
        online_bucket = next((m for m, r in rates if r < th_online), BUCKET_CANDIDATES[-1])
        reco[field_key] = {
            "field": field,
            "src": src,
            "distinct_est": n,
            "recommended_bucket_local": local_bucket,
            "recommended_bucket_online": online_bucket,
            "threshold_local": th_local,
            "threshold_online": th_online,
            "token_share": share,
        }
    table = pa.Table.from_pandas(pd.DataFrame(records))
    write_table(table, out_parquet, overwrite=overwrite)
    write_json(
        out_json,
        {
            "per_field": list(reco.values()),
            "token_share_bands": [
                {"band": ">=0.20", "collision_local": 0.02, "collision_online": 0.01},
                {"band": "[0.05,0.20)", "collision_local": 0.05, "collision_online": 0.02},
                {"band": "<0.05", "collision_local": 0.10, "collision_online": 0.05},
            ],
        },
        overwrite=overwrite,
    )
    return table
