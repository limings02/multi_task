from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data.canonical import build_canonical_aliccp


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(map(str, row)))
            f.write("\n")


def test_canonical_pipeline_small(tmp_path, monkeypatch):
    skeleton_path = tmp_path / "raw" / "skeleton.csv"
    common_path = tmp_path / "raw" / "common.csv"

    skeleton_rows = [
        [101, 1, 0, "0000000000000001", 1, "210\x021\x031.0\x01150_14\x02123\x030.5"],
        [202, 0, 1, "0000000000000002", 2, "210\x021\x032.0"],
        [303, 1, 1, "0000000000000003", 3, "210\x021\x033.0\x01"],
        [404, 0, 0, "0000000000000004", 4, "210\x021\x034.0"],
    ]
    _write_csv(skeleton_path, skeleton_rows)

    common_rows = [
        ["0000000000000001", 2, "220\x021\x030.7"],
        ["0000000000000002", 1, "220\x021\x030.8"],
        ["0000000000000003", 1, "220\x021\x030.9"],
        ["0000000000000004", 1, "220\x021\x031.0"],
    ]
    _write_csv(common_path, common_rows)

    out_dir = tmp_path / "canonical"
    config = {
        "raw": {
            "skeleton_path": "IGNORE_ME",
            "common_features_path": "IGNORE_ME",
        },
        "canonical": {
            "out_dir": str(out_dir),
            "sqlite_path": str(out_dir / "common_features.sqlite"),
            "samples_path": str(out_dir / "samples_train.parquet"),
            "tokens_dir": str(out_dir / "tokens_train"),
        },
        "params": {
            "nrows": 1000,
            "chunksize_sk": 2,
            "chunksize_cf": 2,
            "buffer_max_tokens": 10,
            "rebuild_sqlite": True,
            "overwrite": True,
        },
    }
    config_path = tmp_path / "aliccp.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    # Ensure Windows-only defaults in config are overridden in CI-friendly temp dirs.
    monkeypatch.setenv("ALICCP_SKELETON_PATH", str(skeleton_path))
    monkeypatch.setenv("ALICCP_COMMON_FEATURES_PATH", str(common_path))

    manifest = build_canonical_aliccp(str(config_path))

    samples_path = Path(manifest["output_dir"]) / "samples_train.parquet"
    tokens_dir = Path(manifest["output_dir"]) / "tokens_train"

    assert samples_path.exists()
    samples_table = pq.read_table(samples_path)
    assert samples_table.schema.field("row_id").type == pa.int64()
    assert samples_table.schema.field("y1").type == pa.int8()
    assert samples_table.schema.field("y2").type == pa.int8()
    assert samples_table.schema.field("c4").type == pa.int32()
    assert samples_table.schema.field("c0").type == pa.int64()

    samples_df = samples_table.to_pandas()
    assert samples_df["row_id"].tolist() == list(range(len(samples_df)))

    parts = sorted(tokens_dir.glob("*.parquet"))
    assert parts, "Expected at least one tokens part parquet"
    token_table = pq.read_table(parts[0])
    assert token_table.schema.field("row_id").type == pa.int64()
    assert token_table.schema.field("src").type == pa.int8()
    assert token_table.schema.field("field").type == pa.string()
    assert token_table.schema.field("fid").type == pa.string()
    assert token_table.schema.field("val").type == pa.float32()

    assert manifest["join_hit_ratio"] > 0.99
    assert manifest["tokens_total_rows"] > 0
    assert manifest["samples_rows"] == len(samples_df)
    assert manifest["samples_unique_entity_id"] == 4
