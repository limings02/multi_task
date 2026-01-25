from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.token_filter import build_rowid_membership, filter_tokens_by_rowids


def _write_tokens(path: Path, row_ids, src=0):
    table = pa.Table.from_pydict(
        {
            "row_id": row_ids,
            "src": [src] * len(row_ids),
            "field": ["f"] * len(row_ids),
            "fid": ["1"] * len(row_ids),
            "val": np.ones(len(row_ids), dtype=np.float32),
        },
        schema=pa.schema(
            [
                ("row_id", pa.int64()),
                ("src", pa.int8()),
                ("field", pa.string()),
                ("fid", pa.string()),
                ("val", pa.float32()),
            ]
        ),
    )
    pq.write_table(table, path, compression="snappy")


def test_filter_tokens_by_rowids(tmp_path):
    # 准备 tokens 目录
    tokens_dir = tmp_path / "tokens_train"
    tokens_dir.mkdir()
    _write_tokens(tokens_dir / "tokens_train.part_0_4.parquet", [0, 1, 2, 3, 4])
    _write_tokens(tokens_dir / "tokens_train.part_5_9.parquet", [5, 6, 7, 8, 9])

    # 准备 split 样本（train: 0-4, valid: 5-9）
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
    train_tbl = pa.Table.from_pydict(
        {"row_id": [0, 1, 2, 3, 4], "y1": [0] * 5, "y2": [0] * 5, "entity_id": ["e"] * 5, "c4": [1] * 5, "c0": [1] * 5},
        schema=schema,
    )
    valid_tbl = pa.Table.from_pydict(
        {"row_id": [5, 6, 7, 8, 9], "y1": [0] * 5, "y2": [0] * 5, "entity_id": ["e"] * 5, "c4": [1] * 5, "c0": [1] * 5},
        schema=schema,
    )
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    pq.write_table(train_tbl, split_dir / "samples_train.parquet")
    pq.write_table(valid_tbl, split_dir / "samples_valid.parquet")

    train_mem = build_rowid_membership(split_dir / "samples_train.parquet", chunk_rows=2)
    valid_mem = build_rowid_membership(split_dir / "samples_valid.parquet", chunk_rows=2)

    out_dir = tmp_path / "tokens_split"
    stats = filter_tokens_by_rowids(
        tokens_dir=tokens_dir,
        train_membership=train_mem,
        valid_membership=valid_mem,
        out_dir=out_dir,
        flush_rows=3,
        overwrite=True,
    )

    train_files = list((out_dir / "train_tokens").glob("*.parquet"))
    valid_files = list((out_dir / "valid_tokens").glob("*.parquet"))
    assert train_files and valid_files
    train_rows = sum(pq.read_table(f).num_rows for f in train_files)
    valid_rows = sum(pq.read_table(f).num_rows for f in valid_files)
    assert train_rows == 5 and valid_rows == 5
    assert stats["train_token_rows"] == 5
    assert stats["valid_token_rows"] == 5

