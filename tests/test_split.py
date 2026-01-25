from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.split import materialize_entity_hash_split


def test_materialize_entity_hash_split(tmp_path):
    # 构造样本数据：两个实体重复出现，确保不会跨 split
    data = {
        "row_id": [0, 1, 2, 3, 4, 5],
        "y1": [1, 0, 1, 0, 1, 0],
        "y2": [0, 0, 1, 0, 0, 1],
        "entity_id": ["e1", "e2", "e1", "e3", "e2", "e4"],
        "c4": [1, 1, 2, 2, 3, 3],
        "c0": [10, 11, 12, 13, 14, 15],
    }
    table = pa.Table.from_pydict(data)
    samples_path = tmp_path / "samples_train.parquet"
    pq.write_table(table, samples_path)

    out_dir = tmp_path / "split"
    result = materialize_entity_hash_split(
        samples_path=samples_path,
        out_dir=out_dir,
        seed=42,
        ratios={"train": 0.5, "valid": 0.5},
        overwrite=True,
        chunksize=2,
    )

    train_path = out_dir / "samples_train.parquet"
    valid_path = out_dir / "samples_valid.parquet"
    assert train_path.exists() and valid_path.exists()
    train_tbl = pq.read_table(train_path)
    valid_tbl = pq.read_table(valid_path)
    # 检查同一 entity 不跨 split
    train_entities = set(train_tbl.column("entity_id").to_pylist())
    valid_entities = set(valid_tbl.column("entity_id").to_pylist())
    assert train_entities.isdisjoint(valid_entities)
    # 行数守恒
    assert train_tbl.num_rows + valid_tbl.num_rows == table.num_rows
    # 输出 meta 存在
    assert Path(result["spec"]).exists()
    assert Path(result["stats"]).exists()
