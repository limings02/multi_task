from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.compact_tokens import compact_tokens_dir


def _write_token_part(path: Path, row_ids, src=0):
    table = pa.Table.from_pydict(
        {
            "row_id": row_ids,
            "src": [src] * len(row_ids),
            "field": ["f"] * len(row_ids),
            "fid": ["1"] * len(row_ids),
            "val": [1.0] * len(row_ids),
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


def test_compact_tokens(tmp_path):
    tokens_dir = tmp_path / "tokens_train"
    tokens_dir.mkdir()

    parts = []
    total_rows = 0
    start = 0
    for i in range(6):
        row_ids = list(range(start, start + 10))
        start += 10
        part_path = tokens_dir / f"tokens_train.part_{row_ids[0]}_{row_ids[-1]}.parquet"
        _write_token_part(part_path, row_ids)
        parts.append(
            {
                "path": str(part_path),
                "row_id_min": row_ids[0],
                "row_id_max": row_ids[-1],
                "token_rows": len(row_ids),
            }
        )
        total_rows += len(row_ids)

    manifest = {
        "output_dir": str(tmp_path),
        "tokens_parts": parts,
        "tokens_total_rows": total_rows,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    compact_dir = tmp_path / "tokens_train_compact"

    new_manifest = compact_tokens_dir(
        manifest_path=str(manifest_path),
        out_tokens_dir=str(compact_dir),
        target_parts=2,
        overwrite=True,
    )

    out_files = list(compact_dir.glob("*.parquet"))
    assert len(out_files) == 2
    assert new_manifest["tokens_total_rows"] == total_rows
    assert Path(new_manifest["tokens_dir_compact"]) == compact_dir

    new_parts = new_manifest["tokens_parts"]
    assert sum(p["token_rows"] for p in new_parts) == total_rows
    row_min = min(p["row_id_min"] for p in new_parts)
    row_max = max(p["row_id_max"] for p in new_parts)
    assert row_min == 0 and row_max == total_rows - 1

