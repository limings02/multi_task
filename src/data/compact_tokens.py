"""
Token compaction utilities for Ali-CCP canonical outputs.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.core.logging import get_logger

logger = get_logger(__name__)


def _load_manifest(manifest_path: Path) -> Dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_part_path(part: Dict, output_dir: Path, in_tokens_dir: Optional[Path]) -> Path:
    p = Path(part.get("path", ""))
    if p.is_absolute():
        if p.exists():
            return p
    else:
        candidates = [
            output_dir / p,
            (in_tokens_dir / p) if in_tokens_dir else None,
        ]
        for c in candidates:
            if c and c.exists():
                return c
    return p


def _sorted_parts(parts: List[Dict]) -> List[Dict]:
    return sorted(parts, key=lambda p: (p.get("row_id_min") or 0, str(p.get("path", ""))))


def _concat_and_write(writer: pq.ParquetWriter, tables: List[pa.Table]) -> int:
    if not tables:
        return 0
    table = pa.concat_tables(tables)
    writer.write_table(table)
    return table.num_rows


def compact_tokens_dir(
    manifest_path: str | Path,
    in_tokens_dir: str | Path | None = None,
    out_tokens_dir: str | Path | None = None,
    target_parts: int = 100,
    max_rows_per_file: Optional[int] = None,
    target_chunk_rows: int = 2_000_000,
    overwrite: bool = False,
    inplace: bool = False,
    seed: int = 20260121,
) -> Dict:
    """
    Compact many small token parquet parts into fewer large parts.
    """

    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    output_dir = Path(manifest.get("output_dir", manifest_path.parent))

    if in_tokens_dir:
        in_tokens_dir = Path(in_tokens_dir)
    if out_tokens_dir:
        out_tokens_dir = Path(out_tokens_dir)
    else:
        out_tokens_dir = output_dir / "tokens_train_compact"

    tokens_parts = manifest.get("tokens_parts", [])
    tokens_total_rows = int(manifest.get("tokens_total_rows", 0))

    if tokens_total_rows <= 0:
        raise ValueError("tokens_total_rows must be positive for compaction.")

    rows_threshold = (
        max_rows_per_file
        if max_rows_per_file is not None
        else int(math.ceil(tokens_total_rows / max(1, target_parts)))
    )

    rng = None
    try:
        import random

        rng = random.Random(seed)
    except Exception:
        pass

    sorted_parts = _sorted_parts(tokens_parts)
    logger.info(
        "Starting tokens compaction: %d input parts -> target_parts=%d (rows_threshold=%d)",
        len(sorted_parts),
        target_parts,
        rows_threshold,
    )

    if out_tokens_dir.exists():
        if overwrite:
            logger.info("Removing existing output tokens dir %s (overwrite=True)", out_tokens_dir)
            for f in out_tokens_dir.glob("*.parquet"):
                f.unlink()
        else:
            raise FileExistsError(f"Output tokens dir already exists: {out_tokens_dir}")
    else:
        out_tokens_dir.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    buffer_tables: List[pa.Table] = []
    buffer_rows = 0
    current_rows = 0
    current_row_min: Optional[int] = None
    current_row_max: Optional[int] = None
    temp_path: Optional[Path] = None
    new_parts: List[Dict[str, object]] = []

    def flush_writer():
        nonlocal writer, temp_path, current_rows, current_row_min, current_row_max
        if writer is None:
            return
        if buffer_tables:
            written_rows = _concat_and_write(writer, buffer_tables)
            current_rows += written_rows
            buffer_tables.clear()
        writer.close()
        if current_row_min is None or current_row_max is None:
            # nothing written
            if temp_path and temp_path.exists():
                temp_path.unlink()
            writer = None
            temp_path = None
            current_rows = 0
            current_row_min = None
            current_row_max = None
            return
        final_path = out_tokens_dir / f"tokens_train.part_{current_row_min}_{current_row_max}.parquet"
        if temp_path:
            os.replace(temp_path, final_path)
        new_parts.append(
            {
                "path": str(final_path),
                "row_id_min": current_row_min,
                "row_id_max": current_row_max,
                "token_rows": current_rows,
            }
        )
        logger.info(
            "Wrote compact part %s (rows=%d, row_id_min=%d, row_id_max=%d)",
            final_path,
            current_rows,
            current_row_min,
            current_row_max,
        )
        writer = None
        temp_path = None
        current_rows = 0
        current_row_min = None
        current_row_max = None

    for part in sorted_parts:
        part_path = _resolve_part_path(part, output_dir, in_tokens_dir)
        if not part_path.exists():
            raise FileNotFoundError(f"Token part not found: {part_path}")
        pf = pq.ParquetFile(part_path)
        for batch in pf.iter_batches(batch_size=target_chunk_rows):
            table = pa.Table.from_batches([batch])
            if writer is None:
                temp_path = out_tokens_dir / f"tokens_compact_{len(new_parts)}.parquet.tmp"
                writer = pq.ParquetWriter(temp_path, table.schema, compression="snappy")
                current_rows = 0
                current_row_min = None
                current_row_max = None

            row_ids = table.column("row_id")
            batch_min = pc.min(row_ids).as_py()
            batch_max = pc.max(row_ids).as_py()
            current_row_min = batch_min if current_row_min is None else min(current_row_min, batch_min)
            current_row_max = batch_max if current_row_max is None else max(current_row_max, batch_max)

            buffer_tables.append(table)
            buffer_rows += table.num_rows

            if buffer_rows >= target_chunk_rows or (current_rows + buffer_rows) >= rows_threshold:
                written = _concat_and_write(writer, buffer_tables)
                current_rows += written
                buffer_tables.clear()
                buffer_rows = 0

            if current_rows >= rows_threshold:
                flush_writer()

    flush_writer()

    tokens_dir_compact = out_tokens_dir

    compaction_dump = {
        "target_parts": target_parts,
        "max_rows_per_file": max_rows_per_file,
        "target_chunk_rows": target_chunk_rows,
        "inplace": inplace,
        "overwrite": overwrite,
        "seed": seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    new_manifest = dict(manifest)
    new_manifest["tokens_parts"] = new_parts
    new_manifest["tokens_total_rows"] = tokens_total_rows
    new_manifest["tokens_dir_compact"] = str(tokens_dir_compact)
    new_manifest["compaction_dump"] = compaction_dump

    manifest_compact_path = output_dir / "manifest.compact.json"
    with open(manifest_compact_path, "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, ensure_ascii=False, indent=2)
    logger.info("Compaction manifest written to %s", manifest_compact_path)

    if inplace:
        tokens_dir_orig = Path(manifest["tokens_parts"][0]["path"]).parent
        tmp_dir = tokens_dir_compact
        bak_dir = tokens_dir_orig.with_name(tokens_dir_orig.name + ".bak")
        if bak_dir.exists():
            if overwrite:
                logger.info("Removing existing backup dir %s", bak_dir)
                for f in bak_dir.glob("*.parquet"):
                    f.unlink()
                bak_dir.rmdir()
            else:
                raise FileExistsError(f"Backup dir already exists: {bak_dir}")
        if tokens_dir_orig.exists():
            os.replace(tokens_dir_orig, bak_dir)
        os.replace(tmp_dir, tokens_dir_orig)
        new_manifest["tokens_dir_compact"] = str(tokens_dir_orig)
        with open(manifest_compact_path, "w", encoding="utf-8") as f:
            json.dump(new_manifest, f, ensure_ascii=False, indent=2)
        logger.info("Inplace compaction completed. Original tokens moved to %s", bak_dir)

    return new_manifest
