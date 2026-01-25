"""
按 row_id 过滤 tokens，生成 train/valid 两套 token parquet。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.core.logging import get_logger

logger = get_logger(__name__)


class Membership:
    """
    轻量的 row_id membership 封装。
    优先使用 RoaringBitmap；若不可用则退回排序数组 + 二分。
    """

    def __init__(self, use_bitmap: bool = False):
        self._bitmap = None
        self._arr: Optional[np.ndarray] = None
        self._use_bitmap = use_bitmap
        if use_bitmap:
            try:
                import pyroaring as pr  # type: ignore

                self._bitmap = pr.BitMap()
            except Exception:
                logger.warning("PyRoaring 不可用，退回 numpy 数组实现。")
                self._use_bitmap = False

    def add_batch(self, values: np.ndarray):
        if self._use_bitmap and self._bitmap is not None:
            self._bitmap.update(values.tolist())
        else:
            if self._arr is None:
                self._arr = values.copy()
            else:
                self._arr = np.concatenate([self._arr, values])

    def finalize(self):
        if self._use_bitmap and self._bitmap is not None:
            return
        if self._arr is not None:
            self._arr = np.unique(self._arr)

    def contains(self, values: np.ndarray) -> np.ndarray:
        if self._use_bitmap and self._bitmap is not None:
            return np.fromiter((v in self._bitmap for v in values), dtype=bool, count=len(values))
        if self._arr is None:
            return np.zeros(len(values), dtype=bool)
        # np.isin 对排序数组性能可接受
        return np.isin(values, self._arr, assume_unique=True if np.all(np.diff(self._arr) >= 0) else False)


def build_rowid_membership(samples_split_path: Path, chunk_rows: int = 2_000_000) -> Membership:
    """
    从 split 样本表构建 row_id membership，支持高效批量 contains。
    """

    pq_file = pq.ParquetFile(samples_split_path)
    membership = Membership(use_bitmap=True)
    for batch in pq_file.iter_batches(batch_size=chunk_rows, columns=["row_id"]):
        arr = batch.column(0).to_numpy(zero_copy_only=False)
        membership.add_batch(arr)
    membership.finalize()
    logger.info("Built membership from %s", samples_split_path)
    return membership


def _open_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    if path.exists():
        path.unlink()
    return pq.ParquetWriter(path, schema=schema, compression="snappy")


def filter_tokens_by_rowids(
    tokens_dir: Path,
    train_membership: Membership,
    valid_membership: Membership,
    out_dir: Path,
    flush_rows: int = 2_000_000,
    overwrite: bool = False,
) -> Dict[str, int]:
    """
    将 tokens_dir 中的 tokens 按 row_id 过滤为 train/valid 两套。
    """

    if out_dir.exists() and not overwrite:
        raise FileExistsError(f"Output tokens split dir exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = out_dir / "train_tokens"
    valid_dir = out_dir / "valid_tokens"
    for d in (train_dir, valid_dir):
        if d.exists() and overwrite:
            for f in d.glob("*.parquet"):
                f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    input_files = sorted(tokens_dir.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet parts found in {tokens_dir}")

    schema = pq.ParquetFile(input_files[0]).schema_arrow

    def make_state(prefix: str, directory: Path):
        return {
            "dir": directory,
            "writer": None,
            "buffer": [],
            "buffer_rows": 0,
            "current_rows": 0,
            "row_min": None,
            "row_max": None,
            "file_count": 0,
            "rows_total": 0,
        }

    train_state = make_state("train", train_dir)
    valid_state = make_state("valid", valid_dir)

    def flush(state, prefix: str):
        if state["writer"] is None:
            return
        if state["buffer"]:
            tbl = pa.concat_tables(state["buffer"])
            state["writer"].write_table(tbl)
            state["current_rows"] += tbl.num_rows
            state["buffer"].clear()
        state["writer"].close()
        outfile = state["dir"] / f"tokens_{prefix}.part_{state['row_min']}_{state['row_max']}.parquet"
        temp_path = state.get("temp_path")
        if temp_path and temp_path.exists():
            temp_path.rename(outfile)
        state["rows_total"] += state["current_rows"]
        state["file_count"] += 1
        logger.info(
            "[%s] wrote part %s rows=%d range=[%s,%s]",
            prefix,
            outfile,
            state["current_rows"],
            state["row_min"],
            state["row_max"],
        )
        state.update({"writer": None, "temp_path": None, "current_rows": 0, "row_min": None, "row_max": None})

    for idx, path in enumerate(input_files):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches():
            row_ids = batch.column("row_id").to_numpy(zero_copy_only=False)
            train_mask = train_membership.contains(row_ids)
            valid_mask = valid_membership.contains(row_ids)

            def handle(mask, state, prefix):
                nonlocal batch
                if not mask.any():
                    return
                sub_tbl = pa.Table.from_batches([batch.filter(pa.array(mask))])
                if state["writer"] is None:
                    temp_path = state["dir"] / f"tokens_{prefix}_{state['file_count']}.tmp"
                    state["temp_path"] = temp_path
                    state["writer"] = pq.ParquetWriter(temp_path, schema=schema, compression="snappy")
                state["buffer"].append(sub_tbl)
                state["buffer_rows"] += sub_tbl.num_rows
                rmin = pc.min(sub_tbl.column("row_id")).as_py()
                rmax = pc.max(sub_tbl.column("row_id")).as_py()
                state["row_min"] = rmin if state["row_min"] is None else min(state["row_min"], rmin)
                state["row_max"] = rmax if state["row_max"] is None else max(state["row_max"], rmax)
                if state["buffer_rows"] >= flush_rows:
                    tbl = pa.concat_tables(state["buffer"])
                    state["writer"].write_table(tbl)
                    state["current_rows"] += tbl.num_rows
                    state["buffer"].clear()
                    state["buffer_rows"] = 0
                if state["current_rows"] >= flush_rows:
                    flush(state, prefix)

            handle(train_mask, train_state, "train")
            handle(valid_mask, valid_state, "valid")
        logger.info("Processed tokens file %d/%d: %s", idx + 1, len(input_files), path)

    flush(train_state, "train")
    flush(valid_state, "valid")

    return {
        "train_token_rows": train_state["rows_total"],
        "valid_token_rows": valid_state["rows_total"],
        "train_files_count": train_state["file_count"],
        "valid_files_count": valid_state["file_count"],
    }

