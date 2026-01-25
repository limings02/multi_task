"""
样本切分工具：基于 entity_id 的稳定哈希划分 train/valid（可选 test），避免泄漏公共特征。
"""

from __future__ import annotations

import json
import math
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.core.logging import get_logger

logger = get_logger(__name__)


def stable_hash_u64(s: str) -> int:
    """
    生成跨平台稳定的 64bit 哈希。优先 xxhash，更快；若不可用则退回 blake2b(digest_size=8)。
    """

    try:
        import xxhash  # type: ignore

        return xxhash.xxh64(s).intdigest()
    except Exception:
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
        return int.from_bytes(h.digest(), byteorder="big", signed=False)


def _build_thresholds(ratios: Dict[str, float]) -> List[Tuple[str, float]]:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("ratios sum must be positive.")
    thresholds: List[Tuple[str, float]] = []
    acc = 0.0
    for split, r in ratios.items():
        acc += r / total
        thresholds.append((split, acc))
    if abs(thresholds[-1][1] - 1.0) > 1e-6:
        thresholds[-1] = (thresholds[-1][0], 1.0)
    return thresholds


def assign_split(entity_id: str, seed: int, thresholds: List[Tuple[str, float]]) -> str:
    """按哈希结果落到区间，保证同 seed 同 entity_id 永远落同一 split。"""

    u = stable_hash_u64(f"{seed}:{entity_id}") / (2**64)
    for split, thr in thresholds:
        if u < thr:
            return split
    return thresholds[-1][0]


def materialize_entity_hash_split(
    samples_path: Path,
    out_dir: Path,
    seed: int,
    ratios: Dict[str, float],
    overwrite: bool,
    chunksize: int = 2_000_000,
) -> Dict:
    """
    按 entity_id 稳定哈希切分样本表；流式处理避免一次性读 4e7+ 行。
    """

    thresholds = _build_thresholds(ratios)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_specs = {
        "method": "entity_hash",
        "seed": seed,
        "ratios": ratios,
        "thresholds": thresholds,
        "hash_fn": "xxhash64|blake2b",
    }
    spec_path = out_dir / "split_spec.json"
    stats_path = out_dir / "split_stats.json"

    if not overwrite and spec_path.exists():
        logger.info("Split outputs already exist at %s; overwrite=False, skipping.", out_dir)
        return {"spec": str(spec_path), "stats": str(stats_path)}

    pq_file = pq.ParquetFile(samples_path)
    writers: Dict[str, pq.ParquetWriter] = {}
    schema = pq_file.schema_arrow

    # 统计：行数、y1/y2 总和、funnel 异常、unique entity（用哈希减少内存）
    split_row_counts = defaultdict(int)
    split_y1 = defaultdict(float)
    split_y2 = defaultdict(float)
    split_funnel_bad = defaultdict(int)
    split_entity_hash = defaultdict(set)

    def _get_writer(split_name: str) -> pq.ParquetWriter:
        if split_name not in writers:
            out_path = out_dir / f"samples_{split_name}.parquet"
            if out_path.exists() and overwrite:
                out_path.unlink()
            writers[split_name] = pq.ParquetWriter(out_path, schema=schema, compression="snappy")
        return writers[split_name]

    batch_iter = pq_file.iter_batches(batch_size=chunksize)
    for idx, batch in enumerate(batch_iter):
        tbl = pa.Table.from_batches([batch])
        entity_col = tbl.column("entity_id").to_pylist()
        splits = [assign_split(eid, seed, thresholds) for eid in entity_col]
        split_arr = pa.array(splits)
        for split_name in ratios.keys():
            mask = pc.equal(split_arr, split_name)
            if pc.any(mask).as_py():
                sub_tbl = tbl.filter(mask)
                writer = _get_writer(split_name)
                writer.write_table(sub_tbl)
                rows = sub_tbl.num_rows
                split_row_counts[split_name] += rows
                split_y1[split_name] += pc.sum(sub_tbl.column("y1")).as_py()
                split_y2[split_name] += pc.sum(sub_tbl.column("y2")).as_py()
                bad = pc.sum(
                    pc.and_(
                        pc.equal(sub_tbl.column("y2"), 1),
                        pc.equal(sub_tbl.column("y1"), 0),
                    )
                ).as_py()
                split_funnel_bad[split_name] += bad
                # 记录 entity 哈希，减少内存占用
                for eid in sub_tbl.column("entity_id").to_pylist():
                    split_entity_hash[split_name].add(stable_hash_u64(eid))
        if (idx + 1) % 10 == 0:
            logger.info("Processed %d batches (~%d rows)", idx + 1, (idx + 1) * chunksize)

    for w in writers.values():
        w.close()

    split_stats = {}
    for split_name in ratios.keys():
        rows = split_row_counts[split_name]
        y1_sum = split_y1[split_name]
        y2_sum = split_y2[split_name]
        unique_e = len(split_entity_hash[split_name])
        ctr = y1_sum / rows if rows else 0.0
        cvr = y2_sum / rows if rows else 0.0
        split_stats[split_name] = {
            "rows": rows,
            "unique_entity": unique_e,
            "ctr": ctr,
            "cvr": cvr,
            "funnel_anomaly": split_funnel_bad[split_name],
        }
        logger.info(
            "Split %s: rows=%d unique_entity=%d ctr=%.4f cvr=%.4f funnel_anomaly=%d",
            split_name,
            rows,
            unique_e,
            ctr,
            cvr,
            split_funnel_bad[split_name],
        )

    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(split_specs, f, ensure_ascii=False, indent=2)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)

    return {"spec": str(spec_path), "stats": str(stats_path), "splits": split_stats}

