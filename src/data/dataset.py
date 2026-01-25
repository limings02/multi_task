"""
Dataset/Loader helper for processed data.

Key behaviours:
- Multi-worker safe: shards parquet fragments per worker to avoid duplicate reads.
- Value handling: only pad/return *_val when FeatureSpec.use_value is True.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


def _load_feature_meta(meta_path: Path) -> Tuple[Dict[str, Dict[str, bool]], Dict[str, int]]:
    """
    Read metadata.json produced by processed_builder to recover per-field flags and row counts.
    Returns (feature_meta, rows_dict) where rows_dict maps split name -> row count.
    """
    if not meta_path.exists():
        return {}, {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, {}
    return meta.get("feature_meta", {}), meta.get("rows", {})


class ProcessedIterDataset(IterableDataset):
    """
    Stream rows from processed parquet using pyarrow.dataset.
    Worker sharding is done by slicing the fragment list so each worker reads
    disjoint files and no examples are duplicated.
    """

    def __init__(self, data_dir: str | Path, metadata_path: str | Path | None = None):
        self.data_dir = Path(data_dir)
        self.dataset = ds.dataset(self.data_dir, format="parquet")
        self.fragments = list(self.dataset.get_fragments())
        meta_path = Path(metadata_path) if metadata_path else self.data_dir.parent / "metadata.json"
        self.feature_meta, rows_dict = _load_feature_meta(meta_path)
        # P1-6: prefer pre-computed row count from metadata to avoid expensive count_rows()
        split_name = self.data_dir.name  # e.g., "train" or "valid"
        if split_name in rows_dict:
            self._count = int(rows_dict[split_name])
        else:
            self._count = self.dataset.count_rows()  # fallback

    def __iter__(self) -> Iterator[Dict]:
        info = get_worker_info()
        fragments = self.fragments
        if info is not None and info.num_workers > 1:
            fragments = fragments[info.id :: info.num_workers]
        for frag in fragments:
            for batch in frag.to_batches():
                tbl = batch.to_pydict()
                rows = len(next(iter(tbl.values())))
                for i in range(rows):
                    yield {k: v[i] for k, v in tbl.items()}

    def __len__(self) -> int:
        return self._count


def _pad_indices(seqs: List[List[int]], pad_val: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_val, dtype=torch.int64)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.int64)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.int64)
    return out, lengths


def _pad_values(seqs: List[List[float]], pad_val: float = 0.0) -> torch.Tensor:
    max_len = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), max_len), pad_val, dtype=torch.float32)
    for i, s in enumerate(seqs):
        if s:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.float32)
    return out


def _collate_factory(feature_meta: Dict[str, Dict[str, bool]]):
    """
    Build a collate_fn that honours per-field use_value flag.
    """

    def collate_fn(batch: List[Dict]):
        labels = {
            k: torch.tensor([b[k] for b in batch], dtype=torch.float32)
            for k in ["y_ctr", "y_cvr", "y_ctcvr", "click_mask"]
        }
        labels["row_id"] = torch.tensor([b["row_id"] for b in batch], dtype=torch.int64)
        labels["entity_id"] = [b["entity_id"] for b in batch]

        features: Dict[str, Dict[str, torch.Tensor | None]] = {}
        # Iterate over keys once to avoid repeated string ops.
        for key in batch[0].keys():
            if not (key.startswith("f") and key.endswith("_idx")):
                continue
            base = key[:-4]
            meta = feature_meta.get(base, {})
            use_value = meta.get("use_value")
            if use_value is None:
                use_value = (base + "_val") in batch[0]  # fallback: infer from presence
            use_value = bool(use_value)
            is_multi = bool(meta.get("is_multi_hot", isinstance(batch[0][key], list)))
            if is_multi:
                idx_seqs = [b[key] for b in batch]
                idx_pad, lengths = _pad_indices(idx_seqs, pad_val=0)
                feat = {"type": "multi", "idx": idx_pad, "len": lengths}
                # P1-5: *_off column removed from processed data (was semantically mismatched)
                val_key = base + "_val"
                if use_value and val_key in batch[0]:
                    val_seqs = [b[val_key] for b in batch]
                    feat["val"] = _pad_values(val_seqs, pad_val=0.0)
                else:
                    feat["val"] = None
                features[base] = feat
            else:
                idx_tensor = torch.tensor([b[key] for b in batch], dtype=torch.int64)
                feat = {"type": "single", "idx": idx_tensor}
                val_key = base + "_val"
                if use_value and val_key in batch[0]:
                    feat["val"] = torch.tensor([b[val_key] for b in batch], dtype=torch.float32)
                else:
                    feat["val"] = None
                features[base] = feat
        return {"labels": labels, "features": features}

    return collate_fn


def build_dataloader(
    data_dir: str | Path,
    batch_size: int = 1024,
    shuffle: bool = False,
    num_workers: int = 0,
    metadata_path: str | Path | None = None,
) -> DataLoader:
    # P0-4: IterableDataset does not support shuffle=True in DataLoader
    if shuffle:
        raise ValueError(
            "shuffle=True is not supported with IterableDataset. "
            "For shuffling, either shuffle the parquet fragments offline, "
            "or use a map-style dataset."
        )
    ds_obj = ProcessedIterDataset(data_dir, metadata_path=metadata_path)
    collate = _collate_factory(ds_obj.feature_meta)
    return DataLoader(
        ds_obj,
        batch_size=batch_size,
        shuffle=False,  # always False for IterableDataset
        num_workers=num_workers,
        collate_fn=collate,
    )


# Backward-compatible default collate_fn (uses no metadata; val returned only if present).
collate_fn = _collate_factory({})


__all__ = ["ProcessedIterDataset", "build_dataloader", "collate_fn"]
