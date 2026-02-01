"""
EmbeddingBag-friendly DataLoader utilities for processed parquet data.

Design goals
------------
- Reuse existing IterableDataset from `src.data.dataset` (no IO duplication).
- Produce per-field EmbeddingBag inputs `{indices, offsets, weights}` so model
  can hold one EmbeddingBag per feature without extra reshaping.
- Enforce deterministic base ordering to avoid silent feature shift:
    bases = sorted([k for k in row.keys() if k.startswith("f") and k.endswith("_idx")])
- Handle single-hot / multi-hot / empty bags uniformly; support optional values.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

# Prefer importing the dataset class; provide a clear error if name changes.
try:
    from src.data.dataset import ProcessedIterDataset
except Exception as exc:  # pragma: no cover - defensive
    raise ImportError(
        "Failed to import ProcessedIterDataset from src.data.dataset. "
        "Please ensure dataset.py exposes ProcessedIterDataset."
    ) from exc


# ---------- small helpers ----------

def _load_feature_meta(metadata_path: Path) -> Dict[str, Any]:
    """Load feature_meta from metadata.json; falls back to empty dict."""
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get("feature_meta", {}) or {}


def _ensure_list(obj: Any) -> List[Any]:
    """Normalize scalar or iterable to a Python list."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if torch.is_tensor(obj):
        return obj.tolist()
    return [obj]


def _check_indices(indices: List[int], base: str, row_idx: int, meta: Dict[str, Any], debug: bool) -> None:
    for idx in indices:
        if idx is None:
            raise ValueError(f"{base}: idx is None at row {row_idx}")
        if idx < 0:
            raise ValueError(f"{base}: negative idx {idx} at row {row_idx}")
    if debug:
        upper = meta.get("num_embeddings") or meta.get("vocab_num_embeddings") or meta.get("hash_bucket")
        offset = meta.get("special_base_offset") or meta.get("base_offset") or 0
        if upper is not None:
            limit = offset + int(upper)
            bad = [x for x in indices if x >= limit]
            if bad:
                raise ValueError(f"{base}: {len(bad)} indices exceed num_embeddings={limit}, sample row {row_idx}, first5={bad[:5]}")


# ---------- core collation ----------

def collate_fn_embeddingbag(
    batch: List[Dict],
    feature_meta: Dict[str, Any],
    debug: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convert a list of row dicts into per-field EmbeddingBag inputs.

    Returns:
        labels  : dict of target tensors
        features: {"fields": {base: {indices, offsets, weights/None}}, "field_names": [...]}
        meta    : {"entity_id": list[Any]}
    """
    if not batch:
        raise ValueError("Empty batch is not allowed.")

    # 1) Field discovery & deterministic ordering
    sample0 = batch[0]
    base_keys = sorted(k for k in sample0.keys() if k.startswith("f") and k.endswith("_idx"))
    bases = [k[:-4] for k in base_keys]
    if not bases:
        raise ValueError("No feature *_idx columns found in batch.")

    # Pre-pass: detect if any field uses values so we can align weights on the fly.
    any_use_value = False
    for base in bases:
        meta = feature_meta.get(base, {})
        use_value = bool(meta.get("use_value", False))
        any_use_value = any_use_value or use_value
        if any_use_value:
            break

    # 2) labels/meta with ESMM-friendly guards
    filtered_rows: List[Tuple[Dict, float, float]] = []
    y_ctr_list: List[float] = []
    y_cvr_list: List[float] = []
    y_ctcvr_list: List[float] = []
    click_mask_list: List[float] = []
    row_id_list: List[int] = []
    entity_id_list: List[Any] = []

    for row in batch:
        y_ctr = float(row.get("y_ctr", 0.0))
        raw_y_cvr = row.get("y_cvr", 0.0)
        try:
            y_cvr = float(raw_y_cvr)
        except Exception:
            y_cvr = 0.0

        if y_ctr <= 0.0 and y_cvr > 0.0:
            continue  # drop funnel-inconsistent samples entirely
        if y_ctr <= 0.0:
            y_cvr = 0.0
        filtered_rows.append((row, y_ctr, y_cvr))

    if not filtered_rows:
        raise ValueError("Batch empty after filtering funnel-inconsistent samples.")

    B = len(filtered_rows)

    for row, y_ctr, y_cvr in filtered_rows:
        click_mask = row.get("click_mask")
        if click_mask is None:
            click_mask = 1.0 if y_ctr > 0.0 else 0.0
        click_mask = float(click_mask)

        y_ctcvr = 1.0 if (y_ctr > 0.5 and y_cvr > 0.5) else 0.0

        y_ctr_list.append(y_ctr)
        y_cvr_list.append(y_cvr)
        y_ctcvr_list.append(y_ctcvr)
        click_mask_list.append(click_mask)
        row_id_list.append(int(row.get("row_id", len(row_id_list))))
        entity_id_list.append(row.get("entity_id"))

    labels = {
        "y_ctr": torch.tensor(y_ctr_list, dtype=torch.float32),
        "y_cvr": torch.tensor(y_cvr_list, dtype=torch.float32),
        "y_ctcvr": torch.tensor(y_ctcvr_list, dtype=torch.float32),
        "click_mask": torch.tensor(click_mask_list, dtype=torch.float32),
        "row_id": torch.tensor(row_id_list, dtype=torch.int64),
    }
    meta_out = {"entity_id": entity_id_list}

    # 3) per-field accumulation
    indices_map: Dict[str, List[int]] = {base: [] for base in bases}
    weights_map: Dict[str, List[float]] = {base: [] for base in bases}
    offsets_map: Dict[str, List[int]] = {base: [] for base in bases}

    for row_idx, (row, _, _) in enumerate(filtered_rows):
        for base in bases:
            meta = feature_meta.get(base, {})
            is_multi = bool(meta.get("is_multi_hot", False))
            use_value = bool(meta.get("use_value", False))
            max_len = meta.get("max_len")
            if isinstance(max_len, float) and math.isnan(max_len):
                max_len = None
            if max_len is not None:
                try:
                    max_len = int(max_len)
                except Exception:
                    max_len = None

            # idx list
            raw_idx = row.get(base + "_idx")
            idx_list = _ensure_list(raw_idx if is_multi else int(raw_idx) if raw_idx is not None else None)
            if max_len is not None:
                idx_list = idx_list[:max_len]
            _check_indices(idx_list, base, row_idx, meta, debug)

            # val list
            if use_value:
                raw_val = row.get(base + "_val")
                val_list = _ensure_list(raw_val if is_multi else raw_val)
                if not val_list:
                    val_list = [1.0] * len(idx_list)
                if max_len is not None:
                    val_list = val_list[:max_len]
                if debug and len(val_list) != len(idx_list):
                    raise ValueError(f"{base}: val/idx length mismatch at row {row_idx}")
            else:
                val_list = None

            # offsets & append
            offsets_map[base].append(len(indices_map[base]))
            indices_map[base].extend(int(x) for x in idx_list)

            if use_value:
                weights_map[base].extend(float(v) for v in (val_list or []))
            elif any_use_value:
                weights_map[base].extend(1.0 for _ in idx_list)

    # 4) pack tensors per field
    fields_out: Dict[str, Dict[str, torch.Tensor | None]] = {}
    for base in bases:
        indices = indices_map[base]
        weights = weights_map[base]
        offsets = offsets_map[base]

        indices_t = torch.tensor(indices, dtype=torch.int64) if indices else torch.empty(0, dtype=torch.int64)
        offsets_t = torch.tensor(offsets, dtype=torch.int64)

        if any_use_value:
            weights_t = torch.tensor(weights, dtype=torch.float32) if weights else torch.empty(0, dtype=torch.float32)
            assert len(weights) == len(indices), f"{base}: weights/indices length mismatch ({len(weights)} vs {len(indices)})"
        else:
            weights_t = None

        if debug:
            assert len(offsets) == B, f"{base}: offsets length != batch size"
            if offsets:
                assert offsets[0] == 0, f"{base}: offsets[0] must be 0"
            assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1)), f"{base}: offsets not non-decreasing"
            assert (not indices) or offsets[-1] <= len(indices), f"{base}: last offset exceeds indices length"

        fields_out[base] = {
            "indices": indices_t,
            "offsets": offsets_t,
            "weights": weights_t,
        }

    features = {
        "fields": fields_out,
        "field_names": bases,
    }

    return labels, features, meta_out


# ---------- DataLoader builder ----------

def _seed_worker(worker_id: int, base_seed: int) -> None:  # pragma: no cover - simple
    seed = base_seed + worker_id
    random.seed(seed)
    try:
        import numpy as np  # optional
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)


def _collate_embeddingbag(batch: List[Dict], feature_meta: Dict[str, Any], debug: bool) -> Tuple[Dict, Dict, Dict]:
    return collate_fn_embeddingbag(batch, feature_meta, debug)


def _worker_init_with_seed(worker_id: int, base_seed: int) -> None:
    _seed_worker(worker_id, base_seed)


class _SubsetIterDataset(IterableDataset):
    """
    Deterministic down-sampling wrapper for IterableDataset.
    Used to build a small, fixed validation subset for cheap AUC.
    """

    def __init__(
        self,
        base_ds: IterableDataset,
        subset_ratio: float | None = None,
        max_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        if subset_ratio is not None and (subset_ratio <= 0 or subset_ratio > 1):
            raise ValueError("subset_ratio must be in (0,1].")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        self.base_ds = base_ds
        self.subset_ratio = subset_ratio
        self.max_samples = max_samples
        self.seed = seed if seed is not None else 2026

    def __iter__(self):
        info = get_worker_info()
        worker_seed = self.seed if info is None else (self.seed + info.id)
        rng = random.Random(worker_seed)
        yielded = 0
        for row in iter(self.base_ds):
            if self.max_samples is not None and yielded >= self.max_samples:
                break
            if self.subset_ratio is not None and rng.random() > self.subset_ratio:
                continue
            yielded += 1
            yield row

    def __len__(self) -> int:  # pragma: no cover - best effort estimate
        base_len = len(self.base_ds) if hasattr(self.base_ds, "__len__") else 0
        if self.max_samples is not None:
            base_len = min(base_len, self.max_samples)
        if self.subset_ratio is not None and base_len:
            return int(base_len * self.subset_ratio)
        return base_len


def make_dataloader(
    split: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool | None = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int | None = None,
    feature_meta: Dict[str, Any] | None = None,
    debug: bool = False,
    neg_keep_prob_train: float = 1.0,
    subset_ratio: float | None = None,
    subset_max_samples: int | None = None,
    subset_seed: int | None = None,
) -> DataLoader:
    """
    Build a DataLoader that yields (labels, features, meta) tuples
    suitable for EmbeddingBag-based models.
    """
    if split not in {"train", "valid"}:
        raise ValueError("split must be 'train' or 'valid'")

    data_dir = Path("data/processed") / split
    metadata_path = data_dir.parent / "metadata.json"
    fm = feature_meta or _load_feature_meta(metadata_path)

    # IterableDataset does not support shuffle=True; guard against misuse.
    if shuffle is None:
        shuffle = split == "train"
    if shuffle:
        raise ValueError("shuffle=True is not supported with IterableDataset; shuffle offline instead.")

    ds: IterableDataset = ProcessedIterDataset(
        data_dir,
        metadata_path=metadata_path,
        neg_keep_prob=neg_keep_prob_train if split == "train" else 1.0,
    )
    if split == "valid" and (subset_ratio is not None or subset_max_samples is not None):
        ds = _SubsetIterDataset(
            ds,
            subset_ratio=subset_ratio,
            max_samples=subset_max_samples,
            seed=subset_seed if subset_seed is not None else seed,
        )

    base_seed = seed if seed is not None else torch.initial_seed()
    worker_init = partial(_worker_init_with_seed, base_seed=base_seed) if seed is not None else None
    collate = partial(_collate_embeddingbag, feature_meta=fm, debug=debug)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        collate_fn=collate,
        worker_init_fn=worker_init,
    )
    if num_workers > 0:
        logging.getLogger(__name__).info(
            "DataLoader spawn config: num_workers=%d persistent_workers=%s pin_memory=%s collate_fn=%s.%s",
            num_workers,
            bool(persistent_workers and num_workers > 0),
            pin_memory,
            collate.func.__module__ if isinstance(collate, partial) else type(collate).__module__,
            collate.func.__name__ if isinstance(collate, partial) else getattr(collate, "__name__", type(collate).__name__),
        )
    return dl


# ---------- quick self-check ----------

def _quick_test_embeddingbag_pack() -> None:
    """
    Minimal self-test: two samples with single-hot, multi-hot, and an empty bag.
    Ensures shapes/offsets/weights align with spec.
    """
    feature_meta = {
        "f0_301": {"is_multi_hot": False, "use_value": False},
        "f1_110_14": {"is_multi_hot": True, "use_value": True, "max_len": 3},
    }

    batch = [
        {
            "y_ctr": 1.0,
            "y_cvr": 0.0,
            "y_ctcvr": 0.0,
            "click_mask": 1.0,
            "row_id": 10,
            "entity_id": "uA",
            "f0_301_idx": 5,
            "f1_110_14_idx": [7, 8, 9, 10],  # will be truncated to 3
            "f1_110_14_val": [0.5, 0.4, 0.3, 0.2],
        },
        {
            "y_ctr": 0.0,
            "y_cvr": 0.0,
            "y_ctcvr": 0.0,
            "click_mask": 1.0,
            "row_id": 11,
            "entity_id": "uB",
            "f0_301_idx": 3,
            "f1_110_14_idx": [],  # empty bag allowed
        },
    ]

    labels, features, meta = collate_fn_embeddingbag(batch, feature_meta, debug=True)

    fields = features["fields"]
    # field order
    assert features["field_names"] == ["f0_301", "f1_110_14"]

    # f0_301 single-hot
    f0 = fields["f0_301"]
    assert f0["indices"].tolist() == [5, 3]
    assert f0["offsets"].tolist() == [0, 1]
    assert f0["weights"] is not None and f0["weights"].tolist() == [1.0, 1.0]  # filled because any_use_value=True

    # f1_110_14 multi-hot with truncation + empty bag
    f1 = fields["f1_110_14"]
    assert f1["indices"].tolist() == [7, 8, 9]
    assert f1["offsets"].tolist() == [0, 3]  # second sample empty -> length derived from last offset
    assert f1["weights"] is not None
    assert torch.allclose(f1["weights"], torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32))

    assert meta["entity_id"] == ["uA", "uB"]


if __name__ == "__main__":  # pragma: no cover
    _quick_test_embeddingbag_pack()
