"""
Lightweight validator to turn silent processed-data bugs into hard errors.

Checks (fail -> exit 1):
1) Column existence for every (src, field) from featuremap.
2) vocab / hybrid-head OOV rate after truncation not >0.95.
3) use_value=true fields must show non-1.0 values if the field appears.
4) Hybrid/hash indices must be strictly less than total_num_embeddings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import pyarrow.parquet as pq
from src.data.featuremap_parser import load_featuremap, FeatureSpec, FeatureMap


def _check_columns(batch: Dict[str, Any], feature_map: FeatureMap, errors: List[str]):
    cols = set(batch.keys())
    for spec in feature_map.features:
        base = f"f{spec.src}_{spec.field}"
        idx_col = base + "_idx"
        if idx_col not in cols:
            errors.append(f"missing column {idx_col}")
        if spec.is_multi_hot:
            val_col = base + "_val"
            if val_col not in cols:
                errors.append(f"missing column {val_col} for multi-hot {base}")
        elif spec.use_value:
            val_col = base + "_val"
            if val_col not in cols:
                errors.append(f"missing column {val_col} for single-hot use_value {base}")


def _oov_rate_after_trunc(batch: Dict[str, Any], spec: FeatureSpec, oov_id: int) -> float:
    base = f"f{spec.src}_{spec.field}"
    idx_col = base + "_idx"
    data = batch[idx_col]
    oov = 0
    tot = 0
    if spec.is_multi_hot:
        for arr in data:
            if isinstance(arr, list):
                tot += len(arr)
                oov += sum(1 for x in arr if x == oov_id)
    else:
        tot = len(data)
        oov = sum(1 for x in data if x == oov_id)
    return oov / max(1, tot)


def _check_oov(batch: Dict[str, Any], feature_map: FeatureMap, errors: List[str], warns: List[str]):
    oov_id = feature_map.token_policy.vocab_oov_id
    for spec in feature_map.features:
        if spec.encoding not in {"vocab", "hybrid"}:
            continue
        rate = _oov_rate_after_trunc(batch, spec, oov_id)
        if rate > 0.95:
            errors.append(f"OOV after trunc too high ({rate:.3f}) for f{spec.src}_{spec.field}")
        else:
            warns.append(f"OOV rate f{spec.src}_{spec.field}: {rate:.4f}")


def _check_values(batch: Dict[str, Any], feature_map: FeatureMap, errors: List[str], warns: List[str]):
    for spec in feature_map.features:
        if not spec.use_value:
            continue
        base = f"f{spec.src}_{spec.field}"
        val_col = base + "_val"
        if val_col not in batch:
            continue
        vals = batch[val_col]
        seen = 0
        non_one = 0
        for arr in vals:
            if isinstance(arr, list):
                seen += len(arr)
                non_one += sum(1 for v in arr if abs(float(v) - 1.0) > 1e-6)
            else:
                seen += 1
                non_one += 1 if abs(float(arr) - 1.0) > 1e-6 else 0
            if seen >= 256:  # small budget
                break
        if seen > 0 and non_one == 0:
            errors.append(f"values all 1.0 for use_value field {base}")
        else:
            warns.append(f"values non-1 ratio {base}: {non_one}/{seen}" if seen else f"{base} not present")


def _check_overflow(batch: Dict[str, Any], feature_map: FeatureMap, errors: List[str], warns: List[str]):
    totals = {f"f{spec.src}_{spec.field}": spec.total_num_embeddings() for spec in feature_map.features}
    for base, total in totals.items():
        idx_col = base + "_idx"
        if idx_col not in batch:
            continue
        data = batch[idx_col]
        max_idx = 0
        for arr in data:
            if isinstance(arr, list):
                if arr:
                    max_idx = max(max_idx, max(arr))
            else:
                max_idx = max(max_idx, int(arr))
        if max_idx >= total:
            errors.append(f"{base} idx overflow: {max_idx} >= {total}")
        else:
            warns.append(f"{base} max_idx {max_idx} / {total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", required=True, type=Path)
    ap.add_argument("--featuremap", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=1000)
    args = ap.parse_args()

    feature_map = load_featuremap(args.featuremap)
    pdir = args.processed_dir
    if pdir.is_file() and pdir.suffix == ".parquet":
        parquet_path = pdir
    else:
        # 找目录下的 parquet；优先 train/valid 命名，找不到则取第一个
        candidates = list(pdir.glob("*.parquet"))
        if not candidates:
            for sub in ["train", "valid"]:
                subdir = pdir / sub
                if subdir.is_dir():
                    candidates.extend(subdir.glob("*.parquet"))
        pref = [c for c in candidates if "train" in c.name.lower()] or [c for c in candidates if "valid" in c.name.lower()]
        if pref:
            parquet_path = pref[0]
        elif candidates:
            parquet_path = candidates[0]
        else:
            listing = "\n".join(str(x) for x in pdir.glob("**/*"))
            raise RuntimeError(f"No parquet file found under {pdir}. Dir listing:\n{listing}")

    pf = pq.ParquetFile(parquet_path)
    if pf.metadata.num_rows == 0:
        print(f"processed parquet empty: {parquet_path}", file=sys.stderr)
        sys.exit(1)
    batch = next(pf.iter_batches(batch_size=args.num_samples))
    batch_dict = batch.to_pydict()

    errors: List[str] = []
    warns: List[str] = []
    _check_columns(batch_dict, feature_map, errors)
    _check_oov(batch_dict, feature_map, errors, warns)
    _check_values(batch_dict, feature_map, errors, warns)
    _check_overflow(batch_dict, feature_map, errors, warns)

    for w in warns:
        print(f"[WARN] {w}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    print("validate_processed PASS")


if __name__ == "__main__":
    main()
