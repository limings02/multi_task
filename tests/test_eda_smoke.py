from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.eda.aliccp_eda import run_eda_aliccp


def _write_samples(path: Path, rows) -> None:
    table = pa.Table.from_pydict(
        {
            "row_id": np.array([r[0] for r in rows], dtype=np.int64),
            "y1": np.array([r[1] for r in rows], dtype=np.int8),
            "y2": np.array([r[2] for r in rows], dtype=np.int8),
            "entity_id": [r[3] for r in rows],
            "c4": np.array([r[4] for r in rows], dtype=np.int32),
            "c0": np.array([r[5] for r in rows], dtype=np.int64),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_tokens(path: Path, rows) -> None:
    table = pa.Table.from_pydict(
        {
            "row_id": np.array([r[0] for r in rows], dtype=np.int64),
            "src": np.array([r[1] for r in rows], dtype=np.int8),
            "field": [r[2] for r in rows],
            "fid": [r[3] for r in rows],
            "val": np.array([r[4] for r in rows], dtype=np.float32),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_run_eda_smoke(tmp_path: Path):
    split_dir = tmp_path / "splits" / "aliccp_entity_hash_v1"
    train_samples = split_dir / "samples_train.parquet"
    valid_samples = split_dir / "samples_valid.parquet"
    tokens_root = split_dir / "tokens"
    train_tokens_dir = tokens_root / "train_tokens"
    valid_tokens_dir = tokens_root / "valid_tokens"

    _write_samples(
        train_samples,
        [
            (0, 1, 1, "e1", 1, 10),
            (1, 0, 0, "e2", 2, 11),
            (2, 1, 0, "e3", 3, 12),
            (3, 0, 0, "e4", 4, 13),
        ],
    )
    _write_samples(
        valid_samples,
        [
            (4, 0, 0, "e5", 5, 14),
            (5, 1, 1, "e6", 6, 15),
        ],
    )

    _write_tokens(
        train_tokens_dir / "train_tokens.part_0_3.parquet",
        [
            (0, 0, "f1", "a", 1.0),
            (0, 1, "f1", "b", 2.0),
            (1, 0, "f2", "c", 1.0),
            (2, 0, "f1", "a", 1.0),
            (2, 0, "f2", "c", 1.0),
            (3, 1, "f3", "d", 0.5),
        ],
    )
    _write_tokens(
        valid_tokens_dir / "valid_tokens.part_4_5.parquet",
        [
            (4, 0, "f1", "a", 1.0),
            (4, 0, "f2", "x", 1.0),
            (5, 1, "f3", "z", 1.5),
        ],
    )

    tokens_manifest = {
        "train_token_rows": 6,
        "valid_token_rows": 3,
        "expected_tokens_total_rows": 9,
        "observed_tokens_total_rows": 9,
    }
    tokens_root.mkdir(parents=True, exist_ok=True)
    (tokens_root / "tokens_split_manifest.json").write_text(
        json.dumps(tokens_manifest, indent=2), encoding="utf-8"
    )
    (split_dir / "split_spec.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    (split_dir / "split_stats.json").write_text(json.dumps({"train": {}, "valid": {}}), encoding="utf-8")

    canonical_manifest_dir = tmp_path / "canonical"
    canonical_manifest_dir.mkdir(parents=True, exist_ok=True)
    (canonical_manifest_dir / "manifest.json").write_text(
        json.dumps({"tokens_total_rows": 9}, indent=2), encoding="utf-8"
    )

    config = {
        "canonical": {"out_dir": str(canonical_manifest_dir)},
        "eda": {
            "split_dir": str(split_dir),
            "out_dir": str(split_dir / "eda"),
            "report_dir": str(tmp_path / "reports"),
            "overwrite": True,
            "backend": "pyarrow",
            "batch_size": 2,
            "topk_n": 5,
            "min_support": 1,
            "topk_jaccard_n": 3,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    outputs = run_eda_aliccp(str(config_path))

    required_files = [
        split_dir / "eda" / "sanity.json",
        split_dir / "eda" / "samples_stats_train.json",
        split_dir / "eda" / "samples_stats_valid.json",
        split_dir / "eda" / "entity_freq_train.parquet",
        split_dir / "eda" / "entity_freq_valid.parquet",
        split_dir / "eda" / "row_nnz_hist_train.parquet",
        split_dir / "eda" / "row_nnz_hist_valid.parquet",
        split_dir / "eda" / "field_stats_train.parquet",
        split_dir / "eda" / "field_topk_train.parquet",
        split_dir / "eda" / "fid_lift_train.parquet",
        split_dir / "eda" / "drift_summary.parquet",
        tmp_path / "reports" / "drift_report.md",
    ]
    for path in required_files:
        assert path.exists(), f"Missing output: {path}"

    sanity = json.loads((split_dir / "eda" / "sanity.json").read_text(encoding="utf-8"))
    assert sanity["n_train_samples"] == 4
    assert outputs["backend"] == "pyarrow"
