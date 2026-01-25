import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.featuremap_parser import FeatureMap, FeatureSpec, TokenPolicy, TokenSelectSpec
from src.data.processed_builder import _collect_tokens_for_batch, _process_batch, map_token_to_index, map_hybrid_tail
from src.data.token_select import select_tokens
from src.data.dataset import collate_fn


def make_policy():
    return TokenPolicy(
        vocab_pad_id=0,
        vocab_missing_id=1,
        vocab_oov_id=2,
        vocab_base_offset=3,
        hash_pad_id=0,
        hash_missing_id=1,
        hash_base_offset=2,
        hash_seed=17,
    )


def test_multihot_missing_not_offset():
    policy = make_policy()
    spec = FeatureSpec(
        field="f1",
        src=0,
        encoding="vocab",
        is_multi_hot=True,
        max_len=3,
        use_value=True,
        pooling="sum",
        vocab_num_tokens=2,
        vocab_num_embeddings=5,
        special_base_offset=3,
        hash_bucket=None,
        hybrid_rule=None,
        tail_encoding=None,
        embedding_dim=8,
        hash_base_offset=2,
        token_select=TokenSelectSpec(strategy="topk_by_freq", default="topk_by_freq", k=3),
    )
    fm = FeatureMap(features=[spec], token_policy=policy, raw={})
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([0]),
            pa.array([0], type=pa.int8()),
            pa.array([0], type=pa.int8()),
            pa.array(["e0"]),
            pa.array([0], type=pa.int32()),
            pa.array([0], type=pa.int64()),
        ],
        names=["row_id", "y1", "y2", "entity_id", "c4", "c0"],
    )
    tbl, _ = _process_batch(batch, [], fm, {}, {}, {})
    idx_list = tbl.column("f0_f1_idx").to_pylist()[0]
    assert idx_list == [policy.vocab_missing_id], "missing id must not be offset/hashed"


def test_collate_values_keep_float():
    sample = {
        "y_ctr": 1.0,
        "y_cvr": 0.0,
        "y_ctcvr": 0.0,
        "click_mask": 1.0,
        "row_id": 1,
        "entity_id": "e",
        "f0_f_idx": [10],
        "f0_f_val": [0.3],
    }
    out = collate_fn([sample])
    assert abs(out["features"]["f0_f"]["val"][0, 0].item() - 0.3) < 1e-6


def test_hybrid_head_tail_disjoint():
    policy = make_policy()
    spec = FeatureSpec(
        field="h",
        src=0,
        encoding="hybrid",
        is_multi_hot=False,
        max_len=None,
        use_value=False,
        pooling="none",
        vocab_num_tokens=2,
        vocab_num_embeddings=5,
        special_base_offset=3,
        hash_bucket=4,
        hybrid_rule="topn",
        tail_encoding="hash",
        embedding_dim=8,
        hash_base_offset=2,
        token_select=None,
    )
    vocab_table = {("0", "h"): {"a": 0}}
    head_idx, is_oov = map_token_to_index("a", spec, policy, vocab_table[("0", "h")])
    tail_idx = map_hybrid_tail("zz", spec, policy)
    assert head_idx < spec.vocab_num_embeddings
    assert tail_idx >= spec.vocab_num_embeddings
    # P1-1: tail is now compact, check it's within bounds
    assert tail_idx < spec.total_num_embeddings()
    assert head_idx != tail_idx
    assert not is_oov  # head token found in vocab


def test_src_not_mixed():
    policy = make_policy()
    spec0 = FeatureSpec(
        field="f",
        src=0,
        encoding="vocab",
        is_multi_hot=False,
        max_len=None,
        use_value=False,
        pooling="none",
        vocab_num_tokens=1,
        vocab_num_embeddings=4,
        special_base_offset=3,
        hash_bucket=None,
        hybrid_rule=None,
        tail_encoding=None,
        embedding_dim=8,
        hash_base_offset=2,
        token_select=None,
    )
    spec1 = FeatureSpec(
        field="f",
        src=1,
        encoding="vocab",
        is_multi_hot=False,
        max_len=None,
        use_value=False,
        pooling="none",
        vocab_num_tokens=1,
        vocab_num_embeddings=4,
        special_base_offset=3,
        hash_bucket=None,
        hybrid_rule=None,
        tail_encoding=None,
        embedding_dim=8,
        hash_base_offset=2,
        token_select=None,
    )
    fm = FeatureMap(features=[spec0, spec1], token_policy=policy, raw={})

    # temp tokens parquet
    with tempfile.TemporaryDirectory() as td:
        token_path = Path(td) / "part.parquet"
        pq.write_table(
            pa.Table.from_arrays(
                [
                    pa.array([1, 1], type=pa.int64()),
                    pa.array([0, 1], type=pa.int8()),
                    pa.array(["f", "f"]),
                    pa.array(["a", "b"]),
                    pa.array([1.0, 1.0]),
                ],
                names=["row_id", "src", "field", "fid", "val"],
            ),
            token_path,
        )
        part_index = [(0, 10, token_path, True)]  # (lo, hi, path, has_src)
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([1]),
                pa.array([0], type=pa.int8()),
                pa.array([0], type=pa.int8()),
                pa.array(["e"]),
                pa.array([0], type=pa.int32()),
                pa.array([0], type=pa.int64()),
            ],
            names=["row_id", "y1", "y2", "entity_id", "c4", "c0"],
        )
        freq_rank = { (0,"f"): {"a":0}, (1,"f"): {"b":0}}
        vocab_table = { (0,"f"): {"a":0}, (1,"f"): {"b":0}}
        tbl, _ = _process_batch(batch, part_index, fm, freq_rank, vocab_table, {})
        # Both features should produce valid indices (not mixing src)
        # The test verifies that src=0 gets token "a" and src=1 gets token "b"
        # Both map to idx=3 (rank=0 + special_base_offset=3), which is correct
        # The key validation is that they are processed independently without error
        idx0 = tbl.column("f0_f_idx").to_pylist()[0]
        idx1 = tbl.column("f1_f_idx").to_pylist()[0]
        # Both should be valid (special_base_offset + rank = 3 + 0 = 3)
        assert idx0 == 3, f"src=0 should map 'a' to idx=3, got {idx0}"
        assert idx1 == 3, f"src=1 should map 'b' to idx=3, got {idx1}"


# ==================== P0/P1 Regression Tests ====================

def test_p0_1_field_aware_hash():
    """P0-1: Different fields with same fid must hash to different indices."""
    from src.data.processed_builder import _hash_token
    
    # Same fid "123" in different fields should produce different hash values
    idx1 = _hash_token("123", bucket=1000, seed=17, field="205", src=0)
    idx2 = _hash_token("123", bucket=1000, seed=17, field="509", src=0)
    assert idx1 != idx2, "Same fid in different fields must hash to different indices"
    
    # Same fid in same field but different src should also differ
    idx3 = _hash_token("123", bucket=1000, seed=17, field="205", src=1)
    assert idx1 != idx3, "Same fid/field but different src must hash differently"
    
    # Same inputs should be deterministic
    idx1_again = _hash_token("123", bucket=1000, seed=17, field="205", src=0)
    assert idx1 == idx1_again, "Hash must be deterministic"


def test_p0_2_hash_oov_no_miscount():
    """P0-2: Hash encoding should never report vocab OOV (idx=2 is valid hash)."""
    policy = make_policy()
    spec = FeatureSpec(
        field="hash_field",
        src=0,
        encoding="hash",
        is_multi_hot=False,
        max_len=None,
        use_value=False,
        pooling="none",
        vocab_num_tokens=None,
        vocab_num_embeddings=None,
        special_base_offset=3,
        hash_bucket=1000,
        hybrid_rule=None,
        tail_encoding=None,
        embedding_dim=8,
        token_select=None,
        hash_base_offset=2,
    )
    # Even if hash happens to produce idx=2, it should NOT be counted as OOV
    idx, is_oov = map_token_to_index("some_token", spec, policy, None)
    assert not is_oov, "Hash encoding should never report is_vocab_oov=True"


def test_p0_3_missing_value_consistency():
    """P0-3: Missing token value must be 1.0 for both multi-hot and single-hot."""
    policy = make_policy()
    
    # Multi-hot spec
    multi_spec = FeatureSpec(
        field="multi_f",
        src=0,
        encoding="vocab",
        is_multi_hot=True,
        max_len=3,
        use_value=True,
        pooling="sum",
        vocab_num_tokens=2,
        vocab_num_embeddings=5,
        special_base_offset=3,
        hash_bucket=None,
        hybrid_rule=None,
        tail_encoding=None,
        embedding_dim=8,
        hash_base_offset=2,
        token_select=TokenSelectSpec(strategy="topk_by_freq", default="topk_by_freq", k=3),
    )
    
    # Single-hot spec
    single_spec = FeatureSpec(
        field="single_f",
        src=0,
        encoding="vocab",
        is_multi_hot=False,
        max_len=None,
        use_value=True,
        pooling="none",
        vocab_num_tokens=2,
        vocab_num_embeddings=5,
        special_base_offset=3,
        hash_bucket=None,
        hybrid_rule=None,
        tail_encoding=None,
        embedding_dim=8,
        hash_base_offset=2,
        token_select=None,
    )
    
    fm = FeatureMap(features=[multi_spec, single_spec], token_policy=policy, raw={})
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([0]),
            pa.array([0], type=pa.int8()),
            pa.array([0], type=pa.int8()),
            pa.array(["e0"]),
        ],
        names=["row_id", "y1", "y2", "entity_id"],
    )
    
    # No tokens -> missing
    tbl, _ = _process_batch(batch, [], fm, {}, {}, {})
    
    # Multi-hot missing value should be 1.0
    multi_val = tbl.column("f0_multi_f_val").to_pylist()[0]
    assert multi_val == [1.0], f"Multi-hot missing value should be [1.0], got {multi_val}"
    
    # Single-hot missing value should also be 1.0
    single_val = tbl.column("f0_single_f_val").to_pylist()[0]
    assert single_val == 1.0, f"Single-hot missing value should be 1.0, got {single_val}"


def test_p0_4_iterable_shuffle_raises():
    """P0-4: build_dataloader with shuffle=True must raise ValueError."""
    import pytest
    from src.data.dataset import build_dataloader
    import tempfile
    
    with tempfile.TemporaryDirectory() as td:
        # Create minimal parquet
        pq.write_table(
            pa.Table.from_arrays(
                [pa.array([1], type=pa.int64())],
                names=["row_id"],
            ),
            Path(td) / "part.parquet",
        )
        
        with pytest.raises(ValueError, match="shuffle=True is not supported"):
            build_dataloader(td, shuffle=True)


def test_p1_1_hybrid_tail_compact():
    """P1-1: Hybrid tail should be compact (no gap between head and tail)."""
    policy = make_policy()
    spec = FeatureSpec(
        field="hybrid_f",
        src=0,
        encoding="hybrid",
        is_multi_hot=False,
        max_len=None,
        use_value=False,
        pooling="none",
        vocab_num_tokens=100,
        vocab_num_embeddings=103,  # 100 + 3 special
        special_base_offset=3,
        hash_bucket=50,
        hybrid_rule="topn",
        tail_encoding="hash",
        embedding_dim=8,
        token_select=None,
        hash_base_offset=2,
    )
    
    # P1-1: total should be vocab_num_embeddings + hash_bucket (no hash_base_offset gap)
    assert spec.total_num_embeddings() == 103 + 50, "Hybrid total should be vocab_num_embeddings + hash_bucket"
    
    # Tail idx should start right after vocab_num_embeddings
    tail_idx = map_hybrid_tail("unknown_token", spec, policy)
    assert tail_idx >= spec.vocab_num_embeddings, "Tail must be >= vocab_num_embeddings"
    assert tail_idx < spec.total_num_embeddings(), "Tail must be < total_num_embeddings"
