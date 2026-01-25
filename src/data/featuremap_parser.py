"""
FeatureMap parser and validation helpers.

This module loads ``configs/dataset/featuremap.yaml`` into strongly typed
structures that downstream processing code can rely on.

Key responsibilities
--------------------
1) Parse schema (schemaA) into ``FeatureSpec`` objects.
2) Materialize token policies (vocab/hash special ids & offsets).
3) Validate invariants the repo relies on:
   - vocab_num_embeddings == vocab_num_tokens + special_base_offset
   - is_multi_hot -> max_len > 0 and pooling set
   - single hot  -> max_len is None and pooling is none
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


FeatureKey = Tuple[int, str]  # (src, field)


@dataclasses.dataclass(frozen=True)
class TokenSelectSpec:
    """
    Token selection behaviour for multi-hot fields.

    - strategy: top-level strategy, usually "auto".
    - default: fallback when auto cannot decide (auto_mix or topk_by_freq).
    - k: maximum tokens to retain (should equal max_len).
    - auto_rule: thresholds to decide whether values are informative.
    - mix: mix ratio for auto_mix (value_ratio, freq_ratio).
    """

    strategy: str = "auto"
    default: str = "auto_mix"
    k: Optional[int] = None
    non_one_ratio_gte: float = 0.05
    value_std_gte: float = 1e-3
    mix_value_frac: float = 2 / 3


@dataclasses.dataclass(frozen=True)
class TokenPolicy:
    """Global token policy shared by vocab/hash encodings."""

    vocab_pad_id: int
    vocab_missing_id: int
    vocab_oov_id: int
    vocab_base_offset: int
    hash_pad_id: int
    hash_missing_id: int
    hash_base_offset: int
    hash_seed: int

    @property
    def vocab_special_ids(self) -> Tuple[int, int, int]:
        return self.vocab_pad_id, self.vocab_missing_id, self.vocab_oov_id

    @property
    def hash_special_ids(self) -> Tuple[int, int]:
        return self.hash_pad_id, self.hash_missing_id


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """
    Parsed feature specification for a single field.

    Attributes
    ----------
    field : str
        Field id string matching tokens column ``field`` / ``fid`` prefix.
    encoding : str
        One of {"vocab", "hash", "hybrid"}.
    is_multi_hot : bool
    max_len : Optional[int]
    use_value : bool
        Whether values (weights) should be used during pooling.
    pooling : str
    vocab_num_tokens : Optional[int]
    vocab_num_embeddings : Optional[int]
    special_base_offset : int
    hash_bucket : Optional[int]
    hybrid_rule : Optional[str]
        e.g., "topn" meaning head uses vocab, tail uses hash.
    tail_encoding : Optional[str]
        Expected to be "hash" when hybrid.
    embedding_dim : int
    src : int
    """

    field: str
    encoding: str
    is_multi_hot: bool
    max_len: Optional[int]
    use_value: bool
    pooling: str
    vocab_num_tokens: Optional[int]
    vocab_num_embeddings: Optional[int]
    special_base_offset: int
    hash_bucket: Optional[int]
    hybrid_rule: Optional[str]
    tail_encoding: Optional[str]
    embedding_dim: int
    src: int
    hash_base_offset: int
    token_select: Optional[TokenSelectSpec] = None

    # convenience helpers
    def is_vocab_head(self) -> bool:
        return self.encoding in {"vocab", "hybrid"}

    def is_hashed(self) -> bool:
        return self.encoding == "hash" or (self.encoding == "hybrid" and self.tail_encoding == "hash")

    def total_num_embeddings(self) -> int:
        """
        Total rows of the embedding table this feature can emit.
        - vocab: vocab_num_embeddings
        - hash: hash_base_offset + hash_bucket
        - hybrid: vocab_num_embeddings + hash_bucket (P1-1: no gap, tail is compact)
        """
        if self.encoding == "vocab":
            return int(self.vocab_num_embeddings)
        if self.encoding == "hash":
            return int(self.hash_base_offset + (self.hash_bucket or 0))
        if self.encoding == "hybrid":
            # P1-1: hybrid tail is compact (no hash_base_offset gap)
            return int((self.vocab_num_embeddings or 0) + (self.hash_bucket or 0))
        raise ValueError(f"Unknown encoding {self.encoding}")

    def validate(self, policy: TokenPolicy) -> None:
        if self.encoding in {"vocab", "hybrid"}:
            if self.vocab_num_tokens is None or self.vocab_num_embeddings is None:
                raise ValueError(f"Feature {self.field}: vocab/hybrid must specify vocab_num_tokens and vocab_num_embeddings.")
            expected = self.vocab_num_tokens + self.special_base_offset
            if self.vocab_num_embeddings != expected:
                raise ValueError(
                    f"Feature {self.field}: vocab_num_embeddings({self.vocab_num_embeddings}) "
                    f"!= vocab_num_tokens({self.vocab_num_tokens}) + special_base_offset({self.special_base_offset})"
                )
        if self.is_multi_hot:
            if self.max_len is None or self.max_len <= 0:
                raise ValueError(f"Feature {self.field}: multi-hot must set max_len>0.")
            if self.pooling in {None, "none"}:
                raise ValueError(f"Feature {self.field}: multi-hot pooling must not be none.")
        else:
            if self.max_len is not None:
                raise ValueError(f"Feature {self.field}: single-hot must have max_len=None.")
            if self.pooling not in {None, "none"}:
                raise ValueError(f"Feature {self.field}: single-hot pooling should be none, got {self.pooling}.")
        if self.encoding == "hash":
            if self.hash_bucket is None or self.hash_bucket <= 0:
                raise ValueError(f"Feature {self.field}: hash_bucket must be positive.")
        if self.encoding == "hybrid":
            if self.hash_bucket is None or self.hash_bucket <= 0:
                raise ValueError(f"Feature {self.field}: hybrid requires hash_bucket>0 for tail hashing.")
            if self.tail_encoding not in {"hash"}:
                raise ValueError(f"Feature {self.field}: hybrid tail_encoding must be 'hash', got {self.tail_encoding}.")
            if self.vocab_num_tokens is None or self.vocab_num_embeddings is None:
                raise ValueError(f"Feature {self.field}: hybrid requires vocab_num_tokens/embeddings for head.")


@dataclasses.dataclass(frozen=True)
class FeatureMap:
    """Container for all feature specs plus invariants."""

    features: List[FeatureSpec]
    token_policy: TokenPolicy
    raw: Dict

    def feature_by_key(self) -> Dict[FeatureKey, FeatureSpec]:
        """
        Map (src, field) -> FeatureSpec to avoid silently mixing features that
        share the same field name but come from different src streams.
        """
        return {(f.src, f.field): f for f in self.features}


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_token_policy(raw: Dict) -> TokenPolicy:
    pol = raw.get("token_policy", {})
    vocab = pol.get("vocab", {})
    hash_p = pol.get("hash", {})
    return TokenPolicy(
        vocab_pad_id=int(vocab.get("pad_id", 0)),
        vocab_missing_id=int(vocab.get("missing_id", 1)),
        vocab_oov_id=int(vocab.get("oov_id", 2)),
        vocab_base_offset=int(vocab.get("special_base_offset", 3)),
        hash_pad_id=int(hash_p.get("pad_id", 0)),
        hash_missing_id=int(hash_p.get("missing_id", 1)),
        hash_base_offset=int(hash_p.get("special_base_offset", 2)),
        hash_seed=int(hash_p.get("seed_default", 17)),
    )


def _parse_token_select(raw_feat: Dict, defaults: Dict) -> Optional[TokenSelectSpec]:
    ts = raw_feat.get("token_select")
    if ts is None:
        return None
    strategy = ts.get("strategy", "auto")
    default = ts.get("default", "auto_mix")
    k = ts.get("k", raw_feat.get("max_len"))
    auto_rule = ts.get("auto_rule", {})
    non_one = auto_rule.get("non_one_ratio_gte", 0.05)
    value_std = auto_rule.get("value_std_gte", 1e-3)
    mix = ts.get("mix", {})
    mix_value_frac = mix.get("value_frac", 2 / 3)
    return TokenSelectSpec(
        strategy=strategy,
        default=default,
        k=k,
        non_one_ratio_gte=non_one,
        value_std_gte=value_std,
        mix_value_frac=mix_value_frac,
    )


def _parse_feature(raw_feat: Dict, defaults: Dict, policy: TokenPolicy) -> FeatureSpec:
    def _get(key: str):
        return raw_feat.get(key, defaults.get(key))

    token_select = _parse_token_select(raw_feat, defaults)
    spec = FeatureSpec(
        field=str(raw_feat["field"]),
        src=int(raw_feat.get("src", 0)),
        encoding=str(raw_feat["encoding"]),
        embedding_dim=int(_get("embedding_dim")),
        is_multi_hot=bool(_get("is_multi_hot")),
        max_len=_get("max_len"),
        use_value=bool(_get("use_value")),
        pooling=_get("pooling"),
        vocab_num_tokens=_get("vocab_num_tokens"),
        vocab_num_embeddings=_get("vocab_num_embeddings"),
        special_base_offset=int(_get("special_base_offset")) if _get("special_base_offset") is not None else policy.vocab_base_offset,
        hash_bucket=_get("hash_bucket"),
        hybrid_rule=_get("hybrid_rule"),
        tail_encoding=_get("tail_encoding"),
        hash_base_offset=policy.hash_base_offset,
        token_select=token_select,
    )
    if spec.is_multi_hot and spec.token_select is None:
        # Default strictly follows requirement to avoid silent drift.
        spec = dataclasses.replace(
            spec,
            token_select=TokenSelectSpec(
                strategy="auto",
                default="auto_mix" if spec.use_value else "topk_by_freq",
                k=spec.max_len,
                mix_value_frac=2 / 3,
            ),
        )
    spec.validate(policy)
    return spec


def load_featuremap(path: str | Path) -> FeatureMap:
    """Load featuremap yaml and validate invariants."""

    path = Path(path)
    raw = _load_yaml(path)
    policy = _parse_token_policy(raw)
    defaults = raw.get("defaults", {})
    features_raw = raw.get("features", [])
    feats: List[FeatureSpec] = []
    for f in features_raw:
        feats.append(_parse_feature(f, defaults, policy))

    fm = FeatureMap(features=feats, token_policy=policy, raw=raw)
    # extra invariant: unique (src, field) to prevent silent collisions
    seen_keys = set()
    for f in feats:
        key = (f.src, f.field)
        if key in seen_keys:
            raise ValueError(f"Duplicate feature for src/field: {key}")
        seen_keys.add(key)
    return fm


def featuremap_hash(fm: FeatureMap) -> str:
    """Compute a stable hash of the featuremap (excluding dynamic timestamps)."""

    canon = json.dumps(fm.raw, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


__all__ = ["FeatureSpec", "FeatureMap", "TokenPolicy", "TokenSelectSpec", "FeatureKey", "load_featuremap", "featuremap_hash"]
