"""
Token selection utilities that strictly avoid full-data counters.

Strategies (stable + deterministic):
- topk_by_freq : ascending freq_rank (0 = most frequent), tie by token string.
- topk_by_value: value desc, tie by token string asc.
- auto        : if values informative (non_one>=0.05 or std>=1e-3) -> default path,
                else fallback to frequency.
- auto_mix    : pick ceil(2k/3) by value then fill remaining by freq (no duplicates).

All selectors only consume the head freq_rank that caller passes in (usually
EDA topK or duckdb topN), so they are safe for large vocabularies.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.data.featuremap_parser import FeatureSpec, TokenSelectSpec


def _values_informative(values: Sequence[float], ts: TokenSelectSpec) -> bool:
    """
    Decide whether value weighting carries signal.
    - non_one ratio guards the common case of binary 1.0 weights.
    - std guards continuous scores. NaN/Inf are treated as 0.
    Complexity: O(n) over provided values (already deduped).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return False
    finite = np.isfinite(arr)
    arr = np.where(finite, arr, 0.0)
    non_one = float(np.mean(arr != 1.0))
    std = float(arr.std())
    return (non_one >= ts.non_one_ratio_gte) or (std >= ts.value_std_gte)


def _topk_by_freq(tokens: Sequence[str], k: int, freq_rank: Dict[str, int]) -> List[str]:
    ranked = sorted(tokens, key=lambda t: (freq_rank.get(t, 1_000_000_000), str(t)))
    return ranked[:k]


def _topk_by_value(value_map: Dict[str, float], k: int) -> List[Tuple[str, float]]:
    """
    Sort by value desc, tie token string. Returns (token, value) list.
    Complexity: O(n log n) with numpy lexsort, n = |value_map|.
    """
    if not value_map or k <= 0:
        return []
    tokens = np.array(list(value_map.keys()), dtype=object)
    values = np.array(list(value_map.values()), dtype=float)
    order = np.lexsort((tokens.astype(str), -values))
    order = order[:k]
    return [(str(tokens[i]), float(values[i])) for i in order]


def _auto_mix(
    value_map: Dict[str, float],
    k: int,
    freq_rank: Dict[str, int],
    ts: TokenSelectSpec,
) -> Tuple[List[str], List[float]]:
    k_val = max(0, math.ceil(k * ts.mix_value_frac))
    k_freq = max(0, k - k_val)
    by_val = _topk_by_value(value_map, k_val)
    chosen_tokens = [t for t, _ in by_val]
    chosen_vals = [v for _, v in by_val]
    if k_freq > 0:
        remaining = [t for t in value_map.keys() if t not in set(chosen_tokens)]
        freq_sel = _topk_by_freq(remaining, k_freq, freq_rank)
        chosen_tokens.extend(freq_sel)
        chosen_vals.extend([value_map[t] for t in freq_sel])
    return chosen_tokens, chosen_vals


def select_tokens(
    spec: FeatureSpec,
    token_value_pairs: List[Tuple[str, float]],
    freq_rank: Dict[str, int],
) -> Tuple[List[str], List[float], str, float]:
    """
    Main entry.
    Args:
        spec: FeatureSpec for the field.
        token_value_pairs: deduped (token, value) pairs within one row/field.
        freq_rank: head frequency rank map (token -> rank).
    Returns:
        tokens: selected tokens (<= k, deterministic).
        values: aligned values (empty if use_value=False).
        strategy_used: resolved strategy name (freq/value/auto/auto_mix).
        retained_frac: len(tokens) / max(1, len(token_value_pairs)).
    """
    ts: TokenSelectSpec = spec.token_select or TokenSelectSpec(
        strategy="auto",
        default="auto_mix" if spec.use_value else "topk_by_freq",
        k=spec.max_len,
        non_one_ratio_gte=0.05,
        value_std_gte=1e-3,
    )
    value_map = {t: float(v) for t, v in token_value_pairs}
    if not value_map:
        return [], [], ts.strategy, 0.0
    k = ts.k or spec.max_len or len(value_map)
    if k <= 0:
        return [], [], ts.strategy, 0.0

    tokens = list(value_map.keys())
    values = list(value_map.values())
    strategy = ts.strategy

    if strategy == "topk_by_freq":
        toks = _topk_by_freq(tokens, k, freq_rank)
        return toks, [value_map[t] for t in toks] if spec.use_value else [], "topk_by_freq", len(toks) / len(value_map)

    if strategy == "topk_by_value":
        pairs = _topk_by_value(value_map, k)
        if not pairs:
            return [], [], "topk_by_value", 0.0
        t, v = zip(*pairs)
        return list(t), list(v) if spec.use_value else [], "topk_by_value", len(pairs) / len(value_map)

    if strategy == "auto_mix":
        toks, vals = _auto_mix(value_map, k, freq_rank, ts)
        return toks, vals if spec.use_value else [], "auto_mix", len(toks) / len(value_map)

    # auto or fallback
    if _values_informative(values, ts):
        if ts.default == "auto_mix":
            toks, vals = _auto_mix(value_map, k, freq_rank, ts)
            return toks, vals if spec.use_value else [], "auto_mix", len(toks) / len(value_map)
        if ts.default == "topk_by_value":
            pairs = _topk_by_value(value_map, k)
            if not pairs:
                return [], [], "topk_by_value", 0.0
            t, v = zip(*pairs)
            return list(t), list(v) if spec.use_value else [], "topk_by_value", len(pairs) / len(value_map)

    toks = _topk_by_freq(tokens, k, freq_rank)
    return toks, [value_map[t] for t in toks] if spec.use_value else [], "topk_by_freq", len(toks) / len(value_map)


__all__ = ["select_tokens"]
