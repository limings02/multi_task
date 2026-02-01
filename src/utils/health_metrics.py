"""
Health monitoring metrics computation for multi-task learning experiments.

This module provides utilities to compute:
- Label counts and reliability flags (CTR/CVR/CTCVR positives)
- Logit/probability distribution statistics (mean, std, min, max, percentiles)
- MMoE gate health metrics (mean weights, entropy, top-1 share)

All outputs are JSON-serializable Python scalars or lists.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any

import torch
import numpy as np


def _to_float(x: Any) -> Optional[float]:
    """Convert tensor/numpy/scalar to Python float, return None if invalid."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return float(x.item()) if x.ndim == 0 else float(x.flatten()[0])
    return float(x)


def _to_float_list(x: Any) -> Optional[List[float]]:
    """Convert tensor/numpy array to Python list of floats."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return [float(v) for v in x.detach().cpu().tolist()]
    if isinstance(x, np.ndarray):
        return [float(v) for v in x.tolist()]
    if isinstance(x, list):
        return [float(v) for v in x]
    return None


def compute_label_counts(
    y_ctr: torch.Tensor,
    y_cvr: torch.Tensor,
    click_mask: torch.Tensor,
    y_ctcvr: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Compute label counts and positive rates for CTR/CVR/CTCVR.

    Args:
        y_ctr: CTR labels (full exposure), shape [N]
        y_cvr: CVR labels (conversion given click), shape [N]
        click_mask: Click mask (1 where clicked), shape [N]
        y_ctcvr: CTCVR labels (optional, will be constructed as y_ctr & y_cvr if None)

    Returns:
        Dictionary with:
        - ctr_pos, ctr_pos_rate: full exposure CTR positives
        - cvr_pos_masked, cvr_pos_rate_masked: click=1 subset CVR positives
        - ctcvr_pos, ctcvr_pos_rate: full exposure CTCVR positives
        - masked_n: number of samples in click=1 subset
        - cvr_metric_reliable, ctcvr_metric_reliable: bool flags
    """
    n_total = int(y_ctr.numel())

    # CTR: full exposure
    ctr_pos = int((y_ctr > 0.5).sum().item())
    ctr_pos_rate = ctr_pos / n_total if n_total > 0 else 0.0

    # CVR: click=1 subset (masked)
    click_bool = click_mask > 0.5
    masked_n = int(click_bool.sum().item())
    cvr_pos_masked = int((y_cvr[click_bool] > 0.5).sum().item()) if masked_n > 0 else 0
    cvr_pos_rate_masked = cvr_pos_masked / masked_n if masked_n > 0 else 0.0

    # CTCVR: full exposure
    if y_ctcvr is not None:
        ctcvr_labels = y_ctcvr
    else:
        # Construct CTCVR = y_ctr & y_cvr
        ctcvr_labels = (y_ctr > 0.5) & (y_cvr > 0.5)
    ctcvr_pos = int((ctcvr_labels > 0.5).sum().item())
    ctcvr_pos_rate = ctcvr_pos / n_total if n_total > 0 else 0.0

    # Reliability flags
    cvr_metric_reliable = cvr_pos_masked >= 100
    ctcvr_metric_reliable = ctcvr_pos >= 100

    return {
        "ctr_pos": ctr_pos,
        "ctr_pos_rate": float(ctr_pos_rate),
        "cvr_pos_masked": cvr_pos_masked,
        "cvr_pos_rate_masked": float(cvr_pos_rate_masked),
        "ctcvr_pos": ctcvr_pos,
        "ctcvr_pos_rate": float(ctcvr_pos_rate),
        "masked_n": masked_n,
        "cvr_metric_reliable": bool(cvr_metric_reliable),
        "ctcvr_metric_reliable": bool(ctcvr_metric_reliable),
    }


def compute_logit_stats(
    logits: torch.Tensor,
    prefix: str,
) -> Dict[str, Optional[float]]:
    """
    Compute logit distribution statistics.

    Args:
        logits: 1D tensor of logits, shape [N]
        prefix: field name prefix (e.g., "logit_ctr")

    Returns:
        Dictionary with mean, std, min, max, p01, p99.
    """
    if logits.numel() == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_p01": None,
            f"{prefix}_p99": None,
        }

    logits_flat = logits.view(-1).float()

    mean_val = _to_float(logits_flat.mean())
    std_val = _to_float(logits_flat.std())
    min_val = _to_float(logits_flat.min())
    max_val = _to_float(logits_flat.max())

    # Quantiles
    try:
        p01 = _to_float(torch.quantile(logits_flat, 0.01))
        p99 = _to_float(torch.quantile(logits_flat, 0.99))
    except Exception:
        # Fallback for older PyTorch or edge cases
        sorted_vals = logits_flat.sort().values
        n = sorted_vals.numel()
        p01 = _to_float(sorted_vals[max(0, int(n * 0.01))])
        p99 = _to_float(sorted_vals[min(n - 1, int(n * 0.99))])

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_p01": p01,
        f"{prefix}_p99": p99,
    }


def compute_prob_stats(
    probs: torch.Tensor,
    prefix: str,
) -> Dict[str, Optional[float]]:
    """
    Compute probability distribution statistics (mean, p01, p99).

    Args:
        probs: 1D tensor of probabilities, shape [N]
        prefix: field name prefix (e.g., "prob_ctcvr")

    Returns:
        Dictionary with mean, p01, p99.
    """
    if probs.numel() == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_p01": None,
            f"{prefix}_p99": None,
        }

    probs_flat = probs.view(-1).float()

    mean_val = _to_float(probs_flat.mean())

    try:
        p01 = _to_float(torch.quantile(probs_flat, 0.01))
        p99 = _to_float(torch.quantile(probs_flat, 0.99))
    except Exception:
        sorted_vals = probs_flat.sort().values
        n = sorted_vals.numel()
        p01 = _to_float(sorted_vals[max(0, int(n * 0.01))])
        p99 = _to_float(sorted_vals[min(n - 1, int(n * 0.99))])

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_p01": p01,
        f"{prefix}_p99": p99,
    }


def compute_gate_entropy(gate_probs: torch.Tensor, eps: float = 1e-12) -> Optional[float]:
    """
    Compute average entropy of gate probability distributions.

    H(p) = -sum_i p_i * log(p_i), averaged over batch.

    Args:
        gate_probs: Gate softmax outputs, shape [B, E] where E = num_experts
        eps: Small constant to avoid log(0)

    Returns:
        Mean entropy (float)
    """
    if gate_probs.numel() == 0:
        return 0.0

    # Clamp for numerical stability
    p = gate_probs.clamp(min=eps)
    # Entropy per sample: -sum_i p_i * log(p_i)
    entropy_per_sample = -(p * torch.log(p)).sum(dim=-1)  # [B]
    return _to_float(entropy_per_sample.mean())


def compute_gate_top1_share(gate_probs: torch.Tensor) -> Optional[List[float]]:
    """
    Compute the fraction of samples where each expert has the highest gate weight.

    Args:
        gate_probs: Gate softmax outputs, shape [B, E]

    Returns:
        List of length E with top-1 share for each expert.
    """
    if gate_probs.numel() == 0:
        return None

    B, E = gate_probs.shape
    top1_idx = gate_probs.argmax(dim=-1)  # [B]
    counts = torch.zeros(E, device=gate_probs.device, dtype=torch.float32)
    for i in range(E):
        counts[i] = (top1_idx == i).sum().float()
    shares = counts / B
    return _to_float_list(shares)


def aggregate_gate_metrics(
    gate_probs_list: List[torch.Tensor],
    task: str,
) -> Dict[str, Any]:
    """
    Aggregate gate metrics over multiple batches.

    Args:
        gate_probs_list: List of gate prob tensors, each [B, E]
        task: Task name (e.g., "ctr", "cvr")

    Returns:
        Dictionary with:
        - gate_{task}_mean: list[float] of length E (mean gate weight per expert)
        - gate_{task}_entropy_mean: float (average entropy)
        - gate_{task}_top1_share: list[float] of length E
    """
    if not gate_probs_list or len(gate_probs_list) == 0:
        return {
            f"gate_{task}_mean": None,
            f"gate_{task}_entropy_mean": None,
            f"gate_{task}_top1_share": None,
        }

    # Concatenate all batches
    all_probs = torch.cat(gate_probs_list, dim=0)  # [N_total, E]

    # Mean gate weight per expert
    gate_mean = _to_float_list(all_probs.mean(dim=0))  # [E]

    # Average entropy
    entropy_mean = compute_gate_entropy(all_probs)

    # Top-1 share
    top1_share = compute_gate_top1_share(all_probs)

    return {
        f"gate_{task}_mean": gate_mean,
        f"gate_{task}_entropy_mean": entropy_mean,
        f"gate_{task}_top1_share": top1_share,
    }


__all__ = [
    "compute_label_counts",
    "compute_logit_stats",
    "compute_prob_stats",
    "compute_gate_entropy",
    "compute_gate_top1_share",
    "aggregate_gate_metrics",
]
