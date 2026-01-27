from __future__ import annotations

"""
Basic binary classification metrics used by the evaluation pipeline.

The helpers in this module are deliberately dependency-light and safe:
- `auc` falls back to None when sklearn is unavailable or labels are degenerate.
- Probabilities are always derived from logits via sigmoid to keep behaviour
  consistent between training/eval.
"""

from typing import Dict, Optional

import numpy as np
import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable sigmoid wrapper.
    Using torch.sigmoid keeps behaviour identical to training code.
    """
    return torch.sigmoid(x)


def logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    """
    Binary log loss with clipping for numerical stability.

    Args:
        y_true: array of shape (N,), values in {0,1}.
        y_prob: array of predicted probabilities in [0,1].
        eps: small constant to avoid log(0).
    """
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)).mean())


def auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    ROC-AUC with graceful degradation when sklearn is unavailable or labels
    contain a single class.
    """
    # Degenerate label set -> undefined AUC
    unique = np.unique(y_true)
    if unique.size < 2:
        return None

    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except ImportError:
        return None

    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Any runtime error (e.g., all scores equal) returns None per spec.
        return None


def compute_binary_metrics(
    y_true,
    y_logit,
    mask=None,
) -> Dict[str, Optional[float]]:
    """
    Compute binary metrics from logits (probabilities are derived via sigmoid).

    Args:
        y_true: array-like of shape (N,)
        y_logit: array-like of shape (N,)
        mask: optional boolean mask of shape (N,) to subset CVR on clicked rows
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_logit_arr = np.asarray(y_logit, dtype=np.float64)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        y_true_arr = y_true_arr[mask_arr]
        y_logit_arr = y_logit_arr[mask_arr]

    n = int(y_true_arr.shape[0])
    if n == 0:
        return {
            "logloss": None,
            "auc": None,
            "pos_rate": None,
            "pred_mean": None,
            "n": 0,
        }

    y_prob = sigmoid(torch.from_numpy(y_logit_arr)).numpy()
    return {
        "logloss": logloss(y_true_arr, y_prob),
        "auc": auc(y_true_arr, y_prob),
        "pos_rate": float(y_true_arr.mean()),
        "pred_mean": float(y_prob.mean()),
        "n": n,
    }


__all__ = ["sigmoid", "logloss", "auc", "compute_binary_metrics"]
