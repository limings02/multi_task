from __future__ import annotations

"""
Expected Calibration Error (ECE) utilities.

Both CTR and CVR use the same binning logic; CVR must be computed on clicked
samples only, so callers should supply a mask when needed.
"""

from typing import Dict, List, Optional

import numpy as np
import torch

from src.eval.metrics import sigmoid


def ece_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> Dict[str, object]:
    """
    Compute per-bin calibration stats for reliability diagrams.

    Returns:
        {
            "ece": float | None,
            "bins": [
                {"bin_left": float, "bin_right": float, "n": int,
                 "avg_pred": float, "avg_label": float, "abs_gap": float},
                ...
            ],
        }
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    N = y_true.shape[0]
    if N == 0:
        return {"ece": None, "bins": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, object]] = []
    ece = 0.0

    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        # Include right edge only for the last bin to cover prob==1.0.
        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        n_bin = int(mask.sum())
        if n_bin == 0:
            bins.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "n": 0,
                    "avg_pred": None,
                    "avg_label": None,
                    "abs_gap": None,
                }
            )
            continue

        avg_pred = float(y_prob[mask].mean())
        avg_label = float(y_true[mask].mean())
        abs_gap = abs(avg_pred - avg_label)
        bins.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "n": n_bin,
                "avg_pred": avg_pred,
                "avg_label": avg_label,
                "abs_gap": abs_gap,
            }
        )
        ece += (n_bin / N) * abs_gap

    return {"ece": float(ece), "bins": bins}


def compute_ece_from_logits(
    y_true,
    y_logit,
    mask=None,
    n_bins: int = 20,
) -> Dict[str, object]:
    """
    Convenience wrapper: applies optional mask, converts logits -> prob, then
    delegates to `ece_stats`.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_logit_arr = np.asarray(y_logit, dtype=np.float64)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        y_true_arr = y_true_arr[mask_arr]
        y_logit_arr = y_logit_arr[mask_arr]

    if y_true_arr.shape[0] == 0:
        return {"ece": None, "bins": []}

    y_prob = sigmoid(torch.from_numpy(y_logit_arr)).numpy()
    return ece_stats(y_true_arr, y_prob, n_bins=n_bins)


__all__ = ["ece_stats", "compute_ece_from_logits"]
