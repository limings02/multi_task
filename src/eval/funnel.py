from __future__ import annotations

"""
Funnel consistency metrics for multi-task CTR/CVR models.

The core idea is to compare multiplicative consistency:
    pred_ctcvr = pred_ctr * pred_cvr
against either the true ctcvr label (if present) or the model's explicit ctcvr
head (if available in inputs).
"""

from typing import Dict, Optional

import numpy as np


def _to_numpy(arr):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr
    try:
        import torch

        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)


def funnel_consistency(df_or_arrays, has_ctcvr_label: bool) -> Dict[str, Optional[float]]:
    """
    Compute simple gap statistics between predicted CTCVR (ctr * cvr) and
    observed/target ctcvr.

    Args:
        df_or_arrays: either a pandas/pyarrow table-like object with column
            access, or a dict containing arrays for:
            - "pred_ctr", "pred_cvr" (probabilities)
            - optional "pred_ctcvr_head" (if model outputs a direct head)
            - optional "y_ctcvr"
        has_ctcvr_label: whether y_ctcvr is expected/available.
    """
    # Extract columns defensively
    get_col = (
        df_or_arrays.get
        if isinstance(df_or_arrays, dict)
        else lambda k: df_or_arrays[k] if k in df_or_arrays else None
    )

    pred_ctr = _to_numpy(get_col("pred_ctr"))
    pred_cvr = _to_numpy(get_col("pred_cvr"))
    if pred_ctr is None or pred_cvr is None:
        return {"n": 0}

    pred_ctcvr = pred_ctr * pred_cvr
    pred_ctcvr_head = _to_numpy(get_col("pred_ctcvr")) or _to_numpy(get_col("pred_ctcvr_head"))
    y_ctcvr = _to_numpy(get_col("y_ctcvr")) if has_ctcvr_label else None

    if pred_ctcvr_head is not None and pred_ctcvr_head.shape[0] == pred_ctcvr.shape[0]:
        target = pred_ctcvr_head
    elif y_ctcvr is not None:
        target = y_ctcvr
    else:
        target = None

    n = int(pred_ctcvr.shape[0])
    result: Dict[str, Optional[float]] = {
        "n": n,
        "mean_pred_ctcvr": float(np.mean(pred_ctcvr)) if n > 0 else None,
    }

    if target is None or n == 0:
        # No reference to compare against.
        result.update(
            {
                "mean_y_ctcvr": None,
                "gap_mean": None,
                "gap_p50": None,
                "gap_p90": None,
                "gap_p99": None,
            }
        )
        return result

    gaps = pred_ctcvr - target
    result.update(
        {
            "mean_y_ctcvr": float(np.mean(target)),
            "gap_mean": float(np.mean(gaps)),
            "gap_p50": float(np.quantile(gaps, 0.5)),
            "gap_p90": float(np.quantile(gaps, 0.9)),
            "gap_p99": float(np.quantile(gaps, 0.99)),
        }
    )
    return result


__all__ = ["funnel_consistency"]
