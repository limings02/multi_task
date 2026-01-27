from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

import torch


class LossFn(Protocol):
    """Protocol for loss objects."""

    def compute(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        ...


def get_labels(batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Robustly extract labels dict.

    Supports:
    - batch = {"labels": labels, ...}
    - batch = labels (already a dict)
    """
    if "labels" in batch and isinstance(batch["labels"], dict):
        return batch["labels"]
    if all(isinstance(v, torch.Tensor) for v in batch.values()):
        return batch  # assume already labels
    raise KeyError("Cannot find labels: batch must be a labels dict or contain a 'labels' key.")


__all__ = ["LossFn", "get_labels"]
