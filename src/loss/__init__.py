"""Loss package init."""

from .base import LossFn, get_labels
from .bce import MultiTaskBCELoss

__all__ = ["LossFn", "get_labels", "MultiTaskBCELoss"]
