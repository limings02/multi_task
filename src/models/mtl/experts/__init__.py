"""
Heterogeneous Expert Building Blocks for PLE-Lite.

This module provides:
  - registry.py: Expert factory supporting mlp/crossnet_v2/identity types
  - output_align.py: ExpertOutputAlign for heterogeneous expert output normalization
"""
from __future__ import annotations

from src.models.mtl.experts.registry import EXPERT_REGISTRY, build_expert, build_expert_list
from src.models.mtl.experts.output_align import ExpertOutputAlign, PerTaskExpertAligner

__all__ = [
    "EXPERT_REGISTRY",
    "build_expert",
    "build_expert_list",
    "ExpertOutputAlign",
    "PerTaskExpertAligner",
]
