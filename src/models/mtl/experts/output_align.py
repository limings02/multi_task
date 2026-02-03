"""
Expert Output Alignment Module for Heterogeneous Experts.

When using heterogeneous experts (MLP, CrossNet-v2, Identity, etc.), 
their outputs may have different statistical distributions even after 
ensuring the same output dimension. This module provides alignment 
mechanisms to stabilize gate mixing:

  - LayerNorm: Normalize each expert output (recommended for heterogeneous experts)
  - Learnable Scale: Per-expert learnable scalar multiplier (compensates for variance differences)
  - Dropout: Optional regularization

Usage:
    aligner = ExpertOutputAlign(
        num_experts=6,  # = num_shared + num_private_for_this_task
        dim=128,        # expert output dimension
        layernorm=True,
        learnable_scale=True,
        dropout=0.0,
    )
    
    # In forward: stacked_outputs [B, D, K] -> aligned [B, D, K]
    aligned = aligner(stacked_outputs)

Author: Heterogeneous Expert Extension for PLE-Lite
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ExpertOutputAlign(nn.Module):
    """
    Alignment module for heterogeneous expert outputs.
    
    Addresses the scale/distribution mismatch problem when mixing outputs from 
    different expert architectures (e.g., MLP vs CrossNet-v2).
    
    Operations applied in order:
      1. LayerNorm (optional, per-expert): Normalizes each expert's output
      2. Learnable Scale (optional, per-expert): Scalar multiplier to adjust variance
      3. Dropout (optional): Standard dropout regularization
    
    Args:
        num_experts: Number of experts (K) to align
        dim: Expert output dimension (D)
        layernorm: Whether to apply LayerNorm per expert (default: True)
        learnable_scale: Whether to use learnable scalar per expert (default: True)
        dropout: Dropout probability (default: 0.0)
        scale_init: Initial value for learnable scales (default: 1.0)
    
    Input shape: [B, D, K] (batch, dim, num_experts) - stacked expert outputs
    Output shape: [B, D, K] (same as input)
    """
    
    def __init__(
        self,
        num_experts: int,
        dim: int,
        layernorm: bool = True,
        learnable_scale: bool = True,
        dropout: float = 0.0,
        scale_init: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.use_layernorm = layernorm
        self.use_learnable_scale = learnable_scale
        self.dropout_p = dropout
        
        # Per-expert LayerNorm
        # Note: We use separate LayerNorms for each expert to allow them to learn
        # different normalization statistics (important for heterogeneous experts)
        if self.use_layernorm:
            self.layernorms = nn.ModuleList([
                nn.LayerNorm(dim) for _ in range(num_experts)
            ])
        else:
            self.layernorms = None
        
        # Per-expert learnable scale
        # Initialized to scale_init (typically 1.0) so that alignment is initially identity-like
        if self.use_learnable_scale:
            self.scales = nn.Parameter(torch.full((num_experts,), scale_init))
        else:
            self.scales = None
        
        # Dropout (shared across experts)
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        logger.debug(
            f"[ExpertOutputAlign] num_experts={num_experts}, dim={dim}, "
            f"layernorm={layernorm}, learnable_scale={learnable_scale}, dropout={dropout}"
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Align expert outputs.
        
        Args:
            x: Stacked expert outputs [B, D, K]
               - B: batch size
               - D: expert output dimension
               - K: number of experts
        
        Returns:
            Aligned outputs [B, D, K]
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, D, K], got shape {x.shape}")
        
        B, D, K = x.shape
        if K != self.num_experts:
            raise ValueError(
                f"Input num_experts ({K}) does not match configured ({self.num_experts})"
            )
        if D != self.dim:
            raise ValueError(
                f"Input dim ({D}) does not match configured ({self.dim})"
            )
        
        # Apply per-expert LayerNorm
        if self.layernorms is not None:
            # x: [B, D, K] -> process each expert dimension separately
            # LayerNorm expects [..., D], so we transpose to [B, K, D], apply, then transpose back
            aligned_list = []
            for k in range(K):
                expert_out = x[:, :, k]  # [B, D]
                aligned_list.append(self.layernorms[k](expert_out))
            x = torch.stack(aligned_list, dim=2)  # [B, D, K]
        
        # Apply per-expert learnable scale
        if self.scales is not None:
            # scales: [K] -> broadcast to [1, 1, K]
            x = x * self.scales.view(1, 1, -1)
        
        # Apply dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x
    
    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, dim={self.dim}, "
            f"layernorm={self.use_layernorm}, learnable_scale={self.use_learnable_scale}, "
            f"dropout={self.dropout_p}"
        )


class PerTaskExpertAligner(nn.Module):
    """
    Container for per-task ExpertOutputAlign modules.
    
    Since different tasks may have different numbers of experts 
    (num_shared + num_private_for_task), we need separate aligners per task.
    
    Args:
        task_num_experts: Dict[task_name -> num_experts_for_task]
        dim: Expert output dimension (same for all experts)
        layernorm: Whether to use LayerNorm
        learnable_scale: Whether to use learnable scale
        dropout: Dropout probability
    
    Usage:
        aligner = PerTaskExpertAligner(
            task_num_experts={"ctr": 5, "cvr": 6},
            dim=128,
            layernorm=True,
            learnable_scale=True,
        )
        
        # In forward:
        ctr_outputs = aligner("ctr", ctr_stacked)  # [B, D, 5]
        cvr_outputs = aligner("cvr", cvr_stacked)  # [B, D, 6]
    """
    
    def __init__(
        self,
        task_num_experts: dict[str, int],
        dim: int,
        layernorm: bool = True,
        learnable_scale: bool = True,
        dropout: float = 0.0,
        scale_init: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.task_num_experts = task_num_experts
        
        self.aligners = nn.ModuleDict()
        for task, num_exp in task_num_experts.items():
            self.aligners[task] = ExpertOutputAlign(
                num_experts=num_exp,
                dim=dim,
                layernorm=layernorm,
                learnable_scale=learnable_scale,
                dropout=dropout,
                scale_init=scale_init,
            )
        
        logger.info(
            f"[PerTaskExpertAligner] tasks={list(task_num_experts.keys())}, "
            f"dim={dim}, layernorm={layernorm}, learnable_scale={learnable_scale}"
        )
    
    def forward(self, task: str, x: torch.Tensor) -> torch.Tensor:
        """
        Align expert outputs for a specific task.
        
        Args:
            task: Task name (e.g., "ctr", "cvr")
            x: Stacked expert outputs [B, D, K_task]
        
        Returns:
            Aligned outputs [B, D, K_task]
        """
        if task not in self.aligners:
            raise KeyError(f"Task '{task}' not found in aligner. Available: {list(self.aligners.keys())}")
        return self.aligners[task](x)
    
    def __getitem__(self, task: str) -> "ExpertOutputAlign":
        """Get aligner for a specific task."""
        return self.aligners[task]  # type: ignore[return-value]


__all__ = [
    "ExpertOutputAlign",
    "PerTaskExpertAligner",
]
