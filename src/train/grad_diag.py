"""
Gradient Diagnostics Module for Multi-Task Learning

This module provides dynamic identification of shared parameters based on computational
graph analysis, rather than relying on hardcoded naming conventions.

Key Features:
- Dynamic shared parameter identification via per-task gradient computation
- Support for both dense and sparse gradients (embedding with sparse_grad=True)
- Cosine similarity computation for gradient conflict detection
- Caching of shared parameter identification for efficiency

Usage:
    from src.train.grad_diag import GradientDiagnostics
    
    diag = GradientDiagnostics(model, min_tasks=2)
    metrics = diag.compute_metrics(losses_by_task={"ctr": loss_ctr, "ctcvr": loss_ctcvr})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SharedParamInfo:
    """Information about shared parameters identified from the computational graph."""
    
    # Parameter names that are shared across tasks
    shared_param_names: List[str] = field(default_factory=list)
    # Indices into the model's named_parameters() list
    shared_param_indices: List[int] = field(default_factory=list)
    # Boolean mask indicating if each shared param has sparse gradients
    is_sparse_mask: List[bool] = field(default_factory=list)
    # Count of dense shared params
    dense_count: int = 0
    # Count of sparse shared params
    sparse_count: int = 0
    # Task names used for identification
    task_names: List[str] = field(default_factory=list)
    # Cache key used for this identification
    cache_key: str = ""
    
    @property
    def total_count(self) -> int:
        return self.dense_count + self.sparse_count


def _param_has_nonzero_grad(grad: Optional[torch.Tensor]) -> bool:
    """Check if a gradient tensor is non-zero (works for both dense and sparse)."""
    if grad is None:
        return False
    if grad.is_sparse:
        # For sparse tensors, check if there are any non-zero elements
        return grad._nnz() > 0
    else:
        # For dense tensors, check if norm > 0
        with torch.no_grad():
            return grad.abs().sum().item() > 0


def _compute_sparse_dot(
    g1: torch.Tensor, g2: torch.Tensor
) -> Tuple[float, float, float]:
    """
    Compute dot product and norms for two sparse tensors.
    
    支持 embedding 梯度：稀疏张量的 values 可能是多维的（如 [nnz, embedding_dim]）。
    在这种情况下，每个索引对应一个向量，dot product 需要对向量求内积再求和。
    
    Returns:
        (dot_product, norm1, norm2)
    """
    # Coalesce to ensure indices are sorted and unique
    g1 = g1.coalesce()
    g2 = g2.coalesce()
    
    # Get values and linearized indices
    idx1 = g1.indices()
    idx2 = g2.indices()
    val1 = g1.values().float()
    val2 = g2.values().float()
    
    # Norms are straightforward - flatten values to compute total norm
    norm1 = float(val1.norm().item())
    norm2 = float(val2.norm().item())
    
    # For dot product, we need to find matching indices
    # Linearize indices for comparison
    shape = g1.shape
    if idx1.dim() == 1:
        # 1D sparse tensor
        lin1 = idx1
        lin2 = idx2
    else:
        # Multi-dimensional sparse tensor - linearize
        # indices shape: [sparse_dim, nnz]
        # 只线性化稀疏维度，values 可能有额外的 dense 维度（如 embedding_dim）
        sparse_dim = idx1.size(0)
        strides = []
        for d in range(sparse_dim):
            stride = 1
            for dd in range(d + 1, sparse_dim):
                stride *= shape[dd]
            strides.append(stride)
        strides = torch.tensor(strides, device=idx1.device, dtype=idx1.dtype)
        lin1 = (idx1 * strides.unsqueeze(1)).sum(dim=0)
        lin2 = (idx2 * strides.unsqueeze(1)).sum(dim=0)
    
    # Sort both index arrays and use two-pointer approach for intersection
    sort1 = lin1.argsort()
    sort2 = lin2.argsort()
    sorted_idx1 = lin1[sort1]
    sorted_idx2 = lin2[sort2]
    sorted_val1 = val1[sort1]
    sorted_val2 = val2[sort2]
    
    # Two-pointer intersection for dot product
    dot = 0.0
    i, j = 0, 0
    n1, n2 = sorted_idx1.size(0), sorted_idx2.size(0)
    
    while i < n1 and j < n2:
        if sorted_idx1[i] < sorted_idx2[j]:
            i += 1
        elif sorted_idx1[i] > sorted_idx2[j]:
            j += 1
        else:
            # 处理 values 可能是向量的情况（如 embedding 梯度）
            # sorted_val1[i] 和 sorted_val2[j] 可能是向量，需要计算内积
            v1 = sorted_val1[i].reshape(-1)  # flatten to 1D
            v2 = sorted_val2[j].reshape(-1)  # flatten to 1D
            dot += float(torch.dot(v1, v2).item())
            i += 1
            j += 1
    
    return dot, norm1, norm2


class GradientDiagnostics:
    """
    Dynamic gradient diagnostics for multi-task learning.
    
    Identifies shared parameters by computing per-task gradients and finding
    parameters that receive non-zero gradients from multiple tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        min_tasks: int = 2,
        cache_refresh: str = "never",  # "never" | "epoch" | "always"
    ):
        """
        Initialize gradient diagnostics.
        
        Args:
            model: The model to analyze
            min_tasks: Minimum number of tasks a parameter must be used by to be considered shared
            cache_refresh: When to refresh the shared parameter cache
        """
        self.model = model
        self.min_tasks = min_tasks
        self.cache_refresh = cache_refresh
        
        # Cache for shared parameter info
        self._shared_info_cache: Dict[str, SharedParamInfo] = {}
        self._current_epoch = 0
        
        # Collect all trainable parameters once
        self._named_params: List[Tuple[str, nn.Parameter]] = [
            (name, param) for name, param in model.named_parameters()
            if param.requires_grad
        ]
        self._param_names = [name for name, _ in self._named_params]
    
    def _make_cache_key(self, task_names: Sequence[str]) -> str:
        """Create a cache key from task names and current state."""
        sorted_tasks = sorted(task_names)
        base_key = "_".join(sorted_tasks)
        if self.cache_refresh == "epoch":
            return f"{base_key}_epoch{self._current_epoch}"
        return base_key
    
    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for cache management."""
        self._current_epoch = epoch
    
    def resolve_shared_params(
        self,
        losses_by_task: Dict[str, torch.Tensor],
        force_refresh: bool = False,
    ) -> SharedParamInfo:
        """
        Identify parameters that are shared across multiple tasks.
        
        Uses autograd.grad to compute per-task gradients and identifies parameters
        that receive non-zero gradients from at least `min_tasks` tasks.
        
        兜底策略（当动态识别失败时）：
        - 排除 heads.ctr.* / heads.cvr.* 等 head 专属参数
        - 将剩余参数全部视为共享参数
        
        Args:
            losses_by_task: Dictionary mapping task names to their loss tensors
            force_refresh: Force recomputation even if cached
            
        Returns:
            SharedParamInfo containing identified shared parameters
        """
        task_names = list(losses_by_task.keys())
        cache_key = self._make_cache_key(task_names)
        
        # Check cache
        if not force_refresh and cache_key in self._shared_info_cache:
            return self._shared_info_cache[cache_key]
        
        # Compute per-task gradients to identify dependencies
        params = [p for _, p in self._named_params]
        
        # Track which tasks depend on each parameter
        param_task_deps: Dict[int, Set[str]] = {i: set() for i in range(len(params))}
        param_is_sparse: Dict[int, bool] = {}
        grad_compute_failed = False
        
        for task_name, loss in losses_by_task.items():
            if loss is None or not loss.requires_grad:
                logger.debug("resolve_shared_params: task %s loss is None or not requires_grad", task_name)
                continue
            
            try:
                # Compute gradients for this task
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False,
                )
                
                nonzero_count = 0
                for idx, grad in enumerate(grads):
                    if _param_has_nonzero_grad(grad):
                        param_task_deps[idx].add(task_name)
                        nonzero_count += 1
                        # Track if this param has sparse grads
                        if grad is not None:
                            param_is_sparse[idx] = grad.is_sparse
                
                logger.debug(
                    "resolve_shared_params: task %s has %d params with nonzero grad",
                    task_name, nonzero_count
                )
                            
            except RuntimeError as e:
                logger.warning("Failed to compute gradients for task %s: %s", task_name, e)
                grad_compute_failed = True
                continue
        
        # Identify shared parameters
        shared_names = []
        shared_indices = []
        is_sparse_mask = []
        dense_count = 0
        sparse_count = 0
        
        for idx, deps in param_task_deps.items():
            if len(deps) >= self.min_tasks:
                shared_names.append(self._param_names[idx])
                shared_indices.append(idx)
                is_sparse = param_is_sparse.get(idx, False)
                is_sparse_mask.append(is_sparse)
                if is_sparse:
                    sparse_count += 1
                else:
                    dense_count += 1
        
        # ====================================================================
        # 兜底策略：如果动态识别失败或没有找到共享参数，使用启发式规则
        # 排除 heads.ctr.* / heads.cvr.* / heads.ctcvr.* 参数，其余视为共享
        # ====================================================================
        if dense_count == 0 and sparse_count == 0:
            logger.warning(
                "resolve_shared_params: Dynamic identification found 0 shared params. "
                "Falling back to heuristic: all params except heads.* are shared."
            )
            for idx, name in enumerate(self._param_names):
                name_low = name.lower()
                # 排除 head 专属参数
                if "heads.ctr" in name_low or "heads.cvr" in name_low or "heads.ctcvr" in name_low:
                    continue
                shared_names.append(name)
                shared_indices.append(idx)
                # 通过参数的 grad 属性或参数名猜测是否是 sparse
                is_sparse = "emb" in name_low or "embedding" in name_low
                is_sparse_mask.append(is_sparse)
                if is_sparse:
                    sparse_count += 1
                else:
                    dense_count += 1
            
            logger.info(
                "Fallback: Identified %d shared params (%d dense, %d sparse) by heuristic",
                len(shared_names), dense_count, sparse_count
            )
        
        info = SharedParamInfo(
            shared_param_names=shared_names,
            shared_param_indices=shared_indices,
            is_sparse_mask=is_sparse_mask,
            dense_count=dense_count,
            sparse_count=sparse_count,
            task_names=task_names,
            cache_key=cache_key,
        )
        
        # Cache the result
        self._shared_info_cache[cache_key] = info
        
        logger.debug(
            "Identified %d shared params (%d dense, %d sparse) for tasks %s",
            info.total_count, dense_count, sparse_count, task_names
        )
        
        return info
    
    def compute_task_grads(
        self,
        losses_by_task: Dict[str, torch.Tensor],
        shared_info: Optional[SharedParamInfo] = None,
    ) -> Dict[str, List[Optional[torch.Tensor]]]:
        """
        Compute per-task gradients for shared parameters.
        
        Args:
            losses_by_task: Dictionary mapping task names to their loss tensors
            shared_info: Pre-computed shared parameter info (will compute if None)
            
        Returns:
            Dictionary mapping task names to lists of gradients for shared params
        """
        if shared_info is None:
            shared_info = self.resolve_shared_params(losses_by_task)
        
        if not shared_info.shared_param_indices:
            return {task: [] for task in losses_by_task}
        
        # Get the shared parameters
        shared_params = [self._named_params[i][1] for i in shared_info.shared_param_indices]
        
        task_grads = {}
        for task_name, loss in losses_by_task.items():
            if loss is None or not loss.requires_grad:
                task_grads[task_name] = [None] * len(shared_params)
                continue
            
            try:
                grads = torch.autograd.grad(
                    loss,
                    shared_params,
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=False,
                )
                task_grads[task_name] = list(grads)
            except RuntimeError as e:
                logger.warning("Failed to compute gradients for task %s: %s", task_name, e)
                task_grads[task_name] = [None] * len(shared_params)
        
        return task_grads
    
    def compute_metrics(
        self,
        losses_by_task: Dict[str, torch.Tensor],
        shared_info: Optional[SharedParamInfo] = None,
    ) -> Dict[str, Any]:
        """
        Compute gradient diagnostic metrics.
        
        Args:
            losses_by_task: Dictionary mapping task names to their loss tensors
            shared_info: Pre-computed shared parameter info (will compute if None)
            
        Returns:
            Dictionary containing:
            - shared_dense_count, shared_sparse_count
            - grad_norm_shared_{task} for each task
            - grad_cosine_shared_dense (if applicable)
            - grad_cosine_shared_sparse (if applicable)
            - conflict_detected (bool)
        """
        if shared_info is None:
            shared_info = self.resolve_shared_params(losses_by_task)
        
        metrics: Dict[str, Any] = {
            "shared_dense_count": shared_info.dense_count,
            "shared_sparse_count": shared_info.sparse_count,
            "shared_total_count": shared_info.total_count,
        }
        
        if shared_info.total_count == 0:
            # No shared parameters found
            for task in losses_by_task:
                metrics[f"grad_norm_shared_{task}"] = None
            metrics["grad_cosine_shared_dense"] = None
            metrics["grad_cosine_shared_sparse"] = None
            metrics["grad_cosine_shared_all"] = None
            metrics["conflict_detected"] = None
            return metrics
        
        # Compute per-task gradients
        task_grads = self.compute_task_grads(losses_by_task, shared_info)
        task_names = list(losses_by_task.keys())
        
        # Compute norms for each task
        for task in task_names:
            grads = task_grads[task]
            total_norm_sq = 0.0
            for grad in grads:
                if grad is not None:
                    if grad.is_sparse:
                        # Must coalesce before accessing values
                        grad_c = grad.coalesce()
                        norm_val = float(grad_c.values().float().norm().item())
                    else:
                        norm_val = float(grad.float().norm().item())
                    total_norm_sq += norm_val ** 2
            metrics[f"grad_norm_shared_{task}"] = total_norm_sq ** 0.5 if total_norm_sq > 0 else 0.0
        
        # Compute pairwise cosine similarities
        # For simplicity, focus on first two tasks (typically CTR and CTCVR)
        if len(task_names) >= 2:
            cosine_dense, cosine_sparse, cosine_all = self._compute_pairwise_cosine(
                task_grads[task_names[0]],
                task_grads[task_names[1]],
                shared_info.is_sparse_mask,
            )
            metrics["grad_cosine_shared_dense"] = cosine_dense
            metrics["grad_cosine_shared_sparse"] = cosine_sparse
            metrics["grad_cosine_shared_all"] = cosine_all
            metrics["conflict_detected"] = cosine_all is not None and cosine_all < 0
        else:
            metrics["grad_cosine_shared_dense"] = None
            metrics["grad_cosine_shared_sparse"] = None
            metrics["grad_cosine_shared_all"] = None
            metrics["conflict_detected"] = None
        
        return metrics
    
    def _compute_pairwise_cosine(
        self,
        grads1: List[Optional[torch.Tensor]],
        grads2: List[Optional[torch.Tensor]],
        is_sparse_mask: List[bool],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Compute cosine similarity between two sets of gradients.
        
        Returns:
            (cosine_dense, cosine_sparse, cosine_all)
        """
        # Separate dense and sparse computations
        dense_dot = 0.0
        dense_norm1_sq = 0.0
        dense_norm2_sq = 0.0
        dense_count = 0
        
        sparse_dot = 0.0
        sparse_norm1_sq = 0.0
        sparse_norm2_sq = 0.0
        sparse_count = 0
        
        for g1, g2, is_sparse in zip(grads1, grads2, is_sparse_mask):
            if g1 is None or g2 is None:
                continue
            
            if is_sparse and g1.is_sparse and g2.is_sparse:
                # Sparse gradient handling
                dot, n1, n2 = _compute_sparse_dot(g1, g2)
                sparse_dot += dot
                sparse_norm1_sq += n1 ** 2
                sparse_norm2_sq += n2 ** 2
                sparse_count += 1
            else:
                # Dense gradient handling (or mixed - treat as dense)
                g1_flat = g1.float().reshape(-1) if not g1.is_sparse else g1.to_dense().float().reshape(-1)
                g2_flat = g2.float().reshape(-1) if not g2.is_sparse else g2.to_dense().float().reshape(-1)
                dense_dot += float(torch.dot(g1_flat, g2_flat).item())
                dense_norm1_sq += float((g1_flat ** 2).sum().item())
                dense_norm2_sq += float((g2_flat ** 2).sum().item())
                dense_count += 1
        
        # Compute cosine similarities
        eps = 1e-12
        
        cosine_dense = None
        if dense_count > 0 and dense_norm1_sq > 0 and dense_norm2_sq > 0:
            cosine_dense = dense_dot / ((dense_norm1_sq ** 0.5) * (dense_norm2_sq ** 0.5) + eps)
        
        cosine_sparse = None
        if sparse_count > 0 and sparse_norm1_sq > 0 and sparse_norm2_sq > 0:
            cosine_sparse = sparse_dot / ((sparse_norm1_sq ** 0.5) * (sparse_norm2_sq ** 0.5) + eps)
        
        # Combined cosine (weighted by norm)
        cosine_all = None
        total_dot = dense_dot + sparse_dot
        total_norm1_sq = dense_norm1_sq + sparse_norm1_sq
        total_norm2_sq = dense_norm2_sq + sparse_norm2_sq
        if total_norm1_sq > 0 and total_norm2_sq > 0:
            cosine_all = total_dot / ((total_norm1_sq ** 0.5) * (total_norm2_sq ** 0.5) + eps)
        
        return cosine_dense, cosine_sparse, cosine_all


# Legacy compatibility: wrap old interface
def resolve_shared_params_legacy(
    model: nn.Module,
    losses_by_task: Dict[str, torch.Tensor],
    min_tasks: int = 2,
) -> SharedParamInfo:
    """Legacy interface for resolving shared parameters."""
    diag = GradientDiagnostics(model, min_tasks=min_tasks)
    return diag.resolve_shared_params(losses_by_task)


__all__ = [
    "GradientDiagnostics",
    "SharedParamInfo",
    "resolve_shared_params_legacy",
]
