from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
from torch import nn

from src.models.heads import TaskHead


class SharedBottom(nn.Module):
    """
    Shared-bottom multi-task model that wraps a backbone (e.g., DeepFM)
    and attaches per-task towers while preserving the wide (linear) term.
    """

    def __init__(self, backbone: nn.Module, head_cfg: Dict, enabled_heads: Optional[List[str]] = None):
        super().__init__()
        self.backbone = backbone

        tasks = head_cfg.get("tasks") or ["ctr", "cvr"]
        self.tasks: List[str] = tasks
        # Enabled heads are a subset (or equal) of all tasks; default to all tasks.
        enabled = enabled_heads or tasks
        self.enabled_heads: Set[str] = {h.lower() for h in enabled}

        self._head_cfg_resolved: Dict[str, Dict[str, object]] = {}

        def _pick_cfg(head_cfg: Dict, task: str, key: str, default):
            """
            Fallback only when value is missing or None (not any falsy).
            Priority: task-specific -> default block -> top-level -> default.
            """
            if isinstance(head_cfg.get(task), dict) and key in head_cfg[task] and head_cfg[task][key] is not None:
                return head_cfg[task][key]
            if isinstance(head_cfg.get("default"), dict) and key in head_cfg["default"] and head_cfg["default"][key] is not None:
                return head_cfg["default"][key]
            if key in head_cfg and head_cfg[key] is not None:
                return head_cfg[key]
            return default

        in_dim = getattr(backbone, "out_dim", head_cfg.get("in_dim"))
        if in_dim is None:
            raise ValueError("SharedBottom: backbone must expose out_dim or provide head_cfg['in_dim'].")

        # Normalize shared representation to keep tower inputs in a stable range.
        self.layernorm = nn.LayerNorm(in_dim)

        self.towers = nn.ModuleDict()
        for task in tasks:
            mlp_dims = list(_pick_cfg(head_cfg, task, "mlp_dims", []))
            dropout = float(_pick_cfg(head_cfg, task, "dropout", 0.0))
            use_bn = bool(_pick_cfg(head_cfg, task, "use_bn", False))
            activation = str(_pick_cfg(head_cfg, task, "activation", "relu"))

            self._head_cfg_resolved[task] = {
                "mlp_dims": mlp_dims,
                "dropout": dropout,
                "use_bn": use_bn,
                "activation": activation,
            }

            self.towers[task] = TaskHead(
                in_dim=in_dim,
                mlp_dims=mlp_dims,
                out=1,
                activation=activation,
                dropout=dropout,
                use_bn=use_bn,
            )
            # Bias init to prior logits to avoid saturated outputs at cold start.
            prior_map = {"ctr": 0.07, "cvr": 0.01}
            prior = prior_map.get(task, 0.5)
            bias_init = float(torch.log(torch.tensor(prior / (1 - prior))))
            if self.towers[task].out_proj.bias is not None:
                nn.init.constant_(self.towers[task].out_proj.bias, bias_init)

    def forward(self, features: Dict, dense_x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: dict produced by dataloader (EmbeddingBag inputs).
            dense_x: currently unsupported and passed through to backbone for validation.
        Returns:
            Dict mapping task name -> logit tensor of shape [B].
        """
        out = self.backbone(features, dense_x=dense_x, return_aux=True)
        if "h" not in out or "logit_linear" not in out:
            raise KeyError("Backbone must return {'h', 'logit_linear'} when return_aux=True.")

        h = out["h"]
        h = self.layernorm(h)
        linear = out["logit_linear"]
        if linear.dim() == 2 and linear.size(-1) == 1:
            linear = linear.squeeze(-1)
        if linear.dim() != 1:
            raise ValueError(f"logit_linear must broadcast to [B]; got shape {tuple(linear.shape)}")

        results: Dict[str, torch.Tensor] = {}
        # Keep wide (linear) contribution for each enabled task logit to retain DeepFM's wide+deep benefits.
        for task, head in self.towers.items():
            if task not in self.enabled_heads:
                continue
            task_logit = head(h) + linear
            results[task] = task_logit
        # Propagate optional auxiliary signals for debugging/analysis.
        if "aux" in out:
            results["aux"] = out["aux"]
        return results


__all__ = ["SharedBottom"]
