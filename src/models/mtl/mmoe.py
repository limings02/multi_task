from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch
from torch import nn

from src.models.backbones.layers import MLP
from src.models.heads import TaskHead


class _Gate(nn.Module):
    """
    Per-task gate that outputs a softmax over experts.
    """

    def __init__(self, in_dim: int, num_experts: int, gate_type: str = "linear", hidden_dims: Optional[List[int]] = None):
        super().__init__()
        gate_type = gate_type.lower()
        self.num_experts = num_experts
        if gate_type == "linear":
            self.net = nn.Linear(in_dim, num_experts)
        elif gate_type == "mlp":
            hidden_dims = hidden_dims or [in_dim]
            self.net = nn.Sequential(
                MLP(input_dim=in_dim, hidden_dims=hidden_dims, activation="relu", dropout=0.0, use_bn=False),
                nn.Linear(hidden_dims[-1] if hidden_dims else in_dim, num_experts),
            )
        else:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class MMoE(nn.Module):
    """
    Minimal MMoE block plugged in place of the shared-bottom towers.

    - Uses backbone-provided representations (default deep_h) as expert inputs.
    - Keeps head interface identical to SharedBottom: returns per-task logits with optional wide/FM additions.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_cfg: Dict,
        mmoe_cfg: Dict,
        enabled_heads: Optional[List[str]] = None,
        use_legacy_pseudo_deepfm: bool = True,
        return_logit_parts: bool = False,
        per_head_add: Optional[Dict[str, Dict[str, bool]]] = None,
        head_priors: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_legacy_pseudo_deepfm = bool(use_legacy_pseudo_deepfm)
        self.return_logit_parts = bool(return_logit_parts)
        self.per_head_add = per_head_add or {}
        self.head_priors = head_priors or {}

        tasks = head_cfg.get("tasks") or ["ctr", "cvr"]
        enabled = enabled_heads or tasks
        self.tasks: List[str] = tasks
        self.enabled_heads: Set[str] = {h.lower() for h in enabled}

        # ---- resolve input dim ----
        input_source = str(mmoe_cfg.get("input", "deep_h")).lower()
        self.input_source = input_source

        deep_dim = getattr(backbone, "deep_out_dim", getattr(backbone, "out_dim", None))
        emb_concat_dim = getattr(getattr(backbone, "feat_emb", None), "concat_dim", None)

        if input_source == "deep_h":
            in_dim = deep_dim
        elif input_source == "emb_concat":
            in_dim = emb_concat_dim
        else:
            raise ValueError(f"Unsupported mmoe.input value: {input_source}")

        if in_dim is None:
            raise ValueError("MMoE: could not infer input dimension from backbone; ensure deep_out_dim or feat_emb.concat_dim is available.")

        self.in_dim = in_dim
        self.layernorm = nn.LayerNorm(in_dim)

        # ---- experts ----
        num_experts = int(mmoe_cfg.get("num_experts", 4))
        expert_mlp_dims = list(mmoe_cfg.get("expert_mlp_dims", []))
        expert_output_dims = expert_mlp_dims + [in_dim] if expert_mlp_dims else [in_dim]

        self.experts = nn.ModuleList(
            [
                MLP(
                    input_dim=in_dim,
                    hidden_dims=expert_output_dims[:-1],  # hidden layers (last dim is output)
                    activation=str(mmoe_cfg.get("activation", "relu")),
                    dropout=float(mmoe_cfg.get("dropout", 0.0)),
                    use_bn=bool(mmoe_cfg.get("use_bn", False)),
                )
                for _ in range(num_experts)
            ]
        )

        # ---- gates (per task) ----
        gate_type = str(mmoe_cfg.get("gate_type", "linear"))
        gate_hidden_dims = mmoe_cfg.get("gate_hidden_dims")
        self.gates = nn.ModuleDict(
            {task: _Gate(in_dim=in_dim, num_experts=num_experts, gate_type=gate_type, hidden_dims=gate_hidden_dims) for task in tasks}
        )

        # ---- towers / heads ----
        self._head_cfg_resolved: Dict[str, Dict[str, object]] = {}
        self.towers = nn.ModuleDict()
        for task in tasks:
            def _pick_cfg(cfg: Dict, task_name: str, key: str, default):
                if isinstance(cfg.get(task_name), dict) and key in cfg[task_name] and cfg[task_name][key] is not None:
                    return cfg[task_name][key]
                if isinstance(cfg.get("default"), dict) and key in cfg["default"] and cfg["default"][key] is not None:
                    return cfg["default"][key]
                if key in cfg and cfg[key] is not None:
                    return cfg[key]
                return default

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

            prior_map = {"ctr": 0.07, "cvr": 0.01}
            prior = float(self.head_priors.get(task, prior_map.get(task, 0.5)))
            prior = min(max(prior, 1e-6), 1 - 1e-6)
            bias_init = float(torch.log(torch.tensor(prior / (1 - prior))))
            if self.towers[task].out_proj.bias is not None:
                nn.init.constant_(self.towers[task].out_proj.bias, bias_init)

    def _get_head_add_cfg(self, task: str) -> Dict[str, bool]:
        task = task.lower()
        cfg = self.per_head_add.get(task)
        if cfg is not None:
            return {"use_wide": bool(cfg.get("use_wide", False)), "use_fm": bool(cfg.get("use_fm", False))}
        if task == "cvr":
            return {"use_wide": False, "use_fm": False}
        return {"use_wide": True, "use_fm": True}

    def _select_input(self, features: Dict[str, torch.Tensor], backbone_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.input_source == "deep_h":
            x = backbone_out.get("deep_h")
            if x is None:
                x = backbone_out.get("h")
            if x is None:
                raise KeyError("MMoE requires 'deep_h' (or 'h') from backbone when input=deep_h.")
            return x

        # emb_concat path: recompute via backbone.feat_emb to avoid changing backbone outputs
        if not hasattr(self.backbone, "feat_emb"):
            raise AttributeError("Backbone has no feat_emb; cannot use emb_concat as MMoE input.")
        emb_out = self.backbone.feat_emb(features)
        return emb_out["emb_concat"]

    def forward(self, features: Dict, dense_x: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.backbone(features, dense_x=dense_x, return_aux=True)
        if "logit_linear" not in out:
            raise KeyError("Backbone must return 'logit_linear' when return_aux=True.")

        results: Dict[str, torch.Tensor] = {}
        logit_parts_decomposable = not self.use_legacy_pseudo_deepfm

        base_x = self._select_input(features, out)
        base_x = self.layernorm(base_x)

        expert_outputs = [expert(base_x) for expert in self.experts]

        def _mix(gate_weights: torch.Tensor) -> torch.Tensor:
            # gate_weights: [B, E]; expert_outputs: list of [B, D]
            stacked = torch.stack(expert_outputs, dim=2)  # [B, D, E]
            weights = gate_weights.unsqueeze(1)  # [B,1,E]
            mixed = torch.bmm(stacked, weights.transpose(1, 2)).squeeze(-1)  # [B, D]
            return mixed

        wide_logit = out.get("logit_linear")
        fm_logit = out.get("fm_logit")
        if wide_logit is not None and wide_logit.dim() == 2 and wide_logit.size(-1) == 1:
            wide_logit = wide_logit.squeeze(-1)
        if fm_logit is not None and fm_logit.dim() == 2 and fm_logit.size(-1) == 1:
            fm_logit = fm_logit.squeeze(-1)

        for task, head in self.towers.items():
            if task not in self.enabled_heads:
                continue
            gate = self.gates[task]
            gate_w = gate(base_x)
            task_h = _mix(gate_w)
            task_logit = head(task_h)

            if self.use_legacy_pseudo_deepfm:
                if wide_logit is not None:
                    task_logit = task_logit + wide_logit
                results[task] = task_logit
                if self.return_logit_parts:
                    results[f"{task}_logit_parts"] = {
                        "wide": wide_logit,
                        "fm": None,
                        "deep": task_logit - (wide_logit if wide_logit is not None else 0.0),
                        "total": task_logit,
                    }
                continue

            add_cfg = self._get_head_add_cfg(task)
            wide_term = wide_logit if (add_cfg.get("use_wide", False) and wide_logit is not None) else 0.0
            fm_term = fm_logit if (add_cfg.get("use_fm", False) and fm_logit is not None) else 0.0
            task_logit = task_logit + wide_term + fm_term
            results[task] = task_logit

            if self.return_logit_parts:
                wide_comp = wide_term if torch.is_tensor(wide_term) else torch.tensor(wide_term, device=task_logit.device)
                fm_comp = fm_term if torch.is_tensor(fm_term) else torch.tensor(fm_term, device=task_logit.device)
                results[f"{task}_logit_parts"] = {
                    "wide": wide_comp,
                    "fm": fm_comp,
                    "deep": task_logit - wide_comp - fm_comp,
                    "total": task_logit,
                }

        results["logit_parts_decomposable"] = logit_parts_decomposable
        if "aux" in out:
            results["aux"] = out["aux"]
        return results


__all__ = ["MMoE"]
