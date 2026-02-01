"""
MMoEInputComposer: 可配置的 MMoE 输入组装模块。

支持将多个 backbone 输出（deep_h / fm_vec / embedding 表示）组合成 MMoE 的输入，
通过配置灵活控制各部分的选用、投影、归一化和融合方式。

配置字段说明：
  - input: str = "deep_h"           # 旧配置兼容，始终使用 deep_h
  - add_fm_vec: bool = False        # 是否加入 fm_vec（FM 二阶向量）
  - add_emb: str = "none"           # embedding 聚合方式: none|sum|mean|concat
  - part_proj_dim: int | None       # 每个 part 投影到的统一维度；None 表示不投影
  - fusion: str = "concat"          # 融合方式: concat|sum
  - adapter_mlp_dims: List[int]     # 融合后可选 MLP 层维度
  - dropout: float = 0.0            # 各 part 投影后的 dropout
  - norm: str = "none"              # 各 part 投影后的归一化: none|layernorm

向后兼容：
  - 当旧配置只有 input:"deep_h" 且无 add_fm_vec/add_emb 时，等价于只用 deep_h 经投影/归一化后
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from src.models.backbones.layers import MLP

logger = logging.getLogger(__name__)


class PartProjector(nn.Module):
    """
    单个 part 的投影模块：Linear -> (LayerNorm) -> Dropout
    """

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        norm: str = "none",
        dropout: float = 0.0,
    ):
        """
        Args:
            in_dim: 输入维度
            proj_dim: 投影后的维度
            norm: 归一化方式 ("none" | "layernorm")
            dropout: dropout 比例
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, proj_dim)
        self.norm: Optional[nn.Module] = None
        if norm.lower() == "layernorm":
            self.norm = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim]
        Returns:
            [B, proj_dim]
        """
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.dropout(x)
        return x


class MMoEInputComposer(nn.Module):
    """
    可配置的 MMoE 输入组装器。
    
    根据配置从 backbone forward 输出的 aux dict 中选择并组合多个张量：
      - deep_h: [B, H_deep]（必选，始终使用）
      - fm_vec: [B, D_fm]（可选，当 add_fm_vec=True 时使用）
      - emb 表示: sum_emb/mean_emb/emb_concat（可选，由 add_emb 控制）
    
    每个被选中的 part 会经过：
      1. Linear 投影到 part_proj_dim（如果指定）
      2. LayerNorm（如果 norm="layernorm"）
      3. Dropout
    
    融合方式：
      - fusion="concat": torch.cat(parts, dim=-1)
      - fusion="sum": 各 part 求和（要求投影到同一维度）
    
    融合后可选通过 adapter MLP 进一步变换。
    
    警告：add_emb="concat" 时，emb_concat 可能非常宽（F * D），会显著增加显存占用。
    """

    def __init__(
        self,
        # === 维度信息（从 backbone 获取）===
        deep_dim: int,
        fm_dim: int = 0,  # FM 二阶向量维度；若 FM 未启用则为 0
        emb_concat_dim: int = 0,  # embedding concat 维度（用于 add_emb="concat"）
        emb_single_dim: int = 0,  # 单个 embedding 维度（用于 sum/mean）
        # === 配置参数 ===
        add_fm_vec: bool = False,
        add_emb: str = "none",  # none|sum|mean|concat
        part_proj_dim: Optional[int] = None,  # 每个 part 投影到的维度；None 表示保留原维度
        fusion: str = "concat",  # concat|sum
        adapter_mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        norm: str = "none",  # none|layernorm
    ):
        """
        Args:
            deep_dim: deep_h 的维度
            fm_dim: fm_vec 的维度（若 FM 未启用则为 0）
            emb_concat_dim: emb_concat 的维度（所有 embedding concat 后）
            emb_single_dim: 单个 embedding 的维度（用于 sum/mean 聚合）
            add_fm_vec: 是否使用 fm_vec
            add_emb: embedding 聚合方式 (none|sum|mean|concat)
            part_proj_dim: 投影维度；None 表示各 part 保留原维度（仅 concat 融合有效）
            fusion: 融合方式 (concat|sum)
            adapter_mlp_dims: 融合后 MLP 层维度列表；空表示不使用
            dropout: dropout 比例
            norm: 归一化方式 (none|layernorm)
        """
        super().__init__()
        
        # 配置存储
        self.add_fm_vec = bool(add_fm_vec)
        self.add_emb = add_emb.lower() if add_emb else "none"
        self.fusion = fusion.lower() if fusion else "concat"
        self.norm_type = norm.lower() if norm else "none"
        self.dropout_p = float(dropout)
        
        # 验证配置
        if self.add_emb not in {"none", "sum", "mean", "concat"}:
            raise ValueError(f"add_emb must be one of 'none', 'sum', 'mean', 'concat', got '{self.add_emb}'")
        if self.fusion not in {"concat", "sum"}:
            raise ValueError(f"fusion must be 'concat' or 'sum', got '{self.fusion}'")
        
        # 如果使用 sum 融合，则必须指定 part_proj_dim
        if self.fusion == "sum" and part_proj_dim is None:
            raise ValueError("fusion='sum' requires part_proj_dim to be specified (all parts must project to same dim)")
        
        # 维度验证
        if self.add_fm_vec and fm_dim <= 0:
            raise ValueError("add_fm_vec=True but fm_dim is 0. Is FM enabled in backbone?")
        if self.add_emb == "concat" and emb_concat_dim <= 0:
            raise ValueError("add_emb='concat' but emb_concat_dim is 0.")
        if self.add_emb in {"sum", "mean"} and emb_single_dim <= 0:
            raise ValueError(f"add_emb='{self.add_emb}' but emb_single_dim is 0.")
        
        # ========== 构建各 part 的投影器 ==========
        # 计算每个 part 的原始维度
        part_dims: Dict[str, int] = {"deep_h": deep_dim}
        if self.add_fm_vec:
            part_dims["fm_vec"] = fm_dim
        if self.add_emb == "concat":
            # 警告：emb_concat 可能很宽，会占用大量显存
            logger.warning(
                f"[MMoEInputComposer] add_emb='concat' with dim={emb_concat_dim}. "
                "This can significantly increase memory usage."
            )
            part_dims["emb"] = emb_concat_dim
        elif self.add_emb in {"sum", "mean"}:
            part_dims["emb"] = emb_single_dim
        
        # 存储 part 名称顺序（用于 forward 时一致性）
        self.part_names: List[str] = list(part_dims.keys())
        
        # 决定各 part 的输出维度
        if part_proj_dim is not None:
            proj_dims = {name: part_proj_dim for name in part_dims}
        else:
            # 不投影，保留原维度
            proj_dims = part_dims.copy()
        
        # 构建投影器
        self.projectors = nn.ModuleDict()
        self._part_out_dims: Dict[str, int] = {}
        
        for name, in_dim in part_dims.items():
            out_dim = proj_dims[name]
            # 如果输入维度等于输出维度且不需要 norm/dropout，可以考虑 Identity
            # 但为了简化，统一使用 PartProjector
            self.projectors[name] = PartProjector(
                in_dim=in_dim,
                proj_dim=out_dim,
                norm=self.norm_type,
                dropout=self.dropout_p,
            )
            self._part_out_dims[name] = out_dim
        
        # ========== 计算融合后的维度 ==========
        if self.fusion == "concat":
            fused_dim = sum(self._part_out_dims.values())
        else:  # sum
            # sum 融合要求所有 part 维度相同
            dims_set = set(self._part_out_dims.values())
            if len(dims_set) != 1:
                raise ValueError(
                    f"fusion='sum' requires all parts to have same proj_dim, "
                    f"but got {self._part_out_dims}"
                )
            fused_dim = dims_set.pop()
        
        # ========== 可选的 adapter MLP ==========
        self.adapter: Optional[nn.Module] = None
        if adapter_mlp_dims and len(adapter_mlp_dims) > 0:
            self.adapter = MLP(
                input_dim=fused_dim,
                hidden_dims=adapter_mlp_dims,
                activation="relu",
                dropout=self.dropout_p,
                use_bn=False,
            )
            self.out_dim = self.adapter.output_dim
        else:
            self.out_dim = fused_dim
        
        # 记录初始化信息（首次 forward 时打印）
        self._logged_init = False
        self._init_info = {
            "parts": list(self.part_names),
            "part_in_dims": dict(part_dims),
            "part_out_dims": dict(self._part_out_dims),
            "fusion": self.fusion,
            "fused_dim": fused_dim,
            "adapter_mlp_dims": adapter_mlp_dims,
            "final_out_dim": self.out_dim,
        }
        logger.info(
            f"[MMoEInputComposer] Initialized: parts={self.part_names}, "
            f"fusion={self.fusion}, out_dim={self.out_dim}"
        )

    def forward(self, aux: Dict[str, Any]) -> torch.Tensor:
        """
        从 backbone 输出的 aux dict 中组装 MMoE 输入。
        
        Args:
            aux: backbone forward(return_aux=True) 返回的字典，应至少包含：
                - "deep_h": [B, H_deep]
                - "fm_vec": [B, D_fm]（当 add_fm_vec=True 时需要）
                - "sum_emb": [B, D_emb]（当 add_emb="sum" 时需要）
                - "mean_emb": [B, D_emb]（当 add_emb="mean" 时需要）
                - "emb_concat": [B, concat_dim]（当 add_emb="concat" 时需要）
        
        Returns:
            mmoe_input: [B, out_dim] 组装后的 MMoE 输入张量
        """
        parts: List[torch.Tensor] = []
        
        # ===== deep_h（必选）=====
        deep_h = aux.get("deep_h")
        if deep_h is None:
            deep_h = aux.get("h")
        if deep_h is None:
            raise KeyError(
                "[MMoEInputComposer] aux dict missing 'deep_h' (or 'h'). "
                "Ensure backbone.forward() returns this key."
            )
        parts.append(self.projectors["deep_h"](deep_h))
        
        # ===== fm_vec（可选）=====
        if self.add_fm_vec:
            fm_vec = aux.get("fm_vec")
            if fm_vec is None:
                raise KeyError(
                    "[MMoEInputComposer] add_fm_vec=True but aux missing 'fm_vec'. "
                    "Is FM enabled in backbone config?"
                )
            parts.append(self.projectors["fm_vec"](fm_vec))
        
        # ===== embedding 表示（可选）=====
        if self.add_emb == "sum":
            sum_emb = aux.get("sum_emb")
            if sum_emb is None:
                raise KeyError(
                    "[MMoEInputComposer] add_emb='sum' but aux missing 'sum_emb'. "
                    "Ensure backbone provides this output."
                )
            parts.append(self.projectors["emb"](sum_emb))
        elif self.add_emb == "mean":
            mean_emb = aux.get("mean_emb")
            if mean_emb is None:
                raise KeyError(
                    "[MMoEInputComposer] add_emb='mean' but aux missing 'mean_emb'. "
                    "Ensure backbone provides this output."
                )
            parts.append(self.projectors["emb"](mean_emb))
        elif self.add_emb == "concat":
            emb_concat = aux.get("emb_concat")
            if emb_concat is None:
                raise KeyError(
                    "[MMoEInputComposer] add_emb='concat' but aux missing 'emb_concat'. "
                    "Ensure backbone provides this output."
                )
            parts.append(self.projectors["emb"](emb_concat))
        
        # ===== 融合 =====
        if self.fusion == "concat":
            fused = torch.cat(parts, dim=-1)
        else:  # sum
            fused = sum(parts)  # type: ignore
        
        # ===== 可选 adapter MLP =====
        if self.adapter is not None:
            fused = self.adapter(fused)
        
        # ===== 首次 forward 打印调试信息 =====
        if not self._logged_init:
            self._logged_init = True
            logger.info(
                f"[MMoEInputComposer] First forward: input shape={deep_h.shape[0]}x?, "
                f"output shape={fused.shape}, parts={self.part_names}"
            )
        
        return fused

    def get_config_summary(self) -> Dict[str, Any]:
        """返回配置摘要，用于日志/checkpoint。"""
        return dict(self._init_info)


def build_composer_from_config(
    backbone: nn.Module,
    mmoe_cfg: Dict[str, Any],
) -> Optional[MMoEInputComposer]:
    """
    根据 mmoe 配置构建 MMoEInputComposer。
    
    若配置表明只使用 deep_h 且无额外处理，返回 None（由调用方决定是否直接用 deep_h）。
    
    向后兼容策略：
    - 只有当配置中明确指定了新字段（add_fm_vec=True, add_emb!="none", part_proj_dim,
      adapter_mlp_dims, norm!="none"）时才创建 composer
    - 旧配置（只有 input/num_experts/dropout 等）走 fast-path，不创建 composer
    
    Args:
        backbone: DeepFMBackbone 实例，用于获取各输出维度
        mmoe_cfg: mmoe 配置字典
    
    Returns:
        MMoEInputComposer 实例，或 None（当配置为纯 deep_h 模式时）
    """
    add_fm_vec = bool(mmoe_cfg.get("add_fm_vec", False))
    add_emb = str(mmoe_cfg.get("add_emb", "none")).lower()
    part_proj_dim = mmoe_cfg.get("part_proj_dim")
    fusion = str(mmoe_cfg.get("fusion", "concat")).lower()
    adapter_mlp_dims = mmoe_cfg.get("adapter_mlp_dims") or []
    norm = str(mmoe_cfg.get("norm", "none")).lower()
    # dropout 从 mmoe_cfg 获取，但不作为触发 composer 创建的条件
    dropout = float(mmoe_cfg.get("dropout", 0.0))
    
    # 判断是否需要创建 composer：
    # 只有当明确指定了"新增功能"时才创建
    # 旧配置的 dropout 会在 MMoE 的 expert MLP 中处理，不需要 composer
    needs_composer = (
        add_fm_vec  # 需要 fm_vec
        or add_emb != "none"  # 需要 embedding 聚合
        or part_proj_dim is not None  # 明确指定投影维度
        or adapter_mlp_dims  # 需要 adapter MLP
        or norm != "none"  # 需要归一化
    )
    
    if not needs_composer:
        logger.info("[build_composer_from_config] Pure deep_h mode (legacy), skipping composer.")
        return None
    
    # 从 backbone 获取维度信息
    deep_dim = getattr(backbone, "deep_out_dim", None)
    if deep_dim is None:
        deep_dim = getattr(backbone, "out_dim", None)
    if deep_dim is None:
        raise ValueError("Cannot infer deep_dim from backbone.")
    
    # fm_vec 维度
    fm_dim = getattr(backbone, "fm_dim", 0)
    fm_enabled = getattr(backbone, "fm_enabled", False)
    if add_fm_vec and not fm_enabled:
        raise ValueError("add_fm_vec=True but FM is not enabled in backbone.")
    
    # embedding 维度
    feat_emb = getattr(backbone, "feat_emb", None)
    emb_concat_dim = getattr(feat_emb, "concat_dim", 0) if feat_emb else 0
    
    # 单个 embedding 维度（假设所有 field 维度相同）
    emb_single_dim = 0
    if feat_emb is not None:
        emb_dims = getattr(feat_emb, "embedding_dims", {})
        if emb_dims:
            dims_set = set(emb_dims.values())
            if len(dims_set) == 1:
                emb_single_dim = dims_set.pop()
            elif add_emb in {"sum", "mean"}:
                raise ValueError(
                    f"add_emb='{add_emb}' requires all embeddings to have same dim, "
                    f"but found {dims_set}"
                )
    
    # 如果需要 part_proj_dim 但未指定，使用 deep_dim 作为默认
    if part_proj_dim is None and fusion == "sum":
        part_proj_dim = deep_dim
        logger.info(f"[build_composer_from_config] fusion='sum', auto-setting part_proj_dim={part_proj_dim}")
    
    return MMoEInputComposer(
        deep_dim=deep_dim,
        fm_dim=fm_dim if fm_enabled else 0,
        emb_concat_dim=emb_concat_dim,
        emb_single_dim=emb_single_dim,
        add_fm_vec=add_fm_vec,
        add_emb=add_emb,
        part_proj_dim=part_proj_dim,
        fusion=fusion,
        adapter_mlp_dims=adapter_mlp_dims,
        dropout=dropout,
        norm=norm,
    )


__all__ = ["MMoEInputComposer", "PartProjector", "build_composer_from_config"]
