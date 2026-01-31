from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import logging

import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch import amp as torch_amp


@dataclass
class OptimizerBundle:
    """
    Thin wrapper that unifies dense/sparse optimizers and AMP scaler handling.

    - Supports legacy single-optimizer semantics (only dense_opt is set).
    - sparse_opt can be None or disabled; step/zero_grad no-op if absent.
    - load_state_dict handles both new-format {"dense":..., "sparse":...}
      and legacy {"optimizer": ...}.
    """

    dense_opt: Optimizer
    dense_params: Iterable[nn.Parameter]
    sparse_opt: Optional[Optimizer] = None
    sparse_params: Iterable[nn.Parameter] | None = None
    scaler: Optional[torch_amp.GradScaler] = None

    @property
    def has_sparse(self) -> bool:
        return self.sparse_opt is not None

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.dense_opt is not None:
            self.dense_opt.zero_grad(set_to_none=set_to_none)
        if self.sparse_opt is not None:
            self.sparse_opt.zero_grad(set_to_none=set_to_none)

    def unscale_(self, scaler: torch_amp.GradScaler) -> None:
        """
        Unscale before clipping (only dense params need clipping).
        """
        scaler.unscale_(self.dense_opt)

    def step(self, scaler: Optional[torch_amp.GradScaler] = None) -> None:
        use_scaler = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
        if use_scaler:
            scaler.step(self.dense_opt)
            if self.sparse_opt is not None:
                scaler.step(self.sparse_opt)
            scaler.update()
        else:
            if self.dense_opt is not None:
                self.dense_opt.step()
            if self.sparse_opt is not None:
                self.sparse_opt.step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dense": self.dense_opt.state_dict() if self.dense_opt is not None else None,
            "sparse": self.sparse_opt.state_dict() if self.sparse_opt is not None else None,
            "dense_param_ids": [id(p) for p in self.dense_params] if self.dense_params is not None else None,
            "sparse_param_ids": [id(p) for p in self.sparse_params] if self.sparse_params is not None else None,
        }

    def scaler_state_dict(self) -> Optional[Dict[str, Any]]:
        if self.scaler is None or not getattr(self.scaler, "is_enabled", lambda: False)():
            return None
        return self.scaler.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if "dense" in state or "sparse" in state:
            if self.dense_opt is not None and state.get("dense") is not None:
                self.dense_opt.load_state_dict(state["dense"])
            elif "dense" in state and state.get("dense") is None:
                logger.warning("OptimizerBundle.load_state_dict: dense optimizer state missing/None.")
            if self.sparse_opt is not None and state.get("sparse") is not None:
                self.sparse_opt.load_state_dict(state["sparse"])
            elif self.sparse_opt is not None and "sparse" in state:
                logger.warning("OptimizerBundle.load_state_dict: sparse optimizer state missing/None.")
            return
        if "optimizer" in state and self.dense_opt is not None:
            logger.warning("OptimizerBundle.load_state_dict: loading legacy 'optimizer' into dense slot.")
            self.dense_opt.load_state_dict(state["optimizer"])

    def load_scaler_state(self, state: Optional[Dict[str, Any]]) -> None:
        if self.scaler is None or state is None:
            return
        try:
            self.scaler.load_state_dict(state)
        except Exception:
            # ignore scaler load failure to keep compatibility
            pass


logger = logging.getLogger(__name__)


def _make_adamw(params: Iterable[nn.Parameter], cfg: Dict[str, Any]) -> Optimizer:
    return optim.AdamW(
        params,
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        betas=tuple(cfg.get("betas", (0.9, 0.999))),
        eps=float(cfg.get("eps", 1e-8)),
    )


def _split_sparse_dense_params(model: nn.Module) -> Tuple[list[nn.Parameter], list[nn.Parameter]]:
    sparse_params: list[nn.Parameter] = []
    dense_params: list[nn.Parameter] = []
    sparse_ids = set()

    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.EmbeddingBag)) and getattr(m, "sparse", False):
            p = m.weight
            if p.requires_grad:
                sparse_params.append(p)
                sparse_ids.add(id(p))

    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in sparse_ids:
            continue
        dense_params.append(p)

    # Assert no overlap
    assert not (set(sparse_ids) & {id(p) for p in dense_params}), "Sparse and dense param groups overlap."
    return sparse_params, dense_params


def build_optimizer_bundle(cfg: Dict[str, Any], model: nn.Module, scaler: Optional[torch_amp.GradScaler]) -> OptimizerBundle:
    """
    Build optimizer bundle according to config. Default remains single AdamW.
    """
    optim_cfg = cfg.get("optim", {}) or {}
    opt_type = str(optim_cfg.get("type", "single")).lower()

    # Dense config falls back to legacy flat keys for compatibility.
    dense_cfg = optim_cfg.get("dense", optim_cfg)
    sparse_cfg = optim_cfg.get("sparse", {}) or {}
    sparse_enabled_cfg = bool(sparse_cfg.get("enabled", False))
    allow_fallback = bool(sparse_cfg.get("allow_fallback_if_empty", False))
    if sparse_enabled_cfg and float(sparse_cfg.get("weight_decay", 0.0)) != 0.0:
        logger.warning("Sparse optimizer does not support weight_decay; ignoring sparse.weight_decay.")
        sparse_cfg = {**sparse_cfg, "weight_decay": 0.0}

    sparse_params, dense_params = _split_sparse_dense_params(model)
    num_sparse = len(sparse_params)
    num_dense = len(dense_params)
    num_sparse_elems = sum(p.numel() for p in sparse_params)
    num_dense_elems = sum(p.numel() for p in dense_params)

    sparse_opt = None
    sparse_effective = False

    if opt_type == "dual_sparse_dense" and sparse_enabled_cfg:
        if num_sparse == 0 and not allow_fallback:
            raise ValueError(
                f"dual_sparse_dense requested with sparse.enabled=true but no sparse params were found (num_sparse=0, num_dense={num_dense}). "
                "Enable embedding sparse=True or set optim.sparse.enabled=false (or allow_fallback_if_empty=true to force downgrade)."
            )
        if num_sparse == 0 and allow_fallback:
            logger.warning("dual_sparse_dense requested but no sparse params found; falling back to dense-only because allow_fallback_if_empty=true.")
            sparse_effective = False
        else:
            sparse_opt = optim.SparseAdam(
                sparse_params,
                lr=float(sparse_cfg.get("lr", 1e-3)),
                betas=tuple(sparse_cfg.get("betas", (0.9, 0.999))),
                eps=float(sparse_cfg.get("eps", 1e-8)),
            )
            sparse_effective = True

    dense_target_params = dense_params if sparse_opt is not None else model.parameters()
    dense_opt = _make_adamw(dense_target_params, dense_cfg)

    logger.info(
        "[optim] type=%s sparse_enabled_cfg=%s allow_fallback_if_empty=%s "
        "num_sparse_params=%d num_dense_params=%d num_sparse_elems=%d num_dense_elems=%d sparse_effective=%s dense_lr=%.6g dense_wd=%.3g%s",
        opt_type,
        sparse_enabled_cfg,
        allow_fallback,
        num_sparse,
        num_dense,
        num_sparse_elems,
        num_dense_elems,
        sparse_effective,
        float(dense_cfg.get("lr", 1e-3)),
        float(dense_cfg.get("weight_decay", 0.0)),
        f" sparse_lr={float(sparse_cfg.get('lr', 1e-3))}" if sparse_effective else "",
    )

    return OptimizerBundle(
        dense_opt=dense_opt,
        dense_params=dense_params if sparse_opt is not None else list(model.parameters()),
        sparse_opt=sparse_opt,
        sparse_params=sparse_params if sparse_effective else None,
        scaler=scaler,
    )


__all__ = ["OptimizerBundle", "build_optimizer_bundle"]
