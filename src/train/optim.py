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


# ============================================================================
# Learning Rate Scheduler (warmup + cosine/step decay)
# ============================================================================

@dataclass
class LRSchedulerBundle:
    """
    Thin wrapper for learning rate schedulers supporting both dense and sparse optimizers.
    
    - dense_sch: scheduler for dense optimizer (required if enabled)
    - sparse_sch: scheduler for sparse optimizer (optional)
    - enabled: whether scheduling is active
    """
    dense_sch: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    sparse_sch: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    enabled: bool = False
    target: str = "dense"  # "dense" | "both"
    warmup_steps: int = 0
    total_steps: int = 0
    scheduler_type: str = "cosine"
    min_lr_ratio: float = 0.1
    
    def step(self) -> None:
        """Step both schedulers if enabled."""
        if not self.enabled:
            return
        if self.dense_sch is not None:
            self.dense_sch.step()
        if self.sparse_sch is not None:
            self.sparse_sch.step()
    
    def get_last_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        result = {}
        if self.dense_sch is not None:
            result["lr_dense"] = self.dense_sch.get_last_lr()[0]
        if self.sparse_sch is not None:
            result["lr_sparse"] = self.sparse_sch.get_last_lr()[0]
        return result
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            "dense_sch": self.dense_sch.state_dict() if self.dense_sch is not None else None,
            "sparse_sch": self.sparse_sch.state_dict() if self.sparse_sch is not None else None,
            "enabled": self.enabled,
            "target": self.target,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "scheduler_type": self.scheduler_type,
            "min_lr_ratio": self.min_lr_ratio,
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        if state is None:
            return
        if self.dense_sch is not None and state.get("dense_sch") is not None:
            try:
                self.dense_sch.load_state_dict(state["dense_sch"])
                logger.info("LRSchedulerBundle: loaded dense scheduler state")
            except Exception as e:
                logger.warning("LRSchedulerBundle: failed to load dense scheduler state: %s", e)
        if self.sparse_sch is not None and state.get("sparse_sch") is not None:
            try:
                self.sparse_sch.load_state_dict(state["sparse_sch"])
                logger.info("LRSchedulerBundle: loaded sparse scheduler state")
            except Exception as e:
                logger.warning("LRSchedulerBundle: failed to load sparse scheduler state: %s", e)


def _build_lr_lambda(
    warmup_steps: int,
    total_steps: int,
    scheduler_type: str,
    min_lr_ratio: float,
    step_milestones: Optional[list] = None,
    step_gamma: float = 0.3,
) -> callable:
    """
    Build a learning rate lambda function for LambdaLR.
    
    Supports:
    - warmup phase: linear warmup from 0 to 1
    - cosine decay: cosine annealing from 1 to min_lr_ratio
    - step decay: multiply by gamma at each milestone
    """
    import math
    
    def lr_lambda(step: int) -> float:
        # Warmup phase
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        
        # Post-warmup phase
        progress_step = step - warmup_steps
        post_warmup_steps = max(total_steps - warmup_steps, 1)
        
        if scheduler_type == "cosine":
            # Cosine annealing
            progress = min(progress_step / post_warmup_steps, 1.0)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
        
        elif scheduler_type == "step":
            # Step decay
            if step_milestones is None:
                return 1.0
            lr_mult = 1.0
            for milestone in step_milestones:
                if step >= milestone:
                    lr_mult *= step_gamma
            return max(lr_mult, min_lr_ratio)
        
        else:
            # Linear decay as fallback
            progress = min(progress_step / post_warmup_steps, 1.0)
            return 1.0 - (1.0 - min_lr_ratio) * progress
    
    return lr_lambda


def build_lr_scheduler_bundle(
    cfg: Dict[str, Any],
    optim_bundle: OptimizerBundle,
    train_loader_len: int,
    epochs: int,
    max_steps: Optional[int],
    logger_obj: Optional[logging.Logger] = None,
) -> LRSchedulerBundle:
    """
    Build learning rate scheduler bundle from config.
    
    Config example (cfg.optim.lr_scheduler):
        lr_scheduler:
            enabled: true
            target: "dense"        # "dense" | "both"
            type: "cosine"         # "cosine" | "step"
            warmup_steps: 2000
            total_steps: 80000     # optional, inferred if not set
            min_lr_ratio: 0.1
            # step-specific
            step_milestones: [30000, 50000]
            step_gamma: 0.3
    """
    log = logger_obj or logger
    optim_cfg = cfg.get("optim", {}) or {}
    sch_cfg = optim_cfg.get("lr_scheduler", {}) or {}
    
    enabled = bool(sch_cfg.get("enabled", False))
    if not enabled:
        log.info("[lr_scheduler] disabled (enabled=false or not configured)")
        return LRSchedulerBundle(enabled=False)
    
    # Parse config
    target = str(sch_cfg.get("target", "dense")).lower()
    scheduler_type = str(sch_cfg.get("type", "cosine")).lower()
    warmup_steps = int(sch_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(sch_cfg.get("min_lr_ratio", 0.1))
    step_milestones = sch_cfg.get("step_milestones")
    step_gamma = float(sch_cfg.get("step_gamma", 0.3))
    
    # Compute total_steps
    runtime_cfg = cfg.get("runtime", {}) or {}
    cfg_total_steps = sch_cfg.get("total_steps")
    runtime_max_steps = max_steps or runtime_cfg.get("max_train_steps")
    inferred_total_steps = epochs * train_loader_len
    
    if cfg_total_steps is not None:
        total_steps = int(cfg_total_steps)
    elif runtime_max_steps is not None:
        total_steps = int(runtime_max_steps)
    else:
        total_steps = inferred_total_steps
    
    # Build lambda
    lr_lambda = _build_lr_lambda(
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        scheduler_type=scheduler_type,
        min_lr_ratio=min_lr_ratio,
        step_milestones=step_milestones,
        step_gamma=step_gamma,
    )
    
    # Build schedulers
    dense_sch = None
    sparse_sch = None
    
    if optim_bundle.dense_opt is not None:
        dense_sch = torch.optim.lr_scheduler.LambdaLR(optim_bundle.dense_opt, lr_lambda)
    
    if target == "both" and optim_bundle.sparse_opt is not None:
        sparse_sch = torch.optim.lr_scheduler.LambdaLR(optim_bundle.sparse_opt, lr_lambda)
    
    # Log configuration
    log.info(
        "[lr_scheduler] enabled=true type=%s target=%s warmup_steps=%d total_steps=%d "
        "min_lr_ratio=%.4f (cfg_total=%s runtime_max=%s inferred=%d)",
        scheduler_type, target, warmup_steps, total_steps, min_lr_ratio,
        cfg_total_steps, runtime_max_steps, inferred_total_steps
    )
    if scheduler_type == "step":
        log.info("[lr_scheduler] step_milestones=%s step_gamma=%.3f", step_milestones, step_gamma)
    
    return LRSchedulerBundle(
        dense_sch=dense_sch,
        sparse_sch=sparse_sch,
        enabled=True,
        target=target,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        scheduler_type=scheduler_type,
        min_lr_ratio=min_lr_ratio,
    )


__all__ = ["OptimizerBundle", "build_optimizer_bundle", "LRSchedulerBundle", "build_lr_scheduler_bundle"]
