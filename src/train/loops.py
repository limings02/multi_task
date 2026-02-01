from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import amp as torch_amp
from src.eval.metrics import compute_binary_metrics
from src.train.grad_diag import GradientDiagnostics
from src.utils.health_metrics import (
    compute_label_counts,
    compute_logit_stats,
    compute_prob_stats,
    aggregate_gate_metrics,
)

EPS = 1e-12
_diag_logger = logging.getLogger(__name__)


def _to_device_labels(labels: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in labels.items()}


def _to_device_features(features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move indices/offsets/weights to target device; keep structure identical.
    """
    fields = {}
    for base, fd in features["fields"].items():
        fields[base] = {
            "indices": fd["indices"].to(device),
            "offsets": fd["offsets"].to(device),
            "weights": fd["weights"].to(device) if fd.get("weights") is not None else None,
        }
    return {"fields": fields, "field_names": features["field_names"]}


def _select_shared_params(model, fallback_cap: int = 20) -> List[Tuple[str, torch.nn.Parameter]]:
    """
    Prefer shared embedding/backbone params (name contains emb/embedding under backbone).
    If none found, fall back to the first K backbone params to keep overhead bounded.
    """
    preferred: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_low = name.lower()
        if not name_low.startswith("backbone"):
            continue
        if "emb" in name_low or "embedding" in name_low:
            preferred.append((name, param))
    if preferred:
        return preferred

    fallback: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.lower().startswith("backbone"):
            continue
        fallback.append((name, param))
        if len(fallback) >= fallback_cap:
            break
    return fallback


def _flatten_grads(params: Sequence[Tuple[str, torch.nn.Parameter]]) -> Tuple[torch.Tensor | None, int, int]:
    """
    Flatten dense gradients into a 1D tensor.

    Returns:
        flat_grad: concatenated dense grads or None if empty
        num_dense: number of dense grads included
        num_sparse_skipped: number of sparse grads skipped
    """
    grads = []
    num_sparse_skipped = 0
    num_dense = 0
    for _, p in params:
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            num_sparse_skipped += 1
            continue
        num_dense += 1
        grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return None, num_dense, num_sparse_skipped
    return torch.cat(grads), num_dense, num_sparse_skipped


def _compute_grad_percentiles(values: List[float]) -> Tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    tensor_vals = torch.tensor(values, device="cpu")
    p10, p50, p90 = torch.quantile(tensor_vals, torch.tensor([0.1, 0.5, 0.9]))
    return float(p10.item()), float(p50.item()), float(p90.item())


def train_one_epoch(
    model,
    loader,
    optimizer_bundle,
    loss_fn,
    device,
    logger,
    epoch: int,
    max_steps: Optional[int] = None,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 50,
    global_step: int = 0,
    grad_diag_enabled: bool = False,
    grad_diag_every: Optional[int] = None,  # If None, uses log_every
    grad_diag_min_tasks: int = 2,
    eval_every_steps: Optional[int] = None,
    validate_fn: Optional[Callable[[], Dict[str, float]]] = None,
    eval_callback: Optional[Callable[[Dict[str, float], int, int], None]] = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Optional[torch_amp.GradScaler] = None,
    amp_device_type: str = "cuda",
    debug_logit_every: Optional[int] = None,
) -> Dict[str, float]:
    """
    Train for one epoch and emit aggregated metrics plus optional gradient diagnostics.

    Gradient diagnostics use the new dynamic shared parameter detection that supports
    both dense and sparse gradients. Diagnostics run at the cadence specified by
    grad_diag_every (defaults to log_every).
    """
    model.train()
    enabled_heads = getattr(loss_fn, "enabled_heads", {"ctr", "cvr"})
    use_esmm = bool(getattr(loss_fn, "use_esmm", False))
    esmm_version = str(getattr(loss_fn, "esmm_version", "v2"))
    has_ctr = "ctr" in enabled_heads
    has_cvr = "cvr" in enabled_heads

    tot_loss = tot_ctr = tot_cvr = tot_mask = 0.0
    tot_ctr_raw = tot_cvr_raw = 0.0
    tot_ctr_scaled = tot_cvr_scaled = 0.0
    steps = 0
    n_rows = 0
    n_masked = 0.0
    pos_masked = 0.0

    # ESMM-specific metrics accumulators
    tot_p_ctr_mean = tot_p_cvr_mean = tot_p_ctcvr_mean = 0.0
    tot_pos_rate_ctr = tot_pos_rate_cvr = tot_pos_rate_ctcvr = 0.0
    tot_loss_cvr_aux = 0.0
    esmm_metric_count = 0
    cvr_aux_count = 0

    # Gradient diagnostics accumulators (new dynamic approach)
    grad_norm_ctr_vals: List[float] = []
    grad_norm_cvr_vals: List[float] = []
    grad_norm_ctcvr_vals: List[float] = []  # For ESMM mode
    grad_norm_ratio_vals: List[float] = []
    grad_cos_dense_vals: List[float] = []
    grad_cos_sparse_vals: List[float] = []
    grad_cos_all_vals: List[float] = []
    conflict_hits = 0
    grad_samples = 0
    shared_dense_count_last = 0
    shared_sparse_count_last = 0

    # Initialize gradient diagnostics object (new dynamic approach)
    grad_diag: Optional[GradientDiagnostics] = None
    if grad_diag_enabled:
        grad_diag = GradientDiagnostics(
            model, 
            min_tasks=grad_diag_min_tasks,
            cache_refresh="epoch",
        )
        grad_diag.set_epoch(epoch)
    
    # Effective gradient diagnosis frequency
    diag_every = grad_diag_every if grad_diag_every is not None else log_every

    # Legacy: keep shared_params for backward compat logging, but don't use for actual diag
    shared_params: List[Tuple[str, torch.nn.Parameter]] = _select_shared_params(model) if grad_diag_enabled else []
    ctr_raw_count = cvr_raw_count = 0
    ctr_scaled_count = cvr_scaled_count = 0

    t_start = time.time()

    for step, (labels, features, meta) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        # Update global_step in loss_fn for aux_focal warmup control
        current_global_step = global_step + step
        if hasattr(loss_fn, "global_step"):
            loss_fn.global_step = current_global_step

        labels = _to_device_labels(labels, device)
        features_dev = _to_device_features(features, device)

        optimizer_bundle.zero_grad(set_to_none=True)
        # Autocast only on CUDA to avoid CPU precision issues.
        with torch_amp.autocast(device_type=amp_device_type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(features_dev)
            batch = {"labels": labels, "features": features_dev, "meta": meta}
            loss, loss_dict = loss_fn.compute(outputs, batch)

        # Debug-only logit stats (no effect on gradients).
        current_global_step = global_step + steps + 1
        should_debug_logits = (
            debug_logit_every is not None
            and debug_logit_every > 0
            and (current_global_step % debug_logit_every == 0 or current_global_step == 1)
        )
        if should_debug_logits and hasattr(logger, "info"):
            with torch.no_grad():
                def _stats(t: Optional[torch.Tensor]):
                    if t is None:
                        return None
                    flat = t.detach().float().reshape(-1)
                    if flat.numel() == 0:
                        return None
                    qs = torch.quantile(flat, torch.tensor([0.99, 0.999], device=flat.device))
                    return {
                        "mean": float(flat.mean().item()),
                        "std": float(flat.std().item()),
                        "abs_max": float(flat.abs().max().item()),
                        "p99": float(qs[0].item()),
                        "p999": float(qs[1].item()),
                    }

                ctr_logit = outputs.get("ctr")
                ctr_parts = outputs.get("ctr_logit_parts", {}) if isinstance(outputs, dict) else {}
                fm_logit = ctr_parts.get("fm")
                wide_logit = ctr_parts.get("wide")
                deep_logit = ctr_parts.get("deep")
                stats_total = _stats(ctr_logit)
                stats_fm = _stats(fm_logit)
                stats_wide = _stats(wide_logit)
                stats_deep = _stats(deep_logit)

                # Per-field wide debug collected inside backbone when enabled.
                linear_debug = getattr(getattr(model, "backbone", model), "last_linear_debug", None)
                fm_debug = getattr(getattr(model, "backbone", model), "last_fm_debug", None)
                log_parts = [
                    f"[dbg_logit] step={current_global_step}",
                    f"ctr_total={stats_total}",
                    f"wide={stats_wide}",
                    f"fm={stats_fm}",
                    f"deep={stats_deep}",
                ]
                if linear_debug:
                    log_parts.append(f"wide_fields={linear_debug}")
                if fm_debug:
                    log_parts.append(f"fm_stats={fm_debug}")
                pos_w = {
                    "pos_weight_ctr": loss_dict.get("pos_weight_ctr"),
                    "pos_weight_cvr": loss_dict.get("pos_weight_cvr"),
                    "ctr_pos_rate": float(labels["y_ctr"].mean().item()),
                    "cvr_pos_rate": float(labels["y_cvr"].mean().item()),
                }
                log_parts.append(f"labels={pos_w}")
                logger.info(" ".join(log_parts))

        mask_cvr_sum_val_raw = loss_dict.get("mask_cvr_sum", None)
        mask_cvr_sum_val = None
        if mask_cvr_sum_val_raw is not None:
            if isinstance(mask_cvr_sum_val_raw, torch.Tensor):
                mask_cvr_sum_val = float(mask_cvr_sum_val_raw.item())
            else:
                mask_cvr_sum_val = float(mask_cvr_sum_val_raw)

        log_loss_debug = ((step + 1) % log_every == 0) or not loss.requires_grad
        if log_loss_debug and hasattr(logger, "debug"):
            logger.debug(
                "loss_debug step=%d type=%s requires_grad=%s device=%s mask_cvr_sum=%s enabled_heads=%s",
                step + 1,
                type(loss).__name__,
                getattr(loss, "requires_grad", None),
                getattr(loss, "device", None),
                mask_cvr_sum_val_raw,
                enabled_heads,
            )

        skip_cvr_only_empty_mask = has_cvr and (not has_ctr) and (mask_cvr_sum_val == 0.0)
        if skip_cvr_only_empty_mask:
            if log_loss_debug and hasattr(logger, "debug"):
                logger.debug("skip backward/step at step=%d because mask_cvr_sum==0 in CVR-only batch", step + 1)
        else:
            if amp_enabled and scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    # unscale before clipping, otherwise clipping would operate on scaled grads.
                    optimizer_bundle.unscale_(scaler)
                    clip_grad_norm_(optimizer_bundle.dense_params, grad_clip_norm)
                optimizer_bundle.step(scaler)
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    clip_grad_norm_(optimizer_bundle.dense_params, grad_clip_norm)
                optimizer_bundle.step()

        # aggregate stats
        B = int(labels["y_ctr"].shape[0])
        n_rows += B
        if has_cvr:
            mask = labels["click_mask"]
            n_masked += float(mask.sum().item())
            pos_masked += float((labels["y_cvr"] * (mask > 0.5)).sum().item())

        tot_loss += loss_dict["loss_total"]
        tot_ctr += loss_dict.get("loss_ctr", 0.0)
        tot_cvr += loss_dict.get("loss_cvr", 0.0)
        tot_mask += loss_dict.get("mask_cvr_sum", 0.0)
        if loss_dict.get("loss_ctr_raw") is not None:
            tot_ctr_raw += loss_dict["loss_ctr_raw"]
            ctr_raw_count += 1
        if loss_dict.get("loss_cvr_raw") is not None:
            tot_cvr_raw += loss_dict["loss_cvr_raw"]
            cvr_raw_count += 1
        if loss_dict.get("loss_ctr_scaled") is not None:
            tot_ctr_scaled += loss_dict["loss_ctr_scaled"]
            ctr_scaled_count += 1
        if loss_dict.get("loss_cvr_scaled") is not None:
            tot_cvr_scaled += loss_dict["loss_cvr_scaled"]
            cvr_scaled_count += 1

        # Accumulate ESMM-specific metrics
        if use_esmm:
            if loss_dict.get("p_ctr_mean") is not None:
                tot_p_ctr_mean += loss_dict["p_ctr_mean"]
                tot_p_cvr_mean += loss_dict.get("p_cvr_mean", 0.0)
                tot_p_ctcvr_mean += loss_dict.get("p_ctcvr_mean", 0.0)
                tot_pos_rate_ctr += loss_dict.get("pos_rate_ctr", 0.0)
                tot_pos_rate_ctcvr += loss_dict.get("pos_rate_ctcvr", 0.0)
                if loss_dict.get("pos_rate_cvr") is not None:
                    tot_pos_rate_cvr += loss_dict["pos_rate_cvr"]
                esmm_metric_count += 1
            if loss_dict.get("loss_cvr_aux") is not None:
                tot_loss_cvr_aux += loss_dict["loss_cvr_aux"]
                cvr_aux_count += 1

        steps += 1

        # ============================================================
        # New Dynamic Gradient Diagnostics (supports dense + sparse)
        # ============================================================
        should_sample_grad = (
            grad_diag_enabled
            and grad_diag is not None
            and has_ctr
            and has_cvr
            and (current_global_step % diag_every == 0)
        )
        if should_sample_grad:
            try:
                # Zero gradients before diagnostics
                optimizer_bundle.zero_grad(set_to_none=True)
                
                # Keep diagnostics in full precision
                with torch_amp.autocast(enabled=False, device_type=amp_device_type):
                    diag_outputs = model(features_dev)
                    
                    # Build per-task losses for gradient computation
                    # CTR loss: full batch
                    loss_ctr_diag = F.binary_cross_entropy_with_logits(
                        diag_outputs["ctr"].float(), labels["y_ctr"].float(), reduction="mean"
                    )
                    
                    # For ESMM: use CTCVR loss; for non-ESMM: use masked CVR loss
                    if use_esmm:
                        # Standard ESMM: compute CTCVR loss
                        log_p_ctr = F.logsigmoid(diag_outputs["ctr"].float())
                        log_p_cvr = F.logsigmoid(diag_outputs["cvr"].float())
                        log_p_ctcvr = log_p_ctr + log_p_cvr
                        # Simple BCE approximation for gradient analysis
                        y_ctcvr = labels["y_ctcvr"].float()
                        loss_ctcvr_diag = -(y_ctcvr * log_p_ctcvr + (1 - y_ctcvr) * torch.log1p(-torch.exp(log_p_ctcvr).clamp(max=1-1e-7))).mean()
                        
                        losses_by_task = {"ctr": loss_ctr_diag, "ctcvr": loss_ctcvr_diag}
                    else:
                        # Non-ESMM: masked CVR loss
                        mask = labels["click_mask"].float()
                        mask_sum = mask.sum()
                        if mask_sum > 0:
                            loss_vec = F.binary_cross_entropy_with_logits(
                                diag_outputs["cvr"].float(), labels["y_cvr"].float(), reduction="none"
                            )
                            loss_cvr_diag = (loss_vec * mask).sum() / (mask_sum + EPS)
                        else:
                            loss_cvr_diag = torch.tensor(0.0, device=device, requires_grad=True)
                        
                        losses_by_task = {"ctr": loss_ctr_diag, "cvr": loss_cvr_diag}
                
                # Compute gradient metrics using new dynamic approach
                diag_metrics = grad_diag.compute_metrics(losses_by_task)
                
                # Extract and accumulate metrics
                shared_dense_count_last = diag_metrics.get("shared_dense_count", 0)
                shared_sparse_count_last = diag_metrics.get("shared_sparse_count", 0)
                
                # Norms
                if use_esmm:
                    norm_ctr = diag_metrics.get("grad_norm_shared_ctr")
                    norm_ctcvr = diag_metrics.get("grad_norm_shared_ctcvr")
                    if norm_ctr is not None:
                        grad_norm_ctr_vals.append(norm_ctr)
                    if norm_ctcvr is not None:
                        grad_norm_ctcvr_vals.append(norm_ctcvr)
                    if norm_ctr is not None and norm_ctcvr is not None and norm_ctr > 0:
                        grad_norm_ratio_vals.append(norm_ctcvr / norm_ctr)
                else:
                    norm_ctr = diag_metrics.get("grad_norm_shared_ctr")
                    norm_cvr = diag_metrics.get("grad_norm_shared_cvr")
                    if norm_ctr is not None:
                        grad_norm_ctr_vals.append(norm_ctr)
                    if norm_cvr is not None:
                        grad_norm_cvr_vals.append(norm_cvr)
                    if norm_ctr is not None and norm_cvr is not None and norm_ctr > 0:
                        grad_norm_ratio_vals.append(norm_cvr / norm_ctr)
                
                # Cosine similarities (dense, sparse, all)
                cos_dense = diag_metrics.get("grad_cosine_shared_dense")
                cos_sparse = diag_metrics.get("grad_cosine_shared_sparse")
                cos_all = diag_metrics.get("grad_cosine_shared_all")
                
                if cos_dense is not None:
                    grad_cos_dense_vals.append(cos_dense)
                if cos_sparse is not None:
                    grad_cos_sparse_vals.append(cos_sparse)
                if cos_all is not None:
                    grad_cos_all_vals.append(cos_all)
                    if cos_all < 0:
                        conflict_hits += 1
                
                grad_samples += 1
                
                # Zero gradients after diagnostics
                optimizer_bundle.zero_grad(set_to_none=True)
                
            except Exception as e:
                _diag_logger.warning("grad_diag: error during computation: %s", e)
                optimizer_bundle.zero_grad(set_to_none=True)

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            log_parts = [
                f"[train] epoch={epoch} step={step+1} global_step={current_global_step}",
                f"loss_total={loss_dict['loss_total']:.4f}",
            ]
            log_parts.append(f"mode={'esmm' if use_esmm else 'legacy'}")
            if has_ctr:
                log_parts.append(f"loss_ctr={loss_dict.get('loss_ctr', 0.0):.4f}")
            else:
                log_parts.append("loss_ctr=0.0000(disabled)")
            if has_cvr:
                if use_esmm:
                    log_parts.append(f"loss_ctcvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                    log_parts.append(f"n_exposure_batch={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
                    log_parts.append("aliases(loss_cvr->loss_ctcvr,mask_sum->n_exposure_batch)")
                else:
                    log_parts.append(f"loss_cvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                    log_parts.append(f"mask_sum={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
            else:
                log_parts.append("loss_cvr=0.0000(disabled)")
            if has_ctr:
                mode = loss_dict.get("pos_weight_mode")
                if mode:
                    log_parts.append(f"pos_weight_mode={mode}")
                ctr_pw = loss_dict.get("pos_weight_ctr")
                if ctr_pw is not None:
                    log_parts.append(f"ctr_pos_w={ctr_pw:.3f}")
            log_parts.append(f"time={elapsed:.2f}s")
            logger.info(" ".join(log_parts))

        if eval_every_steps and validate_fn and (current_global_step % eval_every_steps == 0):
            # Mid-epoch validation at the same cadence as training logs.
            # validate_fn must switch the model to eval/no_grad internally; we switch back to train afterwards.
            val_metrics = validate_fn()
            model.train()
            if eval_callback:
                eval_callback(val_metrics, current_global_step, epoch)
            if val_metrics and hasattr(logger, "info"):
                auc_primary = val_metrics.get("auc_primary")
                auc_ctr = val_metrics.get("auc_ctr")
                auc_cvr = val_metrics.get("auc_cvr")
                log_parts = [
                    f"[eval] epoch={epoch} step={step+1} global_step={current_global_step}",
                    f"loss_total={val_metrics.get('loss_total'):.4f}",
                ]
                if auc_primary is not None:
                    log_parts.append(f"auc_primary={auc_primary:.4f}")
                if auc_ctr is not None:
                    log_parts.append(f"auc_ctr={auc_ctr:.4f}")
                if auc_cvr is not None:
                    log_parts.append(f"auc_cvr={auc_cvr:.4f}")
                logger.info(" ".join(log_parts))

    dur = time.time() - t_start
    if steps == 0:
        return {}

    mean_ctr = tot_ctr / steps if has_ctr else None
    mean_cvr = tot_cvr / steps if has_cvr else None
    mean_ctr_raw = (tot_ctr_raw / ctr_raw_count) if ctr_raw_count > 0 else (None if has_ctr else None)
    mean_cvr_raw = (tot_cvr_raw / cvr_raw_count) if cvr_raw_count > 0 else (None if has_cvr else None)
    mean_ctr_scaled = (tot_ctr_scaled / ctr_scaled_count) if ctr_scaled_count > 0 else (None if has_ctr else None)
    mean_cvr_scaled = (tot_cvr_scaled / cvr_scaled_count) if cvr_scaled_count > 0 else (None if has_cvr else None)
    loss_ratio_scaled = None
    if mean_ctr_scaled is not None and mean_cvr_scaled is not None:
        loss_ratio_scaled = float(mean_cvr_scaled / (mean_ctr_scaled + EPS))

    # Compute gradient diagnostic percentiles
    grad_cos_dense_p10, grad_cos_dense_p50, grad_cos_dense_p90 = _compute_grad_percentiles(grad_cos_dense_vals)
    grad_cos_sparse_p10, grad_cos_sparse_p50, grad_cos_sparse_p90 = _compute_grad_percentiles(grad_cos_sparse_vals)
    grad_cos_all_p10, grad_cos_all_p50, grad_cos_all_p90 = _compute_grad_percentiles(grad_cos_all_vals)
    
    grad_metrics = {
        # Norms
        "grad_norm_shared_ctr_mean": (sum(grad_norm_ctr_vals) / len(grad_norm_ctr_vals)) if grad_norm_ctr_vals else None,
        "grad_norm_shared_cvr_mean": (sum(grad_norm_cvr_vals) / len(grad_norm_cvr_vals)) if grad_norm_cvr_vals else None,
        "grad_norm_shared_ctcvr_mean": (sum(grad_norm_ctcvr_vals) / len(grad_norm_ctcvr_vals)) if grad_norm_ctcvr_vals else None,
        "grad_norm_ratio_mean": (sum(grad_norm_ratio_vals) / len(grad_norm_ratio_vals)) if grad_norm_ratio_vals else None,
        
        # Cosine similarities - dense
        "grad_cosine_shared_dense_mean": (sum(grad_cos_dense_vals) / len(grad_cos_dense_vals)) if grad_cos_dense_vals else None,
        "grad_cosine_dense_p10": grad_cos_dense_p10,
        "grad_cosine_dense_p50": grad_cos_dense_p50,
        "grad_cosine_dense_p90": grad_cos_dense_p90,
        
        # Cosine similarities - sparse
        "grad_cosine_shared_sparse_mean": (sum(grad_cos_sparse_vals) / len(grad_cos_sparse_vals)) if grad_cos_sparse_vals else None,
        "grad_cosine_sparse_p10": grad_cos_sparse_p10,
        "grad_cosine_sparse_p50": grad_cos_sparse_p50,
        "grad_cosine_sparse_p90": grad_cos_sparse_p90,
        
        # Cosine similarities - combined
        "grad_cosine_shared_mean": (sum(grad_cos_all_vals) / len(grad_cos_all_vals)) if grad_cos_all_vals else None,
        "grad_cosine_p10": grad_cos_all_p10,
        "grad_cosine_p50": grad_cos_all_p50,
        "grad_cosine_p90": grad_cos_all_p90,
        
        # Conflict detection
        "conflict_rate": (conflict_hits / grad_samples) if grad_samples else None,
        "grad_samples": grad_samples,
        
        # Shared param counts (from last sample)
        "shared_dense_count": shared_dense_count_last if grad_samples > 0 else None,
        "shared_sparse_count": shared_sparse_count_last if grad_samples > 0 else None,
    }

    # ESMM-specific metrics
    esmm_metrics = {}
    if use_esmm and esmm_metric_count > 0:
        esmm_metrics = {
            "p_ctr_mean": tot_p_ctr_mean / esmm_metric_count,
            "p_cvr_mean": tot_p_cvr_mean / esmm_metric_count,
            "p_ctcvr_mean": tot_p_ctcvr_mean / esmm_metric_count,
            "pos_rate_ctr": tot_pos_rate_ctr / esmm_metric_count,
            "pos_rate_cvr": tot_pos_rate_cvr / esmm_metric_count if tot_pos_rate_cvr > 0 else None,
            "pos_rate_ctcvr": tot_pos_rate_ctcvr / esmm_metric_count,
            "loss_cvr_aux": (tot_loss_cvr_aux / cvr_aux_count) if cvr_aux_count > 0 else None,
            "esmm_version": esmm_version,
        }

    return {
        "loss_total": tot_loss / steps,
        "loss_ctr": mean_ctr if has_ctr else 0.0,
        "loss_cvr": mean_cvr if has_cvr else 0.0,
        "loss_ctr_raw": mean_ctr_raw,
        "loss_cvr_raw": mean_cvr_raw,
        "loss_ctr_scaled": mean_ctr_scaled,
        "loss_cvr_scaled": mean_cvr_scaled,
        "loss_ratio_scaled": loss_ratio_scaled,
        "mask_cvr_sum": tot_mask / steps if has_cvr else 0.0,
        "steps": steps,
        "n_rows": n_rows,
        "n_masked_train": int(n_masked) if has_cvr else None,
        "pos_masked_train": int(pos_masked) if has_cvr else None,
        "steps_per_sec": steps / dur if dur > 0 else 0.0,
        **grad_metrics,
        **esmm_metrics,
    }


def validate(
    model,
    loader,
    loss_fn,
    device,
    logger,
    epoch: int,
    max_steps: Optional[int] = None,
    log_every: int = 200,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    amp_device_type: str = "cuda",
    calc_auc: bool = False,
    log_health_metrics: bool = False,
) -> Dict[str, float]:
    model.eval()
    enabled_heads = getattr(loss_fn, "enabled_heads", {"ctr", "cvr"})
    use_esmm = bool(getattr(loss_fn, "use_esmm", False))
    esmm_eps = float(getattr(loss_fn, "esmm_eps", 1e-8))
    has_ctr = "ctr" in enabled_heads
    has_cvr = "cvr" in enabled_heads

    tot_loss = tot_ctr = tot_cvr = tot_mask = 0.0
    tot_ctr_raw = tot_cvr_raw = 0.0
    tot_ctr_scaled = tot_cvr_scaled = 0.0
    ctr_raw_count = cvr_raw_count = 0
    ctr_scaled_count = cvr_scaled_count = 0
    steps = 0
    n_rows = 0
    n_masked = 0.0
    pos_masked = 0.0
    t_start = time.time()

    y_ctr_list: List[torch.Tensor] = []
    y_cvr_list: List[torch.Tensor] = []
    y_ctcvr_list: List[torch.Tensor] = []
    click_mask_list: List[torch.Tensor] = []
    ctr_logit_list: List[torch.Tensor] = []
    cvr_logit_list: List[torch.Tensor] = []
    ctcvr_logit_list: List[torch.Tensor] = []
    p_cvr_logit_list: List[torch.Tensor] = []

    # Health metrics collection (only when log_health_metrics=True)
    gate_ctr_list: List[torch.Tensor] = []
    gate_cvr_list: List[torch.Tensor] = []
    p_ctcvr_list: List[torch.Tensor] = []  # For CTCVR prob stats

    with torch.no_grad():
        for step, (labels, features, meta) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            labels = _to_device_labels(labels, device)
            features_dev = _to_device_features(features, device)

            with torch_amp.autocast(device_type=amp_device_type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(features_dev)
                batch = {"labels": labels, "features": features_dev, "meta": meta}
                loss, loss_dict = loss_fn.compute(outputs, batch)

            B = int(labels["y_ctr"].shape[0])
            n_rows += B
            if has_cvr:
                mask = labels["click_mask"]
                n_masked += float(mask.sum().item())
                pos_masked += float((labels["y_cvr"] * (mask > 0.5)).sum().item())

            if calc_auc:
                if has_ctr:
                    ctr_logit = outputs["ctr"]
                    if ctr_logit.dim() > 1:
                        ctr_logit = ctr_logit.view(-1)
                    ctr_logit_list.append(ctr_logit.detach().cpu())
                    y_ctr_list.append(labels["y_ctr"].detach().cpu())
                if has_cvr:
                    if use_esmm:
                        ctcvr_logit = outputs.get("ctcvr", outputs.get("cvr"))
                        if ctcvr_logit is None:
                            raise KeyError("ctcvr/cvr logit missing in outputs during validation")
                        if ctcvr_logit.dim() > 1:
                            ctcvr_logit = ctcvr_logit.view(-1)
                        ctcvr_logit_list.append(ctcvr_logit.detach().cpu())
                        y_ctcvr_list.append(labels["y_ctcvr"].detach().cpu())

                        # Derived CVR (post-click) probability for optional monitoring
                        ctr_logit = outputs["ctr"]
                        if ctr_logit.dim() > 1:
                            ctr_logit = ctr_logit.view(-1)
                        p_ctr = torch.sigmoid(ctr_logit)
                        p_ctcvr = torch.sigmoid(ctcvr_logit)
                        p_cvr = p_ctcvr / (p_ctr + esmm_eps)
                        p_cvr = torch.clamp(p_cvr, max=1.0)
                        p_cvr_logit = torch.log(p_cvr / torch.clamp(1.0 - p_cvr, min=esmm_eps))
                        p_cvr_logit_list.append(p_cvr_logit.detach().cpu())
                        y_cvr_list.append(labels["y_cvr"].detach().cpu())
                        click_mask_list.append(labels["y_ctr"].detach().cpu())  # click subset mask
                        cvr_logit_list.append(ctcvr_logit.detach().cpu())  # keep for logging symmetry

                        # Collect p_ctcvr for health metrics
                        if log_health_metrics:
                            p_ctcvr_list.append(p_ctcvr.detach().cpu())
                    else:
                        cvr_logit = outputs["cvr"]
                        if cvr_logit.dim() > 1:
                            cvr_logit = cvr_logit.view(-1)
                        cvr_logit_list.append(cvr_logit.detach().cpu())
                        y_cvr_list.append(labels["y_cvr"].detach().cpu())
                        click_mask_list.append(labels["click_mask"].detach().cpu())

            # Collect gate weights for health metrics (MMoE only)
            if log_health_metrics and calc_auc:
                aux = outputs.get("aux", {})
                gates = aux.get("gates", {})
                if "ctr" in gates:
                    gate_ctr_list.append(gates["ctr"].detach().cpu())
                if "cvr" in gates:
                    gate_cvr_list.append(gates["cvr"].detach().cpu())

            tot_loss += loss_dict["loss_total"]
            tot_ctr += loss_dict.get("loss_ctr", 0.0)
            tot_cvr += loss_dict.get("loss_cvr", 0.0)
            tot_mask += loss_dict.get("mask_cvr_sum", 0.0)
            if loss_dict.get("loss_ctr_raw") is not None:
                tot_ctr_raw += loss_dict["loss_ctr_raw"]
                ctr_raw_count += 1
            if loss_dict.get("loss_cvr_raw") is not None:
                tot_cvr_raw += loss_dict["loss_cvr_raw"]
                cvr_raw_count += 1
            if loss_dict.get("loss_ctr_scaled") is not None:
                tot_ctr_scaled += loss_dict["loss_ctr_scaled"]
                ctr_scaled_count += 1
            if loss_dict.get("loss_cvr_scaled") is not None:
                tot_cvr_scaled += loss_dict["loss_cvr_scaled"]
                cvr_scaled_count += 1
            steps += 1

            if (step + 1) % log_every == 0:
                elapsed = time.time() - t_start
                log_parts = [
                    f"[val] epoch={epoch} step={step+1}",
                    f"loss_total={loss_dict['loss_total']:.4f}",
                ]
                log_parts.append(f"mode={'esmm' if use_esmm else 'legacy'}")
                if has_ctr:
                    log_parts.append(f"loss_ctr={loss_dict.get('loss_ctr', 0.0):.4f}")
                else:
                    log_parts.append("loss_ctr=0.0000(disabled)")
                if has_cvr:
                    if use_esmm:
                        log_parts.append(f"loss_ctcvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                        log_parts.append(f"n_exposure_batch={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
                        log_parts.append("aliases(loss_cvr->loss_ctcvr,mask_sum->n_exposure_batch)")
                    else:
                        log_parts.append(f"loss_cvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                        log_parts.append(f"mask_sum={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
                else:
                    log_parts.append("loss_cvr=0.0000(disabled)")
                if has_ctr:
                    mode = loss_dict.get("pos_weight_mode")
                    if mode:
                        log_parts.append(f"pos_weight_mode={mode}")
                    ctr_pw = loss_dict.get("pos_weight_ctr")
                    if ctr_pw is not None:
                        log_parts.append(f"ctr_pos_w={ctr_pw:.3f}")
                log_parts.append(f"time={elapsed:.2f}s")
                logger.info(" ".join(log_parts))

    dur = time.time() - t_start
    if steps == 0:
        return {}

    mean_ctr = tot_ctr / steps if has_ctr else None
    mean_cvr = tot_cvr / steps if has_cvr else None
    mean_ctr_raw = (tot_ctr_raw / ctr_raw_count) if ctr_raw_count > 0 else (None if has_ctr else None)
    mean_cvr_raw = (tot_cvr_raw / cvr_raw_count) if cvr_raw_count > 0 else (None if has_cvr else None)
    mean_ctr_scaled = (tot_ctr_scaled / ctr_scaled_count) if ctr_scaled_count > 0 else (None if has_ctr else None)
    mean_cvr_scaled = (tot_cvr_scaled / cvr_scaled_count) if cvr_scaled_count > 0 else (None if has_cvr else None)
    loss_ratio_scaled = None
    if mean_ctr_scaled is not None and mean_cvr_scaled is not None:
        loss_ratio_scaled = float(mean_cvr_scaled / (mean_ctr_scaled + EPS))

    auc_ctr = auc_cvr = auc_ctcvr = auc_primary = None
    if calc_auc:
        auc_vals = []
        ctr_metrics = cvr_metrics = ctcvr_metrics = None
        if has_ctr and y_ctr_list:
            ctr_metrics = compute_binary_metrics(
                torch.cat(y_ctr_list).numpy(), torch.cat(ctr_logit_list).numpy()
            )
            auc_ctr = ctr_metrics.get("auc")
            if auc_ctr is not None:
                auc_vals.append(auc_ctr)
        if has_cvr and y_cvr_list:
            if use_esmm and p_cvr_logit_list:
                # CVR evaluated on clicked subset via derived p_cvr
                mask_arr = torch.cat(click_mask_list).numpy()  # y_ctr used as mask
                cvr_metrics = compute_binary_metrics(
                    torch.cat(y_cvr_list).numpy(),
                    torch.cat(p_cvr_logit_list).numpy(),
                    mask=mask_arr > 0.5,
                )
                auc_cvr = cvr_metrics.get("auc")
            else:
                mask_arr = torch.cat(click_mask_list).numpy()
                cvr_metrics = compute_binary_metrics(
                    torch.cat(y_cvr_list).numpy(),
                    torch.cat(cvr_logit_list).numpy(),
                    mask=mask_arr > 0.5,
                )
                auc_cvr = cvr_metrics.get("auc")
            if auc_cvr is not None and not use_esmm:  # esmm keeps cvr as optional monitor, not part of primary auc
                auc_vals.append(auc_cvr)
        if use_esmm and y_ctcvr_list and ctcvr_logit_list:
            ctcvr_metrics = compute_binary_metrics(
                torch.cat(y_ctcvr_list).numpy(), torch.cat(ctcvr_logit_list).numpy()
            )
            auc_ctcvr = ctcvr_metrics.get("auc")
            if auc_ctcvr is not None:
                auc_vals.append(auc_ctcvr)
        if auc_vals:
            auc_primary = float(sum(auc_vals) / len(auc_vals))

    # Health monitoring metrics (only computed when log_health_metrics=True and calc_auc=True)
    health_metrics: Dict[str, Any] = {}
    if log_health_metrics and calc_auc:
        # A) Label counts and reliability flags
        if y_ctr_list and y_cvr_list and click_mask_list:
            all_y_ctr = torch.cat(y_ctr_list)
            all_y_cvr = torch.cat(y_cvr_list)
            all_click_mask = torch.cat(click_mask_list)
            all_y_ctcvr = torch.cat(y_ctcvr_list) if y_ctcvr_list else None
            label_counts = compute_label_counts(
                y_ctr=all_y_ctr,
                y_cvr=all_y_cvr,
                click_mask=all_click_mask,
                y_ctcvr=all_y_ctcvr,
            )
            health_metrics.update(label_counts)

        # B) Logit/Prob distribution stats
        if ctr_logit_list:
            all_ctr_logit = torch.cat(ctr_logit_list)
            health_metrics.update(compute_logit_stats(all_ctr_logit, "logit_ctr"))

        if cvr_logit_list:
            all_cvr_logit = torch.cat(cvr_logit_list)
            health_metrics.update(compute_logit_stats(all_cvr_logit, "logit_cvr"))

        # CTCVR probability stats (ESMM mode)
        if p_ctcvr_list:
            all_p_ctcvr = torch.cat(p_ctcvr_list)
            health_metrics.update(compute_prob_stats(all_p_ctcvr, "prob_ctcvr"))

        # C) MMoE gate health metrics
        if gate_ctr_list:
            health_metrics.update(aggregate_gate_metrics(gate_ctr_list, "ctr"))
        if gate_cvr_list:
            health_metrics.update(aggregate_gate_metrics(gate_cvr_list, "cvr"))

    return {
        "loss_total": tot_loss / steps,
        "loss_ctr": mean_ctr if has_ctr else 0.0,
        "loss_cvr": mean_cvr if has_cvr else 0.0,
        "loss_ctr_raw": mean_ctr_raw,
        "loss_cvr_raw": mean_cvr_raw,
        "loss_ctr_scaled": mean_ctr_scaled,
        "loss_cvr_scaled": mean_cvr_scaled,
        "loss_ratio_scaled": loss_ratio_scaled,
        "mask_cvr_sum": tot_mask / steps if has_cvr else 0.0,
        "steps": steps,
        "n_rows": n_rows,
        "n_masked_train": int(n_masked) if has_cvr else None,
        "pos_masked_train": int(pos_masked) if has_cvr else None,
        "steps_per_sec": steps / dur if dur > 0 else 0.0,
        "grad_norm_shared_ctr_mean": None,
        "grad_norm_shared_cvr_mean": None,
        "grad_norm_ratio_mean": None,
        "grad_cosine_shared_mean": None,
        "conflict_rate": None,
        "grad_cosine_p10": None,
        "grad_cosine_p50": None,
        "grad_cosine_p90": None,
        "auc_ctr": auc_ctr,
        "auc_cvr": auc_cvr,
        "auc_ctcvr": auc_ctcvr,
        "auc_primary": auc_primary,
        **health_metrics,
    }


__all__ = ["train_one_epoch", "validate"]
