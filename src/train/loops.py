from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import amp as torch_amp

EPS = 1e-12


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


def _flatten_grads(params: Sequence[Tuple[str, torch.nn.Parameter]]) -> torch.Tensor | None:
    grads = []
    for _, p in params:
        if p.grad is None:
            continue
        grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return None
    return torch.cat(grads)


def _compute_grad_percentiles(values: List[float]) -> Tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    tensor_vals = torch.tensor(values, device="cpu")
    p10, p50, p90 = torch.quantile(tensor_vals, torch.tensor([0.1, 0.5, 0.9]))
    return float(p10.item()), float(p50.item()), float(p90.item())


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    logger,
    epoch: int,
    max_steps: Optional[int] = None,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 50,
    global_step: int = 0,
    grad_diag_enabled: bool = False,
    eval_every_steps: Optional[int] = None,
    validate_fn: Optional[Callable[[], Dict[str, float]]] = None,
    eval_callback: Optional[Callable[[Dict[str, float], int, int], None]] = None,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    scaler: Optional[torch_amp.GradScaler] = None,
    amp_device_type: str = "cuda",
) -> Dict[str, float]:
    """
    Train for one epoch and emit aggregated metrics plus optional gradient diagnostics.

    Gradient diagnostics reuse runtime.log_every as the sampling trigger (no extra freq knob)
    to avoid extra log spam and match existing logging cadence.
    """
    model.train()
    enabled_heads = getattr(loss_fn, "enabled_heads", {"ctr", "cvr"})
    has_ctr = "ctr" in enabled_heads
    has_cvr = "cvr" in enabled_heads

    tot_loss = tot_ctr = tot_cvr = tot_mask = 0.0
    tot_ctr_raw = tot_cvr_raw = 0.0
    tot_ctr_scaled = tot_cvr_scaled = 0.0
    steps = 0
    n_rows = 0
    n_masked = 0.0
    pos_masked = 0.0

    grad_norm_ctr_vals: List[float] = []
    grad_norm_cvr_vals: List[float] = []
    grad_norm_ratio_vals: List[float] = []
    grad_cos_vals: List[float] = []
    conflict_hits = 0
    grad_samples = 0

    shared_params: List[Tuple[str, torch.nn.Parameter]] = _select_shared_params(model) if grad_diag_enabled else []
    ctr_raw_count = cvr_raw_count = 0
    ctr_scaled_count = cvr_scaled_count = 0

    t_start = time.time()

    for step, (labels, features, meta) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        labels = _to_device_labels(labels, device)
        features_dev = _to_device_features(features, device)

        optimizer.zero_grad(set_to_none=True)
        # Autocast only on CUDA to avoid CPU precision issues.
        with torch_amp.autocast(device_type=amp_device_type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(features_dev)
            batch = {"labels": labels, "features": features_dev, "meta": meta}
            loss, loss_dict = loss_fn.compute(outputs, batch)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                # unscale before clipping, otherwise clipping would operate on scaled grads.
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

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
        steps += 1

        current_global_step = global_step + steps
        should_sample_grad = (
            grad_diag_enabled
            and has_ctr
            and has_cvr
            and shared_params
            and (current_global_step % log_every == 0)  # reuse log_every; no extra sampling knob
        )
        if should_sample_grad:
            # Gradient diagnostics run on the same cadence as logging.
            # We isolate them from real updates by zeroing before/after and skipping optimizer.step().
            optimizer.zero_grad(set_to_none=True)
            # Keep diagnostics in full precision to avoid mixing scaled grads into analysis.
            with torch_amp.autocast(enabled=False,device_type='cuda'):
                diag_out_ctr = model(features_dev)
                loss_ctr_diag = F.binary_cross_entropy_with_logits(diag_out_ctr["ctr"], labels["y_ctr"], reduction="mean")
            loss_ctr_diag.backward()
            g_ctr = _flatten_grads(shared_params)
            optimizer.zero_grad(set_to_none=True)

            with torch_amp.autocast(enabled=False,device_type='cuda'):
                diag_out_cvr = model(features_dev)
                mask = labels["click_mask"]
                mask_sum = float(mask.sum().item())
                g_cvr = None
                if mask_sum > 0:
                    # Masked CVR loss keeps denominator=mask_sum so it matches the training objective (conditional on clicks).
                    loss_vec = F.binary_cross_entropy_with_logits(diag_out_cvr["cvr"], labels["y_cvr"], reduction="none")
                    loss_cvr_diag = (loss_vec * mask).sum() / (mask.sum() + EPS)
                    loss_cvr_diag.backward()
                    g_cvr = _flatten_grads(shared_params)
            optimizer.zero_grad(set_to_none=True)

            if g_ctr is not None and g_cvr is not None:
                norm_ctr = torch.norm(g_ctr)
                norm_cvr = torch.norm(g_cvr)
                norm_ratio = norm_cvr / (norm_ctr + EPS)
                cosine = torch.dot(g_ctr, g_cvr) / (norm_ctr * norm_cvr + EPS)
                norm_ctr_val = float(norm_ctr.item())
                norm_cvr_val = float(norm_cvr.item())
                norm_ratio_val = float(norm_ratio.item())
                cosine_val = float(cosine.item())

                grad_norm_ctr_vals.append(norm_ctr_val)
                grad_norm_cvr_vals.append(norm_cvr_val)
                grad_norm_ratio_vals.append(norm_ratio_val)
                grad_cos_vals.append(cosine_val)
                conflict_hits += int(cosine_val < 0.0)
                grad_samples += 1

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            log_parts = [
                f"epoch={epoch} step={step+1}",
                f"loss_total={loss_dict['loss_total']:.4f}",
            ]
            if has_ctr:
                log_parts.append(f"loss_ctr={loss_dict.get('loss_ctr', 0.0):.4f}")
            else:
                log_parts.append("loss_ctr=0.0000(disabled)")
            if has_cvr:
                log_parts.append(f"loss_cvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                log_parts.append(f"mask_sum={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
            else:
                log_parts.append("loss_cvr=0.0000(disabled)")
            log_parts.append(f"time={elapsed:.2f}s")
            logger.info(" ".join(log_parts))

        if eval_every_steps and validate_fn and (current_global_step % eval_every_steps == 0):
            # Mid-epoch validation at the same cadence as training logs.
            # validate_fn must switch the model to eval/no_grad internally; we switch back to train afterwards.
            val_metrics = validate_fn()
            model.train()
            if eval_callback:
                eval_callback(val_metrics, current_global_step, epoch)

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

    grad_cos_p10, grad_cos_p50, grad_cos_p90 = _compute_grad_percentiles(grad_cos_vals)
    grad_metrics = {
        "grad_norm_shared_ctr_mean": (sum(grad_norm_ctr_vals) / grad_samples) if grad_samples else None,
        "grad_norm_shared_cvr_mean": (sum(grad_norm_cvr_vals) / grad_samples) if grad_samples else None,
        "grad_norm_ratio_mean": (sum(grad_norm_ratio_vals) / grad_samples) if grad_samples else None,
        "grad_cosine_shared_mean": (sum(grad_cos_vals) / grad_samples) if grad_samples else None,
        "conflict_rate": (conflict_hits / grad_samples) if grad_samples else None,
        "grad_cosine_p10": grad_cos_p10,
        "grad_cosine_p50": grad_cos_p50,
        "grad_cosine_p90": grad_cos_p90,
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
) -> Dict[str, float]:
    model.eval()
    enabled_heads = getattr(loss_fn, "enabled_heads", {"ctr", "cvr"})
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
                if has_ctr:
                    log_parts.append(f"loss_ctr={loss_dict.get('loss_ctr', 0.0):.4f}")
                else:
                    log_parts.append("loss_ctr=0.0000(disabled)")
                if has_cvr:
                    log_parts.append(f"loss_cvr={loss_dict.get('loss_cvr', 0.0):.4f}")
                    log_parts.append(f"mask_sum={loss_dict.get('mask_cvr_sum', 0.0):.1f}")
                else:
                    log_parts.append("loss_cvr=0.0000(disabled)")
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
    }


__all__ = ["train_one_epoch", "validate"]
