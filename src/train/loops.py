from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from torch.nn.utils import clip_grad_norm_


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
) -> Dict[str, float]:
    model.train()
    tot_loss = tot_ctr = tot_cvr = tot_mask = 0.0
    steps = 0
    t_start = time.time()

    for step, (labels, features, meta) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        labels = _to_device_labels(labels, device)
        features_dev = _to_device_features(features, device)

        outputs = model(features_dev)
        batch = {"labels": labels, "features": features_dev, "meta": meta}
        loss, loss_dict = loss_fn.compute(outputs, batch)

        # One-time sanity log at the very first batch.
        if epoch == 1 and step == 0:
            import torch.nn.functional as F

            lbls = labels
            y_ctr = lbls["y_ctr"].detach().float().cpu()
            y_cvr = lbls["y_cvr"].detach().float().cpu()
            mask = lbls["click_mask"].detach().float().cpu()
            clicked = mask > 0.5
            clicked_count = int(clicked.sum().item())
            conv_in_clicked = float(y_cvr[clicked].sum().item()) if clicked_count > 0 else 0.0
            pos_ctr = int(lbls["y_ctr"].sum().item())
            pos_cvr_clicked = float((lbls["y_cvr"] * (lbls["click_mask"] > 0.5)).sum().item())

            # Recompute backbone aux to inspect linear vs head contribution
            with torch.no_grad():
                aux = model.backbone(features_dev, return_aux=True)
            linear = aux["logit_linear"]
            if linear.dim() == 2 and linear.size(-1) == 1:
                linear = linear.squeeze(-1)
            h = aux["h"]
            ctr_head_logit = model.towers["ctr"](h)
            ctr_logit_full = ctr_head_logit + linear

            ctr_logit = outputs["ctr"]
            cvr_logit = outputs["cvr"]
            if ctr_logit.dim() == 2 and ctr_logit.size(-1) == 1:
                ctr_logit = ctr_logit.squeeze(-1)
            if cvr_logit.dim() == 2 and cvr_logit.size(-1) == 1:
                cvr_logit = cvr_logit.squeeze(-1)
            ctr_logit = ctr_logit.detach().float().cpu()
            cvr_logit = cvr_logit.detach().float().cpu()
            linear_cpu = linear.detach().float().cpu()
            ctr_head_cpu = ctr_head_logit.detach().float().cpu()
            ctr_full_cpu = ctr_logit_full.detach().float().cpu()

            ctr_prob_mean = torch.sigmoid(ctr_logit).mean().item()
            cvr_prob_mean = torch.sigmoid(cvr_logit).mean().item()

            ctr_ref = F.binary_cross_entropy_with_logits(outputs["ctr"].squeeze(-1), labels["y_ctr"], reduction="mean")
            logger.info(
                "SANITY_LOSS ctr_ref=%.6f ctr_fn=%.6f pos_ctr=%d pos_cvr_clicked=%.1f",
                ctr_ref.item(),
                loss_dict["loss_ctr"],
                pos_ctr,
                pos_cvr_clicked,
            )
            logger.info(
                "SANITY y_ctr(min/mean/max)=%.3f/%.3f/%.3f "
                "y_cvr(min/mean/max)=%.3f/%.3f/%.3f "
                "mask_sum=%.1f clicked=%d conv_in_clicked=%.1f",
                y_ctr.min().item(),
                y_ctr.mean().item(),
                y_ctr.max().item(),
                y_cvr.min().item(),
                y_cvr.mean().item(),
                y_cvr.max().item(),
                mask.sum().item(),
                clicked_count,
                conv_in_clicked,
            )
            logger.info(
                "SANITY ctr_logit(mean/std/min/max)=%.3f/%.3f/%.3f/%.3f ctr_prob_mean=%.3f",
                ctr_logit.mean().item(),
                ctr_logit.std().item(),
                ctr_logit.min().item(),
                ctr_logit.max().item(),
                ctr_prob_mean,
            )
            logger.info(
                "SANITY cvr_logit(mean/std/min/max)=%.3f/%.3f/%.3f/%.3f cvr_prob_mean=%.3f",
                cvr_logit.mean().item(),
                cvr_logit.std().item(),
                cvr_logit.min().item(),
                cvr_logit.max().item(),
                cvr_prob_mean,
            )
            logger.info(
                "SANITY linear(mean/std/min/max)=%.3f/%.3f/%.3f/%.3f ctr_head(mean/std/min/max)=%.3f/%.3f/%.3f/%.3f ctr_combined(mean/std/min/max)=%.3f/%.3f/%.3f/%.3f",
                linear_cpu.mean().item(),
                linear_cpu.std().item(),
                linear_cpu.min().item(),
                linear_cpu.max().item(),
                ctr_head_cpu.mean().item(),
                ctr_head_cpu.std().item(),
                ctr_head_cpu.min().item(),
                ctr_head_cpu.max().item(),
                ctr_full_cpu.mean().item(),
                ctr_full_cpu.std().item(),
                ctr_full_cpu.min().item(),
                ctr_full_cpu.max().item(),
            )

            # Inspect per-field weights when use_value=True
            fm = getattr(model.backbone, "feature_meta", {})
            fields = features_dev["fields"]
            for base, meta in fm.items():
                if not meta.get("use_value", False):
                    continue
                fd = fields.get(base)
                if fd is None:
                    continue
                wts = fd.get("weights")
                if wts is None or wts.numel() == 0:
                    continue
                w_cpu = wts.detach().float().cpu()
                offsets = fd["offsets"].cpu()
                indices_len = fd["indices"].shape[0]
                bag_lens = torch.empty_like(offsets)
                if offsets.numel() > 0:
                    bag_lens[:-1] = offsets[1:] - offsets[:-1]
                    last = indices_len - offsets[-1]
                    bag_lens[-1] = last
                logger.info(
                    "SANITY weights[%s] min/mean/max/absmax=%.4f/%.4f/%.4f/%.4f bag_len_max/mean=%.1f/%.2f",
                    base,
                    w_cpu.min().item(),
                    w_cpu.mean().item(),
                    w_cpu.max().item(),
                    w_cpu.abs().max().item(),
                    bag_lens.max().item() if bag_lens.numel() else 0.0,
                    bag_lens.float().mean().item() if bag_lens.numel() else 0.0,
                )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        tot_loss += loss_dict["loss_total"]
        tot_ctr += loss_dict["loss_ctr"]
        tot_cvr += loss_dict["loss_cvr"]
        tot_mask += loss_dict["mask_cvr_sum"]
        steps += 1

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"epoch={epoch} step={step+1} "
                f"loss_total={loss_dict['loss_total']:.4f} "
                f"loss_ctr={loss_dict['loss_ctr']:.4f} "
                f"loss_cvr={loss_dict['loss_cvr']:.4f} "
                f"mask_sum={loss_dict['mask_cvr_sum']:.1f} "
                f"time={elapsed:.2f}s"
            )

    dur = time.time() - t_start
    if steps == 0:
        return {}
    return {
        "loss_total": tot_loss / steps,
        "loss_ctr": tot_ctr / steps,
        "loss_cvr": tot_cvr / steps,
        "mask_cvr_sum": tot_mask / steps,
        "steps": steps,
        "steps_per_sec": steps / dur if dur > 0 else 0.0,
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
) -> Dict[str, float]:
    model.eval()
    tot_loss = tot_ctr = tot_cvr = tot_mask = 0.0
    steps = 0
    t_start = time.time()

    with torch.no_grad():
        for step, (labels, features, meta) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            labels = _to_device_labels(labels, device)
            features_dev = _to_device_features(features, device)

            outputs = model(features_dev)
            batch = {"labels": labels, "features": features_dev, "meta": meta}
            loss, loss_dict = loss_fn.compute(outputs, batch)

            tot_loss += loss_dict["loss_total"]
            tot_ctr += loss_dict["loss_ctr"]
            tot_cvr += loss_dict["loss_cvr"]
            tot_mask += loss_dict["mask_cvr_sum"]
            steps += 1

            if (step + 1) % log_every == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f"[val] epoch={epoch} step={step+1} "
                    f"loss_total={loss_dict['loss_total']:.4f} "
                    f"loss_ctr={loss_dict['loss_ctr']:.4f} "
                    f"loss_cvr={loss_dict['loss_cvr']:.4f} "
                    f"mask_sum={loss_dict['mask_cvr_sum']:.1f} "
                    f"time={elapsed:.2f}s"
                )

    dur = time.time() - t_start
    if steps == 0:
        return {}
    return {
        "loss_total": tot_loss / steps,
        "loss_ctr": tot_ctr / steps,
        "loss_cvr": tot_cvr / steps,
        "mask_cvr_sum": tot_mask / steps,
        "steps": steps,
        "steps_per_sec": steps / dur if dur > 0 else 0.0,
    }


__all__ = ["train_one_epoch", "validate"]
