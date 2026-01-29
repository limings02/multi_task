from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from src.loss.base import LossFn, get_labels


class MultiTaskBCELoss:
    """
    BCE losses for CTR and CVR with click-conditional masking.
    CTCVR is not directly trained here; only its probability is logged.
    """

    def __init__(
        self,
        w_ctr: float = 1.0,
        w_cvr: float = 1.0,
        eps: float = 1e-6,
        enabled_heads: Any = None,
    ):
        self.w_ctr = float(w_ctr)
        self.w_cvr = float(w_cvr)
        self.eps = float(eps)
        default_heads = ["ctr", "cvr"]
        enabled_list = enabled_heads or default_heads
        self.enabled_heads = {h.lower() for h in enabled_list}

    def compute(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        labels = get_labels(batch)

        y_ctr = labels.get("y_ctr")
        y_cvr = labels.get("y_cvr")
        click_mask = labels.get("click_mask")

        # Base device for creating zeros without moving tensors repeatedly.
        base_device = None
        if outputs:
            base_device = next(iter(outputs.values())).device
        zero = lambda: torch.tensor(0.0, device=base_device)  # noqa: E731

        loss_ctr = zero()
        loss_cvr = zero()
        mask_sum_val = 0.0
        loss_ctr_raw_val: float | None = None
        loss_cvr_raw_val: float | None = None
        loss_ctr_scaled_val: float | None = None
        loss_cvr_scaled_val: float | None = None
        pos_weight_ctr_val: float | None = None
        pos_weight_cvr_val: float | None = None

        if "ctr" in self.enabled_heads:
            ctr_logit = outputs["ctr"]
            assert y_ctr is not None, "y_ctr missing from labels"
            assert ctr_logit.shape == y_ctr.shape, "ctr logit/label shape mismatch"
            # 动态计算 CTR 正例权重：neg/pos
            pos_count = torch.clamp(y_ctr.sum(), min=self.eps)
            neg_count = torch.clamp((1.0 - y_ctr).sum(), min=self.eps)
            pos_weight_ctr = neg_count / pos_count
            pos_weight_ctr_val = float(pos_weight_ctr.item())
            loss_ctr = F.binary_cross_entropy_with_logits(
                ctr_logit, y_ctr, reduction="mean", pos_weight=pos_weight_ctr
            )
            loss_ctr_raw_val = float(loss_ctr.item())
            loss_ctr_scaled_val = float(loss_ctr_raw_val * self.w_ctr)

        if "cvr" in self.enabled_heads:
            cvr_logit = outputs["cvr"]
            assert y_cvr is not None, "y_cvr missing from labels"
            assert click_mask is not None, "click_mask missing from labels"
            assert cvr_logit.shape == y_cvr.shape, "cvr logit/label shape mismatch"

            mask = click_mask
            mask_sum = mask.sum()
            mask_sum_val = float(mask_sum.item())
            if mask_sum_val > 0:
                # 仅在有点击的子集上计算 pos_weight，防止未曝光样本干扰
                pos_count_cvr = torch.clamp((y_cvr * mask).sum(), min=self.eps)
                neg_count_cvr = torch.clamp(((1.0 - y_cvr) * mask).sum(), min=self.eps)
                pos_weight_cvr = neg_count_cvr / pos_count_cvr
                pos_weight_cvr_val = float(pos_weight_cvr.item())
                loss_cvr_vec = F.binary_cross_entropy_with_logits(
                    cvr_logit, y_cvr, reduction="none", pos_weight=pos_weight_cvr
                )
                loss_cvr = (loss_cvr_vec * mask).sum() / (mask_sum + self.eps)
                loss_cvr_raw_val = float(loss_cvr.item())
                loss_cvr_scaled_val = float(loss_cvr_raw_val * self.w_cvr)
            else:
                loss_cvr = zero()

        # Total only sums enabled heads
        loss_total = zero()
        if "ctr" in self.enabled_heads:
            loss_total = loss_total + self.w_ctr * loss_ctr
        if "cvr" in self.enabled_heads:
            loss_total = loss_total + self.w_cvr * loss_cvr

        # CTCVR prob for logging (only when both heads are active)
        ctcvr_prob_mean: float | None = None
        if "ctr" in self.enabled_heads and "cvr" in self.enabled_heads:
            ctr_prob = torch.sigmoid(outputs["ctr"])
            cvr_prob = torch.sigmoid(outputs["cvr"])
            ctcvr_prob_mean = float((ctr_prob * cvr_prob).mean().item())

        loss_ratio_scaled: float | None = None
        if loss_ctr_scaled_val is not None and loss_cvr_scaled_val is not None:
            loss_ratio_scaled = float(loss_cvr_scaled_val / (loss_ctr_scaled_val + 1e-12))

        loss_dict = {
            "loss_total": float(loss_total.item()),
            "loss_ctr": float(loss_ctr.item()) if "ctr" in self.enabled_heads else 0.0,
            "loss_cvr": float(loss_cvr.item()) if "cvr" in self.enabled_heads else 0.0,
            "mask_cvr_sum": mask_sum_val if "cvr" in self.enabled_heads else 0.0,
            "ctcvr_prob_mean": ctcvr_prob_mean,
            "loss_ctr_raw": loss_ctr_raw_val,
            "loss_cvr_raw": loss_cvr_raw_val,
            "loss_ctr_scaled": loss_ctr_scaled_val,
            "loss_cvr_scaled": loss_cvr_scaled_val,
            "loss_ratio_scaled": loss_ratio_scaled,
            "pos_weight_ctr": pos_weight_ctr_val,
            "pos_weight_cvr": pos_weight_cvr_val,
        }
        return loss_total, loss_dict


__all__ = ["MultiTaskBCELoss"]
