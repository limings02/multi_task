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

    def __init__(self, w_ctr: float = 1.0, w_cvr: float = 1.0, eps: float = 1e-6):
        self.w_ctr = float(w_ctr)
        self.w_cvr = float(w_cvr)
        self.eps = float(eps)

    def compute(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        labels = get_labels(batch)

        y_ctr = labels["y_ctr"]
        y_cvr = labels["y_cvr"]
        click_mask = labels["click_mask"]

        ctr_logit = outputs["ctr"]
        cvr_logit = outputs["cvr"]

        assert ctr_logit.shape == y_ctr.shape, "ctr logit/label shape mismatch"
        assert cvr_logit.shape == y_cvr.shape, "cvr logit/label shape mismatch"

        # CTR: unconditional BCE
        loss_ctr = F.binary_cross_entropy_with_logits(ctr_logit, y_ctr, reduction="mean")

        # CVR: only on clicked samples (click_mask == 1)
        loss_cvr_vec = F.binary_cross_entropy_with_logits(cvr_logit, y_cvr, reduction="none")
        mask = click_mask
        denom = mask.sum() + self.eps
        loss_cvr = (loss_cvr_vec * mask).sum() / denom

        # Total
        loss_total = self.w_ctr * loss_ctr + self.w_cvr * loss_cvr

        # CTCVR prob for logging (not trained here)
        ctr_prob = torch.sigmoid(ctr_logit)
        cvr_prob = torch.sigmoid(cvr_logit)
        ctcvr_prob_mean = float((ctr_prob * cvr_prob).mean().item())

        loss_dict = {
            "loss_total": float(loss_total.item()),
            "loss_ctr": float(loss_ctr.item()),
            "loss_cvr": float(loss_cvr.item()),
            "mask_cvr_sum": float(mask.sum().item()),
            "ctcvr_prob_mean": ctcvr_prob_mean,
        }
        return loss_total, loss_dict


__all__ = ["MultiTaskBCELoss"]
