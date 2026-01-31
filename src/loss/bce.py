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
        use_esmm: bool = False,
        lambda_ctr: float = 1.0,
        lambda_ctcvr: float = 1.0,
        esmm_eps: float = 1e-8,
        enabled_heads: Any = None,
        pos_weight_dynamic: bool = True,
        static_pos_weight_ctr: float | None = None,
        static_pos_weight_ctcvr: float | None = None,
        pos_weight_clip_ctr: float | None = None,
        pos_weight_clip_ctcvr: float | None = None,
    ):
        self.w_ctr = float(w_ctr)
        self.w_cvr = float(w_cvr)
        self.eps = float(eps)
        self.use_esmm = bool(use_esmm)
        self.lambda_ctr = float(lambda_ctr)
        self.lambda_ctcvr = float(lambda_ctcvr)
        self.esmm_eps = float(esmm_eps)
        self.pos_weight_dynamic = bool(pos_weight_dynamic)
        self.static_pos_weight_ctr = (
            float(static_pos_weight_ctr) if static_pos_weight_ctr is not None else None
        )
        # Reserved for future ESMM usage; stored to keep config round-trip stable.
        self.static_pos_weight_ctcvr = (
            float(static_pos_weight_ctcvr) if static_pos_weight_ctcvr is not None else None
        )
        self.pos_weight_clip_ctr = float(pos_weight_clip_ctr) if pos_weight_clip_ctr is not None else None
        self.pos_weight_clip_ctcvr = (
            float(pos_weight_clip_ctcvr) if pos_weight_clip_ctcvr is not None else None
        )
        default_heads = ["ctr", "cvr"]
        enabled_list = enabled_heads or default_heads
        self.enabled_heads = {h.lower() for h in enabled_list}

    def compute(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        labels = get_labels(batch)

        y_ctr = labels.get("y_ctr")
        y_cvr = labels.get("y_cvr")
        y_ctcvr = labels.get("y_ctcvr")
        click_mask = labels.get("click_mask")

        # Base device for creating zeros without moving tensors repeatedly.
        base_device = None
        if outputs:
            base_device = next(iter(outputs.values())).device
        zero = lambda: torch.tensor(0.0, device=base_device)  # noqa: E731

        # ESMM branch: CTR + CTCVR (full exposure, static pos_weight)
        if self.use_esmm:
            ctr_logit = outputs.get("ctr")
            ctcvr_logit = outputs.get("ctcvr", outputs.get("cvr"))
            assert ctr_logit is not None, "ctr logit missing from outputs"
            assert ctcvr_logit is not None, "ctcvr/cvr logit missing from outputs"
            assert y_ctr is not None, "y_ctr missing from labels"
            assert y_ctcvr is not None, "y_ctcvr missing from labels"
            assert ctr_logit.shape == y_ctr.shape, "ctr logit/label shape mismatch"
            assert ctcvr_logit.shape == y_ctcvr.shape, "ctcvr logit/label shape mismatch"

            # Static pos_weight only (validated at config level).
            ctr_pw = self.static_pos_weight_ctr if self.static_pos_weight_ctr is not None else 1.0
            ctcvr_pw = self.static_pos_weight_ctcvr if self.static_pos_weight_ctcvr is not None else 1.0
            pos_weight_ctr = torch.tensor(ctr_pw, device=ctr_logit.device, dtype=ctr_logit.dtype)
            pos_weight_ctcvr = torch.tensor(ctcvr_pw, device=ctcvr_logit.device, dtype=ctcvr_logit.dtype)
            if self.pos_weight_clip_ctr is not None:
                pos_weight_ctr = torch.clamp(pos_weight_ctr, max=self.pos_weight_clip_ctr)
            if self.pos_weight_clip_ctcvr is not None:
                pos_weight_ctcvr = torch.clamp(pos_weight_ctcvr, max=self.pos_weight_clip_ctcvr)

            loss_ctr = F.binary_cross_entropy_with_logits(ctr_logit, y_ctr, reduction="mean", pos_weight=pos_weight_ctr)
            loss_ctcvr = F.binary_cross_entropy_with_logits(
                ctcvr_logit, y_ctcvr, reduction="mean", pos_weight=pos_weight_ctcvr
            )
            loss_total = self.lambda_ctr * loss_ctr + self.lambda_ctcvr * loss_ctcvr

            p_ctr = torch.sigmoid(ctr_logit)
            p_ctcvr = torch.sigmoid(ctcvr_logit)
            p_cvr = p_ctcvr / (p_ctr + self.esmm_eps)
            consistency_mask = p_ctr > 1e-3
            consistency_delta = None
            if consistency_mask.any():
                ctcvr_recomposed = p_ctr * p_cvr
                consistency_delta = float(torch.abs(p_ctcvr[consistency_mask] - ctcvr_recomposed[consistency_mask]).mean().item())

            loss_dict = {
                "loss_total": float(loss_total.item()),
                "loss_ctr": float(loss_ctr.item()),
                "loss_cvr": float(loss_ctcvr.item()),  # kept key name for compatibility
                "loss_ctcvr": float(loss_ctcvr.item()),
                "mask_cvr_sum": float(y_ctr.numel()),  # exposures count
                "pos_weight_ctr": float(pos_weight_ctr.item()),
                "pos_weight_cvr": float(pos_weight_ctcvr.item()),
                "pos_weight_ctcvr": float(pos_weight_ctcvr.item()),
                "pos_weight_mode": "static_esmm",
                "p_ctr_mean": float(p_ctr.mean().item()),
                "p_ctcvr_mean": float(p_ctcvr.mean().item()),
                "p_cvr_mean": float(p_cvr.mean().item()),
                "consistency_delta": consistency_delta,
            }
            return loss_total, loss_dict

        loss_ctr = zero()
        loss_cvr = zero()
        mask_sum_val = 0.0
        loss_ctr_raw_val: float | None = None
        loss_cvr_raw_val: float | None = None
        loss_ctr_scaled_val: float | None = None
        loss_cvr_scaled_val: float | None = None
        pos_weight_ctr_val: float | None = None
        pos_weight_cvr_val: float | None = None
        pos_weight_mode = "dynamic" if self.pos_weight_dynamic else "static"

        if "ctr" in self.enabled_heads:
            ctr_logit = outputs["ctr"]
            assert y_ctr is not None, "y_ctr missing from labels"
            assert ctr_logit.shape == y_ctr.shape, "ctr logit/label shape mismatch"
            if self.pos_weight_dynamic:
                # 动态计算 CTR 正例权重：neg/pos
                pos_count = torch.clamp(y_ctr.sum(), min=self.eps)
                neg_count = torch.clamp((1.0 - y_ctr).sum(), min=self.eps)
                pos_weight_ctr = neg_count / pos_count
            else:
                # Static mode falls back to 1.0 if not provided to avoid crashing old checkpoints.
                static_val = self.static_pos_weight_ctr if self.static_pos_weight_ctr is not None else 1.0
                pos_weight_ctr = torch.tensor(static_val, device=ctr_logit.device, dtype=ctr_logit.dtype)

            if self.pos_weight_clip_ctr is not None:
                pos_weight_ctr = torch.clamp(pos_weight_ctr, max=self.pos_weight_clip_ctr)

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
            "pos_weight_mode": pos_weight_mode,
        }
        return loss_total, loss_dict


__all__ = ["MultiTaskBCELoss"]
