from __future__ import annotations

"""
Shared evaluation entry point for CLI and Trainer auto-eval.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.core.checkpoint import load_checkpoint
from src.core.logging import get_logger
from src.data.dataloader import make_dataloader
from src.eval.calibration import compute_ece_from_logits
from src.eval.funnel import funnel_consistency
from src.eval.metrics import compute_binary_metrics, sigmoid
from src.models.build import build_model
from src.train.infer import infer_to_parquet
from src.utils.feature_meta import build_model_feature_meta

LOG = get_logger(__name__)


def _move_features_to_device(features: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Copy of the lightweight mover in train.infer to avoid circular imports.
    """
    fields = {}
    for base, fd in features["fields"].items():
        fields[base] = {
            "indices": fd["indices"].to(device),
            "offsets": fd["offsets"].to(device),
            "weights": fd["weights"].to(device) if fd.get("weights") is not None else None,
        }
    return {"fields": fields, "field_names": features["field_names"]}


def _resolve_run_dir(run_dir: Optional[str | Path], ckpt_path: Optional[str | Path], cfg: Dict[str, Any]) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    if ckpt_path is not None:
        return Path(ckpt_path).resolve().parent
    # Fallback: derive from experiment name with timestamp
    exp_name = cfg.get("experiment", {}).get("name", "eval")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"{exp_name}_eval_{ts}"


def _build_dataloader(cfg: Dict[str, Any], split: str, feature_meta: Dict[str, Any]):
    data_cfg = cfg.get("data", {})
    return make_dataloader(
        split=split,
        batch_size=int(data_cfg.get("batch_size", 256)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        shuffle=False,
        drop_last=False,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", False)),
        seed=data_cfg.get("seed"),
        feature_meta=feature_meta,
        debug=bool(data_cfg.get("debug", False)),
    )


def run_eval(
    cfg: Dict[str, Any],
    split: str,
    ckpt_path: Optional[str | Path] = None,
    run_dir: Optional[str | Path] = None,
    save_preds: bool = False,
    max_batches: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Main evaluation routine. Returns the eval summary dict and also writes
    eval.json under the resolved run_dir.
    """
    log = logger or LOG
    device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
    resolved_run_dir = _resolve_run_dir(run_dir, ckpt_path, cfg)

    # Build feature meta + loader + model
    feature_meta = build_model_feature_meta(Path(cfg["data"]["metadata_path"]), cfg.get("embedding", {}))
    loader = _build_dataloader(cfg, split=split, feature_meta=feature_meta)
    model = build_model(cfg).to(device)

    if ckpt_path:
        load_checkpoint(ckpt_path, model, optimizer=None, map_location=device, strict=False)
        log.info("Loaded checkpoint from %s", ckpt_path)

    # Forward pass to collect logits/labels for metrics
    y_ctr_list = []
    y_cvr_list = []
    click_mask_list = []
    y_ctcvr_list = []
    ctr_logit_list = []
    cvr_logit_list = []

    with torch.no_grad():
        for step, (labels, features, meta) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            labels_dev = {k: v.to(device) for k, v in labels.items() if torch.is_tensor(v)}
            features_dev = _move_features_to_device(features, device)

            outputs = model(features_dev)
            ctr_logit = outputs["ctr"]
            cvr_logit = outputs["cvr"]
            if ctr_logit.dim() > 1:
                ctr_logit = ctr_logit.view(-1)
            if cvr_logit.dim() > 1:
                cvr_logit = cvr_logit.view(-1)

            y_ctr_list.append(labels_dev["y_ctr"].cpu().numpy())
            y_cvr_list.append(labels_dev["y_cvr"].cpu().numpy())
            click_mask_list.append(labels_dev["click_mask"].cpu().numpy())
            if "y_ctcvr" in labels_dev:
                y_ctcvr_list.append(labels_dev["y_ctcvr"].cpu().numpy())

            ctr_logit_list.append(ctr_logit.cpu().numpy())
            cvr_logit_list.append(cvr_logit.cpu().numpy())

    y_ctr = np.concatenate(y_ctr_list) if y_ctr_list else np.array([])
    y_cvr = np.concatenate(y_cvr_list) if y_cvr_list else np.array([])
    click_mask = np.concatenate(click_mask_list) if click_mask_list else np.array([])
    y_ctcvr = np.concatenate(y_ctcvr_list) if y_ctcvr_list else None
    ctr_logit = np.concatenate(ctr_logit_list) if ctr_logit_list else np.array([])
    cvr_logit = np.concatenate(cvr_logit_list) if cvr_logit_list else np.array([])

    ctr_metrics = compute_binary_metrics(y_ctr, ctr_logit)
    cvr_metrics_masked = compute_binary_metrics(y_cvr, cvr_logit, mask=click_mask > 0.5)

    ece_ctr = compute_ece_from_logits(y_ctr, ctr_logit)
    ece_cvr_masked = compute_ece_from_logits(y_cvr, cvr_logit, mask=click_mask > 0.5)

    pred_ctr_prob = sigmoid(torch.from_numpy(ctr_logit)).numpy() if ctr_logit.size > 0 else np.array([])
    pred_cvr_prob = sigmoid(torch.from_numpy(cvr_logit)).numpy() if cvr_logit.size > 0 else np.array([])
    funnel_stats = None
    if pred_ctr_prob.size and pred_cvr_prob.size:
        funnel_stats = funnel_consistency(
            {
                "pred_ctr": pred_ctr_prob,
                "pred_cvr": pred_cvr_prob,
                "y_ctcvr": y_ctcvr,
            },
            has_ctcvr_label=y_ctcvr is not None,
        )

    # Optional predictions export (fresh loader to avoid Iterable exhaustion)
    preds_summary = None
    if save_preds:
        preds_path = resolved_run_dir / f"preds_{split}.parquet"
        preds_loader = _build_dataloader(cfg, split=split, feature_meta=feature_meta)
        preds_summary = infer_to_parquet(
            model,
            preds_loader,
            device=device,
            out_path=preds_path,
            max_batches=max_batches,
            include_ctcvr=True,
            split=split,
        )

    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    eval_path = resolved_run_dir / "eval.json"

    result = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split": split,
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "run_dir": str(resolved_run_dir),
        "ctr_auc": ctr_metrics["auc"],
        "cvr_auc_masked": cvr_metrics_masked["auc"],
        "ctr_logloss": ctr_metrics["logloss"],
        "cvr_logloss_masked": cvr_metrics_masked["logloss"],
        "ece_ctr": ece_ctr["ece"],
        "ece_cvr_masked": ece_cvr_masked["ece"],
        "ctr": ctr_metrics,
        "cvr_masked": cvr_metrics_masked,
        "ece_ctr_bins": ece_ctr.get("bins"),
        "ece_cvr_bins": ece_cvr_masked.get("bins"),
        "funnel_gap_stats": funnel_stats,
        "preds": preds_summary,
        "n": int(y_ctr.shape[0]),
        "n_masked": int(int((click_mask > 0.5).sum())) if click_mask.size else 0,
    }

    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log.info(
        "Eval split=%s ctr_auc=%s ctr_logloss=%s cvr_auc_masked=%s cvr_logloss_masked=%s ece_ctr=%s ece_cvr_masked=%s",
        split,
        ctr_metrics["auc"],
        ctr_metrics["logloss"],
        cvr_metrics_masked["auc"],
        cvr_metrics_masked["logloss"],
        ece_ctr["ece"],
        ece_cvr_masked["ece"],
    )
    log.info("Eval written to %s", eval_path)

    return result


__all__ = ["run_eval"]
