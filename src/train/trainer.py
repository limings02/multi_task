from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import amp as torch_amp
import yaml

from src.core.checkpoint import load_checkpoint, save_checkpoint
from src.data.dataloader import make_dataloader
from src.loss.bce import MultiTaskBCELoss
from src.models.build import build_model
from src.eval.run_eval import run_eval
from src.train.loops import train_one_epoch, validate
from src.train.optim import build_optimizer_bundle
from src.utils.feature_meta import build_model_feature_meta
from src.utils.metrics_schema import apply_esmm_schema


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"trainer_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_enabled_heads(model_cfg: Dict[str, Any]) -> list[str]:
    heads = model_cfg.get("enabled_heads") or model_cfg.get("tasks")
    if not heads:
        heads = ["ctr", "cvr"]
    return sorted([str(h).lower() for h in heads])


def get_exp_tags(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unified experiment identifier used in every metrics.jsonl row.
    """
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "unknown_model")
    enabled = model_cfg.get("enabled_heads") or model_cfg.get("tasks") or ["ctr", "cvr"]
    enabled = sorted([str(h).lower() for h in enabled])
    exp_name = f"{model_name}__heads={'-'.join(enabled)}"
    seed = cfg.get("data", {}).get("seed", cfg.get("runtime", {}).get("seed"))
    return {"model_name": model_name, "enabled_heads": enabled, "exp_name": exp_name, "seed": seed}


class Trainer:
    def __init__(self, cfg: Dict[str, Any], config_path: Optional[str | Path] = None):
        self.cfg = cfg
        self.config_path = Path(config_path).resolve() if config_path else None

        self.use_esmm = bool(cfg.get("use_esmm", False))
        sampling_cfg = cfg.get("sampling", {}) or {}
        self.negative_sampling = str(sampling_cfg.get("negative_sampling", "keep_prob")).lower()

        tags = get_exp_tags(cfg)
        self.enabled_heads = tags["enabled_heads"]
        self.model_name = tags["model_name"]
        self.exp_name = tags["exp_name"]
        self.seed_value = tags["seed"]
        self.metrics_meta = {**tags, "config_path": str(self.config_path) if self.config_path else None}
        self.metrics_meta["mode"] = "esmm" if self.use_esmm else "non-esmm"
        exp_name = cfg.get("experiment", {}).get("name", "exp")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("runs") / f"{exp_name}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config snapshot
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        )

        self.logger = _setup_logger(self.run_dir / "train.log")
        seed = int(cfg.get("runtime", {}).get("seed", 2026))
        _set_seed(seed)

        self.device = torch.device(cfg.get("runtime", {}).get("device", "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            # Keep training runnable on machines without CUDA; AMP will be disabled automatically.
            self.device = torch.device("cpu")
            logging.getLogger(__name__).warning("CUDA requested but not available; falling back to CPU and disabling AMP.")
        runtime_cfg = cfg.get("runtime", {})
        self.amp_enabled = bool(runtime_cfg.get("amp", False)) and self.device.type == "cuda"
        self.amp_dtype = torch.float16
        self.amp_device_type = self.device.type
        # GradScaler keeps fp32 master weights while applying scaled grads; disabled on CPU automatically.
        self.scaler = torch_amp.GradScaler(enabled=self.amp_enabled, device="cuda")
        self.debug_logit_every = int(runtime_cfg.get("debug_logit_every", 0) or 0)
        self.debug_linear_stats = bool(runtime_cfg.get("debug_linear_stats", False))
        self.debug_fm_stats = bool(runtime_cfg.get("debug_fm_stats", False))
        self.debug_dataloader_assert = bool(runtime_cfg.get("debug_dataloader_assert", False))

        # feature_meta for dataloader/model
        metadata_path = Path(cfg["data"]["metadata_path"])
        embedding_cfg = cfg.get("embedding", {})
        self.feature_meta = build_model_feature_meta(metadata_path, embedding_cfg)

        loss_cfg = cfg.get("loss", {})
        static_pos_weight_cfg = loss_cfg.get("static_pos_weight", {}) or {}
        pos_weight_clip_cfg = loss_cfg.get("pos_weight_clip", {}) or {}

        def _validate_esmm_requirements() -> None:
            if not self.use_esmm:
                return
            if bool(loss_cfg.get("pos_weight_dynamic", True)):
                raise ValueError("use_esmm=true requires loss.pos_weight_dynamic to be false (static pos_weight only).")
            if self.negative_sampling not in {"none", "off", "disable", "disabled"}:
                raise ValueError(
                    f"use_esmm=true requires sampling.negative_sampling='none'; got '{self.negative_sampling}'."
                )
            if "ctr" not in static_pos_weight_cfg or "ctcvr" not in static_pos_weight_cfg:
                raise ValueError("use_esmm=true requires loss.static_pos_weight.ctr and loss.static_pos_weight.ctcvr to be set.")

        _validate_esmm_requirements()

        # Resolve negative sampling keep prob (enforced to 1.0 for ESMM)
        neg_keep_prob_cfg = float(cfg.get("data", {}).get("neg_keep_prob_train", 1.0))
        if self.negative_sampling in {"none", "off", "disable", "disabled"} or self.use_esmm:
            self.neg_keep_prob_train = 1.0
        else:
            self.neg_keep_prob_train = neg_keep_prob_cfg

        self.logger.info(
            "mode=%s pos_weight_dynamic=%s pos_weight_ctr=%s pos_weight_ctcvr=%s neg_sampling=%s neg_keep_prob_train=%.3f",
            "esmm" if self.use_esmm else "non-esmm",
            bool(loss_cfg.get("pos_weight_dynamic", True)),
            static_pos_weight_cfg.get("ctr"),
            static_pos_weight_cfg.get("ctcvr"),
            self.negative_sampling,
            self.neg_keep_prob_train,
        )

        self.train_loader = make_dataloader(
            split="train",
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=int(cfg["data"].get("num_workers", 0)),
            shuffle=False,
            drop_last=bool(cfg["data"].get("drop_last", False)),
            pin_memory=bool(cfg["data"].get("pin_memory", True)),
            persistent_workers=bool(cfg["data"].get("persistent_workers", False)),
            seed=cfg["data"].get("seed"),
            feature_meta=self.feature_meta,
            debug=bool(cfg["data"].get("debug", False) or self.debug_dataloader_assert),
            neg_keep_prob_train=self.neg_keep_prob_train,
        )
        self.valid_loader = make_dataloader(
            split="valid",
            batch_size=int(cfg["data"]["batch_size"]),
            num_workers=int(cfg["data"].get("num_workers", 0)),
            shuffle=False,
            drop_last=False,
            pin_memory=bool(cfg["data"].get("pin_memory", True)),
            persistent_workers=bool(cfg["data"].get("persistent_workers", False)),
            seed=cfg["data"].get("seed"),
            feature_meta=self.feature_meta,
            debug=bool(cfg["data"].get("debug", False) or self.debug_dataloader_assert),
        )

        self.model = build_model(cfg).to(self.device)
        # Enable debug hooks on backbone/head if supported.
        if hasattr(self.model, "backbone"):
            setattr(self.model.backbone, "debug_linear", self.debug_linear_stats or self.debug_logit_every > 0)
            setattr(self.model.backbone, "debug_fm", self.debug_fm_stats or self.debug_logit_every > 0)
        if hasattr(self.model, "debug_logit"):
            self.model.debug_logit = self.debug_logit_every > 0

        opt_cfg = cfg.get("optim", {})
        self.optim_bundle = build_optimizer_bundle(cfg, self.model, scaler=self.scaler)

        # ===== Load aux_focal configuration (方案1: ESMM 主链路 BCE + CTCVR Aux-Focal) =====
        aux_focal_cfg = loss_cfg.get("aux_focal", {})
        
        self.loss_fn = MultiTaskBCELoss(
            w_ctr=float(loss_cfg.get("w_ctr", 1.0)),
            w_cvr=float(loss_cfg.get("w_cvr", 1.0)),
            eps=float(loss_cfg.get("eps", 1e-6)),
            use_esmm=self.use_esmm,
            esmm_version=str(cfg.get("esmm", {}).get("version", "v2")),  # "v2" (standard) or "legacy"
            lambda_ctr=float(cfg.get("esmm", {}).get("lambda_ctr", 1.0)),
            lambda_ctcvr=float(cfg.get("esmm", {}).get("lambda_ctcvr", 1.0)),
            lambda_cvr_aux=float(cfg.get("esmm", {}).get("lambda_cvr_aux", 0.0)),  # CVR aux loss
            esmm_eps=float(cfg.get("esmm", {}).get("eps", 1e-8)),
            enabled_heads=self.enabled_heads,
            pos_weight_dynamic=bool(loss_cfg.get("pos_weight_dynamic", True)),
            static_pos_weight_ctr=(
                float(static_pos_weight_cfg["ctr"]) if "ctr" in static_pos_weight_cfg else None
            ),
            static_pos_weight_ctcvr=(
                float(static_pos_weight_cfg["ctcvr"]) if "ctcvr" in static_pos_weight_cfg else None
            ),
            pos_weight_clip_ctr=(
                float(pos_weight_clip_cfg["ctr"]) if "ctr" in pos_weight_clip_cfg else None
            ),
            pos_weight_clip_ctcvr=(
                float(pos_weight_clip_cfg["ctcvr"]) if "ctcvr" in pos_weight_clip_cfg else None
            ),
            # Aux Focal parameters (defaults to disabled for backward compatibility)
            aux_focal_enabled=bool(aux_focal_cfg.get("enabled", False)),
            aux_focal_warmup_steps=int(aux_focal_cfg.get("warmup_steps", 2000)),
            aux_focal_target_head=str(aux_focal_cfg.get("target_head", "ctcvr")),
            aux_focal_lambda=float(aux_focal_cfg.get("lambda", 0.1)),
            aux_focal_gamma=float(aux_focal_cfg.get("gamma", 1.0)),
            aux_focal_use_alpha=bool(aux_focal_cfg.get("use_alpha", False)),
            aux_focal_alpha=float(aux_focal_cfg.get("alpha", 0.25)),
            aux_focal_detach_p_for_weight=bool(aux_focal_cfg.get("detach_p_for_weight", True)),
            aux_focal_compute_fp32=bool(aux_focal_cfg.get("compute_fp32", True)),
            aux_focal_log_components=bool(aux_focal_cfg.get("log_components", True)),
            global_step=0,  # Will be updated in training loop
        )

        # Log ESMM configuration
        if self.use_esmm:
            esmm_cfg = cfg.get("esmm", {})
            self.logger.info(
                "ESMM config: version=%s lambda_ctr=%.2f lambda_ctcvr=%.2f lambda_cvr_aux=%.2f",
                esmm_cfg.get("version", "v2"),
                esmm_cfg.get("lambda_ctr", 1.0),
                esmm_cfg.get("lambda_ctcvr", 1.0),
                esmm_cfg.get("lambda_cvr_aux", 0.0),
            )
            
            # Log Aux Focal configuration if enabled
            if aux_focal_cfg.get("enabled", False):
                self.logger.info(
                    "Aux Focal config: enabled=true warmup_steps=%d target_head=%s lambda=%.3f gamma=%.1f use_alpha=%s alpha=%.3f",
                    aux_focal_cfg.get("warmup_steps", 2000),
                    aux_focal_cfg.get("target_head", "ctcvr"),
                    aux_focal_cfg.get("lambda", 0.1),
                    aux_focal_cfg.get("gamma", 1.0),
                    aux_focal_cfg.get("use_alpha", False),
                    aux_focal_cfg.get("alpha", 0.25),
                )

        self.best_metric: float = float("-inf")  # tracks best full AUC
        self.global_step = 0

        # Health metrics logging (disabled by default for backward compatibility)
        self.log_health_metrics = bool(runtime_cfg.get("log_health_metrics", False))
        if self.log_health_metrics:
            self.logger.info("Health metrics logging enabled (log_health_metrics=true)")

        resume_path = cfg.get("runtime", {}).get("resume_path")
        if resume_path:
            info = load_checkpoint(resume_path, self.model, self.optim_bundle, map_location=self.device, strict=False)
            self.best_metric = info.get("best_metric", self.best_metric)
            self.global_step = info.get("step", 0) or 0
            self.logger.info(f"Resumed from {resume_path}, step={self.global_step}, best_metric={self.best_metric}")

    def _write_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = {
            **self.metrics_meta,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": "fp16" if self.amp_enabled else "none",
            **apply_esmm_schema(metrics, self.use_esmm),
        }
        if self.amp_enabled and hasattr(self, "scaler") and self.scaler is not None:
            payload["scaler_scale"] = float(self.scaler.get_scale())
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def run(self) -> None:
        runtime = self.cfg.get("runtime", {})
        epochs = int(runtime.get("epochs", 1))
        log_every = int(runtime.get("log_every", 50))
        max_train_steps = runtime.get("max_train_steps")
        max_valid_steps = runtime.get("max_valid_steps")
        grad_clip_norm = runtime.get("grad_clip_norm")
        grad_diag_enabled = bool(runtime.get("grad_diag_enabled", False))
        eval_every_steps = log_every  # compute full AUC at the same cadence as logging
        save_last = bool(runtime.get("save_last", True))

        for epoch in range(1, epochs + 1):
            def _validate_full():
                return validate(
                    self.model,
                    self.valid_loader,
                    self.loss_fn,
                    self.device,
                    self.logger,
                    epoch=epoch,
                    max_steps=max_valid_steps,
                    log_every=max(1, log_every * 4),
                    amp_enabled=self.amp_enabled,
                    amp_dtype=self.amp_dtype,
                    amp_device_type=self.amp_device_type,
                    calc_auc=True,
                    log_health_metrics=self.log_health_metrics,
                )

            def _on_eval(val_metrics: Dict[str, Any], g_step: int, ep: int) -> None:
                record = {"epoch": ep, "split": "valid", "global_step": g_step, **val_metrics}
                self._write_metrics(record)
                full_auc = val_metrics.get("auc_primary")
                if full_auc is not None and full_auc > self.best_metric:
                    self.best_metric = full_auc
                    save_checkpoint(
                        self.run_dir / "ckpt_best.pt",
                        self.model,
                        self.optim_bundle,
                        cfg=self.cfg,
                        step=g_step,
                        best_metric=self.best_metric,
                        extra={"epoch": ep, "kind": "full"},
                    )
                    self.logger.info(f"New best at step {g_step}: auc={self.best_metric:.4f}")

            # Resolve gradient diagnostics parameters
            grad_diag_every = runtime.get("grad_diag_every")  # None means use log_every
            grad_diag_min_tasks = int(runtime.get("grad_diag_min_tasks", 2))

            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.optim_bundle,
                self.loss_fn,
                self.device,
                self.logger,
                epoch=epoch,
                max_steps=max_train_steps,
                grad_clip_norm=grad_clip_norm,
                log_every=log_every,
                global_step=self.global_step,
                grad_diag_enabled=grad_diag_enabled,
                grad_diag_every=grad_diag_every,
                grad_diag_min_tasks=grad_diag_min_tasks,
                eval_every_steps=eval_every_steps,
                validate_fn=_validate_full,
                eval_callback=_on_eval,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                scaler=self.scaler,
                amp_device_type=self.amp_device_type,
                debug_logit_every=self.debug_logit_every,
            )
            self.global_step += train_metrics.get("steps", 0)
            train_record = {"epoch": epoch, "split": "train", **train_metrics}
            self._write_metrics(train_record)

            valid_metrics = validate(
                self.model,
                self.valid_loader,
                self.loss_fn,
                self.device,
                self.logger,
                epoch=epoch,
                max_steps=max_valid_steps,
                log_every=log_every * 4,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                calc_auc=True,
                log_health_metrics=self.log_health_metrics,
            )
            valid_record = {"epoch": epoch, "split": "valid", **valid_metrics}
            self._write_metrics(valid_record)

            # checkpointing
            if save_last:
                save_checkpoint(
                    self.run_dir / "ckpt_last.pt",
                    self.model,
                    self.optim_bundle,
                    cfg=self.cfg,
                    step=self.global_step,
                    best_metric=self.best_metric,
                    extra={"epoch": epoch},
                )
            if valid_metrics:
                full_auc = valid_metrics.get("auc_primary")
                if full_auc is not None and full_auc > self.best_metric:
                    self.best_metric = full_auc
                    save_checkpoint(
                        self.run_dir / "ckpt_best.pt",
                        self.model,
                        self.optim_bundle,
                        cfg=self.cfg,
                        step=self.global_step,
                        best_metric=self.best_metric,
                        extra={"epoch": epoch, "kind": "full"},
                    )
                    self.logger.info(f"New best at epoch {epoch}: auc={self.best_metric:.4f}")

        self.logger.info("Training finished.")
        # Optional auto-eval after training completes
        runtime = self.cfg.get("runtime", {})
        if runtime.get("auto_eval", False):
            ckpt_choice = str(runtime.get("auto_eval_ckpt", "best")).lower()
            ckpt_name = "ckpt_best.pt" if ckpt_choice == "best" else "ckpt_last.pt"
            ckpt_path = self.run_dir / ckpt_name
            if not ckpt_path.exists():
                self.logger.warning("Auto-eval skipped: checkpoint not found at %s", ckpt_path)
                return
            split = runtime.get("auto_eval_split", "valid")
            save_preds = bool(runtime.get("auto_eval_save_preds", True))
            self.logger.info(
                "Auto-eval starting on split=%s using %s (save_preds=%s)", split, ckpt_name, save_preds
            )
            eval_config_path = self.config_path or (self.run_dir / "config.yaml")
            run_eval(
                cfg=self.cfg,
                split=split,
                ckpt_path=ckpt_path,
                run_dir=self.run_dir,
                save_preds=save_preds,
                max_batches=None,
                logger=self.logger,
                config_path=eval_config_path,
            )


__all__ = ["Trainer"]
