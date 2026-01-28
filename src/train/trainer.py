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
from torch import optim, amp as torch_amp
import yaml

from src.core.checkpoint import load_checkpoint, save_checkpoint
from src.data.dataloader import make_dataloader
from src.loss.bce import MultiTaskBCELoss
from src.models.build import build_model
from src.eval.run_eval import run_eval
from src.train.loops import train_one_epoch, validate
from src.utils.feature_meta import build_model_feature_meta


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

        tags = get_exp_tags(cfg)
        self.enabled_heads = tags["enabled_heads"]
        self.model_name = tags["model_name"]
        self.exp_name = tags["exp_name"]
        self.seed_value = tags["seed"]
        self.metrics_meta = {**tags, "config_path": str(self.config_path) if self.config_path else None}
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

        # feature_meta for dataloader/model
        metadata_path = Path(cfg["data"]["metadata_path"])
        embedding_cfg = cfg.get("embedding", {})
        self.feature_meta = build_model_feature_meta(metadata_path, embedding_cfg)

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
            debug=bool(cfg["data"].get("debug", False)),
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
            debug=bool(cfg["data"].get("debug", False)),
        )

        self.model = build_model(cfg).to(self.device)

        opt_cfg = cfg.get("optim", {})
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(opt_cfg.get("lr", 1e-3)),
            weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )

        loss_cfg = cfg.get("loss", {})
        self.loss_fn = MultiTaskBCELoss(
            w_ctr=float(loss_cfg.get("w_ctr", 1.0)),
            w_cvr=float(loss_cfg.get("w_cvr", 1.0)),
            eps=float(loss_cfg.get("eps", 1e-6)),
            enabled_heads=self.enabled_heads,
        )

        self.best_metric: float = float("inf")
        self.global_step = 0

        resume_path = cfg.get("runtime", {}).get("resume_path")
        if resume_path:
            info = load_checkpoint(resume_path, self.model, self.optimizer, map_location=self.device, strict=False)
            self.best_metric = info.get("best_metric", self.best_metric)
            self.global_step = info.get("step", 0) or 0
            self.logger.info(f"Resumed from {resume_path}, step={self.global_step}, best_metric={self.best_metric}")

    def _write_metrics(self, metrics: Dict[str, Any]) -> None:
        payload = {
            **self.metrics_meta,
            "amp_enabled": self.amp_enabled,
            "amp_dtype": "fp16" if self.amp_enabled else "none",
            **metrics,
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
        eval_every_steps = log_every  # user request: validate every log_every steps

        for epoch in range(1, epochs + 1):
            def _validate_once():
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
                )

            def _on_eval(val_metrics: Dict[str, Any], g_step: int, ep: int) -> None:
                record = {"epoch": ep, "split": "valid", "global_step": g_step, **val_metrics}
                self._write_metrics(record)
                if val_metrics and val_metrics.get("loss_total") is not None and val_metrics["loss_total"] < self.best_metric:
                    self.best_metric = val_metrics["loss_total"]
                    save_checkpoint(
                        self.run_dir / "ckpt_best.pt",
                        self.model,
                        self.optimizer,
                        cfg=self.cfg,
                        step=g_step,
                        best_metric=self.best_metric,
                        extra={"epoch": ep},
                    )
                    self.logger.info(f"New best at step {g_step}: loss_total={self.best_metric:.4f}")

            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.loss_fn,
                self.device,
                self.logger,
                epoch=epoch,
                max_steps=max_train_steps,
                grad_clip_norm=grad_clip_norm,
                log_every=log_every,
                global_step=self.global_step,
                grad_diag_enabled=grad_diag_enabled,
                eval_every_steps=eval_every_steps,
                validate_fn=_validate_once,
                eval_callback=_on_eval,
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
                scaler=self.scaler,
                amp_device_type=self.amp_device_type,
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
            )
            valid_record = {"epoch": epoch, "split": "valid", **valid_metrics}
            self._write_metrics(valid_record)

            # checkpointing
            save_checkpoint(
                self.run_dir / "ckpt_last.pt",
                self.model,
                self.optimizer,
                cfg=self.cfg,
                step=self.global_step,
                best_metric=self.best_metric,
                extra={"epoch": epoch},
            )
            if valid_metrics and valid_metrics["loss_total"] < self.best_metric:
                self.best_metric = valid_metrics["loss_total"]
                save_checkpoint(
                    self.run_dir / "ckpt_best.pt",
                    self.model,
                    self.optimizer,
                    cfg=self.cfg,
                    step=self.global_step,
                    best_metric=self.best_metric,
                    extra={"epoch": epoch},
                )
                self.logger.info(f"New best at epoch {epoch}: loss_total={self.best_metric:.4f}")

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
