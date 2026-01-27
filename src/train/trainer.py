from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json

import numpy as np
import torch
from torch import optim
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


class Trainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
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
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def run(self) -> None:
        runtime = self.cfg.get("runtime", {})
        epochs = int(runtime.get("epochs", 1))
        log_every = int(runtime.get("log_every", 50))
        max_train_steps = runtime.get("max_train_steps")
        max_valid_steps = runtime.get("max_valid_steps")
        grad_clip_norm = runtime.get("grad_clip_norm")

        for epoch in range(1, epochs + 1):
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
            run_eval(
                cfg=self.cfg,
                split=split,
                ckpt_path=ckpt_path,
                run_dir=self.run_dir,
                save_preds=save_preds,
                max_batches=None,
                logger=self.logger,
            )


__all__ = ["Trainer"]
