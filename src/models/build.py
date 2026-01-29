from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List

import torch.nn as nn

from src.models.backbones.deepfm import DeepFMBackbone
from src.models.mtl.shared_bottom import SharedBottom
from src.utils.config import load_yaml

try:
    from src.utils.feature_meta import build_model_feature_meta
except ImportError:  # pragma: no cover - fallback if module missing
    build_model_feature_meta = None


def _resolve_feature_meta(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build merged model_feature_meta using metadata.json + embedding config.
    """
    data_cfg = cfg.get("data", {})
    embedding_cfg = cfg.get("embedding", {})
    metadata_path = Path(data_cfg["metadata_path"])

    if build_model_feature_meta is None:
        raise ImportError("build_model_feature_meta not available; please ensure src.utils.feature_meta exists.")
    return build_model_feature_meta(metadata_path, embedding_cfg)


def _load_label_priors(cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Read train split positive rates from metadata.json for bias init.
    Falls back to empty dict if missing.
    """
    try:
        data_cfg = cfg.get("data", {})
        metadata_path = Path(data_cfg["metadata_path"])
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        train_stats = meta.get("split_stats", {}).get("train", {})
        priors = {}
        if "ctr" in train_stats:
            priors["ctr"] = float(train_stats["ctr"])
        if "cvr" in train_stats:
            priors["cvr"] = float(train_stats["cvr"])
        return priors
    except Exception:
        return {}


def _build_backbone(cfg: Dict[str, Any], feature_meta: Dict[str, Any]) -> nn.Module:
    model_cfg = cfg.get("model", {})
    # backward-compatible: model/backbone nesting
    backbone_cfg = model_cfg.get("backbone", model_cfg)

    use_legacy = bool(backbone_cfg.get("use_legacy_pseudo_deepfm", True))
    return_parts = bool(backbone_cfg.get("return_logit_parts", False))

    return DeepFMBackbone(
        feature_meta=feature_meta,
        deep_hidden_dims=list(backbone_cfg.get("deep_hidden_dims", [])),
        deep_dropout=float(backbone_cfg.get("deep_dropout", 0.0)),
        deep_activation=str(backbone_cfg.get("deep_activation", "relu")),
        deep_use_bn=bool(backbone_cfg.get("deep_use_bn", False)),
        fm_enabled=bool(backbone_cfg.get("fm_enabled", True)),
        fm_projection_dim=(
            None
            if backbone_cfg.get("fm_projection_dim") is None
            else int(backbone_cfg.get("fm_projection_dim"))
        ),
        out_dim=int(backbone_cfg.get("out_dim", 128)),
        use_legacy_pseudo_deepfm=use_legacy,
        return_logit_parts=return_parts,
    )


def build_model(cfg: Dict[str, Any], feature_map: Dict[str, Any] | None = None, meta: Dict[str, Any] | None = None) -> nn.Module:
    """
    Assemble model according to cfg. Currently supports only DeepFM + SharedBottom.
    """
    model_cfg = cfg.get("model", {})
    enabled_heads = model_cfg.get("enabled_heads") or ["ctr", "cvr"]
    name = model_cfg.get("name") or "deepfm_shared_bottom"

    feature_meta = _resolve_feature_meta(cfg)
    backbone = _build_backbone(cfg, feature_meta)
    label_priors = _load_label_priors(cfg)

    head_cfg = model_cfg.get("heads", {})
    head_cfg.setdefault("tasks", model_cfg.get("tasks", ["ctr", "cvr"]))
    head_cfg.setdefault("default", {})
    head_cfg["default"].setdefault("mlp_dims", model_cfg.get("tower_hidden_dims", []))
    head_cfg["default"].setdefault("dropout", model_cfg.get("head_dropout", 0.0))
    head_cfg["default"].setdefault("use_bn", model_cfg.get("head_use_bn", False))
    head_cfg["default"].setdefault("activation", model_cfg.get("head_activation", model_cfg.get("deep_activation", "relu")))

    per_head_add = model_cfg.get("backbone", {}).get("per_head_add") or model_cfg.get("per_head_add") or {}
    use_legacy = bool(model_cfg.get("backbone", {}).get("use_legacy_pseudo_deepfm", model_cfg.get("use_legacy_pseudo_deepfm", True)))
    return_parts = bool(model_cfg.get("backbone", {}).get("return_logit_parts", model_cfg.get("return_logit_parts", False)))

    return SharedBottom(
        backbone=backbone,
        head_cfg=head_cfg,
        enabled_heads=enabled_heads,
        use_legacy_pseudo_deepfm=use_legacy,
        return_logit_parts=return_parts,
        per_head_add=per_head_add,
        head_priors=label_priors,
    )


__all__ = ["build_model"]
