import torch
import torch.nn.functional as F

from pathlib import Path

from src.data.dataloader import make_dataloader
from src.models.build import build_model
from src.utils.config import load_yaml
from src.utils.feature_meta import build_model_feature_meta


def _bce_with_mask(pred, target, mask):
    # mask-aware loss used in user request
    return (F.binary_cross_entropy_with_logits(pred, target, reduction="none") * mask).sum() / (mask.sum() + 1e-6)


def test_model_forward_shared_bottom():
    cfg = load_yaml("configs/models/backbone_deepfm.yaml")

    # build feature_meta for dataloader & model
    feature_meta = build_model_feature_meta(Path(cfg["data"]["metadata_path"]), cfg["embedding"])

    loader = make_dataloader(
        split="train",
        batch_size=8,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        seed=42,
        feature_meta=feature_meta,
        debug=False,
    )

    model = build_model(cfg)
    model.train()

    labels, features, meta = next(iter(loader))
    out = model(features)

    assert "ctr" in out and "cvr" in out
    B = labels["y_ctr"].shape[0]
    assert out["ctr"].shape == (B,)
    assert out["cvr"].shape == (B,)
    assert torch.isfinite(out["ctr"]).all()
    assert torch.isfinite(out["cvr"]).all()

    loss_ctr = F.binary_cross_entropy_with_logits(out["ctr"], labels["y_ctr"])
    loss_cvr = _bce_with_mask(out["cvr"], labels["y_cvr"], labels["click_mask"])
    loss = loss_ctr + loss_cvr
    loss.backward()  # graph should be valid


def test_falsy_head_cfg_not_overridden():
    cfg = load_yaml("configs/models/backbone_deepfm.yaml")

    # inject head overrides to test falsy handling
    cfg.setdefault("model", {})
    cfg["model"]["name"] = "deepfm_shared_bottom"
    cfg["model"]["heads"] = {
        "tasks": ["ctr", "cvr"],
        "default": {"mlp_dims": [64], "dropout": 0.5, "use_bn": True},
        "ctr": {"mlp_dims": [], "dropout": 0.0, "use_bn": False},
    }

    feature_meta = build_model_feature_meta(Path(cfg["data"]["metadata_path"]), cfg["embedding"])
    loader = make_dataloader(
        split="train",
        batch_size=4,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        seed=7,
        feature_meta=feature_meta,
        debug=False,
    )

    model = build_model(cfg)
    shared = model  # SharedBottom

    assert shared._head_cfg_resolved["ctr"]["mlp_dims"] == []
    assert shared._head_cfg_resolved["ctr"]["dropout"] == 0.0
    assert shared._head_cfg_resolved["ctr"]["use_bn"] is False

    labels, features, meta = next(iter(loader))
    out = model(features)
    assert out["ctr"].shape[0] == labels["y_ctr"].shape[0]
