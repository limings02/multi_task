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
    cfg = load_yaml("configs/model/backbone_deepfm.yaml")

    # Add minimal data/runtime overrides for quick test
    cfg.setdefault("data", {})
    cfg["data"].setdefault("metadata_path", "data/processed/metadata.json")
    cfg["data"].setdefault("batch_size", 8)
    cfg["data"].setdefault("num_workers", 0)
    cfg["data"].setdefault("pin_memory", False)
    cfg["data"].setdefault("persistent_workers", False)
    cfg["data"].setdefault("drop_last", False)
    cfg["data"].setdefault("debug", False)

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
    cfg = load_yaml("configs/model/backbone_deepfm.yaml")

    # inject head overrides to test falsy handling
    cfg.setdefault("model", {})
    cfg["model"]["name"] = "deepfm_shared_bottom"
    cfg["model"]["heads"] = {
        "tasks": ["ctr", "cvr"],
        "default": {"mlp_dims": [64], "dropout": 0.5, "use_bn": True},
        "ctr": {"mlp_dims": [], "dropout": 0.0, "use_bn": False},
    }
    cfg.setdefault("data", {})
    cfg["data"].setdefault("metadata_path", "data/processed/metadata.json")
    cfg["data"].setdefault("batch_size", 4)
    cfg["data"].setdefault("num_workers", 0)
    cfg["data"].setdefault("pin_memory", False)
    cfg["data"].setdefault("persistent_workers", False)

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


def test_model_forward_mmoe():
    cfg = load_yaml("configs/experiments/mtl_mmoe.yaml")

    # speed up for unit test
    cfg["data"]["batch_size"] = 8
    cfg["data"]["num_workers"] = 0
    cfg["runtime"]["device"] = "cpu"

    feature_meta = build_model_feature_meta(Path(cfg["data"]["metadata_path"]), cfg["embedding"])

    loader = make_dataloader(
        split="train",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
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
    loss.backward()


def test_model_forward_ple():
    """
    PLE-Lite 模型 forward 测试用例。
    
    断言：
    1. 输出包含 ctr 和 cvr
    2. 输出 shape 正确
    3. 如果 gate_stabilize.enabled=True，aux 中 gate_reg_loss 有限（无 NaN/Inf）
    4. 如果 log_gates=True，aux["gates"] 中各任务的 gate 权重 shape 正确
    5. 梯度图有效，可以反传
    """
    cfg = load_yaml("configs/model/mtl_ple.yaml")

    # 加速单测
    cfg["data"]["batch_size"] = 8
    cfg["data"]["num_workers"] = 0
    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = "cpu"

    feature_meta = build_model_feature_meta(Path(cfg["data"]["metadata_path"]), cfg["embedding"])

    loader = make_dataloader(
        split="train",
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
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

    # ===== 基本输出断言 =====
    assert "ctr" in out and "cvr" in out, "PLE must output ctr and cvr"
    B = labels["y_ctr"].shape[0]
    assert out["ctr"].shape == (B,), f"ctr shape mismatch: {out['ctr'].shape} vs ({B},)"
    assert out["cvr"].shape == (B,), f"cvr shape mismatch: {out['cvr'].shape} vs ({B},)"
    assert torch.isfinite(out["ctr"]).all(), "ctr has NaN/Inf"
    assert torch.isfinite(out["cvr"]).all(), "cvr has NaN/Inf"

    # ===== gate_reg_loss 断言（gate_stabilize.enabled=True 时）=====
    aux = out.get("aux", {})
    if "gate_reg_loss" in aux:
        gate_reg_loss = aux["gate_reg_loss"]
        assert torch.isfinite(gate_reg_loss), f"gate_reg_loss has NaN/Inf: {gate_reg_loss}"

    # ===== gate 权重 shape 断言（log_gates=True 时）=====
    # PLE 配置：shared_num_experts=4, specific_num_experts={ctr:1, cvr:1}
    # 所以 gate 维度 = 4+1 = 5
    if "gates" in aux:
        gates = aux["gates"]
        expected_gate_dim = 4 + 1  # shared + specific
        if "ctr" in gates:
            assert gates["ctr"].shape[-1] == expected_gate_dim, \
                f"ctr gate dim mismatch: {gates['ctr'].shape[-1]} vs {expected_gate_dim}"
        if "cvr" in gates:
            assert gates["cvr"].shape[-1] == expected_gate_dim, \
                f"cvr gate dim mismatch: {gates['cvr'].shape[-1]} vs {expected_gate_dim}"

    # ===== 梯度反传断言 =====
    loss_ctr = F.binary_cross_entropy_with_logits(out["ctr"], labels["y_ctr"])
    loss_cvr = _bce_with_mask(out["cvr"], labels["y_cvr"], labels["click_mask"])
    loss = loss_ctr + loss_cvr
    
    # 如果有 gate_reg_loss，也加入总 loss（验证梯度链路完整）
    if "gate_reg_loss" in aux:
        loss = loss + aux["gate_reg_loss"]
    
    loss.backward()  # 梯度图应该有效
