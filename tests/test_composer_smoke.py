"""
Smoke test: 验证 MMoEInputComposer 在不同配置下的行为。

测试三组配置：
  1) deep_h only（旧配置兼容）
  2) deep_h + sum_emb
  3) deep_h + sum_emb + fm_vec

验证内容：
  - forward 能正常跑通
  - 输出维度符合预期
"""

import sys
from pathlib import Path

# 确保可以 import src
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from src.models.mtl.composer import MMoEInputComposer, build_composer_from_config


def make_dummy_backbone_output(batch_size: int = 4) -> dict:
    """
    模拟 DeepFMBackbone.forward(return_aux=True) 的输出。
    
    假设：
      - deep_h: [B, 128]
      - fm_vec: [B, 16]  (假设 embedding_dim=16)
      - sum_emb: [B, 16]
      - mean_emb: [B, 16]
      - emb_concat: [B, 80]  (假设 5 个 field * 16 dim)
    """
    B = batch_size
    return {
        "h": torch.randn(B, 128),
        "deep_h": torch.randn(B, 128),
        "logit_linear": torch.randn(B, 1),
        "fm_vec": torch.randn(B, 16),
        "sum_emb": torch.randn(B, 16),
        "mean_emb": torch.randn(B, 16),
        "emb_concat": torch.randn(B, 80),
    }


def test_config_1_deep_h_only():
    """
    配置 1: 只使用 deep_h（旧配置兼容）。
    
    当 add_fm_vec=False, add_emb="none", part_proj_dim=None 时，
    build_composer_from_config 返回 None（走 fast-path）。
    """
    print("\n" + "=" * 60)
    print("测试配置 1: deep_h only (旧配置兼容)")
    print("=" * 60)
    
    mmoe_cfg = {
        "input": "deep_h",
        "add_fm_vec": False,
        "add_emb": "none",
        # 不指定 part_proj_dim, norm, dropout 等
    }
    
    # 模拟 backbone（只需要提供维度信息）
    class DummyBackbone:
        deep_out_dim = 128
        fm_dim = 16
        fm_enabled = True
        class FeatEmb:
            concat_dim = 80
            embedding_dims = {"f0": 16, "f1": 16, "f2": 16, "f3": 16, "f4": 16}
        feat_emb = FeatEmb()
    
    backbone = DummyBackbone()
    composer = build_composer_from_config(backbone, mmoe_cfg)
    
    if composer is None:
        print("✓ composer 为 None（fast-path：直接使用 deep_h）")
        print("  预期输入维度: 128 (deep_h)")
    else:
        aux = make_dummy_backbone_output()
        out = composer(aux)
        print(f"✓ forward 成功，输出 shape: {out.shape}")
        print(f"  预期输入维度: {composer.out_dim}")


def test_config_2_deep_h_plus_sum_emb():
    """
    配置 2: deep_h + sum_emb。
    
    add_emb="sum"，需要投影并拼接。
    """
    print("\n" + "=" * 60)
    print("测试配置 2: deep_h + sum_emb")
    print("=" * 60)
    
    mmoe_cfg = {
        "input": "deep_h",
        "add_fm_vec": False,
        "add_emb": "sum",
        "part_proj_dim": 64,  # 每个 part 投影到 64 维
        "fusion": "concat",
        "dropout": 0.1,
        "norm": "layernorm",
    }
    
    class DummyBackbone:
        deep_out_dim = 128
        fm_dim = 16
        fm_enabled = True
        class FeatEmb:
            concat_dim = 80
            embedding_dims = {"f0": 16, "f1": 16, "f2": 16, "f3": 16, "f4": 16}
        feat_emb = FeatEmb()
    
    backbone = DummyBackbone()
    composer = build_composer_from_config(backbone, mmoe_cfg)
    
    assert composer is not None, "composer 应该被创建"
    
    aux = make_dummy_backbone_output()
    out = composer(aux)
    
    # 预期维度：deep_h (64) + emb (64) = 128
    expected_dim = 64 + 64
    assert out.shape == (4, expected_dim), f"输出维度不符：预期 {expected_dim}，实际 {out.shape[1]}"
    
    print(f"✓ forward 成功，输出 shape: {out.shape}")
    print(f"  parts: {composer.part_names}")
    print(f"  预期输入维度: {expected_dim} (deep_h[64] + sum_emb[64])")
    print(f"  实际 out_dim: {composer.out_dim}")


def test_config_3_deep_h_plus_sum_emb_plus_fm_vec():
    """
    配置 3: deep_h + sum_emb + fm_vec。
    
    三路输入全部启用。
    """
    print("\n" + "=" * 60)
    print("测试配置 3: deep_h + sum_emb + fm_vec")
    print("=" * 60)
    
    mmoe_cfg = {
        "input": "deep_h",
        "add_fm_vec": True,
        "add_emb": "sum",
        "part_proj_dim": 64,
        "fusion": "concat",
        "adapter_mlp_dims": [128],  # 拼接后过一层 MLP
        "dropout": 0.1,
        "norm": "layernorm",
    }
    
    class DummyBackbone:
        deep_out_dim = 128
        fm_dim = 16
        fm_enabled = True
        class FeatEmb:
            concat_dim = 80
            embedding_dims = {"f0": 16, "f1": 16, "f2": 16, "f3": 16, "f4": 16}
        feat_emb = FeatEmb()
    
    backbone = DummyBackbone()
    composer = build_composer_from_config(backbone, mmoe_cfg)
    
    assert composer is not None, "composer 应该被创建"
    
    aux = make_dummy_backbone_output()
    out = composer(aux)
    
    # 预期维度：
    # - concat 后: deep_h (64) + fm_vec (64) + emb (64) = 192
    # - adapter MLP 后: 128
    expected_dim = 128
    assert out.shape == (4, expected_dim), f"输出维度不符：预期 {expected_dim}，实际 {out.shape[1]}"
    
    print(f"✓ forward 成功，输出 shape: {out.shape}")
    print(f"  parts: {composer.part_names}")
    print(f"  预期输入维度: {expected_dim} (经 adapter MLP 变换后)")
    print(f"  实际 out_dim: {composer.out_dim}")
    print(f"  配置摘要: {composer.get_config_summary()}")


def test_fusion_sum():
    """
    测试 fusion="sum" 模式。
    """
    print("\n" + "=" * 60)
    print("测试 fusion='sum' 模式")
    print("=" * 60)
    
    mmoe_cfg = {
        "add_fm_vec": True,
        "add_emb": "sum",
        "part_proj_dim": 64,  # 所有 part 投影到 64 维
        "fusion": "sum",      # sum 融合
        "dropout": 0.0,
        "norm": "none",
    }
    
    class DummyBackbone:
        deep_out_dim = 128
        fm_dim = 16
        fm_enabled = True
        class FeatEmb:
            concat_dim = 80
            embedding_dims = {"f0": 16, "f1": 16, "f2": 16, "f3": 16, "f4": 16}
        feat_emb = FeatEmb()
    
    backbone = DummyBackbone()
    composer = build_composer_from_config(backbone, mmoe_cfg)
    
    assert composer is not None
    
    aux = make_dummy_backbone_output()
    out = composer(aux)
    
    # sum 融合后维度 = part_proj_dim = 64
    expected_dim = 64
    assert out.shape == (4, expected_dim), f"输出维度不符：预期 {expected_dim}，实际 {out.shape[1]}"
    
    print(f"✓ forward 成功，输出 shape: {out.shape}")
    print(f"  fusion='sum'，维度为 part_proj_dim={expected_dim}")


def test_backward_compatibility():
    """
    测试向后兼容：旧配置（只有 input:"deep_h"）应该正常工作。
    """
    print("\n" + "=" * 60)
    print("测试向后兼容：旧配置")
    print("=" * 60)
    
    # 模拟旧配置：只有 input, num_experts 等基本字段
    mmoe_cfg = {
        "input": "deep_h",
        "num_experts": 4,
        "expert_mlp_dims": [128],
        "gate_type": "linear",
        "dropout": 0.1,
    }
    
    class DummyBackbone:
        deep_out_dim = 128
        fm_dim = 16
        fm_enabled = True
        class FeatEmb:
            concat_dim = 80
            embedding_dims = {"f0": 16, "f1": 16, "f2": 16, "f3": 16, "f4": 16}
        feat_emb = FeatEmb()
    
    backbone = DummyBackbone()
    composer = build_composer_from_config(backbone, mmoe_cfg)
    
    # 旧配置没有 add_fm_vec/add_emb，但有 dropout，所以可能会创建 composer
    # 或者返回 None（取决于 build_composer_from_config 的逻辑）
    if composer is None:
        print("✓ 旧配置走 fast-path（composer=None）")
    else:
        aux = make_dummy_backbone_output()
        out = composer(aux)
        print(f"✓ 旧配置也能跑通，输出 shape: {out.shape}")


def test_emb_concat_warning():
    """
    测试 add_emb="concat" 时的显存警告。
    """
    print("\n" + "=" * 60)
    print("测试 add_emb='concat' (显存警告)")
    print("=" * 60)
    
    mmoe_cfg = {
        "add_fm_vec": False,
        "add_emb": "concat",
        "part_proj_dim": 64,
        "fusion": "concat",
    }
    
    class DummyBackbone:
        deep_out_dim = 128
        fm_dim = 16
        fm_enabled = True
        class FeatEmb:
            concat_dim = 80
            embedding_dims = {"f0": 16, "f1": 16, "f2": 16, "f3": 16, "f4": 16}
        feat_emb = FeatEmb()
    
    backbone = DummyBackbone()
    
    # 应该会打印警告
    print("(预期会看到显存警告...)")
    composer = build_composer_from_config(backbone, mmoe_cfg)
    
    assert composer is not None
    aux = make_dummy_backbone_output()
    out = composer(aux)
    
    print(f"✓ forward 成功，输出 shape: {out.shape}")


def main():
    print("=" * 60)
    print("MMoEInputComposer Smoke Test")
    print("=" * 60)
    
    test_config_1_deep_h_only()
    test_config_2_deep_h_plus_sum_emb()
    test_config_3_deep_h_plus_sum_emb_plus_fm_vec()
    test_fusion_sum()
    test_backward_compatibility()
    test_emb_concat_warning()
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
    
    # 维度总结
    print("\n维度变化总结：")
    print("-" * 60)
    print("| 配置                          | 输入维度              |")
    print("|-------------------------------|----------------------|")
    print("| deep_h only                   | 128                  |")
    print("| deep_h + sum_emb (proj=64)    | 64 + 64 = 128        |")
    print("| deep_h + sum_emb + fm_vec     | 64 + 64 + 64 = 192   |")
    print("|   + adapter_mlp_dims=[128]    | -> 128 (MLP 输出)    |")
    print("| fusion='sum' (proj=64)        | 64                   |")
    print("-" * 60)


if __name__ == "__main__":
    main()
