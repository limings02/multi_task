"""
集成测试：验证完整 MMoE 模型构建（新旧配置）。
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from src.models.build import build_model


def test_old_config():
    """测试旧配置（向后兼容）"""
    print("=" * 60)
    print("测试旧配置（向后兼容）")
    print("=" * 60)
    
    cfg = {
        'data': {'metadata_path': 'data/processed/metadata.json'},
        'embedding': {
            'default_embedding_dim': 8,
            'sparse_grad': False,
        },
        'model': {
            'name': 'deepfm_mmoe',
            'mtl': 'mmoe',
            'deep_hidden_dims': [64],
            'deep_dropout': 0.1,
            'deep_activation': 'relu',
            'fm_enabled': True,
            'fm_projection_dim': 8,
            'out_dim': 64,
            'mmoe': {
                'input': 'deep_h',
                'num_experts': 2,
                'expert_mlp_dims': [64],
                'gate_type': 'linear',
                'dropout': 0.1,
            },
            'heads': {'tasks': ['ctr', 'cvr']},
        },
    }
    
    model = build_model(cfg)
    print(f"✓ Model built: {type(model).__name__}")
    print(f"  Composer: {model.composer}")
    print(f"  in_dim: {model.in_dim}")
    return model


def test_new_config_with_composer():
    """测试新配置（使用 composer）"""
    print("\n" + "=" * 60)
    print("测试新配置（使用 composer）")
    print("=" * 60)
    
    cfg = {
        'data': {'metadata_path': 'data/processed/metadata.json'},
        'embedding': {
            'default_embedding_dim': 8,
            'sparse_grad': False,
        },
        'model': {
            'name': 'deepfm_mmoe',
            'mtl': 'mmoe',
            'deep_hidden_dims': [64],
            'deep_dropout': 0.1,
            'deep_activation': 'relu',
            'fm_enabled': True,
            'fm_projection_dim': 8,
            'out_dim': 64,
            'mmoe': {
                'input': 'deep_h',
                'num_experts': 2,
                'expert_mlp_dims': [],
                'gate_type': 'linear',
                'add_fm_vec': True,
                'add_emb': 'sum',
                'part_proj_dim': 32,
                'fusion': 'concat',
                'dropout': 0.1,
                'norm': 'layernorm',
            },
            'heads': {'tasks': ['ctr', 'cvr']},
        },
    }
    
    model = build_model(cfg)
    print(f"✓ Model built: {type(model).__name__}")
    print(f"  Composer: {model.composer}")
    if model.composer:
        print(f"  Composer parts: {model.composer.part_names}")
        print(f"  Composer out_dim: {model.composer.out_dim}")
    print(f"  in_dim: {model.in_dim}")
    return model


def main():
    test_old_config()
    test_new_config_with_composer()
    print("\n" + "=" * 60)
    print("SUCCESS: 新旧配置都能正常工作！")
    print("=" * 60)


if __name__ == "__main__":
    main()
