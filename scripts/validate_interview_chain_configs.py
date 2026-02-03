#!/usr/bin/env python3
"""
配置文件一致性校验脚本
=======================

功能：
  - 检查 interview_chain 中 7 个配置文件的公共字段是否一致
  - 输出差异报告（如有）
  - 作为 CI/CD 的一部分确保实验可比性

用法：
  python scripts/validate_interview_chain_configs.py

作者：资深 MTL 算法工程师
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install via: pip install pyyaml")
    sys.exit(1)


# ============================================================================
# 公共字段定义（所有实验必须一致）
# ============================================================================
COMMON_KEYS = [
    # Data
    "data.batch_size",
    "data.seed",
    "data.neg_keep_prob_train",
    "sampling.negative_sampling",
    
    # Embedding
    "embedding.default_embedding_dim",
    "embedding.sparse_grad",
    
    # Model Backbone（部分字段）
    "model.backbone.deep_hidden_dims",
    "model.backbone.deep_dropout",
    "model.backbone.fm_projection_dim",
    "model.deep_hidden_dims",
    "model.deep_dropout",
    "model.fm_projection_dim",
    
    # Optimizer
    "optim.type",
    "optim.dense.lr",
    "optim.sparse.lr",
    "optim.lr_scheduler.total_steps",
    
    # Runtime
    "runtime.max_train_steps",
    "runtime.seed",
    "runtime.amp",
]


def get_nested_value(d: Dict[str, Any], key: str) -> Any:
    """
    获取嵌套字典的值
    
    例如：get_nested_value(cfg, "data.batch_size") 等价于 cfg["data"]["batch_size"]
    """
    keys = key.split(".")
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, {})
        else:
            return None
    return d


def load_configs(config_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    加载所有实验配置文件
    
    返回：{exp_id: config_dict}
    """
    configs = {}
    for config_file in sorted(config_dir.glob("E*.yaml")):
        exp_id = config_file.stem.split("_")[0]  # E0a, E0b, E1, ...
        with config_file.open("r", encoding="utf-8") as f:
            configs[exp_id] = yaml.safe_load(f)
    return configs


def validate_consistency(configs: Dict[str, Dict[str, Any]], common_keys: List[str]) -> bool:
    """
    校验公共字段一致性
    
    返回：True（一致），False（不一致）
    """
    all_consistent = True
    
    print("=" * 80)
    print("配置文件一致性校验")
    print("=" * 80)
    print(f"检查 {len(configs)} 个配置文件：{', '.join(configs.keys())}")
    print(f"检查 {len(common_keys)} 个公共字段")
    print()
    
    for key in common_keys:
        values = {exp_id: get_nested_value(cfg, key) for exp_id, cfg in configs.items()}
        
        # 将值转为字符串以便比较（处理 list/dict 等复杂类型）
        str_values = {exp_id: str(val) for exp_id, val in values.items()}
        unique_values = set(str_values.values())
        
        if len(unique_values) == 1:
            # 一致
            value_repr = unique_values.pop()
            if value_repr == "None" or value_repr == "{}":
                print(f"⚠️  {key}: {value_repr} (可能缺失)")
            else:
                print(f"✓  {key}: {value_repr}")
        else:
            # 不一致
            all_consistent = False
            print(f"✗  {key}: 不一致！")
            for exp_id, val in values.items():
                print(f"     {exp_id}: {val}")
    
    print()
    if all_consistent:
        print("✅ 所有公共字段一致！")
    else:
        print("❌ 存在不一致字段，请修复后重新运行")
    
    return all_consistent


def main():
    config_dir = Path("configs/experiments/interview_chain")
    
    if not config_dir.exists():
        print(f"Error: 配置目录不存在: {config_dir}")
        sys.exit(1)
    
    # 加载配置
    configs = load_configs(config_dir)
    
    if not configs:
        print(f"Error: 未找到任何配置文件（E*.yaml）在 {config_dir}")
        sys.exit(1)
    
    # 校验一致性
    is_consistent = validate_consistency(configs, COMMON_KEYS)
    
    # 退出码
    sys.exit(0 if is_consistent else 1)


if __name__ == "__main__":
    main()
