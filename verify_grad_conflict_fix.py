#!/usr/bin/env python3
"""
验证梯度冲突诊断指标修复的测试脚本

使用方法：
    python verify_grad_conflict_fix.py <path_to_metrics.jsonl>

示例：
    python verify_grad_conflict_fix.py runs/deepfm_ple_lite_dual_sparse_20260203_123206/metrics.jsonl
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    """加载 metrics.jsonl 文件"""
    metrics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def analyze_conflict_consistency(metrics: List[Dict[str, Any]]) -> None:
    """分析 conflict_rate 与 cosine 分位数的一致性"""
    print("\n=== 冲突率与 Cosine 分位数一致性分析 ===\n")
    
    train_metrics = [m for m in metrics if m.get("split") == "train" and m.get("conflict_rate") is not None]
    
    if not train_metrics:
        print("❌ 未找到 split=train 且有 conflict_rate 的记录")
        return
    
    for idx, m in enumerate(train_metrics, 1):
        epoch = m.get("epoch")
        conflict_rate = m.get("conflict_rate", 0.0)
        
        cos_dense_p10 = m.get("grad_cosine_dense_p10")
        cos_sparse_p10 = m.get("grad_cosine_sparse_p10")
        cos_all_p10 = m.get("grad_cosine_p10")
        
        cos_sparse_mean = m.get("grad_cosine_shared_sparse_mean")
        shared_sparse_count = m.get("shared_sparse_count", 0)
        grad_samples = m.get("grad_samples", 0)
        
        print(f"Epoch {epoch}:")
        print(f"  conflict_rate (all):       {conflict_rate:.4f}")
        print(f"  grad_samples:              {grad_samples}")
        print(f"  shared_sparse_count:       {shared_sparse_count}")
        print(f"  cosine_p10:")
        print(f"    - dense:  {cos_dense_p10:.4f}" if cos_dense_p10 is not None else "    - dense:  None")
        print(f"    - sparse: {cos_sparse_p10:.4f}" if cos_sparse_p10 is not None else "    - sparse: None")
        print(f"    - all:    {cos_all_p10:.4f}" if cos_all_p10 is not None else "    - all:    None")
        print(f"  cosine_sparse_mean:        {cos_sparse_mean:.4f}" if cos_sparse_mean is not None else "  cosine_sparse_mean:        None")
        
        # 检查一致性
        inconsistency = []
        if cos_sparse_p10 is not None and cos_sparse_p10 < 0 and conflict_rate == 0.0:
            inconsistency.append("⚠️  sparse_p10 < 0 但 conflict_rate = 0")
        if cos_all_p10 is not None and cos_all_p10 < 0 and conflict_rate == 0.0:
            inconsistency.append("⚠️  all_p10 < 0 但 conflict_rate = 0")
        
        if inconsistency:
            print("\n  🔴 不一致问题：")
            for issue in inconsistency:
                print(f"     {issue}")
        else:
            print("\n  ✅ 一致性检查通过")
        
        print()


def analyze_grad_norm_coverage(metrics: List[Dict[str, Any]]) -> None:
    """分析 grad_norm 覆盖情况"""
    print("\n=== Grad Norm 覆盖分析 ===\n")
    
    train_metrics = [m for m in metrics if m.get("split") == "train" and m.get("grad_samples", 0) > 0]
    
    if not train_metrics:
        print("❌ 未找到 split=train 且有 grad_samples 的记录")
        return
    
    for idx, m in enumerate(train_metrics, 1):
        epoch = m.get("epoch")
        mode = m.get("mode", "unknown")
        
        norm_ctr = m.get("grad_norm_shared_ctr_mean")
        norm_cvr = m.get("grad_norm_shared_cvr_mean")
        norm_ctcvr = m.get("grad_norm_shared_ctcvr_mean")
        
        print(f"Epoch {epoch} (mode={mode}):")
        print(f"  grad_norm_shared_ctr_mean:    {norm_ctr:.4f}" if norm_ctr is not None else "  grad_norm_shared_ctr_mean:    None")
        print(f"  grad_norm_shared_cvr_mean:    {norm_cvr:.4f}" if norm_cvr is not None else "  grad_norm_shared_cvr_mean:    None")
        print(f"  grad_norm_shared_ctcvr_mean:  {norm_ctcvr:.4f}" if norm_ctcvr is not None else "  grad_norm_shared_ctcvr_mean:  None")
        
        # 检查 ESMM 模式的预期行为
        if mode == "esmm":
            if norm_ctr is not None and norm_ctcvr is not None:
                print("  ✅ ESMM 模式：ctr 和 ctcvr norm 均有值")
            else:
                print("  ⚠️  ESMM 模式：预期 ctr 和 ctcvr 都有值")
            
            if norm_cvr is None:
                print("  ✅ ESMM 模式：cvr norm 为 None（符合预期，因为不训练独立 cvr）")
            else:
                print("  ⚠️  ESMM 模式：cvr norm 不应有值")
        else:
            if norm_ctr is not None and norm_cvr is not None:
                print("  ✅ 非 ESMM 模式：ctr 和 cvr norm 均有值")
            else:
                print("  ⚠️  非 ESMM 模式：预期 ctr 和 cvr 都有值")
        
        print()


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_grad_conflict_fix.py <path_to_metrics.jsonl>")
        print("\n示例:")
        print("  python verify_grad_conflict_fix.py runs/deepfm_ple_lite_dual_sparse_20260203_123206/metrics.jsonl")
        sys.exit(1)
    
    metrics_path = Path(sys.argv[1])
    
    if not metrics_path.exists():
        print(f"❌ 文件不存在: {metrics_path}")
        sys.exit(1)
    
    print(f"📊 加载 metrics 文件: {metrics_path}")
    metrics = load_metrics(metrics_path)
    print(f"✅ 加载了 {len(metrics)} 条记录")
    
    # 分析 1: 冲突率一致性
    analyze_conflict_consistency(metrics)
    
    # 分析 2: grad_norm 覆盖
    analyze_grad_norm_coverage(metrics)
    
    print("\n=== 总结 ===\n")
    print("修复验证完成！")
    print("\n如果你看到新训练运行的日志中有：")
    print("  [grad_conflict_diagnosis] epoch=X samples=Y | conflict_rate: dense=... sparse=... all=...")
    print("则说明修复已生效。")
    print("\n预期行为：")
    print("  1. 当 cosine_sparse_p10 < 0 时，日志中的 conflict_rate_sparse > 0")
    print("  2. ESMM 模式下，grad_norm_shared_ctr_mean 和 grad_norm_shared_ctcvr_mean 有值")
    print("  3. ESMM 模式下，grad_norm_shared_cvr_mean 为 None（符合预期）")
    print()


if __name__ == "__main__":
    main()
