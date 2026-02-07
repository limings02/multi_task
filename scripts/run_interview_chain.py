#!/usr/bin/env python3
"""
Interview Chain Runner - 一键运行所有对比实验
====================================================

功能：
  - 按顺序执行 7 个实验（E0a/E0b/E1/E2/E3/E4/E5）
  - 实时监控进度，失败时停止并给出诊断
  - 自动汇总所有实验的 best 指标到 summary.csv
  - 支持跳过已完成的实验（断点续跑）

用法：
  python scripts/run_interview_chain.py [--dry-run] [--resume] [--skip EXPS]

选项：
  --dry-run       只打印命令，不实际执行
  --resume        跳过已有 run_dir 的实验（断点续跑）
  --skip EXPS     跳过指定实验（例如 --skip E0a,E1）
  --output DIR    输出目录（默认 runs/interview_chain）

实现细节：
  1. 定位 run_dir：
     - 从 trainer.py 代码可知：run_dir = "runs" / f"{exp_name}_{ts}"
     - 训练启动时会打印 run_dir 到 stdout（我们解析这个）
  
  2. 提取 best 指标：
     - BestSelector 会在 metrics.jsonl 中记录每次验证的决策信息
     - ckpt_best.pt 保存时的 step 对应 best 指标
     - 我们解析 metrics.jsonl，找到 split="valid" 且 global_step=best_step 的行
  
  3. 错误诊断：
     - 训练失败时，打印最后 200 行 stdout/stderr
     - 检查 run_dir/train.log 的最后 100 行
     - 给出可能的失败原因（CUDA OOM、配置错误等）

作者：资深 MTL 算法工程师（for interview）
日期：2026-02-03
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# 实验清单（顺序固定，体现渐进改进的研究逻辑）
# ============================================================================
EXPERIMENT_MANIFEST = [
    {
        "id": "E0a",
        "name": "interview_E0a_deepfm_st_ctr",
        "config": "configs/experiments/interview_chain/E0a_deepfm_st_ctr.yaml",
        "description": "单任务 CTR 基线",
        "key_vars": ["enabled_heads=[ctr]", "use_esmm=false"],
    },
    {
        "id": "E0b",
        "name": "interview_E0b_deepfm_st_cvr",
        "config": "configs/experiments/interview_chain/E0b_deepfm_st_cvr.yaml",
        "description": "单任务 CVR 基线",
        "key_vars": ["enabled_heads=[cvr]", "use_esmm=false"],
    },
    {
        "id": "E1",
        "name": "interview_E1_deepfm_shared_dualhead",
        "config": "configs/experiments/interview_chain/E1_deepfm_shared_dualhead.yaml",
        "description": "Hard Sharing（双任务，无 ESMM）",
        "key_vars": ["mtl=sharedbottom", "use_esmm=false"],
    },
    {
        "id": "E2",
        "name": "interview_E2_deepfm_shared_esmm",
        "config": "configs/experiments/interview_chain/E2_deepfm_shared_esmm.yaml",
        "description": "Hard Sharing + ESMM v2",
        "key_vars": ["mtl=sharedbottom", "use_esmm=true"],
    },
    {
        "id": "E3",
        "name": "interview_E3_deepfm_mmoe_esmm",
        "config": "configs/experiments/interview_chain/E3_deepfm_mmoe_esmm.yaml",
        "description": "MMoE + ESMM v2",
        "key_vars": ["mtl=mmoe", "num_experts=4"],
    },
    {
        "id": "E3.1",
        "name": "interview_E3_1_deepfm_mmoe_gate_stabilize_esmm",
        "config": "configs/experiments/interview_chain/E3_1_deepfm_mmoe_gate_stabilize_esmm.yaml",
        "description": "MMoE + Gate Stabilize + ESMM v2（E3 消融：验证 gate 稳定化的效果）",
        "key_vars": ["mtl=mmoe", "gate_stabilize=enabled", "num_experts=4"],
    },
    {
        "id": "E4",
        "name": "interview_E4_deepfm_ple_lite_homo_esmm",
        "config": "configs/experiments/interview_chain/E4_deepfm_ple_lite_homo_esmm.yaml",
        "description": "PLE-Lite（同构专家）+ ESMM v2",
        "key_vars": ["mtl=ple", "hetero_enabled=false"],
    },
    {
        "id": "E5",
        "name": "interview_E5_deepfm_ple_lite_hetero_esmm",
        "config": "configs/experiments/interview_chain/E5_deepfm_ple_lite_hetero_esmm.yaml",
        "description": "PLE-Lite（异构专家）+ ESMM v2（终态）",
        "key_vars": ["mtl=ple", "hetero_enabled=true", "experts=[mlp+crossnet]"],
    },
]


# ============================================================================
# 工具函数
# ============================================================================

def print_header(msg: str):
    """打印醒目的分隔线"""
    print("\n" + "=" * 80)
    print(f"  {msg}")
    print("=" * 80 + "\n")


def print_step(msg: str):
    """打印步骤信息"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def find_existing_run_dir(
    exp_name: str,
    runs_root: Path = Path("runs"),
    require_metrics: bool = True,
) -> Optional[Path]:
    """
    查找已存在的 run_dir（用于 --resume）
    
        策略：
            - 搜索 runs/{exp_name}_* 目录
            - 返回最新的（按 mtime 排序）
            - require_metrics=True 时确保目录内有 config.yaml 和 metrics.jsonl（完整性校验）
    """
    if not runs_root.exists():
        return None
    
    pattern = f"{exp_name}_*"
    candidates = sorted(runs_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    
    for run_dir in candidates:
        if run_dir.is_dir():
            # 完整性校验
            if require_metrics:
                if (run_dir / "config.yaml").exists() and (run_dir / "metrics.jsonl").exists():
                    return run_dir
            else:
                if (run_dir / "config.yaml").exists():
                    return run_dir
    
    return None


def parse_run_dir_from_stdout(stdout: str) -> Optional[Path]:
    """
    从训练 stdout/stderr 合并输出中解析 run_dir
    
    预期日志格式（来自 trainer.py）：
      - INFO: Trainer initialized. run_dir=runs/exp_name_20260203_123456 ...
      - 或者直接包含 "run_dir=..." 或 "run_dir: ..."
    
    返回：
      - Path 对象（如果找到）
      - None（如果未找到，可能因为训练启动前就失败了）
    """
    # 尝试多种模式
    patterns = [
        r"run_dir[=:]\s*([^\s]+)",                  # run_dir=... 或 run_dir: ...
        r"Trainer initialized.*run_dir[=:]\s*([^\s]+)",
        r"runs/[a-zA-Z0-9_]+_\d{8}_\d{6}",          # 直接匹配目录格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, stdout)
        if match:
            path_str = match.group(1) if match.lastindex else match.group(0)
            # 清理可能的尾部符号（逗号、括号等）
            path_str = path_str.rstrip(",.;)]}\"'")
            return Path(path_str)
    
    return None


def extract_best_metrics_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    """
    从 run_dir 提取 best 指标
    
    策略：
      1. 读取 ckpt_best.pt（torch checkpoint）获取 best_step 和 best_metric
      2. 解析 metrics.jsonl，找到 split="valid" 且 global_step=best_step 的行
      3. 提取 auc_ctr, auc_cvr, auc_ctcvr（如存在）, loss 等
    
    备用策略（如果 ckpt_best.pt 无法读取）：
      - 解析 metrics.jsonl 中所有 split="valid_decision" 行
      - 找到 should_update=true 的最后一次记录
      - 从对应的 split="valid" 行提取指标
    
    返回：
      {
        "run_dir": str,
        "best_step": int,
        "auc_ctr": float,
        "auc_cvr": float,
        "auc_cvr_masked": float,  # 如存在
        "auc_ctcvr": float,       # 如存在
        "loss": float,
        # ... 其他指标
      }
    """
    result = {"run_dir": str(run_dir)}
    
    # 尝试读取 ckpt_best.pt（需要 torch，但可能在 Windows 上有问题）
    best_step = None
    ckpt_path = run_dir / "ckpt_best.pt"
    if ckpt_path.exists():
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu")
            best_step = ckpt.get("step")
            result["best_metric"] = ckpt.get("best_metric")
        except Exception as e:
            print_step(f"Warning: 无法读取 ckpt_best.pt: {e}，使用备用策略")
    
    # 解析 metrics.jsonl
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        result["error"] = "metrics.jsonl not found"
        return result
    
    valid_records = []
    decision_records = []
    
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            split = record.get("split")
            if split == "valid":
                valid_records.append(record)
            elif split == "valid_decision":
                decision_records.append(record)
    
    # 策略 1：使用 best_step 定位（如果已知）
    if best_step is not None:
        for rec in valid_records:
            if rec.get("global_step") == best_step:
                result.update({
                    "best_step": best_step,
                    "auc_ctr": rec.get("auc_ctr"),
                    "auc_cvr": rec.get("auc_cvr"),
                    "auc_cvr_masked": rec.get("auc_cvr_masked"),
                    "auc_ctcvr": rec.get("auc_ctcvr"),
                    "auc_primary": rec.get("auc_primary"),
                    "loss": rec.get("loss"),
                    "loss_ctr": rec.get("loss_ctr"),
                    "loss_cvr": rec.get("loss_cvr"),
                    "loss_ctcvr": rec.get("loss_ctcvr"),
                })
                return result
    
    # 策略 2：使用 decision_records 找到最后一次 should_update=true
    for rec in reversed(decision_records):
        if rec.get("should_update") is True:
            best_step = rec.get("global_step")
            break
    
    if best_step is not None:
        for rec in valid_records:
            if rec.get("global_step") == best_step:
                result.update({
                    "best_step": best_step,
                    "auc_ctr": rec.get("auc_ctr"),
                    "auc_cvr": rec.get("auc_cvr"),
                    "auc_cvr_masked": rec.get("auc_cvr_masked"),
                    "auc_ctcvr": rec.get("auc_ctcvr"),
                    "auc_primary": rec.get("auc_primary"),
                    "loss": rec.get("loss"),
                    "loss_ctr": rec.get("loss_ctr"),
                    "loss_cvr": rec.get("loss_cvr"),
                    "loss_ctcvr": rec.get("loss_ctcvr"),
                })
                return result
    
    # 策略 3：如果都失败，取最后一个 valid 记录（不推荐，但总比没有强）
    if valid_records:
        rec = valid_records[-1]
        result.update({
            "best_step": rec.get("global_step"),
            "auc_ctr": rec.get("auc_ctr"),
            "auc_cvr": rec.get("auc_cvr"),
            "auc_cvr_masked": rec.get("auc_cvr_masked"),
            "auc_ctcvr": rec.get("auc_ctcvr"),
            "auc_primary": rec.get("auc_primary"),
            "loss": rec.get("loss"),
            "loss_ctr": rec.get("loss_ctr"),
            "loss_cvr": rec.get("loss_cvr"),
            "loss_ctcvr": rec.get("loss_ctcvr"),
        })
        result["warning"] = "使用最后一个 valid 记录（非 best）"
    else:
        result["error"] = "no valid records found"
    
    return result


def diagnose_failure(returncode: int, stdout: str, stderr: str, run_dir: Optional[Path]) -> str:
    """
    诊断训练失败原因
    
    返回诊断字符串（可能包含多行）
    """
    lines = []
    lines.append(f"训练失败，退出码 {returncode}")
    
    # 常见错误模式
    error_patterns = {
        r"CUDA out of memory": "CUDA OOM（显存不足）：尝试减小 batch_size 或关闭 AMP",
        r"unrecognized arguments": "CLI 参数错误：本仓库的 CLI 不支持 --key value 覆盖，只能修改 YAML",
        r"FileNotFoundError.*metadata\.json": "数据文件缺失：检查 data.metadata_path 是否正确",
        r"KeyError|AttributeError": "配置字段错误：可能缺少必需字段或字段名拼写错误",
        r"RuntimeError.*Expected.*but got": "模型维度不匹配：检查配置中的 *_dim 字段是否一致",
        r"AssertionError": "断言失败：可能是数据校验或模型不变量违反",
    }
    
    combined_output = stdout + "\n" + stderr
    for pattern, hint in error_patterns.items():
        if re.search(pattern, combined_output, re.IGNORECASE):
            lines.append(f"  ⚠️  可能原因：{hint}")
    
    # 打印最后 200 行 stdout/stderr（通常包含最关键的错误信息）
    lines.append("\n--- 最后 200 行 stdout ---")
    lines.append("\n".join(stdout.splitlines()[-200:]))

    lines.append("\n--- 最后 200 行 stderr ---")
    lines.append("\n".join(stderr.splitlines()[-200:]))
    
    # 如果 run_dir 存在，打印 train.log 最后 100 行
    if run_dir and run_dir.exists():
        train_log = run_dir / "train.log"
        if train_log.exists():
            lines.append("\n--- train.log 最后 100 行 ---")
            with train_log.open("r", encoding="utf-8") as f:
                log_lines = f.readlines()
            lines.append("".join(log_lines[-100:]))
    
    return "\n".join(lines)


# ============================================================================
# 主流程
# ============================================================================

def run_experiment(
    exp: Dict[str, Any],
    dry_run: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    运行单个实验
    
    返回：
      {
        "id": str,
        "status": "success" | "failed" | "skipped",
        "run_dir": Path | None,
        "metrics": Dict | None,  # 如果 success
        "error": str | None,     # 如果 failed
      }
    """
    exp_id = exp["id"]
    exp_name = exp["name"]
    config_path = exp["config"]
    description = exp["description"]
    
    print_header(f"{exp_id}: {description}")
    print_step(f"配置文件: {config_path}")
    print_step(f"关键变量: {', '.join(exp['key_vars'])}")
    
    # 检查配置文件是否存在
    if not Path(config_path).exists():
        return {
            "id": exp_id,
            "status": "failed",
            "run_dir": None,
            "metrics": None,
            "error": f"配置文件不存在: {config_path}",
        }
    
    # 检查是否已有 run_dir（--resume）
    if resume:
        existing_run_dir = find_existing_run_dir(exp_name)
        if existing_run_dir:
            print_step(f"✓ 发现已完成的 run_dir: {existing_run_dir}（跳过训练）")
            metrics = extract_best_metrics_from_run_dir(existing_run_dir)
            return {
                "id": exp_id,
                "status": "skipped",
                "run_dir": existing_run_dir,
                "metrics": metrics,
                "error": None,
            }
    
    # 构造训练命令
    cmd = [sys.executable, "-m", "src.cli.main", "train", "--config", config_path]
    print_step(f"执行命令: {' '.join(cmd)}")
    
    if dry_run:
        print_step("（--dry-run 模式，不实际执行）")
        return {
            "id": exp_id,
            "status": "skipped",
            "run_dir": None,
            "metrics": None,
            "error": None,
        }
    
    # 执行训练
    start_time = datetime.now()
    print_step(f"开始训练 - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=3600 * 12,  # 12 小时超时（max_train_steps=40000 预计 1-3 小时）
        )
        returncode = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {
            "id": exp_id,
            "status": "failed",
            "run_dir": None,
            "metrics": None,
            "error": "训练超时（12 小时）",
        }
    except Exception as e:
        return {
            "id": exp_id,
            "status": "failed",
            "run_dir": None,
            "metrics": None,
            "error": f"执行异常: {e}",
        }
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print_step(f"训练结束 - {end_time.strftime('%Y-%m-%d %H:%M:%S')} (耗时 {duration:.0f}s)")
    
    # 解析 run_dir
    combined_output = stdout + "\n" + stderr
    run_dir = parse_run_dir_from_stdout(combined_output)
    if run_dir is None:
        print_step("⚠️  Warning: 无法从 stdout/stderr 解析 run_dir，尝试查找最新目录")
        run_dir = find_existing_run_dir(exp_name, require_metrics=True)
        if run_dir is None:
            run_dir = find_existing_run_dir(exp_name, require_metrics=False)
    
    # 检查退出码
    if returncode != 0:
        diagnosis = diagnose_failure(returncode, stdout, stderr, run_dir)
        print_step(f"✗ 训练失败\n{diagnosis}")
        return {
            "id": exp_id,
            "status": "failed",
            "run_dir": run_dir,
            "metrics": None,
            "error": diagnosis,
        }
    
    print_step(f"✓ 训练成功 - run_dir: {run_dir}")
    
    # 提取 best 指标
    if run_dir and run_dir.exists():
        metrics = extract_best_metrics_from_run_dir(run_dir)
        auc_ctr_str = f"{metrics.get('auc_ctr'):.4f}" if metrics.get('auc_ctr') else 'N/A'
        auc_cvr_str = f"{metrics.get('auc_cvr'):.4f}" if metrics.get('auc_cvr') else 'N/A'
        auc_ctcvr_str = f"{metrics.get('auc_ctcvr'):.4f}" if metrics.get('auc_ctcvr') else 'N/A'
        print_step(f"Best 指标: step={metrics.get('best_step')}, "
                   f"auc_ctr={auc_ctr_str}, "
                   f"auc_cvr={auc_cvr_str}, "
                   f"auc_ctcvr={auc_ctcvr_str}")
    else:
        metrics = {"error": "run_dir not found"}
    
    return {
        "id": exp_id,
        "status": "success",
        "run_dir": run_dir,
        "metrics": metrics,
        "error": None,
    }


def generate_summary(results: List[Dict[str, Any]], output_dir: Path):
    """
    生成汇总报告（CSV + JSON）
    
    输出文件：
      - summary.csv：表格形式，方便复制到论文/PPT
      - summary.json：完整信息，包含 run_dir 路径等
      - delta_analysis.txt：增量分析（ΔAUC 计算）
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== summary.csv ==========
    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "exp_id",
        "description",
        "status",
        "best_step",
        "auc_ctr",
        "auc_cvr",
        "auc_cvr_masked",
        "auc_ctcvr",
        "auc_primary",
        "loss",
        "run_dir",
    ]
    
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for res in results:
            exp_id = res["id"]
            exp = next((e for e in EXPERIMENT_MANIFEST if e["id"] == exp_id), {})
            metrics = res.get("metrics") or {}
            
            row = {
                "exp_id": exp_id,
                "description": exp.get("description", ""),
                "status": res["status"],
                "best_step": metrics.get("best_step", ""),
                "auc_ctr": f"{metrics['auc_ctr']:.6f}" if metrics.get("auc_ctr") is not None else "",
                "auc_cvr": f"{metrics['auc_cvr']:.6f}" if metrics.get("auc_cvr") is not None else "",
                "auc_cvr_masked": f"{metrics['auc_cvr_masked']:.6f}" if metrics.get("auc_cvr_masked") is not None else "",
                "auc_ctcvr": f"{metrics['auc_ctcvr']:.6f}" if metrics.get("auc_ctcvr") is not None else "",
                "auc_primary": f"{metrics['auc_primary']:.6f}" if metrics.get("auc_primary") is not None else "",
                "loss": f"{metrics['loss']:.6f}" if metrics.get("loss") is not None else "",
                "run_dir": str(res.get("run_dir", "")),
            }
            writer.writerow(row)
    
    print_step(f"✓ CSV 汇总已保存: {csv_path}")
    
    # ========== summary.json ==========
    json_path = output_dir / "summary.json"
    summary_data = {
        "generated_at": datetime.now().isoformat(),
        "experiments": []
    }
    
    for res in results:
        exp_id = res["id"]
        exp = next((e for e in EXPERIMENT_MANIFEST if e["id"] == exp_id), {})
        summary_data["experiments"].append({
            "id": exp_id,
            "name": exp.get("name", ""),
            "description": exp.get("description", ""),
            "key_vars": exp.get("key_vars", []),
            "status": res["status"],
            "run_dir": str(res.get("run_dir", "")),
            "metrics": res.get("metrics", {}),
            "error": res.get("error"),
        })
    
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print_step(f"✓ JSON 汇总已保存: {json_path}")
    
    # ========== delta_analysis.txt ==========
    delta_path = output_dir / "delta_analysis.txt"
    lines = []
    lines.append("=" * 80)
    lines.append("增量分析（ΔAUC 计算）")
    lines.append("=" * 80)
    lines.append("")
    lines.append("说明：")
    lines.append("  - 单任务基线（E0a/E0b）：测量各任务的天花板")
    lines.append("  - Hard Sharing（E1）：测量多任务负迁移")
    lines.append("  - ESMM（E2）：测量 ESMM 收益")
    lines.append("  - MMoE（E3）：测量软参数共享收益")
    lines.append("  - PLE（E4/E5）：测量专家分工收益")
    lines.append("  - 异构专家（E5 vs E4）：测量专家多样性收益")
    lines.append("")
    
    # 构建 metrics 查询表
    metrics_map = {}
    for res in results:
        if res["status"] in ["success", "skipped"] and res.get("metrics"):
            metrics_map[res["id"]] = res["metrics"]
    
    # 定义对比对
    comparisons = [
        ("E1 vs E0a (CTR)", "E1", "E0a", "auc_ctr", "Hard Sharing 相比单任务 CTR"),
        ("E1 vs E0b (CVR)", "E1", "E0b", "auc_cvr", "Hard Sharing 相比单任务 CVR"),
        ("E2 vs E1 (CTCVR)", "E2", "E1", "auc_ctcvr", "ESMM 收益（CTCVR）"),
        ("E3 vs E2 (CTCVR)", "E3", "E2", "auc_ctcvr", "MMoE 相比 Hard Sharing（ESMM 下）"),
        ("E4 vs E3 (CTCVR)", "E4", "E3", "auc_ctcvr", "PLE-Lite 相比 MMoE（同构专家）"),
        ("E5 vs E4 (CTCVR)", "E5", "E4", "auc_ctcvr", "异构专家收益（PLE-Lite）"),
    ]
    
    lines.append("对比结果：")
    lines.append("")
    
    for label, exp_new, exp_base, metric_key, desc in comparisons:
        if exp_new not in metrics_map or exp_base not in metrics_map:
            lines.append(f"  {label}: 数据缺失")
            continue
        
        val_new = metrics_map[exp_new].get(metric_key)
        val_base = metrics_map[exp_base].get(metric_key)
        
        if val_new is None or val_base is None:
            lines.append(f"  {label}: 指标缺失")
            continue
        
        delta = val_new - val_base
        delta_pct = (delta / val_base) * 100 if val_base != 0 else 0
        
        lines.append(f"  {label}:")
        lines.append(f"    {desc}")
        lines.append(f"    {exp_base}: {metric_key}={val_base:.6f}")
        lines.append(f"    {exp_new}: {metric_key}={val_new:.6f}")
        lines.append(f"    Δ{metric_key} = {delta:+.6f} ({delta_pct:+.2f}%)")
        lines.append("")
    
    with delta_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print_step(f"✓ 增量分析已保存: {delta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Interview Chain Runner - 一键运行所有对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 完整运行（首次）
  python scripts/run_interview_chain.py
  
  # 断点续跑（跳过已完成的实验）
  python scripts/run_interview_chain.py --resume
  
  # 跳过前两个单任务基线（加速调试）
  python scripts/run_interview_chain.py --skip E0a,E0b
  
  # 只打印命令（调试用）
  python scripts/run_interview_chain.py --dry-run
        """
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际执行")
    parser.add_argument("--resume", action="store_true", help="跳过已有 run_dir 的实验（断点续跑）")
    parser.add_argument("--skip", type=str, default="", help="跳过指定实验（逗号分隔，例如 E0a,E1）")
    parser.add_argument("--output", type=str, default="runs/interview_chain", help="输出目录")
    
    args = parser.parse_args()
    
    skip_set = set(args.skip.split(",")) if args.skip else set()
    output_dir = Path(args.output)
    
    print_header("Interview Chain Runner - 主线 7 个实验")
    print_step(f"输出目录: {output_dir}")
    print_step(f"模式: {'Dry-run' if args.dry_run else 'Resume' if args.resume else 'Full'}")
    if skip_set:
        print_step(f"跳过实验: {', '.join(skip_set)}")
    
    # 执行所有实验
    results = []
    for exp in EXPERIMENT_MANIFEST:
        if exp["id"] in skip_set:
            print_header(f"{exp['id']}: {exp['description']} [SKIPPED by --skip]")
            results.append({
                "id": exp["id"],
                "status": "skipped",
                "run_dir": None,
                "metrics": None,
                "error": "skipped by user",
            })
            continue
        
        result = run_experiment(exp, dry_run=args.dry_run, resume=args.resume)
        results.append(result)
        
        # 如果失败且非 dry-run，询问是否继续
        if result["status"] == "failed" and not args.dry_run:
            print_step("\n❌ 实验失败！")
            user_input = input("是否继续下一个实验？(y/n): ").strip().lower()
            if user_input != "y":
                print_step("用户中止")
                break
    
    # 生成汇总报告
    if not args.dry_run:
        print_header("生成汇总报告")
        generate_summary(results, output_dir)
    
    # 打印最终统计
    print_header("最终统计")
    success_count = sum(1 for r in results if r["status"] in ["success", "skipped"])
    failed_count = sum(1 for r in results if r["status"] == "failed")
    print_step(f"成功/跳过: {success_count}/{len(results)}")
    print_step(f"失败: {failed_count}/{len(results)}")
    
    if failed_count > 0:
        print_step("\n失败的实验：")
        for r in results:
            if r["status"] == "failed":
                print_step(f"  - {r['id']}: {r.get('error', 'unknown error')[:100]}")
    
    print_header("完成")
    print_step(f"查看汇总: {output_dir / 'summary.csv'}")
    print_step(f"增量分析: {output_dir / 'delta_analysis.txt'}")
    
    # 退出码
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
