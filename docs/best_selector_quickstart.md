# Gate-based Best Model Selection - Quick Start

## 快速开始

### 使用 Gate 策略（新功能）

在你的配置文件中添加 `runtime.best_selection` 配置：

```yaml
runtime:
  best_selection:
    strategy: "gate"
    primary_key: "auc_ctcvr"
    aux_keys: ["auc_ctr", "auc_cvr"]
    tol_primary: 0.002
    tol_aux:
      auc_ctr: 0.003
      auc_cvr: 0.008
    use_primary_ma: true
    ma_window: 5
    confirm_times: 2
    cooldown_evals: 1
    log_decision: true
```

### 保持传统行为（向后兼容）

不配置 `best_selection`，或显式设置：

```yaml
runtime:
  best_selection:
    strategy: "auc_primary"
```

## Gate 策略说明

**更新 best checkpoint 的条件**：
1. ✅ 主目标（auc_ctcvr）必须提升 >= `tol_primary`
2. ✅ 辅助任务（auc_ctr, auc_cvr）不能回撤超过 `tol_aux`
3. ✅ 连续 `confirm_times` 次满足上述条件
4. ✅ 不在冷却期内（更新后 `cooldown_evals` 次 eval 跳过）

**可选增强**：
- `use_primary_ma=true`：主目标使用移动平均，减少尖峰误选
- `confirm_times>1`：需连续确认，增加鲁棒性
- `cooldown_evals>0`：防止抖动反复切换 best

## 日志输出

训练时会看到类似日志：

```
[BestSelector] GATE FAILED: primary auc_ctcvr=0.7010 (delta=0.001000 < tol=0.002000)
[BestSelector] GATE FAILED: aux metrics degraded - auc_ctr=0.6400 (delta=-0.010000 < -0.003000)
[BestSelector] GATE PASSED but awaiting confirmation: 1/2
[BestSelector] NEW BEST: auc_ctcvr=0.7030 (delta=+0.003000), aux=[auc_ctr=0.6510, auc_cvr=0.5510] at step 5000
✓ Saved best checkpoint at step 5000
```

每次 eval 的决策也会记录到 `metrics.jsonl`（`split="valid_decision"`）。

## 完整示例

参考 [configs/experiments/mmoe_gate_example.yaml](../configs/experiments/mmoe_gate_example.yaml)

## 单元测试

```bash
python tests/test_best_selector.py
```

## 详细文档

见 [docs/best_selector_implementation.md](best_selector_implementation.md)

## 参数建议

| 参数 | 保守 | 平衡 | 激进 |
|------|------|------|------|
| `tol_primary` | 0.003 | 0.002 | 0.001 |
| `tol_aux.auc_ctr` | 0.002 | 0.003 | 0.005 |
| `tol_aux.auc_cvr` | 0.005 | 0.008 | 0.010 |
| `confirm_times` | 3 | 2 | 1 |
| `cooldown_evals` | 2 | 1 | 0 |
| `use_primary_ma` | true | true | false |

- **保守**：减少误更新，但可能错过一些好的 checkpoint
- **激进**：响应快，但可能选到不稳定的 checkpoint
- **平衡**：推荐起点

根据你的任务波动特性调整！
