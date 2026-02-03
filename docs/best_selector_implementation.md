# Gate-based Best Model Selection - Implementation Summary

## 概述

本改动实现了可配置的 best checkpoint 选择策略，从原先的简单加权 `auc_primary` 改为支持 **Gate 规则**（主升 + 次不崩 + 可选移动平均/确认/冷却）。

## 改动文件列表

### 新增文件

1. **`src/train/best_selector.py`** (新增 470 行)
   - 封装了 `BestSelector` 类，支持两种策略：
     - `auc_primary`：传统单一标量比较（向后兼容）
     - `gate`：主目标提升 + 辅助任务不崩溃的复合判定
   - 支持功能：
     - 主目标移动平均（Moving Average）
     - 连续确认（Confirmation）
     - 冷却期（Cooldown）
     - 详细决策日志

2. **`configs/experiments/mmoe_gate_example.yaml`** (新增 213 行)
   - Gate 策略的完整配置示例
   - 包含详细注释和参数说明

3. **`tests/test_best_selector.py`** (新增 285 行)
   - 完整的单元测试，覆盖所有 gate 场景
   - 可直接运行验证：`python tests/test_best_selector.py`

### 修改文件

4. **`src/train/trainer.py`** (修改 3 处)
   - 导入 `BestSelector`
   - 在 `__init__` 中初始化 `best_selector`，支持从 checkpoint resume
   - 修改 `_on_eval` 使用 `best_selector.should_update_best()` 判定
   - 保存 `best_selector` 状态到 checkpoint

## 核心设计

### BestSelector 类

```python
class BestSelector:
    def __init__(
        self,
        strategy: str = "auc_primary",        # "auc_primary" | "gate"
        primary_key: str = "auc_ctcvr",       # 主目标 key
        aux_keys: List[str] = ["auc_ctr", "auc_cvr"],  # 辅助目标
        use_primary_ma: bool = False,         # 主目标使用移动平均
        ma_window: int = 5,                   # MA 窗口大小
        tol_primary: float = 0.0,             # 主目标最小提升容忍
        tol_aux: Dict[str, float] = {},       # 辅助目标最大回撤容忍
        confirm_times: int = 1,               # 连续确认次数
        cooldown_evals: int = 0,              # 冷却期 eval 次数
        log_decision: bool = True,            # 记录决策日志
        logger: Optional[logging.Logger] = None,
    )
    
    def should_update_best(
        self, metrics: Dict[str, Any], step: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """返回 (should_update, decision_info)"""
```

### Gate 策略判定流程

```
1. 检查冷却期 (cooldown_remaining > 0)
   └─> 如果在冷却中：return False, reason="cooldown_active"

2. 提取主目标值 (primary_key)
   └─> 如果缺失：return False, reason="primary_key missing"

3. 计算主目标比较值 (raw 或 MA)
   └─> 如果 use_primary_ma=True：primary_value = mean(last W values)

4. 检查主目标提升 (primary_delta >= tol_primary)
   └─> 如果不足：return False, reason="primary insufficient"

5. 检查辅助目标 (aux_delta >= -tol_aux)
   └─> 如果任一回撤过大：return False, reason="aux degraded"

6. Gate 通过，检查连续确认 (confirm_count >= confirm_times)
   └─> 如果确认不足：return False, reason="confirmation pending"

7. 所有检查通过
   └─> 更新 best 值，设置冷却期，return True
```

## 配置示例

### 方式 1：使用 Gate 策略（推荐新实验）

```yaml
runtime:
  best_selection:
    strategy: "gate"
    primary_key: "auc_ctcvr"
    aux_keys: ["auc_ctr", "auc_cvr"]
    use_primary_ma: true
    ma_window: 5
    tol_primary: 0.002       # 主目标至少提升 0.2%
    tol_aux:
      auc_ctr: 0.003         # CTR 允许回撤 0.3%
      auc_cvr: 0.008         # CVR 允许回撤 0.8%
    confirm_times: 2         # 需连续 2 次通过 gate
    cooldown_evals: 1        # 更新后跳过 1 次 eval
    log_decision: true
```

### 方式 2：传统策略（默认，向后兼容）

```yaml
runtime:
  # 不配置 best_selection，或显式指定 auc_primary
  best_selection:
    strategy: "auc_primary"
    log_decision: true
```

**注意**：如果完全不配置 `best_selection`，默认使用 `auc_primary` 策略，行为与原版完全一致。

## 日志输出示例

### Gate 策略日志

```
[BestSelector] GATE FAILED: primary auc_ctcvr=0.7010 (delta=0.001000 < tol=0.002000)
[BestSelector] GATE FAILED: aux metrics degraded - auc_ctr=0.6400 (delta=-0.010000 < -0.003000)
[BestSelector] GATE PASSED but awaiting confirmation: 1/2
[BestSelector] NEW BEST: auc_ctcvr=0.7030 (delta=+0.003000), aux=[auc_ctr=0.6510, auc_cvr=0.5510] at step 123
✓ Saved best checkpoint at step 123
[BestSelector] COOLDOWN: 1 evals remaining, skipping update
```

### metrics.jsonl 中的决策记录

每次 eval 会写入一条 `split="valid_decision"` 记录：

```json
{
  "epoch": 1,
  "split": "valid_decision",
  "global_step": 5000,
  "strategy": "gate",
  "should_update": false,
  "reason": "primary improvement insufficient: 0.001500 < 0.002000",
  "primary_value": 0.7015,
  "best_primary": 0.7000,
  "primary_delta": 0.0015,
  "ok_primary": false,
  "aux_status": {
    "auc_ctr": {"value": 0.65, "best": 0.65, "delta": 0.0, "ok": true},
    "auc_cvr": {"value": 0.55, "best": 0.55, "delta": 0.0, "ok": true}
  }
}
```

## 验证方法

### 1. 单元测试

```bash
cd e:\my_project\multi_task
python tests/test_best_selector.py
```

应看到所有测试通过：
```
✓ Test 1: Legacy auc_primary strategy
✓ Test 2: Gate strategy - primary insufficient
✓ Test 3: Gate strategy - aux degraded
✓ Test 4: Gate strategy - gate passes
✓ Test 5: Confirmation requirement
✓ Test 6: Cooldown period
✓ Test 7: Moving average
✓✓✓ ALL TESTS PASSED ✓✓✓
```

### 2. 实际训练验证

使用示例配置运行训练：

```bash
python -m src.cli.main train --config configs/experiments/mmoe_gate_example.yaml
```

观察日志中的 `[BestSelector]` 输出，确认：
- Gate 决策逻辑正确触发
- 日志清晰解释为何某次 eval 没有更新 best
- `ckpt_best.pt` 按 gate 规则保存

### 3. 检查 metrics.jsonl

```python
import json
import pandas as pd

# 读取决策记录
decisions = []
with open("runs/YOUR_RUN/metrics.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("split") == "valid_decision":
            decisions.append(rec)

df = pd.DataFrame(decisions)
print(df[["global_step", "should_update", "reason", "primary_delta"]])
```

## 向后兼容性

1. **默认行为不变**：不配置 `best_selection` 时，使用 `auc_primary` 策略
2. **Checkpoint 格式兼容**：`best_selector` 状态保存在 `extra` 字段，不影响旧代码加载
3. **Metrics 不变**：仍计算并记录 `auc_primary`，只是判定逻辑改用 gate（当 strategy="gate" 时）

## 设计特点（满足需求）

✅ **最小侵入**：只新增 1 个文件（best_selector.py），修改 trainer.py 3 处  
✅ **无新依赖**：只使用标准库（collections.deque, logging）  
✅ **完全可配置**：所有参数通过 YAML 控制  
✅ **向后兼容**：默认 auc_primary 策略，已有实验不受影响  
✅ **可解释日志**：每次决策详细记录原因，便于调试  
✅ **支持 resume**：best_selector 状态持久化到 checkpoint  
✅ **测试覆盖**：7 个场景的单元测试，验证所有逻辑分支

## 参数调优建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `tol_primary` | 0.001 - 0.003 | 主目标最小提升，太小易误更新，太大难触发 |
| `tol_aux.auc_ctr` | 0.002 - 0.005 | CTR 允许回撤，根据任务波动调整 |
| `tol_aux.auc_cvr` | 0.005 - 0.010 | CVR 通常波动更大，容忍可更宽松 |
| `confirm_times` | 1 - 3 | 1=不确认，2=保守，3=非常保守 |
| `cooldown_evals` | 0 - 2 | 0=不冷却，1-2=防止抖动 |
| `use_primary_ma` | true/false | true=平滑尖峰，false=响应快 |
| `ma_window` | 3 - 7 | 窗口越大越平滑，但滞后也大 |

## 故障排查

**Q: 日志中看不到 [BestSelector] 输出？**  
A: 检查 `runtime.best_selection.log_decision: true`，并确认 logger 级别

**Q: Gate 策略永远不更新 best？**  
A: 检查 `tol_primary` 和 `tol_aux` 是否设置过严，尝试放宽容忍

**Q: 想临时禁用 gate，用回 auc_primary？**  
A: 配置中改为 `strategy: "auc_primary"` 即可，无需改代码

**Q: Resume 后 best_selector 状态正确吗？**  
A: 检查 checkpoint 中 `extra.best_selector` 是否存在，trainer 初始化时会自动加载

## 未来扩展可能

- 支持更多策略（如 Pareto 前沿、加权混合等）
- 自适应容忍（根据历史波动自动调整 tol）
- 多目标优化框架（NSGA-II 等）
- Web UI 可视化决策过程

---

**实现完成日期**：2026-02-03  
**实现者**：GitHub Copilot (Claude Sonnet 4.5)
