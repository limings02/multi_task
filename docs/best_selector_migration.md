# 从 auc_primary 迁移到 Gate 策略 - 迁移指南

## 为什么要迁移？

**旧方式（auc_primary）的问题**：
- ❌ 单一标量无法反映多任务平衡
- ❌ 可能选到主目标好但辅助任务崩溃的 checkpoint
- ❌ 对指标尖峰敏感，易误选不稳定的 checkpoint

**Gate 策略的优势**：
- ✅ 主目标必须提升，辅助任务不能崩溃
- ✅ 支持移动平均，过滤尖峰
- ✅ 连续确认和冷却期，增加鲁棒性
- ✅ 详细日志，可解释为什么选/不选

---

## 零风险迁移步骤

### 第 1 步：不改配置，先运行一次（验证向后兼容）

```bash
# 使用现有配置运行训练
python -m src.cli.main train --config configs/experiments/your_existing_config.yaml
```

**预期结果**：
- 训练正常运行，日志中出现 `BestSelector initialized: strategy=auc_primary`
- Best checkpoint 选择逻辑与之前完全一致

如果这步失败，说明集成有问题，需要先修复。

---

### 第 2 步：添加 gate 配置，做对比实验

复制你的配置文件，添加 `best_selection` 配置：

```yaml
# your_config_gate.yaml
runtime:
  # ... 其他配置保持不变
  
  best_selection:
    strategy: "gate"
    primary_key: "auc_ctcvr"
    aux_keys: ["auc_ctr", "auc_cvr"]
    
    # 先用宽松的容忍，避免永不更新
    tol_primary: 0.001   # 主目标提升 0.1%
    tol_aux:
      auc_ctr: 0.005     # CTR 允许回撤 0.5%
      auc_cvr: 0.010     # CVR 允许回撤 1.0%
    
    # 先不用 MA/confirm/cooldown，逐步加强
    use_primary_ma: false
    confirm_times: 1
    cooldown_evals: 0
    log_decision: true
```

运行训练：
```bash
python -m src.cli.main train --config configs/experiments/your_config_gate.yaml
```

---

### 第 3 步：分析日志，调整参数

观察训练日志中的 `[BestSelector]` 输出：

**场景 A：Best 更新频率合理（每隔几次 eval 更新一次）**
```
[BestSelector] NEW BEST: auc_ctcvr=0.7030 (delta=+0.003000), aux=[...] at step 5000
[BestSelector] GATE FAILED: primary improvement insufficient: 0.000800 < 0.001000
[BestSelector] NEW BEST: auc_ctcvr=0.7055 (delta=+0.002500), aux=[...] at step 10000
```
✅ 参数合适，可以进入第 4 步

**场景 B：Best 永远不更新**
```
[BestSelector] GATE FAILED: primary improvement insufficient: 0.000800 < 0.001000
[BestSelector] GATE FAILED: aux metrics degraded - auc_ctr=0.6480 (delta=-0.002000 < -0.005000)
```
❌ `tol_primary` 或 `tol_aux` 过严，需要放宽：
```yaml
tol_primary: 0.0005  # 减半
tol_aux:
  auc_ctr: 0.010     # 翻倍
  auc_cvr: 0.015     # 翻倍
```

**场景 C：Best 更新过于频繁（几乎每次 eval 都更新）**
```
[BestSelector] NEW BEST: ... at step 5000
[BestSelector] NEW BEST: ... at step 5500
[BestSelector] NEW BEST: ... at step 6000
```
❌ 容忍过宽或有噪声尖峰，需要收紧并启用 MA：
```yaml
tol_primary: 0.002      # 提高容忍
use_primary_ma: true    # 启用移动平均
ma_window: 5            # 5 次 eval 的均值
```

---

### 第 4 步：逐步加强鲁棒性（可选）

#### 启用移动平均（减少尖峰干扰）
```yaml
use_primary_ma: true
ma_window: 5  # 或 3/7，根据 eval 频率调整
```

**适用场景**：训练曲线有明显尖峰，希望选更平滑的 checkpoint

#### 启用连续确认（避免偶然达标）
```yaml
confirm_times: 2  # 连续 2 次通过 gate 才更新
```

**适用场景**：指标波动大，希望确认趋势稳定

#### 启用冷却期（防止抖动反复切换）
```yaml
cooldown_evals: 1  # 更新后跳过 1 次 eval
```

**适用场景**：更新 best 后短期内指标可能回撤，避免立即切回旧 checkpoint

---

### 第 5 步：对比最终 best checkpoint

训练结束后，对比两个配置的 best checkpoint：

```bash
# auc_primary 策略的 best
python -m src.cli.main eval --config runs/run_auc_primary/config.yaml \
    --ckpt runs/run_auc_primary/ckpt_best.pt --split test

# gate 策略的 best
python -m src.cli.main eval --config runs/run_gate/config.yaml \
    --ckpt runs/run_gate/ckpt_best.pt --split test
```

**对比指标**：
- auc_ctcvr（主目标）
- auc_ctr / auc_cvr（辅助任务）
- 各指标的方差（稳定性）

如果 gate 策略选出的 checkpoint 各方面都更好或相当，迁移成功！

---

## 回滚方案

如果 gate 策略效果不佳，随时可以回退：

**方案 1：改回 auc_primary 策略**
```yaml
runtime:
  best_selection:
    strategy: "auc_primary"
```

**方案 2：完全移除 best_selection 配置**
```yaml
runtime:
  # 删除或注释掉 best_selection
  # best_selection: ...
```

两种方案都会恢复到原始行为，无需改代码。

---

## 常见场景的参数建议

### 场景 1：ESMM 任务（CTCVR 为主，CTR/CVR 为辅）
```yaml
best_selection:
  strategy: "gate"
  primary_key: "auc_ctcvr"
  aux_keys: ["auc_ctr", "auc_cvr"]
  tol_primary: 0.002
  tol_aux:
    auc_ctr: 0.003
    auc_cvr: 0.008  # CVR 波动通常更大
  use_primary_ma: true
  ma_window: 5
  confirm_times: 2
  cooldown_evals: 1
```

### 场景 2：单任务（只关心一个指标）
```yaml
best_selection:
  strategy: "auc_primary"  # 单任务用传统策略即可
```

或者：
```yaml
best_selection:
  strategy: "gate"
  primary_key: "auc_ctr"
  aux_keys: []  # 无辅助任务
  tol_primary: 0.001
  use_primary_ma: true
```

### 场景 3：指标波动大，需要非常保守
```yaml
best_selection:
  strategy: "gate"
  primary_key: "auc_ctcvr"
  aux_keys: ["auc_ctr", "auc_cvr"]
  tol_primary: 0.003       # 高门槛
  tol_aux:
    auc_ctr: 0.002         # 严格控制回撤
    auc_cvr: 0.005
  use_primary_ma: true
  ma_window: 7             # 更大的窗口
  confirm_times: 3         # 需要 3 次确认
  cooldown_evals: 2        # 更长的冷却期
```

---

## 检查点：你应该看到的日志

### ✅ 正确的日志示例

```
INFO | BestSelector initialized: strategy=gate
INFO | [BestSelector] GATE FAILED: primary auc_ctcvr=0.7010 (delta=0.001000 < tol=0.002000)
INFO | [BestSelector] GATE FAILED: aux metrics degraded - auc_ctr=0.6400 (delta=-0.010000 < -0.003000)
INFO | [BestSelector] GATE PASSED but awaiting confirmation: 1/2
INFO | [BestSelector] NEW BEST: auc_ctcvr=0.7030 (delta=+0.003000), aux=[auc_ctr=0.6510, auc_cvr=0.5510] at step 5000
INFO | ✓ Saved best checkpoint at step 5000
INFO | [BestSelector] COOLDOWN: 1 evals remaining, skipping update
```

### ❌ 需要调整的日志示例

**问题：永不更新**
```
INFO | [BestSelector] GATE FAILED: primary improvement insufficient: ...
INFO | [BestSelector] GATE FAILED: primary improvement insufficient: ...
```
→ 降低 `tol_primary`

**问题：更新过于频繁**
```
INFO | [BestSelector] NEW BEST: ... at step 5000
INFO | [BestSelector] NEW BEST: ... at step 5500
INFO | [BestSelector] NEW BEST: ... at step 6000
```
→ 提高 `tol_primary`，启用 `confirm_times` 或 `cooldown_evals`

---

## 分析决策记录（高级）

每次 eval 的决策都记录在 `metrics.jsonl` 中：

```python
import json
import pandas as pd

# 读取所有决策记录
with open("runs/YOUR_RUN/metrics.jsonl") as f:
    records = [json.loads(line) for line in f]

decisions = [r for r in records if r.get("split") == "valid_decision"]
df = pd.DataFrame(decisions)

# 统计 gate 失败原因
print(df["reason"].value_counts())

# 可视化主目标变化
import matplotlib.pyplot as plt
df["primary_value"].plot()
plt.axhline(y=df[df["should_update"]]["primary_value"].max(), color='r', linestyle='--', label='best')
plt.legend()
plt.show()
```

---

## FAQ

**Q: 迁移会影响已有的实验结果吗？**  
A: 不会。旧的 checkpoint 和 metrics 完全不受影响。

**Q: 迁移后还能用旧的 checkpoint resume 吗？**  
A: 可以。旧 checkpoint 缺失 `best_selector` 状态时会自动初始化。

**Q: 能同时用 auc_primary 和 gate 策略做 A/B 测试吗？**  
A: 可以！只需配置两个不同的 config 文件，分别指定 `strategy`。

**Q: Gate 策略适合所有任务吗？**  
A: 不一定。单任务或辅助任务不重要时，auc_primary 可能更简单有效。

**Q: 参数调优有自动化工具吗？**  
A: 目前需要手动调整。未来可以加入自适应容忍或贝叶斯优化。

---

## 总结

| 步骤 | 操作 | 预期耗时 |
|------|------|----------|
| 1. 验证向后兼容 | 运行现有配置 | 5 分钟 |
| 2. 添加 gate 配置 | 复制配置并添加 best_selection | 10 分钟 |
| 3. 调整参数 | 观察日志，调整 tol | 30 分钟 |
| 4. 加强鲁棒性 | 启用 MA/confirm/cooldown | 10 分钟 |
| 5. 对比评估 | eval 两个 best checkpoint | 15 分钟 |

总计约 **1 小时**即可完成迁移评估。

**建议迁移路径**：
1. 先在小规模实验上试用 gate 策略
2. 调优参数找到合适配置
3. 在大规模实验上正式采用
4. 定期对比 auc_primary 和 gate 策略的效果

祝迁移顺利！
