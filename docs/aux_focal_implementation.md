# 方案1: ESMM 主链路 BCE + CTCVR Aux-Focal（配置化 + warmup）

## 概述

本实现为 ESMM v2 的 CTCVR 损失增加了可配置的辅助 Focal Loss，用于缓解极端类别不平衡问题。

### 核心特性

1. **主损失不变**：CTR 和 CTCVR 的主损失仍使用 BCEWithLogitsLoss + static pos_weight
2. **辅助 Focal**：仅对 CTCVR 增加辅助 Focal Loss，CTR 不受影响
3. **Warmup 机制**：前 N 步不启用 Focal，避免训练初期梯度不稳定
4. **配置化开关**：通过 YAML 配置可完全禁用，disabled 时行为与原实现完全一致
5. **AMP 兼容**：数值稳定，支持混合精度训练
6. **可观测性**：可选记录 loss_ctcvr_bce 和 loss_ctcvr_focal 组件

## 实现细节

### 损失函数

```
L_ctr = BCEWithLogits(logit_ctr, y_ctr, pos_weight=24.7)

L_ctcvr_bce = BCEWithLogits(logit_ctcvr, y_ctcvr, pos_weight=4800)
L_ctcvr_focal = FocalLoss(logit_ctcvr, y_ctcvr, gamma, alpha)  # 辅助项

L_ctcvr = L_ctcvr_bce + lambda * L_ctcvr_focal  (仅当 enabled=true 且 step >= warmup_steps)

L_total = lambda_ctr * L_ctr + lambda_ctcvr * L_ctcvr + lambda_cvr_aux * L_cvr_aux
```

### Focal Loss 公式

```
focal_factor = (1 - p_t)^gamma
p_t = p * y + (1-p) * (1-y)  # 真实类别的预测概率

loss = focal_factor * BCE(logits, y)
```

**关键特性**：
- Easy negatives (y=0, p≈0): p_t ≈ 1 → focal_factor ≈ 0 → 强烈降权
- Hard examples: focal_factor ≈ 1 → 保持完整权重
- gamma 越大，easy samples 降权越强

### 配置示例

```yaml
loss:
  static_pos_weight:
    ctr: 24.7
    ctcvr: 4800
  
  aux_focal:
    enabled: true              # 开关
    warmup_steps: 2000         # 前 2000 step 不启用
    target_head: "ctcvr"       # 固定为 ctcvr
    lambda: 0.1                # focal 系数
    gamma: 1.0                 # focusing 参数
    use_alpha: false           # 是否使用 alpha 平衡
    detach_p_for_weight: true  # detach focal 权重梯度
    compute_fp32: true         # AMP 下用 fp32 计算权重
    log_components: true       # 记录 BCE/Focal 组件
```

### 关键参数说明

| 参数 | 说明 | 推荐值 | Sweep 范围 |
|------|------|--------|-----------|
| `enabled` | 总开关 | `true` | - |
| `warmup_steps` | Warmup 步数 | `2000` | 1000, 2000 |
| `lambda` | Focal 系数 | `0.1` | 0.05, 0.1, 0.2 |
| `gamma` | Focusing 参数 | `1.0` | 1.0, 2.0 |
| `use_alpha` | 使用 alpha 平衡 | `false` | - |
| `detach_p_for_weight` | Detach 权重梯度 | `true` | - |
| `compute_fp32` | FP32 权重计算 | `true` | - |

## 使用方式

### 1. 训练（启用 Aux Focal）

```bash
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml
```

配置文件中确保：
```yaml
loss:
  aux_focal:
    enabled: true
    warmup_steps: 2000
    lambda: 0.1
    gamma: 1.0
```

### 2. 训练（禁用 Aux Focal，等价于原实现）

将配置中 `enabled` 设为 `false` 或删除整个 `aux_focal` 段：

```yaml
loss:
  aux_focal:
    enabled: false  # 或直接删除整个 aux_focal 块
```

### 3. 超参数 Sweep

推荐 sweep 以下参数组合：

```bash
# Lambda sweep
lambda: [0.05, 0.1, 0.2]

# Gamma sweep
gamma: [1.0, 2.0]

# Warmup sweep
warmup_steps: [1000, 2000]
```

示例命令（手动修改配置文件后运行）：
```bash
# Baseline (disabled)
# config: aux_focal.enabled = false
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml

# Focal lambda=0.1, gamma=1.0
# config: aux_focal.enabled=true, lambda=0.1, gamma=1.0
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml

# Focal lambda=0.2, gamma=2.0
# config: aux_focal.enabled=true, lambda=0.2, gamma=2.0
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml
```

## 监控指标

训练日志中新增以下字段（当 `log_components=true` 时）：

| 字段 | 说明 | 备注 |
|------|------|------|
| `aux_focal_enabled` | 是否启用 | bool |
| `aux_focal_on` | 当前 step 是否激活 | warmup 期间为 false |
| `aux_focal_warmup_steps` | Warmup 步数 | - |
| `aux_focal_lambda` | Lambda 系数 | - |
| `aux_focal_gamma` | Gamma 参数 | - |
| `loss_ctcvr_bce` | CTCVR BCE 分量 | 主损失 |
| `loss_ctcvr_focal` | CTCVR Focal 分量 | 辅助损失 |
| `loss_ctcvr` | CTCVR 总损失 | BCE + lambda*Focal |

### 日志示例

```json
{
  "epoch": 1,
  "split": "train",
  "loss_ctr": 0.1234,
  "loss_ctcvr": 0.5678,
  "loss_ctcvr_bce": 0.5234,
  "loss_ctcvr_focal": 0.4440,
  "aux_focal_on": true,
  "aux_focal_lambda": 0.1,
  "aux_focal_gamma": 1.0,
  "global_step": 2500
}
```

## 验证与测试

### 单元测试

```bash
cd e:\my_project\multi_task
set PYTHONPATH=e:\my_project\multi_task
python tests/test_aux_focal_smoke.py
```

测试覆盖：
1. ✅ Focal loss 基本计算（gamma=0 应等于 BCE）
2. ✅ Warmup 机制（step < warmup_steps 时不启用）
3. ✅ AMP 数值稳定性（fp16 不 NaN）
4. ✅ 向后兼容性（缺少 aux_focal 配置不报错）

### Smoke Test（小规模训练）

```bash
# 200 steps smoke test (disabled)
python -m src.cli.main train --config configs/experiments/test_aux_focal_disabled.yaml

# 200 steps smoke test (enabled)
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml
```

## 实现文件清单

### 核心实现
- **[src/loss/bce.py](../../src/loss/bce.py)**
  - 新增 `focal_on_logits_aux()` 函数
  - 修改 `MultiTaskBCELoss.__init__()` 增加 aux_focal 参数
  - 修改 `_compute_esmm()` 在 CTCVR loss 中接入 focal

- **[src/train/trainer.py](../../src/train/trainer.py)**
  - 从配置读取 `loss.aux_focal` 字段
  - 初始化 loss_fn 时传入 aux_focal 参数
  - 启用时打印配置日志

- **[src/train/loops.py](../../src/train/loops.py)**
  - 训练循环中更新 `loss_fn.global_step` 用于 warmup 控制

### 配置文件
- **[configs/experiments/mtl_mmoe.yaml](../../configs/experiments/mtl_mmoe.yaml)**
  - 主配置文件，默认 enabled=true
  - 包含完整的 aux_focal 配置段和注释

- **[configs/experiments/test_aux_focal_disabled.yaml](../../configs/experiments/test_aux_focal_disabled.yaml)**
  - Smoke test 配置，enabled=false

### 测试
- **[tests/test_aux_focal_smoke.py](../../tests/test_aux_focal_smoke.py)**
  - 单元测试和烟雾测试

## 约束与注意事项

### ✅ 已遵守的约束

1. **禁止 hard mining 丢样本**：无任何 `loss[mask].mean()` 样本过滤
2. **Logits 版本**：Focal Loss 直接在 logits 上计算，数值稳定
3. **仅 CTCVR**：只对 CTCVR 加 focal，CTR 一律不加
4. **不重复 pos_weight**：Focal 不叠加 pos_weight（pos_weight 只在主 BCE 中）
5. **向后兼容**：enabled=false 或缺少 aux_focal 配置时，行为与原实现一致
6. **保持现有结构**：训练入口、目录结构、日志格式均未破坏

### ⚠️ 注意事项

1. **Warmup 必须充足**：过小的 warmup_steps 可能导致训练初期不稳定
2. **Lambda 不宜过大**：过大会使辅助 focal 主导梯度，建议 <= 0.2
3. **Gamma 平衡**：gamma=2 降权更强，但可能过度抑制负样本学习
4. **AMP 稳定性**：务必保持 `compute_fp32=true`，否则 fp16 可能数值溢出
5. **日志性能**：`log_components=true` 会增加少量日志开销，生产环境可考虑关闭

## 理论背景

### 为什么需要 Focal Loss？

在 CTCVR（点击后转化）任务中：
- 正样本（转化）极少：~0.02%
- 负样本（未转化）海量：~99.98%
- Pos_weight 可以平衡类别权重，但无法区分 easy/hard negatives

Focal Loss 进一步：
- **Easy negatives**（p≈0 的负样本）：已经预测准确，降权以减少梯度主导
- **Hard negatives**（p≈0.5 的负样本）：预测困难，保持高权重
- **Hard positives**：稀有但重要，保持高权重

### Focal Loss 与 BCE + Pos_weight 的关系

| 方法 | 作用 | 效果 |
|------|------|------|
| BCE | 标准交叉熵 | 所有样本权重一致 |
| BCE + pos_weight | 类别平衡 | 放大正样本权重 |
| Focal Loss | Hard mining | 降权 easy samples |
| **方案1 (本实现)** | **BCE(pos_weight) + Aux-Focal** | **类别平衡 + Hard mining** |

组合使用原理：
1. **主 BCE + pos_weight**：确保稀有正样本得到足够关注
2. **辅助 Focal**：进一步降权大量 easy negatives，让模型关注 hard samples

### 参数影响

#### Gamma (γ)
```
focal_factor = (1 - p_t)^γ
```

| γ | Easy sample 权重 | Hard sample 权重 | 说明 |
|---|------------------|------------------|------|
| 0 | 1.0 | 1.0 | 退化为标准 BCE |
| 1 | 线性降权 | 1.0 | 温和 focusing |
| 2 | 二次降权 | 1.0 | 强 focusing（论文推荐） |

示例：对于 y=0 的负样本
- p=0.01 (easy): focal_factor = (1-0.99)^2 = 0.0001 → 强烈降权
- p=0.50 (hard): focal_factor = (1-0.50)^2 = 0.2500 → 保持较高权重

#### Lambda (λ)
```
L_ctcvr = L_bce + λ * L_focal
```

| λ | BCE 主导 | Focal 主导 | 推荐场景 |
|---|----------|-----------|----------|
| 0.05 | 高 | 低 | 保守尝试 |
| 0.1 | 中 | 中 | 平衡（推荐） |
| 0.2 | 低 | 高 | 激进 hard mining |

## 性能与效率

### 额外计算开销

相比原 ESMM v2 实现，Aux-Focal 的额外开销：

1. **前向**：
   - 额外 sigmoid、pow、乘法操作
   - 预估增加 ~5-10% 训练时间（取决于 batch size 和模型复杂度）

2. **内存**：
   - 需存储 focal_factor 用于反向传播（如果 `detach_p_for_weight=false`）
   - `detach_p_for_weight=true`（推荐）时无额外内存

3. **反向**：
   - Focal 项引入额外梯度计算
   - `detach_p_for_weight=true` 时梯度仅通过 BCE 路径

### 优化建议

1. **禁用时零开销**：`enabled=false` 时完全不执行 focal 计算
2. **Warmup 延迟**：前期不执行 focal，减少总训练时间开销
3. **日志开关**：`log_components=false` 可减少日志写入开销
4. **Detach 权重**：`detach_p_for_weight=true` 可减少反向传播计算量

## 已知限制

1. **仅支持 ESMM v2**：Legacy ESMM 模式未集成（可按需扩展）
2. **仅 CTCVR**：当前实现只对 CTCVR 加 focal，如需对 CTR 加 focal 需修改代码
3. **静态 pos_weight**：Focal 与动态 pos_weight 未充分测试

## 引用

如果本实现对你的工作有帮助，请引用：

```
方案1: ESMM 主链路 BCE + CTCVR Aux-Focal（配置化 + warmup）
实现于 multi_task 项目，2026年2月
```

相关论文：
- [Focal Loss for Dense Object Detection (Lin et al., ICCV 2017)](https://arxiv.org/abs/1708.02002)
- [Entire Space Multi-Task Model (Ma et al., SIGIR 2018)](https://arxiv.org/abs/1804.07931)

---

**实现日期**：2026年2月1日  
**实现者**：GitHub Copilot (Claude Sonnet 4.5)  
**版本**：v1.0
