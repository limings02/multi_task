# 🚀 Aux Focal 快速参考

## 一键启用

### 配置文件 (configs/experiments/mtl_mmoe.yaml)

```yaml
loss:
  aux_focal:
    enabled: true          # 开关
    warmup_steps: 2000     # 前 2000 step 不启用
    lambda: 0.1            # focal 系数（推荐 0.05~0.2）
    gamma: 1.0             # focusing 参数（推荐 1.0~2.0）
```

### 训练命令

```bash
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml
```

## 关键参数速查

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `enabled` | false | true/false | 总开关 |
| `warmup_steps` | 2000 | 1000-2000 | warmup 步数 |
| `lambda` | 0.1 | 0.05-0.2 | focal 系数 |
| `gamma` | 1.0 | 1.0-2.0 | focusing 参数 |
| `use_alpha` | false | false | 是否用 alpha 平衡 |
| `detach_p_for_weight` | true | true | detach 权重梯度 |
| `compute_fp32` | true | true | fp32 权重计算 |
| `log_components` | true | true | 记录 BCE/Focal |

## 日志监控

训练日志新增字段：

```json
{
  "loss_ctcvr_bce": 0.5234,      // CTCVR BCE 主损失
  "loss_ctcvr_focal": 0.4440,    // CTCVR Focal 辅助损失
  "loss_ctcvr": 0.5678,          // 总 CTCVR 损失
  "aux_focal_on": true,          // 是否激活
  "aux_focal_lambda": 0.1,       // lambda 系数
  "aux_focal_gamma": 1.0,        // gamma 参数
  "global_step": 2500            // 当前步数
}
```

## Sweep 模板

```yaml
# Baseline
aux_focal:
  enabled: false

# Experiment 1: lambda=0.05
aux_focal:
  enabled: true
  lambda: 0.05
  gamma: 1.0
  warmup_steps: 2000

# Experiment 2: lambda=0.1 (推荐)
aux_focal:
  enabled: true
  lambda: 0.1
  gamma: 1.0
  warmup_steps: 2000

# Experiment 3: lambda=0.2
aux_focal:
  enabled: true
  lambda: 0.2
  gamma: 1.0
  warmup_steps: 2000

# Experiment 4: gamma=2.0
aux_focal:
  enabled: true
  lambda: 0.1
  gamma: 2.0
  warmup_steps: 2000
```

## 测试命令

```bash
cd e:\my_project\multi_task
set PYTHONPATH=e:\my_project\multi_task
python tests/test_aux_focal_smoke.py
```

## 常见问题

### Q: 如何关闭 aux_focal？
**A**: 设置 `enabled: false` 或删除整个 `aux_focal` 块

### Q: 为什么需要 warmup？
**A**: 训练初期模型不稳定，直接加 focal 可能导致梯度爆炸

### Q: lambda 应该设多大？
**A**: 推荐 0.1，不建议超过 0.2

### Q: gamma 应该设多大？
**A**: 推荐 1.0（温和）或 2.0（激进）

### Q: 会影响 CTR 损失吗？
**A**: 不会，focal 只作用于 CTCVR

### Q: 如何验证是否生效？
**A**: 查看日志中的 `aux_focal_on` 字段，应为 `true`

### Q: 出现 NaN 怎么办？
**A**: 确保 `compute_fp32: true` 并检查 warmup_steps 是否足够

## 实现文件

- **核心逻辑**: [src/loss/bce.py](../src/loss/bce.py)
- **配置解析**: [src/train/trainer.py](../src/train/trainer.py)
- **训练循环**: [src/train/loops.py](../src/train/loops.py)
- **配置文件**: [configs/experiments/mtl_mmoe.yaml](../configs/experiments/mtl_mmoe.yaml)
- **单元测试**: [tests/test_aux_focal_smoke.py](../tests/test_aux_focal_smoke.py)
- **完整文档**: [docs/aux_focal_implementation.md](aux_focal_implementation.md)

---

**快速联系**: 查看 [完整文档](aux_focal_implementation.md) 了解更多细节
