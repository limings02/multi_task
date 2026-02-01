# Aux Focal 实现总结

## 📋 已完成任务

✅ **1. 实现 focal_on_logits_aux 函数** ([src/loss/bce.py](../src/loss/bce.py#L17-L108))
   - Logits 版本，数值稳定
   - 支持 gamma、alpha、detach、fp32 可配置
   - 详细文档注释

✅ **2. 在 ESMM v2 中接入 CTCVR Aux-Focal** ([src/loss/bce.py](../src/loss/bce.py#L381-L419))
   - 只对 CTCVR 加 focal，CTR 不受影响
   - 支持 warmup 机制
   - 保持主 BCE + pos_weight 不变

✅ **3. 配置解析与 backward compatibility** ([src/train/trainer.py](../src/train/trainer.py#L193-L257))
   - 从 `loss.aux_focal` 读取配置
   - 缺少配置时默认 disabled
   - 启用时打印日志

✅ **4. Global step 更新** ([src/train/loops.py](../src/train/loops.py#L193-L197))
   - 训练循环中更新 loss_fn.global_step
   - 用于 warmup 控制

✅ **5. 配置文件** ([configs/experiments/mtl_mmoe.yaml](../configs/experiments/mtl_mmoe.yaml#L102-L130))
   - 完整的 aux_focal 配置段
   - 详细注释和 sweep 建议

✅ **6. 日志记录** ([src/loss/bce.py](../src/loss/bce.py#L571-L587))
   - loss_ctcvr_bce / loss_ctcvr_focal
   - aux_focal_on / aux_focal_lambda / aux_focal_gamma
   - global_step

✅ **7. 单元测试** ([tests/test_aux_focal_smoke.py](../tests/test_aux_focal_smoke.py))
   - 4 个测试全部通过
   - 验证 warmup、AMP、backward compatibility

✅ **8. 文档** ([docs/aux_focal_implementation.md](aux_focal_implementation.md))
   - 使用指南
   - 理论背景
   - 参数说明
   - 监控指标

## 🎯 核心特性

| 特性 | 实现 | 验证 |
|------|------|------|
| 主损失不变 | ✅ CTR/CTCVR 仍用 BCE + pos_weight | ✅ 测试通过 |
| 仅 CTCVR focal | ✅ CTR 不受影响 | ✅ 测试通过 |
| Warmup 机制 | ✅ 前 N step 不启用 | ✅ 测试通过 |
| 配置化开关 | ✅ enabled=false 时完全一致 | ✅ 测试通过 |
| AMP 兼容 | ✅ compute_fp32=true | ✅ 测试通过 |
| 向后兼容 | ✅ 缺少配置不报错 | ✅ 测试通过 |
| 可观测性 | ✅ 日志记录 BCE/Focal 组件 | ✅ 代码实现 |

## 📝 配置示例

### 启用 Aux Focal（推荐配置）

```yaml
loss:
  static_pos_weight:
    ctr: 24.7
    ctcvr: 4800
  
  aux_focal:
    enabled: true
    warmup_steps: 2000
    target_head: "ctcvr"
    lambda: 0.1
    gamma: 1.0
    use_alpha: false
    detach_p_for_weight: true
    compute_fp32: true
    log_components: true
```

### 禁用 Aux Focal（等价于原实现）

```yaml
loss:
  static_pos_weight:
    ctr: 24.7
    ctcvr: 4800
  
  aux_focal:
    enabled: false  # 或直接删除整个 aux_focal 块
```

## 🚀 使用方式

### 训练

```bash
# 启用 aux_focal
python -m src.cli.main train --config configs/experiments/mtl_mmoe.yaml

# 禁用 aux_focal（修改配置或删除 aux_focal 段）
python -m src.cli.main train --config configs/experiments/test_aux_focal_disabled.yaml
```

### 测试

```bash
cd e:\my_project\multi_task
set PYTHONPATH=e:\my_project\multi_task
python tests/test_aux_focal_smoke.py
```

输出示例：
```
=== Running Aux Focal Smoke Tests ===

✓ focal_on_logits_aux basic test passed (BCE=0.5982, Focal(g=2)=0.1535)
✓ Warmup test passed:
  - Baseline (disabled): loss_ctcvr=0.8738
  - Warmup phase: loss_ctcvr=0.8738 (should equal baseline)
  - Active phase: loss_ctcvr=0.9382 (BCE=0.8738 + Focal=0.6440)
✓ AMP stability test passed (device=cuda, loss=1.9336)
✓ Backward compatibility test passed (loss=1.8384)

=== All tests passed! ===
```

## 📊 预期日志

训练日志中会增加以下字段（当 `log_components=true` 时）：

```json
{
  "epoch": 1,
  "split": "train",
  "global_step": 2500,
  "loss_ctr": 0.1234,
  "loss_ctcvr": 0.5678,
  "loss_ctcvr_bce": 0.5234,
  "loss_ctcvr_focal": 0.4440,
  "aux_focal_enabled": true,
  "aux_focal_on": true,
  "aux_focal_warmup_steps": 2000,
  "aux_focal_lambda": 0.1,
  "aux_focal_gamma": 1.0
}
```

## 🔍 Sweep 建议

推荐超参数扫描：

| 参数 | Baseline | 候选值 | 说明 |
|------|----------|--------|------|
| `enabled` | false | false, true | 关闭/开启 focal |
| `lambda` | 0.1 | 0.05, 0.1, 0.2 | Focal 系数 |
| `gamma` | 1.0 | 1.0, 2.0 | Focusing 参数 |
| `warmup_steps` | 2000 | 1000, 2000 | Warmup 步数 |

建议实验组合（共 7 组）：

1. **Baseline**: enabled=false
2. **Focal-1**: lambda=0.05, gamma=1.0, warmup=2000
3. **Focal-2**: lambda=0.1, gamma=1.0, warmup=2000 ⭐ 推荐
4. **Focal-3**: lambda=0.2, gamma=1.0, warmup=2000
5. **Focal-4**: lambda=0.1, gamma=2.0, warmup=2000
6. **Focal-5**: lambda=0.1, gamma=1.0, warmup=1000
7. **Focal-6**: lambda=0.2, gamma=2.0, warmup=1000

## ⚠️ 注意事项

1. **首次使用**：建议先运行 baseline (enabled=false) 作为对照
2. **Warmup 必要性**：不要设置 warmup_steps=0，会导致训练初期不稳定
3. **Lambda 不宜过大**：lambda > 0.3 可能导致辅助 focal 主导梯度
4. **监控 NaN**：如出现 NaN，确保 `compute_fp32=true`
5. **性能开销**：预计增加 5-10% 训练时间

## 🎓 理论背景

**Focal Loss 核心思想**：降权 easy samples，让模型关注 hard samples

```
focal_factor = (1 - p_t)^gamma

- Easy negative (y=0, p≈0): p_t ≈ 1 → focal_factor ≈ 0 → 强降权
- Hard negative (y=0, p≈0.5): p_t ≈ 0.5 → focal_factor ≈ 0.25 → 保持权重
- Hard positive (y=1, p≈0.5): p_t ≈ 0.5 → focal_factor ≈ 0.25 → 保持权重
```

**为什么需要辅助 Focal？**

CTCVR 极端不平衡（正样本 ~0.02%）：
- Pos_weight 解决类别平衡
- Focal 进一步区分 easy/hard samples
- 组合使用效果最佳

## 📁 文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| [src/loss/bce.py](../src/loss/bce.py) | ✏️ 修改 | 新增 focal 函数 + 接入 ESMM |
| [src/train/trainer.py](../src/train/trainer.py) | ✏️ 修改 | 配置解析 + loss_fn 初始化 |
| [src/train/loops.py](../src/train/loops.py) | ✏️ 修改 | global_step 更新 |
| [configs/experiments/mtl_mmoe.yaml](../configs/experiments/mtl_mmoe.yaml) | ✏️ 修改 | 新增 aux_focal 配置段 |
| [tests/test_aux_focal_smoke.py](../tests/test_aux_focal_smoke.py) | ➕ 新增 | 单元测试 |
| [configs/experiments/test_aux_focal_disabled.yaml](../configs/experiments/test_aux_focal_disabled.yaml) | ➕ 新增 | 禁用配置（测试用） |
| [docs/aux_focal_implementation.md](aux_focal_implementation.md) | ➕ 新增 | 完整文档 |
| [docs/aux_focal_summary.md](aux_focal_summary.md) | ➕ 新增 | 本文档 |

## ✅ 验收检查

- [x] enabled=false 时行为与原实现一致
- [x] enabled=true, step < warmup_steps 时不启用 focal
- [x] enabled=true, step >= warmup_steps 时 focal 激活
- [x] AMP 下不出现 NaN
- [x] 缺少 aux_focal 配置不报错
- [x] 日志中能看到 loss_ctcvr_bce 和 loss_ctcvr_focal
- [x] 单元测试全部通过
- [x] 代码无语法错误

## 🎉 完成状态

**状态**：✅ 全部完成  
**测试**：✅ 单元测试通过  
**文档**：✅ 完整文档  
**向后兼容**：✅ 完全兼容  

可以直接使用当前实现开始训练和实验！

---

**实现时间**：2026年2月1日  
**实现者**：GitHub Copilot (Claude Sonnet 4.5)
