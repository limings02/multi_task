加工数据 Schema 契约（train / valid）
===================================

本文约束 `data/processed/train` 与 `data/processed/valid` 下的 **processed parquet** 与 `src/data/dataset.py` 中 `build_dataloader` / `collate_fn` 的对接方式。所有字段均为强约束，未在本文声明的字段一律视为违规。

## 1. 标签与 ID 列（行级必备）
| 列名 | parquet dtype | 取值/形状 | collate 后 dtype | 说明 |
| --- | --- | --- | --- | --- |
| `y_ctr` | float32 | 标量 | float32, 形状 `[B]` | 点击二分类标签，{0,1} |
| `y_cvr` | float32 | 标量 | float32, `[B]` | 成交二分类标签，{0,1} |
| `y_ctcvr` | float32 | 标量 | float32, `[B]` | 点击后成交标签，{0,1} |
| `click_mask` | float32 | 标量 | float32, `[B]` | 是否曝光，{0,1} |
| `row_id` | int64 | 标量，全局唯一 | int64, `[B]` | 样本行号（用于去重/追踪） |
| `entity_id` | string | 标量 | Python `list[str]`，长度 `B` | entity 分桶 ID，不转张量 |

以上 6 列在所有 split 行中必须存在；缺失视为数据损坏。

## 2. 特征列命名与存储约定
- 单值（single-hot）特征：列对 `{f..._idx}` 必有；若 `use_value=True` 则再有 `{f..._val}`。两列均为标量。
- 多值（multi-hot）特征：列对 `{f..._idx}`、可选 `{f..._val}`，类型为 `list<...>`；无 offset 列（已在 P1-5 移除）。
- 所有 `_idx` 列使用 `int64`，`_val` 列使用 `float32`。列表长度已在写入阶段截断到 featuremap 的 `max_len`，未做 padding。
- 列前缀与语义请见下表；`src` 与 `field` 对应 featuremap。

### 2.1 特征清单
| 前缀 | is_multi_hot | use_value | encoding | max_len | 说明 |
| --- | --- | --- | --- | --- | --- |
| `f0_508` | 否 | 是 | vocab | — | 单值，类别型，带权重 |
| `f0_301` | 否 | 否 | vocab | — | 单值 |
| `f0_205` | 否 | 否 | hash | — | 单值，哈希 |
| `f1_121` | 否 | 否 | vocab | — | 单值 |
| `f1_129` | 否 | 否 | vocab | — | 单值 |
| `f1_110_14` | 是 | 是 | hash | 60 | 多值，权重，auto_mix 截断 |
| `f0_210` | 是 | 否 | hybrid(topn+hash) | 20 | 多值，无 value |
| `f0_206` | 否 | 否 | vocab | — | 单值 |
| `f1_127` | 否 | 否 | vocab | — | 单值 |
| `f1_127_14` | 是 | 是 | hash | 60 | 多值，权重 |
| `f0_509` | 否 | 是 | hash | — | 单值，带权重 |
| `f1_124` | 否 | 否 | vocab | — | 单值 |
| `f1_150_14` | 是 | 是 | hybrid(topn+hash) | 80 | 多值，权重 |
| `f1_109_14` | 是 | 是 | vocab | 80 | 多值，权重 |
| `f0_216` | 否 | 否 | hybrid(topn+hash) | — | 单值 |
| `f0_207` | 否 | 否 | hash | — | 单值 |
| `f0_702` | 否 | 是 | hybrid(topn+hash) | — | 单值，带权重 |
| `f1_126` | 否 | 否 | vocab | — | 单值 |
| `f1_101` | 否 | 否 | hash | — | 单值 |
| `f1_125` | 否 | 否 | vocab | — | 单值 |
| `f1_128` | 否 | 否 | vocab | — | 单值 |
| `f1_122` | 否 | 否 | vocab | — | 单值 |
| `f0_853` | 是 | 是 | hybrid(topn+hash) | 10 | 多值，权重 |

## 3. 截断与 padding 规则
- **截断（写 parquet 时）**：多值特征按 featuremap `max_len` 先截断，再落盘；`select_tokens` 采用 auto_mix/topk 策略（详见 `reports/eda/token_truncation_strategy.md`）。
- **缺失与空序列**：
  - 单值：若该行无 token，写入 `missing_id`（见下）与 `val=1.0`（对 use_value=True）。
  - 多值：若无 token，写入长度为 1 的列表 `[missing_id]`，`val=[1.0]`（如适用），`len=1`。
- **ID 约定（来自 featuremap token_policy）**  
  - vocab / hybrid-head：`pad_id=0`，`missing_id=1`，`oov_id=2`，实际 token 起始为 `3`。  
  - hash / hybrid-tail：`pad_id=0`，`missing_id=1`，有效哈希 ID = `2 + (hash % bucket)`（无 oov 概念）。  
- **batch 内 padding（在 collate_fn）**：  
  - 多值 `idx` 右侧用 `pad_id=0` 补齐到本 batch 的 `max(seq_len)`；`len` 记录原始长度。  
  - 多值 `val` 补齐为 `0.0`；若该特征 `use_value=False`，`val` 为 `None`。  
  - 单值无 padding，直接转标量张量。

## 4. collate_fn 输出结构（`build_dataloader(..., collate_fn)`）
```python
batch = {
  "labels": {
    "y_ctr": FloatTensor[B],
    "y_cvr": FloatTensor[B],
    "y_ctcvr": FloatTensor[B],
    "click_mask": FloatTensor[B],
    "row_id": LongTensor[B],
    "entity_id": list[str]  # 长度 B
  },
  "features": {
    "<prefix>": {
      "type": "single" | "multi",
      "idx": LongTensor[B]                # single
            or LongTensor[B, Lmax]        # multi, 已 padding
      "len": LongTensor[B]                # 仅 multi
      "val": FloatTensor[B]               # single + use_value
            or FloatTensor[B, Lmax]       # multi + use_value
            or None                       # use_value=False
    },
    ...
  }
}
```
`Lmax` 为该 batch 中该特征的最大实际长度（≤ featuremap.max_len）。

## 5. 允许/不允许的缺失与填充策略
- **必填字段**：第 1 节 6 个标签列；第 2 节列出的所有 `{f..._idx}`。缺失视为数据错误。
- **可省略的列**：仅 `_val` 列，且只针对 `use_value=False` 的特征；若省略则在 collate 后 `val=None`。
- **缺失填充值**：写 parquet 时用 `missing_id=1`（或 `[1]` 列表）；`val` 缺失时置 `1.0`（单值/多值一致）；batch padding 只使用 `pad_id=0` 与 `val=0.0`。

## 6. 读取与批处理流程简述
1. `ProcessedIterDataset` 用 PyArrow dataset 流式读取分片，逐行生成 Python dict。
2. `collate_fn` 根据 `feature_meta` 判断 `is_multi_hot` / `use_value`，对多值做 batch padding，生成上节结构。
3. 训练侧依赖上述 contract，不得对列名、dtype、填充值、padding 规则做变更；若需变更，必须同步更新本文件与数据生成逻辑。

## 7. 依赖文件
- `configs/dataset/featuremap.yaml`：max_len、token_policy（pad/missing/oov）、encoding 定义。
- `src/data/processed_builder.py`：写入 parquet 的字段与缺失填充逻辑。
- `src/data/dataset.py`：`ProcessedIterDataset` 与 `collate_fn` 的 batch 结构与 padding 细节。

（版本：2026-01-27，覆盖 schema_version=2 数据）
