"""
最小 QA：检查 processed parquet 中每个 field 的 idx 是否落在 [0, num_embeddings)（或你的 offset 规则）内。

你要做的事只有两步：
1) 改 DATA_GLOB 指向你实际 processed parquet
2) 改 COLS：填你实际存 idx 的列名（或按你的存储结构写解析）
"""

import glob
import json
import pyarrow.parquet as pq
import numpy as np

META_PATH = "data\\processed\\metadata.json"   # 改成你的
DATA_GLOB = "data\\processed\\train\\part-00004.parquet" # 改成你的

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

num_embeddings = meta["total_num_embeddings"]

# 你需要把这里改成“你 processed 里 idx 的列名”
# 常见命名："{field}_idx" / "{field}.idx" / "X_{field}" 等
def iter_field_indices(batch, field: str):
    """
    产出一个 batch 中某个 field 的所有 idx（扁平化）。
    你根据自己的 schema 改这里的取列逻辑。
    """
    col = f"{field}_idx"  # <- 你很可能需要改
    if col not in batch.schema.names:
        return None
    arr = batch.column(col)
    # 如果是 List<int>，to_numpy 之前先 flatten
    if arr.type.__class__.__name__ == "ListType":
        flat = arr.flatten()
        return flat.to_numpy(zero_copy_only=False)
    return arr.to_numpy(zero_copy_only=False)

files = sorted(glob.glob(DATA_GLOB))[:3]  # 先抽 3 个文件够判定
assert files, "没找到 parquet 文件，请检查 DATA_GLOB"

for field, nemb in num_embeddings.items():
    mx = -1
    mn = 1 << 60
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=20000):
            idx = iter_field_indices(batch, field)
            if idx is None:
                continue
            if len(idx) == 0:
                continue
            mx = max(mx, int(np.max(idx)))
            mn = min(mn, int(np.min(idx)))
    if mx >= 0:
        print(f"{field}: min={mn}, max={mx}, num_embeddings={nemb}, OOB={mx >= nemb or mn < 0}")
