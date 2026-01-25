"""Performance test for token collection."""
import duckdb
import time
import pyarrow as pa
import numpy as np
from pathlib import Path
from collections import defaultdict

token_dir = Path('data/splits/aliccp_entity_hash_v1/tokens/train_tokens')
glob_pattern = str(token_dir / '*.parquet').replace('\\', '/')

print(f"Token dir: {token_dir}")
print(f"Glob pattern: {glob_pattern}")

# Test DuckDB query performance for small batch
con = duckdb.connect()
con.execute('SET memory_limit = "6GB"')
con.execute('SET temp_directory = "data/.duckdb_tmp"')
con.execute('SET threads TO 6')
con.execute('SET preserve_insertion_order = false')

# Simulated row_ids for 100k samples (more realistic batch size)
row_ids = list(range(100000))
min_id, max_id = 0, 110000
con.register('row_ids_tbl', pa.table({'row_id': pa.array(row_ids, type=pa.int64())}))

sql = f"""
    SELECT
        t.row_id,
        CAST(COALESCE(t.src, 0) AS INTEGER) AS src,
        CAST(t.field AS VARCHAR) AS field,
        CAST(t.fid AS VARCHAR) AS fid,
        CAST(t.val AS DOUBLE) AS val
    FROM read_parquet('{glob_pattern}') t
    JOIN row_ids_tbl r ON t.row_id = r.row_id
    WHERE t.row_id BETWEEN {min_id} AND {max_id}
    ORDER BY t.row_id, src, field
"""

print('Running DuckDB query for 100k samples...')
t0 = time.perf_counter()
raw_tbl = con.execute(sql).fetch_arrow_table()
t1 = time.perf_counter()
print(f'DuckDB query: {t1-t0:.1f}s, {raw_tbl.num_rows:,} rows')

con.close()

print('Converting to NumPy...')
t2 = time.perf_counter()
rids = raw_tbl["row_id"].to_numpy()
srcs = raw_tbl["src"].to_numpy()
fields = raw_tbl["field"].to_pandas().values
fids = raw_tbl["fid"].to_pandas().values
vals = raw_tbl["val"].to_numpy()
del raw_tbl
t3 = time.perf_counter()
print(f'Column extraction: {t3-t2:.1f}s')

print('NaN handling...')
t4 = time.perf_counter()
nan_mask = np.isnan(vals) | np.isinf(vals)
vals = np.where(nan_mask, 0.0, vals)
t5 = time.perf_counter()
print(f'NaN handling: {t5-t4:.1f}s')

print('Building token pairs (single pass)...')
t6 = time.perf_counter()
out = defaultdict(list)
n = len(rids)
if n > 0:
    prev_key = (rids[0], srcs[0], fields[0])
    pairs = []
    
    for i in range(n):
        curr_key = (rids[i], srcs[i], fields[i])
        if curr_key != prev_key:
            out[prev_key].extend(pairs)
            pairs = []
            prev_key = curr_key
        pairs.append((fids[i], vals[i]))
    
    out[prev_key].extend(pairs)
t7 = time.perf_counter()
print(f'Build pairs: {t7-t6:.1f}s, {len(out):,} groups')

total = t7 - t0
print(f'\nTotal for 100k samples: {total:.1f}s')
batches_needed = 40179176 / 100000
print(f'Estimated for 40M samples ({batches_needed:.0f} batches): {total * batches_needed / 3600:.1f} hours')
