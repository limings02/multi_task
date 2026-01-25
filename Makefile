Canonical preprocessing pipeline (scheme C: samples + tokens) for the Ali-CCP dataset.

- Debug first 100000 rows:
  ```powershell
  python -m src.cli.main canonical --config configs/dataset/aliccp.yaml --nrows 100000 --overwrite
  ```
- Full run (default `nrows=null` in config):
  ```powershell
  python -m src.cli.main canonical --config configs/dataset/aliccp.yaml
  ```

## Compact Tokens (optional)

After canonical, you can compact many small token parquet files into ~100 larger parts:

- Bash helper:
  ```bash
  bash scripts/compact.sh configs/dataset/aliccp.yaml --target-parts 100 --overwrite
  ```
- In-place replace tokens (backup to tokens_train.bak):
  ```powershell
  python -m src.cli.main compact --config configs/dataset/aliccp.yaml --target-parts 100 --overwrite --inplace
  ```