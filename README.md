# Ali-CCP MTL Ranker

Canonical preprocessing pipeline (scheme C: samples + tokens) for the Ali-CCP dataset.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # On PowerShell: .\\.venv\\Scripts\\Activate.ps1
pip install -e .
```

## Generate Canonical Data

The CLI builds canonical data via `src/data/canonical.py` (`build_canonical_aliccp`, seed defaults to 20260121 for stability). Paths in `configs/dataset/aliccp.yaml` are relative to the repo root. Override raw inputs via env vars `ALICCP_SKELETON_PATH` and `ALICCP_COMMON_FEATURES_PATH`.

### Run via Python
```bash
python -m src.cli.main canonical --config configs/dataset/aliccp.yaml
python -m src.cli.main canonical --config configs/dataset/aliccp.yaml --nrows 100000 --overwrite
```

### Run via Bash helper
```bash
bash scripts/canonical.sh configs/dataset/aliccp.yaml --nrows 100000 --overwrite
```

Tune/debug with `--nrows` (row cap) or `--overwrite` to rebuild outputs. Outputs are written under `data/interim/aliccp_canonical` by default (all relative to repo root):

- `data/interim/aliccp_canonical/common_features.sqlite` - entity_id -> feat_str lookup
- `data/interim/aliccp_canonical/samples_train.parquet` - row-wise labels/index columns
- `data/interim/aliccp_canonical/tokens_train/tokens_train.part_*.parquet` - long token table shards
- `data/interim/aliccp_canonical/manifest.json` - reproducibility metadata (paths, params, seed, parts)

## Quick Commands (PowerShell)

- Debug first 100000 rows:
  ```powershell
  python -m src.cli.main canonical --config configs/dataset/aliccp.yaml --nrows 100000 --overwrite
  ```
- Full run (default `nrows=null` in config):
  ```powershell
  python -m src.cli.main canonical --config configs/dataset/aliccp.yaml
  ```

