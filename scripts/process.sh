#!/usr/bin/env bash
set -euo pipefail
python -m src.cli.main process \
  --config configs/dataset/featuremap.yaml \
  --split-dir data/splits/aliccp_entity_hash_v1 \
  --out data/processed/aliccp_entity_hash_v1 \
  --batch-size 500000
