# ============================================================================
# Makefile for Multi-Task Learning Project
# ============================================================================

.PHONY: help interview-chain interview-chain-resume interview-chain-dry-run

help:
	@echo "Available targets:"
	@echo "  interview-chain          - Run all 7 experiments in the interview chain (E0a-E5)"
	@echo "  interview-chain-resume   - Resume interview chain (skip completed experiments)"
	@echo "  interview-chain-dry-run  - Dry-run interview chain (print commands only)"
	@echo ""
	@echo "For more details, see docs/interview_chain.md"

# ============================================================================
# Interview Chain - 主线 7 个实验一键运行
# ============================================================================
interview-chain:
	python scripts/run_interview_chain.py

interview-chain-resume:
	python scripts/run_interview_chain.py --resume

interview-chain-dry-run:
	python scripts/run_interview_chain.py --dry-run

# ============================================================================
# Legacy Documentation (preserved for reference)
# ============================================================================
# Canonical preprocessing pipeline (scheme C: samples + tokens) for the Ali-CCP dataset.
#
# - Debug first 100000 rows:
#   python -m src.cli.main canonical --config configs/dataset/aliccp.yaml --nrows 100000 --overwrite
#
# - Full run (default `nrows=null` in config):
#   python -m src.cli.main canonical --config configs/dataset/aliccp.yaml
#
# ## Compact Tokens (optional)
#
# After canonical, you can compact many small token parquet files into ~100 larger parts:
#
# - Bash helper:
#   bash scripts/compact.sh configs/dataset/aliccp.yaml --target-parts 100 --overwrite
#
# - In-place replace tokens (backup to tokens_train.bak):
#   python -m src.cli.main compact --config configs/dataset/aliccp.yaml --target-parts 100 --overwrite --inplace