#!/usr/bin/env python3
"""
Batch evaluation script for multiple experiment runs.
Evaluates best checkpoints and generates comparison report.

Usage:
    python scripts/batch_eval.py --runs E1,E2,E3 --split valid
    python scripts/batch_eval.py --config batch_eval_config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_experiment_dir(run_identifier: str, runs_root: Path) -> Optional[Path]:
    """
    Find experiment directory by partial identifier.
    Supports both full dir name or short identifier (e.g., "E1" for "interview_E1_*")
    """
    if (runs_root / run_identifier).exists():
        return runs_root / run_identifier
    
    # Try to find by pattern
    candidates = list(runs_root.glob(f"*{run_identifier}*"))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        logger.warning(
            f"Multiple matches for '{run_identifier}': {[c.name for c in candidates]}"
        )
        # Return the most recent one
        return max(candidates, key=lambda p: p.stat().st_mtime)
    
    return None


def validate_experiment_dir(exp_dir: Path) -> bool:
    """Validate that experiment directory has required files."""
    if not exp_dir.exists():
        logger.error(f"Directory does not exist: {exp_dir}")
        return False
    
    config_file = exp_dir / "config.yaml"
    ckpt_file = exp_dir / "ckpt_best.pt"
    
    if not config_file.exists():
        logger.error(f"Missing config.yaml in {exp_dir}")
        return False
    
    if not ckpt_file.exists():
        logger.error(f"Missing ckpt_best.pt in {exp_dir}")
        return False
    
    return True


def run_eval(
    exp_dir: Path,
    split: str = "valid",
    save_preds: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run evaluation for a single experiment using CLI.
    
    Returns:
        Dict with eval results and metadata
    """
    config_file = exp_dir / "config.yaml"
    ckpt_file = exp_dir / "ckpt_best.pt"
    
    # Prepare eval command
    cmd = [
        sys.executable, "-m", "src.cli.main", "eval",
        "--config", str(config_file),
        "--ckpt", str(ckpt_file),
        "--split", split,
        "--run-dir", str(exp_dir),
    ]
    
    if save_preds:
        cmd.append("--save-preds")
    
    if max_batches is not None:
        cmd.extend(["--max-batches", str(max_batches)])
    
    logger.info(f"Running eval for {exp_dir.name} on {split} split...")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=Path.cwd(),
        )
        
        # Parse JSON output
        output_data = json.loads(result.stdout)
        
        # Load eval.json for detailed metrics
        eval_json_path = Path(output_data["eval_json"])
        if eval_json_path.exists():
            with open(eval_json_path, "r") as f:
                eval_data = json.load(f)
        else:
            eval_data = output_data["summary"]
        
        logger.info(f"✓ Evaluation completed for {exp_dir.name}")
        return {
            "exp_dir": str(exp_dir),
            "exp_name": exp_dir.name,
            "split": split,
            "success": True,
            "eval_data": eval_data,
            "eval_json_path": str(eval_json_path),
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Evaluation failed for {exp_dir.name}")
        logger.error(f"Error output:\n{e.stderr}")
        return {
            "exp_dir": str(exp_dir),
            "exp_name": exp_dir.name,
            "split": split,
            "success": False,
            "error": e.stderr,
        }
    except json.JSONDecodeError as e:
        logger.error(f"✗ Failed to parse output for {exp_dir.name}")
        return {
            "exp_dir": str(exp_dir),
            "exp_name": exp_dir.name,
            "split": split,
            "success": False,
            "error": f"JSON parse error: {e}",
        }


def extract_metrics(eval_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from eval data for comparison."""
    metrics = {}
    
    # Extract AUC metrics
    for key in ["auc_ctr", "auc_cvr", "auc_ctcvr", "auc_primary"]:
        if key in eval_data:
            metrics[key] = eval_data[key]
    
    # Extract loss metrics
    for key in ["loss_ctr", "loss_cvr", "loss_ctcvr", "loss_total"]:
        if key in eval_data:
            metrics[key] = eval_data[key]
    
    # Extract calibration metrics
    for key in ["ece_ctr", "ece_cvr", "ece_ctcvr"]:
        if key in eval_data:
            metrics[key] = eval_data[key]
    
    # Extract other metrics
    for key in ["funnel_consistency", "n_samples"]:
        if key in eval_data:
            metrics[key] = eval_data[key]
    
    return metrics


def generate_comparison_report(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate comparison report and save to file."""
    successful_results = [r for r in results if r["success"]]
    
    if not successful_results:
        logger.error("No successful evaluations to compare")
        return
    
    # Build comparison dataframe
    rows = []
    for result in successful_results:
        exp_name = result["exp_name"]
        metrics = extract_metrics(result["eval_data"])
        row = {"experiment": exp_name, **metrics}
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by experiment name
    df = df.sort_values("experiment")
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("BATCH EVALUATION COMPARISON REPORT")
    report.append("=" * 80)
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Split: {successful_results[0]['split']}")
    report.append(f"Number of experiments: {len(successful_results)}")
    report.append("")
    
    # Key metrics comparison
    report.append("-" * 80)
    report.append("KEY METRICS COMPARISON")
    report.append("-" * 80)
    
    key_metrics = ["auc_ctr", "auc_cvr", "auc_ctcvr", "auc_primary"]
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    if available_metrics:
        comparison_df = df[["experiment"] + available_metrics].copy()
        
        # Find best for each metric and mark with *
        for metric in available_metrics:
            best_idx = comparison_df[metric].idxmax()
            comparison_df.loc[best_idx, metric] = f"{comparison_df.loc[best_idx, metric]:.6f} *"
        
        report.append(comparison_df.to_string(index=False))
    
    report.append("")
    
    # Loss metrics comparison
    report.append("-" * 80)
    report.append("LOSS METRICS COMPARISON")
    report.append("-" * 80)
    
    loss_metrics = ["loss_ctr", "loss_cvr", "loss_ctcvr", "loss_total"]
    available_loss_metrics = [m for m in loss_metrics if m in df.columns]
    
    if available_loss_metrics:
        loss_df = df[["experiment"] + available_loss_metrics].copy()
        
        # Find best (lowest) for each metric and mark with *
        for metric in available_loss_metrics:
            best_idx = loss_df[metric].idxmin()
            loss_df.loc[best_idx, metric] = f"{loss_df.loc[best_idx, metric]:.6f} *"
        
        report.append(loss_df.to_string(index=False))
    
    report.append("")
    
    # Calibration metrics comparison (if available)
    calibration_metrics = ["ece_ctr", "ece_cvr", "ece_ctcvr"]
    available_cal_metrics = [m for m in calibration_metrics if m in df.columns]
    
    if available_cal_metrics:
        report.append("-" * 80)
        report.append("CALIBRATION METRICS COMPARISON (ECE - lower is better)")
        report.append("-" * 80)
        
        cal_df = df[["experiment"] + available_cal_metrics].copy()
        
        # Find best (lowest) for each metric and mark with *
        for metric in available_cal_metrics:
            best_idx = cal_df[metric].idxmin()
            cal_df.loc[best_idx, metric] = f"{cal_df.loc[best_idx, metric]:.6f} *"
        
        report.append(cal_df.to_string(index=False))
        report.append("")
    
    # Summary statistics
    report.append("-" * 80)
    report.append("SUMMARY")
    report.append("-" * 80)
    
    # Find overall best by primary metric
    if "auc_primary" in df.columns:
        best_exp = df.loc[df["auc_primary"].idxmax(), "experiment"]
        best_auc = df["auc_primary"].max()
        report.append(f"Best overall (by auc_primary): {best_exp} ({best_auc:.6f})")
    elif "auc_ctcvr" in df.columns:
        best_exp = df.loc[df["auc_ctcvr"].idxmax(), "experiment"]
        best_auc = df["auc_ctcvr"].max()
        report.append(f"Best overall (by auc_ctcvr): {best_exp} ({best_auc:.6f})")
    
    # Save report
    report_text = "\n".join(report)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info(f"Comparison report saved to: {output_path}")
    print("\n" + report_text)
    
    # Save detailed data as CSV
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed metrics saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for multiple experiment runs"
    )
    parser.add_argument(
        "--runs",
        type=str,
        required=True,
        help="Comma-separated list of run identifiers or directory names",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory containing experiment runs (default: runs)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="Dataset split to evaluate (default: valid)",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Save predictions parquet files",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Max batches for quick evaluation (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison report (default: logs/eval_runs/comparison_TIMESTAMP.txt)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip experiment directory validation",
    )
    
    args = parser.parse_args()
    
    # Parse run identifiers
    run_ids = [r.strip() for r in args.runs.split(",")]
    runs_root = Path(args.runs_root)
    
    logger.info(f"Batch evaluation started")
    logger.info(f"Runs: {run_ids}")
    logger.info(f"Split: {args.split}")
    
    # Find and validate experiment directories
    exp_dirs = []
    for run_id in run_ids:
        exp_dir = find_experiment_dir(run_id, runs_root)
        if exp_dir is None:
            logger.error(f"Could not find experiment directory for: {run_id}")
            continue
        
        if not args.skip_validation and not validate_experiment_dir(exp_dir):
            continue
        
        exp_dirs.append(exp_dir)
        logger.info(f"✓ Found: {exp_dir.name}")
    
    if not exp_dirs:
        logger.error("No valid experiment directories found")
        sys.exit(1)
    
    logger.info(f"\nEvaluating {len(exp_dirs)} experiments...")
    
    # Run evaluations
    results = []
    for i, exp_dir in enumerate(exp_dirs, 1):
        logger.info(f"\n[{i}/{len(exp_dirs)}] Evaluating {exp_dir.name}...")
        result = run_eval(
            exp_dir=exp_dir,
            split=args.split,
            save_preds=args.save_preds,
            max_batches=args.max_batches,
        )
        results.append(result)
    
    # Generate comparison report
    logger.info("\n" + "=" * 80)
    logger.info("Generating comparison report...")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("logs/eval_runs") / f"comparison_{args.split}_{timestamp}.txt"
    
    generate_comparison_report(results, output_path)
    
    # Save results metadata
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "split": args.split,
                "runs": [r["exp_name"] for r in results],
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"Results metadata saved to: {metadata_path}")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    logger.info("\n" + "=" * 80)
    logger.info(f"BATCH EVALUATION COMPLETED")
    logger.info(f"Successful: {successful}/{len(results)}")
    if failed > 0:
        logger.warning(f"Failed: {failed}/{len(results)}")
    logger.info(f"Report: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
