#!/usr/bin/env python
"""
Merge many small parquet files into fewer large files (in-place).

Combines 402 small part files into 50 large files for better I/O performance during training.
Original files are deleted after successful merge.
"""

import sys
from pathlib import Path
import shutil
import tempfile
import logging

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. Install via: pip install pyarrow")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_parquet_files(
    input_dir: Path,
    target_parts: int = 50,
    split: str = "train",
    dry_run: bool = False,
    batch_size_rows: int = 50000,
) -> dict:
    """
    Merge many small parquet files into fewer large files.
    
    Args:
        input_dir: Directory containing part-*.parquet files
        target_parts: Number of output files to create
        split: split name (train/valid) for logging
        dry_run: If True, only report what would be done
        batch_size_rows: Process in batches of this many rows (memory efficient)
    
    Returns:
        Statistics dict with merge info
    """
    
    input_dir = Path(input_dir)
    if not input_dir.exists():
        logger.error(f"Directory not found: {input_dir}")
        return {"success": False, "error": f"Directory not found: {input_dir}"}
    
    # Find all parquet files
    part_files = sorted(input_dir.glob("part-*.parquet"))
    if not part_files:
        logger.error(f"No parquet files found in {input_dir}")
        return {"success": False, "error": "No parquet files found"}
    
    logger.info(f"Found {len(part_files)} input files in {input_dir}")
    logger.info(f"Target: {target_parts} output files")
    
    # Calculate total rows (use cached metadata for speed)
    total_rows = 0
    row_counts = []
    logger.info("Scanning input files for row counts...")
    for idx, pf_path in enumerate(part_files):
        try:
            pf = pq.ParquetFile(pf_path)
            num_rows = pf.metadata.num_rows
            row_counts.append((pf_path, num_rows))
            total_rows += num_rows
            if (idx + 1) % 50 == 0:
                logger.info(f"  Scanned {idx + 1}/{len(part_files)}")
        except Exception as e:
            logger.error(f"Failed to read {pf_path}: {e}")
            return {"success": False, "error": f"Failed to read {pf_path}"}
    
    logger.info(f"Total rows: {total_rows:,}")
    rows_per_output = total_rows // target_parts
    logger.info(f"Target rows per output file: {rows_per_output:,}")
    logger.info(f"Batch processing size: {batch_size_rows:,} rows (memory efficient)")
    
    if dry_run:
        logger.info("[DRY RUN] Would merge into following structure:")
        for i in range(target_parts):
            logger.info(f"  part-{i:05d}.parquet")
        return {
            "success": True,
            "dry_run": True,
            "input_files": len(part_files),
            "total_rows": total_rows,
            "output_files": target_parts,
            "rows_per_file": rows_per_output,
            "message": f"[DRY RUN] Would merge {len(part_files)} files into {target_parts} files"
        }
    
    # Use temp directory for atomic operation
    temp_dir = input_dir.parent / f".merge_temp_{split}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using temp directory: {temp_dir}")
    
    try:
        output_writers = {}  # output_id -> ParquetWriter
        accumulated_rows = {}  # output_id -> row_count
        output_schemas = {}  # output_id -> schema
        current_output_id = 0
        total_rows_processed = 0
        rows_written_per_output = []
        schema = None
        
        logger.info("Starting merge process (streaming with batches)...")
        
        # Streaming write with memory-efficient batching
        for input_idx, (pf_path, num_rows) in enumerate(row_counts):
            try:
                pf = pq.ParquetFile(pf_path)
                if schema is None:
                    schema = pf.schema_arrow
                
                # Read in chunks to avoid memory spike
                for batch in pf.iter_batches(batch_size=batch_size_rows):
                    table = pa.Table.from_batches([batch])
                    batch_rows = table.num_rows
                    
                    # Determine which output file(s) this batch goes to
                    batch_offset = 0
                    while batch_offset < batch_rows:
                        remaining_in_current = rows_per_output - accumulated_rows.get(current_output_id, 0)
                        rows_to_write = min(batch_rows - batch_offset, remaining_in_current)
                        
                        # Extract slice
                        slice_table = table.slice(batch_offset, rows_to_write)
                        
                        # Get or create writer for this output
                        if current_output_id not in output_writers:
                            output_path = temp_dir / f"part-{current_output_id:05d}.parquet"
                            output_writers[current_output_id] = pq.ParquetWriter(
                                output_path, schema, compression="snappy"
                            )
                            accumulated_rows[current_output_id] = 0
                        
                        # Write
                        output_writers[current_output_id].write_table(slice_table)
                        accumulated_rows[current_output_id] += rows_to_write
                        batch_offset += rows_to_write
                        total_rows_processed += rows_to_write
                        
                        # Check if current output is full (but not the last file yet)
                        if accumulated_rows[current_output_id] >= rows_per_output:
                            # Only advance to next file if not already at the last file
                            if current_output_id < target_parts - 1:
                                writer = output_writers[current_output_id]
                                writer.close()
                                rows_written_per_output.append(accumulated_rows[current_output_id])
                                logger.info(f"  Completed output {current_output_id}: {accumulated_rows[current_output_id]:,} rows")
                                del output_writers[current_output_id]
                                current_output_id += 1
                
                # Progress logging
                if (input_idx + 1) % 50 == 0:
                    logger.info(f"  Processed {input_idx + 1}/{len(part_files)} input files, {total_rows_processed:,} rows written")
                    
            except Exception as e:
                logger.error(f"Error processing {pf_path}: {e}")
                # Close all open writers
                for w in output_writers.values():
                    w.close()
                shutil.rmtree(temp_dir)
                raise
        
        # Close remaining writers
        for output_id in sorted(output_writers.keys()):
            writer = output_writers[output_id]
            writer.close()
            rows_written_per_output.append(accumulated_rows[output_id])
            logger.info(f"  Completed output {output_id}: {accumulated_rows[output_id]:,} rows (final)")
        
        logger.info(f"Merge streaming complete: {len(rows_written_per_output)} output files created")
        logger.info(f"Total rows written: {total_rows_processed:,}")
        
        # Verify output
        output_files = sorted(temp_dir.glob("part-*.parquet"))
        total_output_rows = 0
        for out_path in output_files:
            pf = pq.ParquetFile(out_path)
            total_output_rows += pf.metadata.num_rows
        
        if total_output_rows != total_rows:
            logger.error(f"Row count mismatch: {total_output_rows} != {total_rows}")
            shutil.rmtree(temp_dir)
            return {"success": False, "error": "Row count mismatch after merge"}
        
        logger.info(f"Verification passed: {total_output_rows:,} rows in output")
        
        # Backup and replace
        backup_dir = input_dir.parent / f"{split}_backup_old"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        logger.info(f"Backing up original files to {backup_dir}...")
        input_dir.rename(backup_dir)
        temp_dir.rename(input_dir)
        logger.info("Replacement complete!")
        
        # Clean up backup after successful completion
        logger.info(f"Removing backup {backup_dir}...")
        shutil.rmtree(backup_dir)
        
        return {
            "success": True,
            "input_files": len(part_files),
            "output_files": len(rows_written_per_output),
            "total_rows": total_output_rows,
            "rows_per_output": rows_written_per_output,
            "message": f"Successfully merged {len(part_files)} files into {len(rows_written_per_output)} files"
        }
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        # Restore from backup if exists
        if backup_dir.exists() and temp_dir.exists():
            logger.info("Restoring from backup...")
            shutil.rmtree(temp_dir)
            backup_dir.rename(input_dir)
        return {"success": False, "error": str(e)}


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge small parquet files into larger chunks for better I/O performance"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data directory containing train/valid subdirs"
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "both"],
        default="both",
        help="Which split to merge"
    )
    parser.add_argument(
        "--target-parts",
        type=int,
        default=50,
        help="Number of output parquet files to create"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done without modifying files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for streaming read (memory efficient, default: 50000 rows)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    splits_to_merge = ["train", "valid"] if args.split == "both" else [args.split]
    
    logger.info(f"Parquet Merge Tool")
    logger.info(f"==================")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Splits: {splits_to_merge}")
    logger.info(f"Target parts: {args.target_parts}")
    if args.dry_run:
        logger.info(f"Mode: DRY RUN (no changes)")
    logger.info("")
    
    for split in splits_to_merge:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*60}")
        
        result = merge_parquet_files(
            split_dir,
            target_parts=args.target_parts,
            split=split,
            dry_run=args.dry_run,
            batch_size_rows=args.batch_size,
        )
        
        if result["success"]:
            logger.info(f"✓ {split}: {result['message']}")
            if "rows_per_output" in result:
                min_rows = min(result["rows_per_output"])
                max_rows = max(result["rows_per_output"])
                avg_rows = sum(result["rows_per_output"]) / len(result["rows_per_output"])
                logger.info(f"  Output file stats:")
                logger.info(f"    Min rows: {min_rows:,}")
                logger.info(f"    Max rows: {max_rows:,}")
                logger.info(f"    Avg rows: {avg_rows:,.0f}")
        else:
            logger.error(f"✗ {split}: {result.get('error', 'Unknown error')}")
            return 1
    
    logger.info(f"\n{'='*60}")
    logger.info("All done!")
    logger.info(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
