#!/usr/bin/env python
"""
Fast parquet file merger using PyArrow's dataset API and partitioned writes.
Merges 402 small files into 50 large files with optimal performance.
"""

import sys
from pathlib import Path
import shutil
import logging
import gc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
except ImportError:
    print("ERROR: pyarrow required. Install via: pip install pyarrow")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_fast(
    input_dir: Path,
    output_dir: Path,
    target_parts: int = 50,
) -> dict:
    """
    Fast merge using dataset API and row-group optimization.
    """
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all parquet files
    part_files = sorted(input_dir.glob("part-*.parquet"))
    logger.info(f"Found {len(part_files)} input files")
    
    if not part_files:
        return {"success": False, "error": "No parquet files found"}
    
    # Prepare output dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Scanning file sizes...")
    total_rows = 0
    for idx, pf_path in enumerate(part_files):
        pf = pq.ParquetFile(pf_path)
        total_rows += pf.metadata.num_rows
        if (idx + 1) % 100 == 0:
            logger.info(f"  Scanned {idx + 1}/{len(part_files)}")
    
    logger.info(f"Total rows: {total_rows:,}")
    rows_per_file = total_rows // target_parts
    logger.info(f"Target rows per file: {rows_per_file:,}")
    
    # Read all data and partition
    logger.info("Reading all input files...")
    all_tables = []
    total_read = 0
    
    for idx, pf_path in enumerate(part_files):
        table = pq.read_table(pf_path)
        all_tables.append(table)
        total_read += table.num_rows
        
        if (idx + 1) % 50 == 0:
            logger.info(f"  Read {idx + 1}/{len(part_files)}, {total_read:,} rows")
    
    logger.info(f"Concatenating {len(all_tables)} tables...")
    combined_table = pa.concat_tables(all_tables)
    del all_tables
    gc.collect()
    
    logger.info(f"Combined table: {combined_table.num_rows:,} rows")
    
    # Write in partitions with optimized row group size
    logger.info(f"Writing {target_parts} output files...")
    rows_written = 0
    output_files = []
    
    for part_id in range(target_parts):
        start_row = (part_id * rows_per_file)
        if part_id == target_parts - 1:
            # Last file gets remaining rows
            end_row = combined_table.num_rows
        else:
            end_row = ((part_id + 1) * rows_per_file)
        
        partition_table = combined_table.slice(start_row, end_row - start_row)
        part_rows = partition_table.num_rows
        
        output_file = output_dir / f"part-{part_id:05d}.parquet"
        
        # Write with optimized row group size (larger = fewer files written)
        pq.write_table(
            partition_table,
            output_file,
            compression="snappy",
            row_group_size=min(100000, part_rows),  # Optimize for large batches
        )
        
        rows_written += part_rows
        output_files.append(output_file)
        
        if (part_id + 1) % 10 == 0 or part_id == target_parts - 1:
            logger.info(f"  Written {part_id + 1}/{target_parts}: {part_rows:,} rows")
    
    logger.info(f"Total written: {rows_written:,} rows")
    
    return {
        "success": True,
        "input_files": len(part_files),
        "output_files": len(output_files),
        "total_rows": rows_written,
        "output_dir": str(output_dir),
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fast merge of parquet files"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid"],
        default="train",
        help="Which split to merge"
    )
    parser.add_argument(
        "--target-parts",
        type=int,
        default=50,
        help="Number of output files"
    )
    
    args = parser.parse_args()
    
    split_dir = args.data_dir / args.split
    backup_dir = args.data_dir / f"{args.split}_backup_old"
    temp_dir = args.data_dir / f".merge_{args.split}_temp"
    
    if not split_dir.exists():
        logger.error(f"Not found: {split_dir}")
        return 1
    
    logger.info(f"Input:  {split_dir}")
    logger.info(f"Backup: {backup_dir}")
    logger.info(f"Temp:   {temp_dir}")
    
    try:
        logger.info("\nStarting merge...")
        result = merge_fast(split_dir, temp_dir, args.target_parts)
        
        if not result["success"]:
            logger.error(f"Merge failed: {result.get('error')}")
            return 1
        
        logger.info(f"Merge successful: {result['input_files']} -> {result['output_files']} files")
        
        # Swap: move original to backup, move temp to original
        logger.info(f"\nReplacing original with merged files...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        
        split_dir.rename(backup_dir)
        logger.info(f"Backed up original to {backup_dir}")
        
        temp_dir.rename(split_dir)
        logger.info(f"Moved merged files to {split_dir}")
        
        # Clean up backup
        logger.info(f"Removing backup...")
        shutil.rmtree(backup_dir)
        
        logger.info("\n✓ Merge complete!")
        logger.info(f"  Output: {result['output_files']} files")
        logger.info(f"  Rows: {result['total_rows']:,}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        # Try to cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return 1


if __name__ == "__main__":
    sys.exit(main())
