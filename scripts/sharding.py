#!/usr/bin/env python3
"""
Script to shard parquet files from source location into 64 data directories
"""

import subprocess
import sys
from typing import List
import math

def run_command(cmd: List[str]) -> str:
    """Run a command and return its output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)

def get_all_parquet_files() -> List[str]:
    """Get list of all parquet files from source location"""
    print("Getting list of all parquet files...")
    cmd = [
        "gsutil", "ls", 
        "gs://consus-dataproc/ritvik/ramanujan2_data/stem/4_join/output/nemotron-cc-fineweb-edu-merged/**/*.parquet"
    ]
    output = run_command(cmd)
    files = [line.strip() for line in output.split('\n') if line.strip()]
    print(f"Found {len(files)} parquet files")
    return files
def copy_files_to_shard(files: List[str], shard_id: int):
    """Copy files to a specific shard directory"""
    if not files:
        print(f"Shard {shard_id}: No files to copy")
        return
    
    target_dir = f"gs://consus-dataproc/infinigram/ramanujan2_data/stem/4_join/output/nemotron-cc-fineweb-edu-merged/data/{shard_id}/"
    
    print(f"Shard {shard_id}: Copying {len(files)} files...")
    
    # Use gsutil -m cp for parallel copying
    cmd = ["gsutil", "-m", "cp"] + files + [target_dir]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Shard {shard_id}: Successfully copied {len(files)} files")
    except subprocess.CalledProcessError as e:
        print(f"Shard {shard_id}: Error copying files: {e}")
        sys.exit(1)

def main():
    """Main function to distribute files across shards"""
    TOTAL_SHARDS = 64
    
    # Get all parquet files
    all_files = get_all_parquet_files()
    total_files = len(all_files)
    
    if total_files == 0:
        print("No parquet files found!")
        sys.exit(1)
    
    # Calculate distribution
    files_per_shard = total_files // TOTAL_SHARDS
    remainder = total_files % TOTAL_SHARDS
    
    print(f"\nDistribution plan:")
    print(f"Total files: {total_files}")
    print(f"Total shards: {TOTAL_SHARDS}")
    print(f"Base files per shard: {files_per_shard}")
    print(f"Shards with extra file: {remainder}")
    print(f"Shards 0-{remainder-1}: {files_per_shard + 1} files each")
    print(f"Shards {remainder}-{TOTAL_SHARDS-1}: {files_per_shard} files each")
    print()
    
    # Distribute files
    current_index = 0
    
    for shard_id in range(TOTAL_SHARDS):
        # Calculate number of files for this shard
        if shard_id < remainder:
            files_for_this_shard = files_per_shard + 1
        else:
            files_for_this_shard = files_per_shard
        
        # Skip if no files for this shard
        if files_for_this_shard == 0:
            print(f"Shard {shard_id}: No files assigned")
            continue
        
        # Get files for this shard
        end_index = current_index + files_for_this_shard
        shard_files = all_files[current_index:end_index]
        
        print(f"Shard {shard_id}: Files {current_index} to {end_index-1} ({len(shard_files)} files)")
        
        # Copy files to shard
        copy_files_to_shard(shard_files, shard_id)
        
        current_index = end_index
    
    print(f"\n✅ Distribution complete! All {total_files} files distributed across {TOTAL_SHARDS} shards.")

if __name__ == "__main__":
    main()
