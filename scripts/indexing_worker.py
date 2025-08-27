#!/usr/bin/env python3
"""
InfiniGram Distributed Indexing Worker
Processes assigned data shards and creates indexes
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, check=True, shell=True):
    """Run a shell command and return result"""
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        logger.error(f"Error: {result.stderr}")
        sys.exit(result.returncode)
    return result

def install_dependencies():
    """Install required system and Python dependencies"""
    logger.info("=== Installing Dependencies ===")
    
    # Install system dependencies
    run_command("apt-get update && apt-get install -y curl build-essential cmake pkg-config libprotobuf-dev protobuf-compiler")
    
    # Install GCP CLI
    run_command("curl -sSL https://sdk.cloud.google.com | bash")
    os.environ['PATH'] = f"{os.environ['PATH']}:/root/google-cloud-sdk/bin"
    
    # Install Python dependencies
    os.environ['CFLAGS'] = "-O2"
    os.environ['CXXFLAGS'] = "-O2"
    
    run_command("pip install --upgrade pip setuptools wheel")
    run_command("pip install pandas pyarrow google-cloud-storage tqdm")
    run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    run_command("pip install transformers")
    run_command("pip install sentencepiece --no-cache-dir --verbose")
    run_command("pip install huggingface_hub")
    
    # Verify installations
    logger.info("=== Checking Dependencies ===")
    try:
        import sentencepiece
        logger.info(f"sentencepiece version: {sentencepiece.__version__}")
    except ImportError:
        logger.error("sentencepiece NOT installed")
    
    try:
        import transformers
        logger.info(f"transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("transformers NOT installed")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch NOT installed")
    
    # Test tokenizer
    logger.info("=== Testing GPT-2 Tokenizer ===")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
        logger.info("GPT-2 tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")

def get_worker_id():
    """Calculate worker ID from environment variables"""
    worker_id = os.environ.get('JOB_COMPLETION_INDEX', '0')
    
    if not worker_id or worker_id == '0':
        # Fallback: extract from pod name
        pod_name = os.environ.get('POD_NAME', '')
        import re
        match = re.search(r'(\d+)$', pod_name)
        worker_id = match.group(1) if match else '0'
    
    worker_id = int(worker_id) % 64  # Ensure within range [0, 63]
    
    logger.info(f"Raw JOB_COMPLETION_INDEX: {os.environ.get('JOB_COMPLETION_INDEX', 'not set')}")
    logger.info(f"Raw POD_NAME: {os.environ.get('POD_NAME', 'not set')}")
    logger.info(f"Calculated WORKER_ID: {worker_id}")
    
    return worker_id

def check_shard_data(worker_id):
    """Check if worker's data shard exists and has files"""
    data_shard_dir = f"gs://consus-dataproc/infinigram/ramanujan2_data/stem/4_join/output/nemotron-cc-fineweb-edu-merged/data/{worker_id}"
    index_output_dir = f"gs://consus-dataproc/infinigram/ramanujan2_data/stem/4_join/output/nemotron-cc-fineweb-edu-merged/index/{worker_id}"
    
    logger.info(f"Worker {worker_id} processing shard:")
    logger.info(f"  Input data: {data_shard_dir}")
    logger.info(f"  Output index: {index_output_dir}")
    
    # Check if shard exists and has files
    logger.info(f"Checking worker {worker_id} data shard...")
    
    try:
        result = run_command(f'gsutil ls "{data_shard_dir}/" > /dev/null 2>&1', check=False)
        if result.returncode == 0:
            # Count parquet files
            result = run_command(f'gsutil ls "{data_shard_dir}/*.parquet" 2>/dev/null | wc -l', check=False)
            shard_files = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
            
            logger.info(f"Found {shard_files} parquet files in worker {worker_id} shard")
            
            if shard_files > 0:
                logger.info(f"Worker {worker_id} will process {shard_files} files from shard")
                return True, data_shard_dir, index_output_dir
            else:
                logger.info(f"Worker {worker_id} shard exists but has no parquet files")
                return False, data_shard_dir, index_output_dir
        else:
            logger.info(f"Worker {worker_id} shard directory does not exist")
            return False, data_shard_dir, index_output_dir
    except Exception as e:
        logger.error(f"Error checking shard: {e}")
        return False, data_shard_dir, index_output_dir

def monitor_resources():
    """Monitor system resources"""
    logger.info("=== System Resources ===")
    run_command("df -h /tmp", check=False)
    run_command("free -h", check=False)
    logger.info("========================")
    
    logger.info("=== File System Check ===")
    run_command("ls -la /tmp/", check=False)
    logger.info("========================")

def test_tokenizer():
    """Test tokenizer loading"""
    logger.info("=== Testing Tokenizer ===")
    hf_token = os.environ.get('HF_TOKEN', 'NOT SET')
    logger.info(f"HF_TOKEN: {hf_token}")
    
    try:
        from transformers import AutoTokenizer
        logger.info("Transformers imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import transformers: {e}")
    
    logger.info("=========================")

def process_shard(worker_id, data_shard_dir, index_output_dir):
    """Process worker's data shard"""
    local_dir = f"/tmp/indexing/{worker_id}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Worker {worker_id} downloading shard data...")
    logger.info(f"Downloading from: {data_shard_dir}")
    
    # Download shard data
    run_command(f'gsutil -m cp "{data_shard_dir}/*.parquet" "{local_dir}/"')
    
    # Verify download
    result = run_command(f'ls -1 "{local_dir}/"*.parquet 2>/dev/null | wc -l', check=False)
    local_files = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
    logger.info(f"Downloaded {local_files} parquet files to local storage")
    
    if local_files > 0:
        logger.info(f"Worker {worker_id} starting indexing process...")
        logger.info(f"Local data dir: {local_dir}")
        logger.info(f"Output will be uploaded to: {index_output_dir}")
        
        # Create output directories
        output_dir = f"{local_dir}/output"
        temp_dir = f"{local_dir}/temp"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Run indexing
        indexing_cmd = [
            "python3", "pkg/infini_gram/indexing.py",
            "--data_dir", local_dir,
            "--save_dir", output_dir,
            "--temp_dir", temp_dir,
            "--version", "4",
            "--tokenizer", "gpt2",
            "--token_dtype", "u16",
            "--add_metadata",
            "--add_unigram",
            "--shards", "1",
            "--workers", "1",
            "--worker_id", "0",
            "--batch_size", "65536",
            "--cpus", "25",
            "--mem", "20",
            "--ulimit", "1048576"
        ]
        
        logger.info(f"Running indexing: {' '.join(indexing_cmd)}")
        run_command(' '.join(indexing_cmd))
        
        # Upload results
        logger.info(f"Uploading index files to: {index_output_dir}")
        if Path(output_dir).exists():
            run_command(f'gsutil -m cp -r "{output_dir}/*" "{index_output_dir}/"')
            logger.info("Index files uploaded successfully")
        else:
            logger.warning("No output directory found after indexing")
    else:
        logger.error("Failed to download shard data")
        sys.exit(1)

def create_empty_placeholder(worker_id, index_output_dir):
    """Create empty placeholder for workers with no data"""
    local_dir = f"/tmp/indexing/{worker_id}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Worker {worker_id} has no files to process. Creating empty index placeholder...")
    
    placeholder_file = f"{local_dir}/empty_shard.txt"
    with open(placeholder_file, 'w') as f:
        f.write("empty")
    
    run_command(f'gsutil cp "{placeholder_file}" "{index_output_dir}/"')
    logger.info("Empty shard placeholder created")

def main():
    """Main indexing worker function"""
    logger.info("Starting InfiniGram Indexing Worker")
    
    try:
        # Set up environment
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/gcp-credentials/service-account.json"
        os.environ['PATH'] = f"{os.environ['PATH']}:/usr/local/bin"
        
        # Install dependencies
        install_dependencies()
        
        # Get worker ID
        worker_id = get_worker_id()
        
        # Check shard data
        has_files, data_shard_dir, index_output_dir = check_shard_data(worker_id)
        
        # Monitor resources
        monitor_resources()
        
        # Test tokenizer
        test_tokenizer()
        
        # Process data or create placeholder
        if has_files:
            process_shard(worker_id, data_shard_dir, index_output_dir)
        else:
            create_empty_placeholder(worker_id, index_output_dir)
        
        logger.info(f"Worker {worker_id} completed successfully")
        logger.info(f"Final status: Worker {worker_id} finished processing")
        
    except Exception as e:
        logger.error(f"Worker failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
