import argparse
from collections import defaultdict
import glob
import gzip
import json
import multiprocessing as mp
import numpy as np
import os
import resource
import shutil
import sys
import time
from tqdm import tqdm
import pandas as pd
from google.cloud import storage

HACK = 100000

tokenizer = None
token_dtype = None
version = None

def upload_index_to_gcs(local_dir, gcs_bucket, gcs_prefix):
    """Upload index files from local directory to GCS bucket"""
    print(f"Uploading index files to gs://{gcs_bucket}/{gcs_prefix}")
    
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    
    # Upload all files in the local directory
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Calculate relative path from local_dir
            rel_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_prefix, rel_path).replace('\\', '/')
            
            print(f"Uploading {local_path} to gs://{gcs_bucket}/{gcs_path}")
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
    
    print("Index files uploaded successfully to GCS")

def load_gcs_parquet_files(gcs_bucket, gcs_prefix, temp_dir, max_files=None, file_start=0, file_end=None):
    """Download parquet files from GCS bucket to temporary local directory"""
    print(f"Loading parquet files from gs://{gcs_bucket}/{gcs_prefix}")
    
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    
    # List all parquet files in the bucket
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    parquet_blobs = [blob for blob in blobs if blob.name.endswith('.parquet')]
    
    print(f"Found {len(parquet_blobs)} total parquet files")
    print(f"First 5 parquet files:")
    for i, blob in enumerate(parquet_blobs[:5]):
        print(f"  {i}: {blob.name}")
    
    # Apply file range filtering
    if file_end is None:
        file_end = len(parquet_blobs)
    
    # Filter to only the files this worker should process
    worker_files = parquet_blobs[file_start:file_end]
    
    if max_files:
        worker_files = worker_files[:max_files]
    
    print(f"Worker processing files {file_start} to {file_end-1} ({len(worker_files)} files)")
    
    # Download files to temp directory
    local_files = []
    for i, blob in enumerate(tqdm(worker_files, desc="Downloading files")):
        local_path = os.path.join(temp_dir, f"part_{file_start + i:05d}.parquet")
        print(f"Downloading {blob.name} to {local_path}")
        blob.download_to_filename(local_path)
        local_files.append(local_path)
    
    return local_files

def load_parquet_file(path):
    """Load a single parquet file and extract text content"""
    try:
        print(f"Loading parquet file: {path}")
        df = pd.read_parquet(path)
        print(f"Parquet file shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Extract text column and convert to list of strings
        if 'text' not in df.columns:
            print(f"Warning: 'text' column not found in {path}. Available columns: {df.columns.tolist()}")
            return []
            
        texts = df['text'].dropna().astype(str).tolist()
        print(f"Extracted {len(texts)} text entries from {path}")
        return texts
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []

def load_file(path):
    if path.endswith('.parquet'):
        return load_parquet_file(path)
    elif path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            lines = f.readlines()
    elif path.endswith('.zst'):
        with open(path, 'rb') as f:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                decompressed_data = reader.read().decode('utf-8')
            lines = decompressed_data.split('\n')
            if lines[-1] == '':
                lines = lines[:-1]
    elif path.endswith('.jsonl'):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
    else:
        raise ValueError(f'Unknown file type: {path}')
    return lines

def tok(line):
    global tokenizer, token_dtype, version
    # For parquet data, line is already the text string
    if isinstance(line, str):
        text = line
        metadata = {'path': 'parquet_source', 'linenum': 0}
    else:
        # Handle legacy JSON format
        metadata = json.loads(line.strip('\n'))
        text = metadata['text']
    
    if tokenizer is None:
        byte_arr = text.encode('utf-8')
        if version == 5:
            byte_arr = byte_arr[::-1].copy()
    else:
        text_tokens = tokenizer.encode(text)
        if version == 5:
            text_tokens = text_tokens[::-1].copy()
        byte_arr = np.array(text_tokens, dtype=token_dtype).view(np.uint8).tobytes()
    
    if 'text' in metadata:
        del metadata['text']
    return byte_arr, metadata

def tokenize(args):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    od_paths = [os.path.join(args.save_dir, f'offset.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    mt_paths = [os.path.join(args.save_dir, f'metadata.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    om_paths = [os.path.join(args.save_dir, f'metaoff.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    ug_paths = [os.path.join(args.save_dir, f'unigram.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    if all([os.path.exists(ds_path) for ds_path in ds_paths]) \
        and all([os.path.exists(od_path) for od_path in od_paths]):
        print('Step 1 (tokenize): Skipped. All tokenized files already exist.')
        return

    print('Step 1 (tokenize): Starting ...')

    import transformers
    transformers.utils.logging.set_verbosity(40) # suppress warnings
    global tokenizer, token_dtype
    if args.tokenizer is None:
        tokenizer = None
    elif args.tokenizer == 'gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    elif args.tokenizer == 'olmo':
        tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False)
        # # The following is a faster version, but the result is a bit different
        # from dolma.tokenizer import Tokenizer
        # tokenizer = Tokenizer.from_pretrained('allenai/gpt-neox-olmo-dolma-v1_5', bos_token_id=None, eos_token_id=None, pad_token_id=1, segment_before_tokenization=True)
    else:
        raise ValueError(f'Unknown tokenizer: {args.tokenizer}')

    # Look for both parquet and json files
    data_paths = glob.glob(f'{args.data_dir}/**/*.parquet', recursive=True) + glob.glob(f'{args.data_dir}/**/*.json*', recursive=True)
    data_paths = list(sorted(data_paths))
    print(f"Found {len(data_paths)} data files to process")
    
    # Open file handles for the shards this worker is responsible for
    print(f"Worker {args.worker_id} responsible for shards: {list(range(args.worker_id, args.shards, args.workers))}")
    ds_fouts = [open(ds_path, 'wb') for ds_path in ds_paths]
    od_fouts = [open(od_path, 'wb') for od_path in od_paths]
    if args.add_metadata:
        mt_fouts = [open(mt_path, 'w') for mt_path in mt_paths]
        om_fouts = [open(om_path, 'wb') for om_path in om_paths]
    if args.add_unigram:
        ug_fouts = [open(ug_path, 'w') for ug_path in ug_paths]
        unigram_counts = [defaultdict(int) for ug_path in ug_paths]
    with mp.get_context('fork').Pool(args.cpus) as p:
        ods = [0 for _ in od_fouts]
        if args.add_metadata:
            oms = [0 for _ in om_fouts]
        for data_path in tqdm(data_paths):
            rel_data_path = data_path[len(args.data_dir)+1:]
            print(f"Processing file: {data_path}")
            lines = load_file(data_path)
            print(f"Loaded {len(lines)} lines from {data_path}")
            if len(lines) == 0:
                print(f"Warning: {data_path} has no lines, skipping...")
                continue
            for offset in tqdm(range(0, len(lines), args.workers*args.batch_size), total=len(range(0, len(lines), args.workers*args.batch_size))):
                batch_lines = lines[(offset+args.worker_id):(offset+args.workers*args.batch_size):args.workers]
                print(f"  Processing batch: offset={offset}, worker_id={args.worker_id}, batch_size={len(batch_lines)}")
                if len(batch_lines) == 0:
                    print(f"  Warning: Empty batch at offset {offset}, skipping...")
                    continue
                results = p.map(tok, batch_lines)
                for i, (byte_arr, metadata) in enumerate(results):
                    content = args.doc_sep + byte_arr
                    j = i % (args.shards // args.workers)
                    ds_fouts[j].write(content)
                    od_fouts[j].write(np.array([ods[j]], dtype=np.uint64).view(np.uint8).tobytes())
                    ods[j] += len(content)
                    if args.add_metadata:
                        linenum = (offset + args.worker_id) + args.workers * i
                        mt = json.dumps({'path': rel_data_path, 'linenum': linenum, 'metadata': metadata}) + '\n'
                        mt_fouts[j].write(mt)
                        om_fouts[j].write(np.array([oms[j]], dtype=np.uint64).view(np.uint8).tobytes())
                        oms[j] += len(mt)
                    if args.add_unigram:
                        token_ids = np.frombuffer(content, dtype=np.uint8).view(token_dtype)
                        for token_id in token_ids:
                            unigram_counts[j][token_id] += 1
            del lines

    for ds_fout in ds_fouts:
        ds_fout.close()
    for od_fout in od_fouts:
        od_fout.close()
    if args.add_metadata:
        for mt_fout in mt_fouts:
            mt_fout.close()
        for om_fout in om_fouts:
            om_fout.close()
    if args.add_unigram:
        for j, ug_fout in enumerate(ug_fouts):
            for token_id, count in sorted(unigram_counts[j].items()):
                ug_fout.write(f'{token_id} {count}\n')
            ug_fout.close()

    for ds_path in ds_paths:
        if os.path.getsize(ds_path) == 0:
            print(f'{ds_path} is empty. Please make sure the documents exist!', flush=True)
            exit(1)

def build_sa(args):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]

    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    for t, ds_path in enumerate(ds_paths):
        print(f'Shard {t} / {len(ds_paths)}', flush=True)

        sa_path = ds_path.replace('tokenized', 'table')
        if os.path.exists(sa_path):
            print(f'Step 2 (build_sa): Skipped. File already exists.', flush=True)
            continue

        print('Step 2 (build_sa): Starting ...', flush=True)
        start_time_all = time.time()

        # -------- Step 2.1 (make-part) -------- #

        print(f'\tStep 2.1 (make-part): Starting ...', flush=True)
        start_time = time.time()

        ds_size = os.path.getsize(ds_path)
        if ds_size < args.cpus * args.token_width + HACK:
            print(f'{ds_path} is too small to parallelize. Please use fewer CPUs!', flush=True)
            exit(1)
        ratio = int(np.ceil(np.log2(ds_size) / 8))
        mem_bytes = args.mem * 1024**3
        num_job_batches = 1
        while num_job_batches * (mem_bytes // (12 if args.token_width == 1 else 8)) < ds_size:
            num_job_batches *= 2
        parallel_jobs = args.cpus
        total_jobs = num_job_batches * parallel_jobs
        print(f'Using {num_job_batches} batches of {parallel_jobs} jobs each, for a total of {total_jobs} jobs.', flush=True)

        S = ds_size // total_jobs
        # Make sure that parts contain whole tokens
        if S % args.token_width != 0:
            S += args.token_width - S % args.token_width

        parts_dir = os.path.join(args.temp_dir, f'parts-{args.worker_id}')
        shutil.rmtree(parts_dir, ignore_errors=True)
        os.makedirs(parts_dir)

        for batch_start in tqdm(list(range(0, total_jobs, parallel_jobs))):
            batch_end = min(batch_start+parallel_jobs, total_jobs)
            batch_ranges = []
            for i in range(batch_start, batch_end):
                s, e = i*S, min((i+1)*S+HACK, ds_size)
                batch_ranges.append((s, e))
            pipes = []
            for (s, e) in batch_ranges:
                pipes.append(os.popen(f'./rust_indexing make-part --data-file {ds_path} --parts-dir {parts_dir} --start-byte {s} --end-byte {e} --ratio {ratio} --token-width {args.token_width}'))
            [pipe.read() for pipe in pipes]
            if any([pipe.close() is not None for pipe in pipes]):
                print('\tStep 2.1 (make-part): Something went wrong', flush=True)
                exit(1)

        end_time = time.time()
        print(f'\tStep 2.1 (make-part): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        # -------- Step 2.2 (merge) -------- #

        print(f'\tStep 2.2 (merge): Starting ...', flush=True)
        start_time = time.time()

        merged_dir = os.path.join(args.temp_dir, f'merged-{args.worker_id}')
        shutil.rmtree(merged_dir, ignore_errors=True)
        os.makedirs(merged_dir)

        pipe = os.popen(f'./rust_indexing merge --data-file {ds_path} --parts-dir {parts_dir} --merged-dir {merged_dir} --num-threads {args.cpus} --hacksize {HACK} --ratio {ratio} --token-width {args.token_width}')
        pipe.read()
        if pipe.close() is not None:
            print('\tStep 2.2 (merge): Something went wrong', flush=True)
            exit(1)

        shutil.rmtree(parts_dir)

        end_time = time.time()
        print(f'\tStep 2.2 (merge): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        # -------- Step 2.3 (concat) -------- #

        print(f'\tStep 2.3 (concat): Starting ...', flush=True)
        start_time = time.time()

        pipe = os.popen(f'./rust_indexing concat --data-file {ds_path} --merged-dir {merged_dir} --merged-file {sa_path} --num-threads {args.cpus} --ratio {ratio} --token-width {args.token_width}')
        pipe.read()
        if pipe.close() is not None:
            print('\tStep 2.3 (concat): Something went wrong', flush=True)
            exit(1)

        shutil.rmtree(merged_dir)

        end_time = time.time()
        print(f'\tStep 2.3 (concat): Done. Took {end_time-start_time:.2f} seconds', flush=True)

        end_time_all = time.time()
        print(f'Step 2 (build_sa): Done. Took {end_time_all-start_time_all:.2f} seconds', flush=True)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory containing the raw text corpus. Must be absolute path.')
    parser.add_argument('--gcs_bucket', type=str, help='GCS bucket name for input data')
    parser.add_argument('--gcs_prefix', type=str, help='GCS prefix/path within bucket for input data')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory where temporary indexing files are stored. Must be absolute path.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where the final index files are stored. Must be absolute path.')
    parser.add_argument('--version', type=int, default=4, choices=[4, 5], help='Version of the index.')
    parser.add_argument('--tokenizer', type=str, default=None, choices=[None, 'gpt2', 'llama', 'olmo'])
    parser.add_argument('--token_dtype', type=str, default='u16', choices=['u8', 'u16', 'u32'], help='Data type for tokens.')
    parser.add_argument('--add_metadata', default=False, action='store_true', help='Whether to store document metadata in the index.')
    parser.add_argument('--add_unigram', default=False, action='store_true', help='Whether to precompute unigram counts.')
    parser.add_argument('--shards', type=int, default=1, help='Number of shards to split the index into.')
    parser.add_argument('--workers', type=int, default=1, help='Total number of workers. Must be a divisor of shards.')
    parser.add_argument('--worker_id', type=int, default=0, help='The worker ID of this process. Must be in range [0, workers).')
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size for tokenization.')
    parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Number of CPU cores available to the program.')
    parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GiB available to the program.')
    parser.add_argument('--ulimit', type=int, default=1048576, help='Maximum number of open files allowed.')
    parser.add_argument('--max_files', type=int, help='Maximum number of parquet files to process (for testing)')
    parser.add_argument('--file_start', type=int, default=0, help='Starting file index for this worker (0-based)')
    parser.add_argument('--file_end', type=int, help='Ending file index for this worker (exclusive)')
    parser.add_argument('--gcs_output_bucket', type=str, help='GCS bucket for output index files')
    parser.add_argument('--gcs_output_prefix', type=str, help='GCS prefix/path for output index files')
    args = parser.parse_args()

    # Validate input source
    if not args.data_dir and not (args.gcs_bucket and args.gcs_prefix):
        parser.error("Either --data_dir or both --gcs_bucket and --gcs_prefix must be specified")

    if args.temp_dir is None:
        args.temp_dir = args.save_dir
    if args.data_dir:
        args.data_dir = args.data_dir.rstrip('/')
    args.temp_dir = args.temp_dir.rstrip('/')
    args.save_dir = args.save_dir.rstrip('/')

    print(f"Validation: batch_size={args.batch_size}, cpus={args.cpus}, shards={args.shards}, workers={args.workers}, worker_id={args.worker_id}")
    assert args.batch_size > 0
    assert args.cpus > 0
    assert args.shards > 0
    assert args.workers > 0
    assert 0 <= args.worker_id < args.workers
    assert args.shards % args.workers == 0
    print(f"Validation passed: worker_id {args.worker_id} is valid for {args.workers} workers")
    
    # Validate file range parameters
    if args.file_start < 0:
        parser.error("--file_start must be non-negative")
    if args.file_end is not None and args.file_end <= args.file_start:
        parser.error("--file_end must be greater than --file_start")

    global token_dtype, version
    if args.token_dtype == 'u8':
        token_dtype = np.uint8
        args.token_width = 1
        args.doc_sep = b'\xff'
    elif args.token_dtype == 'u16':
        token_dtype = np.uint16
        args.token_width = 2
        args.doc_sep = b'\xff\xff'
    elif args.token_dtype == 'u32':
        token_dtype = np.uint32
        args.token_width = 4
        args.doc_sep = b'\xff\xff\xff\xff'
    else:
        raise ValueError(f'Unknown token_dtype: {args.token_dtype}')
    version = args.version

    # Handle GCS input first
    if args.gcs_bucket and args.gcs_prefix:
        print(f"Using GCS input: gs://{args.gcs_bucket}/{args.gcs_prefix}")
        # Create a temporary directory for downloaded files
        gcs_temp_dir = os.path.join(args.temp_dir, 'gcs_downloads')
        os.makedirs(gcs_temp_dir, exist_ok=True)
        
        # Download parquet files from GCS
        local_files = load_gcs_parquet_files(args.gcs_bucket, args.gcs_prefix, gcs_temp_dir, args.max_files, args.file_start, args.file_end)
        
        # Update data_dir to point to local downloaded files
        args.data_dir = gcs_temp_dir
        print(f"Downloaded {len(local_files)} files to {gcs_temp_dir}")
        
        # Check if this worker has any files to process
        if len(local_files) == 0:
            print(f"Warning: Worker {args.worker_id} has no files to process (file_start={args.file_start}, file_end={args.file_end})")
            print("This might happen if there are more workers than files. Creating empty index files...")
            # Create empty index files for this worker's shards
            for i in range(args.worker_id, args.shards, args.workers):
                shard_path = os.path.join(args.save_dir, f'tokenized.{i}')
                with open(shard_path, 'wb') as f:
                    pass  # Create empty file
                offset_path = os.path.join(args.save_dir, f'offset.{i}')
                with open(offset_path, 'wb') as f:
                    pass  # Create empty file
            print("Empty index files created. Exiting.")
            return
    
    # Now validate data_dir exists (either original or downloaded from GCS)
    assert os.path.exists(args.data_dir), f"Data directory {args.data_dir} does not exist"
    
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    assert sys.byteorder == 'little'
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.ulimit, args.ulimit))

    tokenize(args)
    build_sa(args)
    
    # Upload index files to GCS if output bucket is specified
    if args.gcs_output_bucket and args.gcs_output_prefix:
        print("Uploading index files to GCS...")
        upload_index_to_gcs(args.save_dir, args.gcs_output_bucket, args.gcs_output_prefix)
        print("Indexing completed and files uploaded to GCS successfully!")
    else:
        print("Indexing completed successfully!")

if __name__ == '__main__':
    main()
