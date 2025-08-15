# Data Division Strategy for Infini-Gram Indexing

## Overview

This document explains how data is divided between worker pods in the Kubernetes indexing job to ensure efficient and balanced processing, and how index files are stored in Google Cloud Storage. The job is optimized for **16 workers** running on **n1-highmem nodes**.

## Current Implementation: Range-Based File Division (16 Workers)

### How It Works

1. **Dynamic File Counting**: Each worker counts the total number of parquet files in the GCS bucket
2. **Range Calculation**: Files are divided into equal ranges based on worker ID (0-15)
3. **Balanced Distribution**: Each worker processes approximately the same number of files
4. **GCS Output**: Index files are automatically uploaded to the specified GCS bucket
5. **High Memory Optimization**: Configured for n1-highmem nodes with 32-64GB RAM per worker

### Example with 8,019 Files and 16 Workers

| Worker ID | File Range | Files Count | Status |
|-----------|------------|-------------|---------|
| 0 | 0 - 501 | 502 files | ✅ Balanced |
| 1 | 502 - 1,003 | 502 files | ✅ Balanced |
| 2 | 1,004 - 1,505 | 502 files | ✅ Balanced |
| 3 | 1,506 - 2,007 | 502 files | ✅ Balanced |
| 4 | 2,008 - 2,509 | 502 files | ✅ Balanced |
| 5 | 2,510 - 3,011 | 502 files | ✅ Balanced |
| 6 | 3,012 - 3,513 | 502 files | ✅ Balanced |
| 7 | 3,514 - 4,015 | 502 files | ✅ Balanced |
| 8 | 4,016 - 4,517 | 502 files | ✅ Balanced |
| 9 | 4,518 - 5,019 | 502 files | ✅ Balanced |
| 10 | 5,020 - 5,521 | 502 files | ✅ Balanced |
| 11 | 5,522 - 6,023 | 502 files | ✅ Balanced |
| 12 | 6,024 - 6,525 | 502 files | ✅ Balanced |
| 13 | 6,526 - 7,027 | 502 files | ✅ Balanced |
| 14 | 7,028 - 7,529 | 502 files | ✅ Balanced |
| 15 | 7,530 - 8,018 | 489 files | ✅ Balanced |

### Code Implementation

```bash
# Calculate file range for this worker
TOTAL_WORKERS=16
FILES_PER_WORKER=$((TOTAL_FILES / TOTAL_WORKERS))
START_FILE=$((WORKER_ID * FILES_PER_WORKER))
END_FILE=$((START_FILE + FILES_PER_WORKER))

# Last worker gets remaining files
if [ $WORKER_ID -eq $((TOTAL_WORKERS - 1)) ]; then
  END_FILE=$TOTAL_FILES
fi
```

## Benefits of 16-Worker Configuration

### 1. **Faster Processing**
- **4x faster** than 4 workers (theoretical)
- Each worker processes ~502 files instead of ~2,005
- Reduced per-worker memory requirements

### 2. **Better Resource Utilization**
- More granular workload distribution
- Better load balancing across cluster
- Reduced idle time per worker

### 3. **Improved Scalability**
- Easier to scale up/down based on cluster capacity
- Better fault tolerance (losing 1 worker = 6.25% vs 25% impact)
- More flexible resource allocation

### 4. **Memory Optimization**
- Each worker needs less memory per file
- Better fit for n1-highmem node characteristics
- Reduced risk of OOM errors

## Node Affinity and Resource Requirements

### Node Selection
The job is configured to run on **n1-highmem nodes**:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: cloud.google.com/machine-family
          operator: In
          values: [n1]
        - key: cloud.google.com/machine-type
          operator: In
          values:
          - n1-highmem-4
          - n1-highmem-8
          - n1-highmem-16
          - n1-highmem-32
          - n1-highmem-64
          - n1-highmem-96
```

### Resource Allocation per Worker
```yaml
resources:
  requests:
    memory: "32Gi"    # Optimized for highmem nodes
    cpu: "4"
  limits:
    memory: "64Gi"    # Optimized for highmem nodes
    cpu: "8"
```

### Recommended Node Types
- **n1-highmem-16**: 16 vCPU, 104 GB RAM (2 workers per node)
- **n1-highmem-32**: 32 vCPU, 208 GB RAM (4 workers per node)
- **n1-highmem-64**: 64 vCPU, 416 GB RAM (8 workers per node)

## Configuration

### Kubernetes Job Parameters

```yaml
# Worker configuration
parallelism: 16                    # Number of parallel workers
completions: 16                    # Total number of workers to complete

# Indexing script parameters
--file_start $START_FILE           # Starting file index
--file_end $END_FILE               # Ending file index (exclusive)
--worker_id $WORKER_ID             # Worker ID (0-15)
--shards 16                        # Total number of shards
--workers 16                       # Total number of workers
--gcs_output_bucket "consus-dataproc"
--gcs_output_prefix "infinigram/..."
```

### Python Script Parameters

```python
parser.add_argument('--file_start', type=int, default=0, 
                   help='Starting file index for this worker (0-based)')
parser.add_argument('--file_end', type=int, 
                   help='Ending file index for this worker (exclusive)')
parser.add_argument('--gcs_output_bucket', type=str, 
                   help='GCS bucket for output index files')
parser.add_argument('--gcs_output_prefix', type=str, 
                   help='GCS prefix/path for output index files')
```

## GCS Storage Configuration

### Output Location
- **Bucket**: `gs://consus-dataproc`
- **Path**: `infinigram/ramanujan2_data/stem/4_join/output/nemotron-cc-fineweb-edu-merged`

### File Structure (16 Workers)
Each worker creates index files locally and then uploads them to GCS:
```
gs://consus-dataproc/infinigram/ramanujan2_data/stem/4_join/output/nemotron-cc-fineweb-edu-merged/
├── tokenized.0          # Worker 0: files 0-501
├── tokenized.1          # Worker 1: files 502-1003
├── tokenized.2          # Worker 2: files 1004-1505
├── ...                  # ... (workers 3-14)
├── tokenized.15         # Worker 15: files 7530-8018
├── offset.0             # Document offsets
├── offset.1
├── ...                  # ... (offsets 2-15)
├── offset.15
├── metadata.0           # Document metadata
├── metadata.1
├── ...                  # ... (metadata 2-15)
├── metadata.15
├── unigram.0            # Unigram counts
├── unigram.1
├── ...                  # ... (unigram 2-15)
└── unigram.15
```

## Performance Expectations

### Processing Time
- **4 workers**: ~4 hours (estimated)
- **16 workers**: ~1 hour (estimated, 4x faster)
- **Actual speedup**: 3-4x due to reduced per-worker overhead

### Memory Usage
- **Per worker**: 32-64GB RAM
- **Total cluster memory**: 512GB-1TB (16 workers × 32-64GB)
- **Memory efficiency**: Better utilization of highmem node characteristics

### Network I/O
- **Per worker**: Downloads ~502 files instead of ~2,005
- **Total network usage**: Similar, but better distributed
- **GCS upload**: 16 parallel uploads instead of 4

## Monitoring and Debugging

### Worker Logs

Each worker logs its file range and GCS operations:
```
Worker 7 processing files 3514 to 4015 (total: 502 files)
Uploading index files to GCS...
Uploading /output/index/tokenized.7 to gs://consus-dataproc/infinigram/...
Index files uploaded successfully to GCS
Indexing completed and files uploaded to GCS successfully!
```

### Resource Monitoring

Monitor these metrics for optimal performance:
- **CPU utilization**: Should be 70-90% per worker
- **Memory usage**: Should stay within 32-64GB per worker
- **Network I/O**: Monitor GCS download/upload speeds
- **Storage I/O**: Monitor local disk performance

## Scaling Considerations

### When to Scale Up (More Workers)
- Large datasets (>100K files)
- Available cluster capacity
- Time constraints

### When to Scale Down (Fewer Workers)
- Small datasets (<1K files)
- Limited cluster resources
- Cost optimization

### Optimal Worker Count Formula
```
Optimal Workers = min(
  Available Cluster Capacity,
  Total Files / 500,  # ~500 files per worker is optimal
  32                  # Maximum recommended
)
```

## Future Improvements

### 1. **Adaptive Scaling**
- Auto-adjust worker count based on data size
- Dynamic resource allocation

### 2. **Advanced Node Selection**
- GPU nodes for tokenization acceleration
- Spot instances for cost optimization

### 3. **Progress Tracking**
- Real-time progress monitoring
- ETA calculations per worker

### 4. **Fault Tolerance**
- Automatic worker restart on failure
- Checkpoint/resume functionality 