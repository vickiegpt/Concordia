# Batch Scheduler User Guide

**Version:** 1.0.0  
**Last Updated:** 2026-08-19  
**Status:** Production Ready

## Overview
The FPGA Batch Scheduler optimizes matmul operations across 16 FPGA instances, providing significant TPS improvements for Kimi and matmulfreellm workloads.

### Architecture Flow
The batch scheduler follows this execution pipeline:

**Request Aggregator → Instance Scheduler → Memory Pipeline → Response Demux**

- **Request Aggregator**: Collects individual matmul operations into batches
- **Instance Scheduler**: Assigns operations to FPGA instances using various policies  
- **Memory Pipeline**: Manages memory transfers and double buffering
- **Response Demux**: Routes completed operations back to callers

## Quick Start

### Enable Batch Scheduler
```bash
export BATCH_SCHEDULER_ENABLED=1
export BATCH_SIZE=64
export INSTANCE_COUNT=16
```

### Run Kimi IQ1S Hybrid
```bash
KIMI_BITLINEAR_TMATMUL=1 \
BATCH_SCHEDULER_ENABLED=1 \
bench/kimi_k26_tps/run_kimi_k26_tps.sh
```

### Run matmulfreellm
```bash
BATCH_SCHEDULER_ENABLED=1 \
BATCH_SIZE=64 \
INSTANCE_COUNT=16 \
TARGET_TPS=2000 \
python tests/matmulfreellm_tps_benchmark.py
```

## Configuration
- `BATCH_SIZE`: Number of operations per batch (default: 64)
- `INSTANCE_COUNT`: Number of FPGA instances (default: 16)
- `TARGET_TPS`: Target tokens per second

## Performance Targets

### Specific Performance Expectations

**Kimi IQ1S Hybrid Workload:**
- **Baseline**: 0.62 TPS (without batch scheduler)
- **Target**: 8+ TPS (with batch scheduler)
- **Expected Improvement**: 13x throughput increase
- **Verification**: Run `bench/kimi_k26_tps/run_kimi_k26_tps.sh` and measure TPS
- **Acceptance Criteria**: Sustained TPS >= 8.0 for 5+ minutes

**matmulfreellm 2.7B Workload:**
- **Baseline**: 12.85 TPS (without batch scheduler)
- **Target**: 2000+ TPS (with batch scheduler)
- **Expected Improvement**: 155x throughput increase
- **Verification**: Run benchmark with `TARGET_TPS=2000` and measure achieved TPS
- **Acceptance Criteria**: Sustained TPS >= 2000 for 5+ minutes


## Monitoring
Check scheduler statistics:
```bash
# View instance utilization
cat /var/log/batch_scheduler_stats.jsonl | tail -1 | jq '.instance_utilizations'

# View average latency
cat /var/log/batch_scheduler_stats.jsonl | tail -1 | jq '.average_latency_us'

# View fallback count
cat /var/log/batch_scheduler_stats.jsonl | tail -1 | jq '.fallback_count'

# View all recent statistics
tail -10 /var/log/batch_scheduler_stats.jsonl | jq '.'
```

