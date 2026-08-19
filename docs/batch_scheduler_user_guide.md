# Batch Scheduler User Guide

**Version:** 1.0.0  
**Last Updated:** 2026-08-19  
**Status:** Production Ready

## Overview
The FPGA Batch Scheduler optimizes matmul operations across 16 FPGA instances, providing significant TPS improvements for Kimi and matmulfreellm workloads.

## Prerequisites

### Hardware Requirements
- **FPGA Instances**: 16 CXL-based FPGA instances with ternary matmul acceleration
- **GPU**: NVIDIA GPU with CUDA support (tested with RTX PRO 6000, 98GB memory)
- **Memory**: Minimum 16GB system RAM, 32GB+ recommended
- **Storage**: 10GB free disk space

### Software Dependencies
- **Rust**: 1.70+ (for building ZLUDA components)
- **CUDA Toolkit**: 11.0+ (for GPU integration)
- **Python**: 3.8+ (for benchmark scripts)
- **Bash**: 4.0+ (for shell scripts)
- **jq**: 1.5+ (for JSON log parsing)

### System Requirements
- Linux kernel with CXL device support
- Access to CXL device nodes (/dev/cxl_tmatmul*)
- GPU driver installation with CUDA runtime
- Development tools: gcc, make, cmake

## Installation

### 1. Build the Batch Scheduler

```bash
# Navigate to the repository
cd /home/victoryang00/hetGPU

# Build release version with batch scheduler enabled
cargo build --release --features batch_scheduler

# Verify build
ls -la target/release/libzluda.so
```

### 2. Verify Device Access

```bash
# Check CXL devices are available
ls -la /dev/cxl_tmatmul*

# Check GPU availability
nvidia-smi

# Verify device permissions
groups | grep -E "video|render"
```

If device permissions are incorrect:
```bash
# Add user to video group for GPU access
sudo usermod -a -G video $USER

# Add user to render group for CXL device access
sudo usermod -a -G render $USER

# Log out and back in for changes to take effect
```

### 3. Create Log Directory

```bash
# Create stats log directory
sudo mkdir -p /var/log/batch_scheduler/

# Set permissions
sudo chown $USER:$USER /var/log/batch_scheduler/

# Verify permissions
ls -la /var/log/batch_scheduler/
```

### 4. Run Health Check

```bash
# Verify system is ready for batch scheduler
bench/batch_scheduler/health_check.sh

# Expected output: Health Score >= 80%
```

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

### Performance Verification Methods

**Method 1: Direct Benchmarking**
```bash
# Kimi workload verification
KIMI_BITLINEAR_TMATMUL=1 \
BATCH_SCHEDULER_ENABLED=1 \
BATCH_SIZE=64 \
INSTANCE_COUNT=16 \
bench/kimi_k26_tps/run_kimi_k26_tps.sh

# matmulfreellm workload verification  
BATCH_SCHEDULER_ENABLED=1 \
BATCH_SIZE=64 \
INSTANCE_COUNT=16 \
python tests/matmulfreellm_tps_benchmark.py
```

**Method 2: Real-time Monitoring**
```bash
# Monitor live statistics
watch -n 1 'tail -1 /var/log/batch_scheduler_stats.jsonl | jq'
```

**Method 3: Performance Regression Testing**
```bash
# Run comprehensive performance suite
bench/batch_scheduler/performance_test.sh
```

### Additional Performance Metrics
- **Instance Utilization**: Target >85% average across 16 instances
- **Average Latency**: Target <1000μs per operation
- **Fallback Rate**: Target <1% operations falling back to GPU
- **Memory Efficiency**: Target >90% memory allocation reuse

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

## Architecture
The batch scheduler consists of:
- Request Aggregator: Collects individual matmul operations into batches
- Instance Scheduler: Assigns operations to FPGA instances using various policies
- Memory Pipeline: Manages memory transfers and double buffering
- Response Demux: Routes completed operations back to callers
- Health Monitor: Tracks instance health and manages fallback

## Scheduling Policies
- RoundRobin: Distribute operations evenly across instances
- SizeAware: Load balance based on operation sizes
- LoadBalanced: Distribute based on current queue depth
- Adaptive: Smart load balancing (recommended)

## Advanced Configuration
```bash
# Enable prefetching
export PIPELINE_ENABLE_PREFETCH=1

# Enable double buffering
export PIPELINE_ENABLE_DOUBLE_BUFFER=1

# Set health check interval
export HEALTH_CHECK_INTERVAL=100
```

## Error Handling
The batch scheduler includes automatic fallback to GPU execution when FPGA instances are unhealthy or operations fail.