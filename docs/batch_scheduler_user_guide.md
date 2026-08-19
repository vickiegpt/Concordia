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

### Scheduling Policies
The Instance Scheduler supports multiple load-balancing strategies:
- **RoundRobin**: Distribute operations evenly across instances
- **SizeAware**: Load balance based on operation sizes
- **LoadBalanced**: Distribute based on current queue depth
- **Adaptive**: Smart load balancing (recommended for production)

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

### Verification Steps
```bash
# Check CXL devices are available
ls -la /dev/cxl_tmatmul*

# Check GPU availability
nvidia-smi

# Verify device permissions
groups | grep -E "video|render"
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

### Core Parameters
- **`BATCH_SCHEDULER_ENABLED`**: Enable/disable batch scheduler (default: 0, range: 0-1)
  - Impact: Controls whether batch scheduling is active
  - Production: Set to 1 for production workloads
  
- **`BATCH_SIZE`**: Number of operations per batch (default: 64, range: 8-256)
  - Impact: Larger batches improve throughput but increase memory usage
  - Production: Start with 64, tune based on memory availability
  
- **`INSTANCE_COUNT`**: Number of FPGA instances (default: 16, range: 1-16)
  - Impact: More instances provide parallel processing capacity
  - Production: Set to match available FPGA hardware
  
- **`TARGET_TPS`**: Target tokens per second (default: 1000, range: 100-5000)
  - Impact: Guides scheduler optimization goals
  - Production: Set based on workload requirements

### Memory Configuration
- **`PIPELINE_ENABLE_PREFETCH`**: Enable memory prefetching (default: 1, range: 0-1)
  - Impact: Improves throughput by preparing data in advance
  - Production: Keep enabled unless memory constrained
  
- **`PIPELINE_ENABLE_DOUBLE_BUFFER`**: Enable double buffering (default: 1, range: 0-1)
  - Impact: Hides memory transfer latency
  - Production: Enable for better performance
  
- **`PIPELINE_ENABLE_POOLING`**: Enable memory pooling (default: 1, range: 0-1)
  - Impact: Reuses memory allocations to reduce overhead
  - Production: Enable for reduced memory fragmentation

### Timeout Configuration
- **`HEALTH_CHECK_INTERVAL`**: Health check interval in milliseconds (default: 100, range: 50-1000)
  - Impact: Frequency of instance health monitoring
  - Production: Lower values catch issues faster but increase overhead
  
- **`OPERATION_TIMEOUT_MS`**: Operation timeout in milliseconds (default: 5000, range: 1000-30000)
  - Impact: Maximum time before operation is considered failed
  - Production: Increase for slower workloads

### Scheduling Configuration
- **`SCHEDULING_POLICY`**: Load balancing policy (default: Adaptive)
  - Options: RoundRobin, SizeAware, LoadBalanced, Adaptive
  - Impact: How operations are distributed across instances
  - Production: Adaptive recommended for mixed workloads

### Logging Configuration
- **`BATCH_SCHEDULER_LOG_LEVEL`**: Logging verbosity (default: info)
  - Options: error, warn, info, debug
  - Production: Use info for normal operation, debug for troubleshooting
  
- **`BATCH_SCHEDULER_DEBUG`**: Enable debug output (default: 0, range: 0-1)
  - Impact: Generates detailed debugging information
  - Production: Disable unless troubleshooting

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

### Logging Setup
```bash
# Create log directory for statistics
sudo mkdir -p /var/log/batch_scheduler/
sudo chown $USER:$USER /var/log/batch_scheduler/

# Set environment variables for logging
export BATCH_SCHEDULER_LOG_FILE=/var/log/batch_scheduler_stats.jsonl
export BATCH_SCHEDULER_LOG_LEVEL=info
```

### Real-time Monitoring
```bash
# View instance utilization (target: >85% average)
tail -1 /var/log/batch_scheduler_stats.jsonl | jq '.instance_utilizations'

# View average latency (target: <1000μs)
tail -1 /var/log/batch_scheduler_stats.jsonl | jq '.average_latency_us'

# View fallback count (target: <1% rate)
tail -1 /var/log/batch_scheduler_stats.jsonl | jq '.fallback_count'

# Monitor all recent statistics
tail -10 /var/log/batch_scheduler_stats.jsonl | jq '.'

# Real-time monitoring watch
watch -n 1 'tail -1 /var/log/batch_scheduler_stats.jsonl | jq'
```

### Performance Metrics Interpretation
- **Instance Utilization**: >85% indicates good load balancing
- **Average Latency**: <1000μs optimal, >2000μs needs investigation
- **Fallback Count**: >5% indicates FPGA or configuration issues
- **Memory Usage**: Monitor system memory during operation

## Production Deployment

### Best Practices
1. **Start with conservative settings** (BATCH_SIZE=32) and gradually increase
2. **Enable all pipeline optimizations** (prefetching, double buffering, pooling)
3. **Monitor utilization** before scaling batch sizes
4. **Use Adaptive scheduling** for mixed workloads
5. **Set appropriate timeouts** based on workload characteristics
6. **Enable health monitoring** for production environments

### Deployment Procedure
```bash
# 1. Pre-deployment verification
bench/batch_scheduler/health_check.sh

# 2. Start with conservative configuration
export BATCH_SCHEDULER_ENABLED=1
export BATCH_SIZE=32
export INSTANCE_COUNT=16
export SCHEDULING_POLICY=Adaptive

# 3. Monitor initial performance
watch -n 5 'tail -1 /var/log/batch_scheduler_stats.jsonl | jq'

# 4. Gradually optimize based on metrics
export BATCH_SIZE=64  # After confirming stability

# 5. Enable full production configuration
export PIPELINE_ENABLE_PREFETCH=1
export PIPELINE_ENABLE_DOUBLE_BUFFER=1
export PIPELINE_ENABLE_POOLING=1
```

### Performance Tuning Methodology
1. **Baseline Measurement**: Run with default settings, record TPS and latency
2. **Batch Size Tuning**: Increase BATCH_SIZE incrementally (32→64→128)
3. **Memory Optimization**: Enable pipeline features if memory allows
4. **Policy Selection**: Test different scheduling policies with workload
5. **Target Validation**: Verify achieved TPS meets production requirements
6. **Stability Testing**: Run sustained tests (30+ minutes) at target configuration

### Rollback Procedures
```bash
# Immediate rollback to GPU-only mode
unset BATCH_SCHEDULER_ENABLED

# Rollback to conservative batch scheduler settings
export BATCH_SIZE=16
export INSTANCE_COUNT=8
export PIPELINE_ENABLE_PREFETCH=0
```

## Cross-Reference
For troubleshooting common issues, device problems, and error resolution, see [Troubleshooting Guide](docs/batch_scheduler_troubleshooting.md).

## FAQ

### Common Operational Questions

**Q: What should I do if TPS is below target?**
A: Check batch size configuration, verify instance utilization, enable pipeline optimizations, and see the [Low TPS Performance](docs/batch_scheduler_troubleshooting.md#low-tps-performance) section.

**Q: How do I know if FPGA instances are healthy?**
A: Run the health check script and monitor fallback rates. See [High Fallback Rate](docs/batch_scheduler_troubleshooting.md#high-fallback-rate) troubleshooting.

**Q: What causes high memory usage?**
A: Large batch sizes and disabled memory pooling. Reduce BATCH_SIZE and enable PIPELINE_ENABLE_POOLING. See [Memory Issues](docs/batch_scheduler_troubleshooting.md#memory-issues).

**Q: How do I tune for maximum performance?**
A: Follow the Performance Tuning Methodology above. Start conservative, monitor metrics, and incrementally optimize based on observed performance.

**Q: When should I use different scheduling policies?**
A: Use Adaptive for mixed workloads, SizeAware for variable operation sizes, RoundRobin for uniform operations.

**Q: What's the impact of increasing batch size?**
A: Higher throughput but increased memory usage and latency per batch. Monitor memory metrics when increasing.

