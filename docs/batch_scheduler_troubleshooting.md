# Batch Scheduler Troubleshooting Guide

**Version:** 1.0.0  
**Last Updated:** 2026-08-19  
**Status:** Production Ready

## Cross-Reference
For user guide, configuration details, and monitoring setup, see [User Guide](docs/batch_scheduler_user_guide.md).

## Escalation Procedures

### Severity Levels
**CRITICAL (Severity 1)**
- Complete system failure
- All FPGA instances down
- Data corruption suspected
- Production workload completely blocked
- **Response Time**: < 1 hour
- **Escalation**: Immediate management notification

**HIGH (Severity 2)**  
- >50% FPGA instances unavailable
- Performance degradation >50%
- Frequent crashes/hangs
- Production impact but partial functionality
- **Response Time**: < 4 hours
- **Escalation**: Engineering lead + management

**MEDIUM (Severity 3)**
- Single instance failures
- Performance degradation <20%
- Intermittent issues
- Workarounds available
- **Response Time**: < 1 business day
- **Escalation**: Engineering team

**LOW (Severity 4)**
- Configuration questions
- Performance optimization requests
- Documentation improvements
- Minor issues with no production impact
- **Response Time**: < 2 business days
- **Escalation**: Standard support channel

### Support Contacts
**Primary Support:**
- Email: support@batchscheduler.example.com
- Response Time: 4 hours (business days)
- Available: Mon-Fri 9AM-5PM PST

**Emergency Support:**
- Email: emergency@batchscheduler.example.com  
- Response Time: 1 hour (24/7)
- For: Severity 1 and 2 issues only

**Development Team:**
- Internal: @batch-scheduler-team
- For: Bug reports, feature requests, code issues

## Common Issues

### Low TPS Performance
**Symptoms:** TPS below target (8 for Kimi, 2000 for matmulfreellm)

**Diagnostic Steps:**
1. Check current TPS: Monitor `/var/log/batch_scheduler_stats.jsonl`
2. Verify batch size: `echo $BATCH_SIZE`
3. Check instance count: `ls /dev/cxl_tmatmul* | wc -l`
4. Monitor utilization: Check scheduler stats for instance utilization
5. Verify GPU availability: `nvidia-smi`

**Solutions:**
1. **Increase batch size** if utilization is low: `export BATCH_SIZE=128`
2. **Enable prefetching**: `export PIPELINE_ENABLE_PREFETCH=1`
3. **Enable double buffering**: `export PIPELINE_ENABLE_DOUBLE_BUFFER=1`
4. **Try Adaptive scheduling**: `export SCHEDULING_POLICY=Adaptive`
5. **Check for system bottlenecks**: CPU, memory, disk I/O
6. **Review configuration** against [User Guide](docs/batch_scheduler_user_guide.md) recommendations

**Expected Results:**
- Kimi IQ1S: TPS should reach 8+ with proper configuration
- matmulfreellm: TPS should reach 2000+ with optimized settings

### High Fallback Rate
**Symptoms:** Many operations falling back to GPU (fallback rate >1%)

**Diagnostic Steps:**
1. Check fallback rate: Monitor `fallback_count` in stats logs
2. Verify FPGA devices: `ls /dev/cxl_tmatmul*`
3. Check device permissions: `ls -la /dev/cxl_tmatmul*`
4. Monitor instance health: Check health stats for blacklisted instances
5. Review system logs: `dmesg | grep -i tmatmul`

**Solutions:**
1. **Fix device permissions**: Add user to render group
2. **Reset unhealthy instances**: `bench/batch_scheduler/health_check.sh`
3. **Reduce batch size** to decrease instance load
4. **Check system resources**: Memory, CPU availability
5. **Verify CXL driver** is loaded and functioning
6. **Review configuration** for incorrect parameters

**Expected Results:**
- Fallback rate should drop below 1% with healthy instances
- All 16 FPGA instances should be available and operational

### Memory Issues
**Symptoms:** OOM errors or high memory usage

**Diagnostic Steps:**
1. Check system memory: `free -h`
2. Monitor GPU memory: `nvidia-smi`
3. Review batch size settings: `echo $BATCH_SIZE`
4. Check for memory leaks in application logs
5. Verify memory pooling is enabled

**Solutions:**
1. **Reduce batch size**: `export BATCH_SIZE=32` (or 16 for severe cases)
2. **Enable memory pooling**: `export PIPELINE_ENABLE_POOLING=1`
3. **Disable prefetching** if memory constrained: `export PIPELINE_ENABLE_PREFETCH=0`
4. **Disable double buffering** to save memory: `export PIPELINE_ENABLE_DOUBLE_BUFFER=0`
5. **Check GPU memory** availability: `nvidia-smi`
6. **Clear system caches** if needed: `sync; echo 3 > /proc/sys/vm/drop_caches`

**Expected Results:**
- Memory usage should stabilize within available system RAM
- No OOM errors with appropriate batch size
- GPU memory usage should stay under 90% capacity

## Debug Mode
```bash
# Enable detailed logging
export BATCH_SCHEDULER_DEBUG=1
export BATCH_SCHEDULER_LOG_LEVEL=debug
```

## Health Check
```bash
# Run health check script
bench/batch_scheduler/health_check.sh
```

## Common Error Messages

### "Aggregator lock failed"
**Cause:** Mutex poisoning or deadlock
**Solution:** Restart application, check for concurrent access issues

### "Scheduler lock failed"  
**Cause:** Scheduler synchronization issue
**Solution:** Verify thread-safe access, reduce concurrent submissions

### "Pipeline lock failed"
**Cause:** Memory pipeline synchronization error
**Solution:** Check for memory allocation failures, reduce batch size

### Instance health errors
**Cause:** FPGA instance unhealthy or blacklisted
**Solution:** Check instance health stats, restart if needed

## Performance Tuning

### Batch Size Tuning
- Start with default (64)
- Increase if throughput is low and memory is available
- Decrease if experiencing memory issues

### Instance Count
- Verify FPGA availability: `ls /dev/cxl_tmatmul* | wc -l`
- Set INSTANCE_COUNT to match available FPGA instances

### Pipeline Configuration
- Enable prefetching for better throughput
- Enable double buffering to hide latency
- Adjust max concurrent transfers based on memory bandwidth

## Operational Procedures

### Deployment Runbook
**Pre-Deployment Checklist:**
- [ ] Verify all 16 FPGA instances are available
- [ ] Run health check: `bench/batch_scheduler/health_check.sh`
- [ ] Confirm GPU access: `nvidia-smi`
- [ ] Set up logging directory: `/var/log/batch_scheduler/`
- [ ] Test with conservative configuration first
- [ ] Monitor initial performance metrics

**Deployment Steps:**
```bash
# 1. Environment verification
export BATCH_SCHEDULER_ENABLED=0  # Start disabled
bench/batch_scheduler/health_check.sh

# 2. Conservative start
export BATCH_SCHEDULER_ENABLED=1
export BATCH_SIZE=16
export INSTANCE_COUNT=16
export SCHEDULING_POLICY=Adaptive

# 3. Monitor for stability
watch -n 5 'tail -1 /var/log/batch_scheduler_stats.jsonl | jq'

# 4. Gradual optimization
export BATCH_SIZE=32  # After 5 minutes of stability
export BATCH_SIZE=64  # After 5 more minutes of stability

# 5. Full configuration
export PIPELINE_ENABLE_PREFETCH=1
export PIPELINE_ENABLE_DOUBLE_BUFFER=1
export PIPELINE_ENABLE_POOLING=1
```

### Scaling Operations
**Scale Up (Increase Throughput):**
```bash
# Increase batch size
export BATCH_SIZE=128  # If memory allows

# Enable all optimizations
export PIPELINE_ENABLE_PREFETCH=1
export PIPELINE_ENABLE_DOUBLE_BUFFER=1
export PIPELINE_ENABLE_POOLING=1

# Try different scheduling policies
export SCHEDULING_POLICY=SizeAware
```

**Scale Down (Reduce Load):**
```bash
# Reduce batch size
export BATCH_SIZE=32

# Disable memory-intensive features
export PIPELINE_ENABLE_PREFETCH=0
export PIPELINE_ENABLE_DOUBLE_BUFFER=0

# Reduce instance count if needed
export INSTANCE_COUNT=8
```

### Maintenance Procedures
**Scheduled Maintenance:**
```bash
# 1. Notify users of maintenance window
# 2. Gradual drain of existing operations
export BATCH_SCHEDULER_ACCEPTING_REQUESTS=0
timeout 60 bash -c 'while pgrep -f batch_scheduler > /dev/null; do sleep 1; done'

# 3. Disable batch scheduler
unset BATCH_SCHEDULER_ENABLED

# 4. Perform maintenance
# - Update drivers/firmware
# - System upgrades
# - Hardware maintenance

# 5. Post-maintenance verification
bench/batch_scheduler/health_check.sh

# 6. Gradual restart
export BATCH_SCHEDULER_ENABLED=1
export BATCH_SIZE=16  # Conservative start
```

**Rolling Updates:**
```bash
# For zero-downtime updates, use instance subsets
export INSTANCE_COUNT=8  # Use half the instances
# Perform update on unused instances
export INSTANCE_COUNT=16  # Gradually restore full capacity
```

### Performance Tuning Runbook
**Baseline Establishment:**
```bash
# 1. Start with defaults
export BATCH_SIZE=64
export INSTANCE_COUNT=16
export SCHEDULING_POLICY=RoundRobin

# 2. Run baseline test (10 minutes)
# Record TPS, latency, memory usage

# 3. Test different batch sizes
for BATCH_SIZE in 32 64 96 128; do
    # Run 5-minute test at each size
    # Record metrics
done

# 4. Test scheduling policies
for POLICY in RoundRobin SizeAware LoadBalanced Adaptive; do
    export SCHEDULING_POLICY=$POLICY
    # Run 5-minute test for each policy
    # Record metrics
done

# 5. Select optimal configuration based on test results
```

**Performance Validation:**
```bash
# Verify target TPS is sustained
watch -n 10 'tail -1 /var/log/batch_scheduler_stats.jsonl | jq ".tps, .average_latency_us"'

# Run sustained test (30+ minutes)
# Monitor for degradation over time

# Validate under load
# - Multiple concurrent users
# - Peak traffic patterns
# - Extended operation duration
```

## Monitoring and Statistics

### Setup Monitoring
```bash
# Create log directory if it doesn't exist
sudo mkdir -p /var/log/batch_scheduler/
sudo chown $USER:$USER /var/log/batch_scheduler/

# Set up log rotation if needed
# Add to logrotate configuration for long-running operations
```

### Instance Utilization Monitoring
```bash
# Check utilization across all 16 instances
# Target: >85% average utilization
# Interpretation:
#   - Low utilization (<50%): Consider reducing batch size
#   - High utilization (>95%): Good load balancing
#   - Uneven distribution: Try different scheduling policy

tail -1 /var/log/batch_scheduler_stats.jsonl | jq '.instance_utilizations'
```

### Average Latency Monitoring
```bash
# Monitor average operation latency
# Target: <1000μs for optimal performance
# Interpretation:
#   - <500μs: Excellent performance
#   - 500-1000μs: Good performance
#   - 1000-2000μs: Acceptable, monitor closely
#   - >2000μs: Investigate bottlenecks

tail -1 /var/log/batch_scheduler_stats.jsonl | jq '.average_latency_us'
```

### Fallback Count Monitoring
```bash
# Track GPU fallback operations
# Target: <1% fallback rate
# Interpretation:
#   - <1%: Healthy FPGA operation
#   - 1-5%: Some instance issues, investigate
#   - >5%: Significant FPGA problems, take action

tail -1 /var/log/batch_scheduler_stats.jsonl | jq '.fallback_count'
```

### Comprehensive Health Monitoring
```bash
# All-in-one monitoring command
watch -n 5 'tail -1 /var/log/batch_scheduler_stats.jsonl | jq "{
  tps: .tps,
  avg_latency_us: .average_latency_us, 
  instance_utilization: (.instance_utilizations | add / length),
  fallback_rate: (.fallback_count / .total_operations * 100)
}"'
```

