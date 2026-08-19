# Batch Scheduler Troubleshooting Guide

**Version:** 1.0.0  
**Last Updated:** 2026-08-19  
**Status:** Production Ready

## Emergency Procedures

### System Failure Recovery

**Complete System Failure Detection:**
```bash
# Check if batch scheduler is responsive
ps aux | grep batch_scheduler

# Check if devices are accessible
ls -la /dev/cxl_tmatmul*

# Check for kernel errors
dmesg | tail -20 | grep -i "error\|fail"
```

**Recovery Steps:**

1. **Immediate System Assessment**
   ```bash
   # Run health check to assess damage
   bench/batch_scheduler/health_check.sh

   # Check system logs for critical errors
   journalctl -xe | grep -i "batch\|scheduler\|cxl" | tail -50
   ```

2. **Emergency Batch Scheduler Shutdown**
   ```bash
   # Disable batch scheduler immediately
   unset BATCH_SCHEDULER_ENABLED

   # Kill any running scheduler processes
   pkill -f batch_scheduler

   # Flush pending operations
   killall -9 zluda  # Only if absolutely necessary
   ```

3. **Device Reset Sequence**
   ```bash
   # Reset CXL devices (requires root)
   sudo modprobe -r cxl_tmatmul_driver
   sudo modprobe cxl_tmatmul_driver

   # Verify devices are back
   ls -la /dev/cxl_tmatmul*
   ```

4. **Fallback to GPU-Only Mode**
   ```bash
   # Ensure GPU fallback is working
   export CUDA_VISIBLE_DEVICES=0
   export BATCH_SCHEDULER_ENABLED=0

   # Test basic GPU functionality
   nvidia-smi
   ```

5. **System Validation**
   ```bash
   # Test GPU-only matmul operations
   # Run your workload without batch scheduler
   # Verify system stability
   ```

### Data Recovery Procedures

**Operation State Recovery:**
```bash
# Check for incomplete operations in logs
grep "incomplete\|pending\|timeout" /var/log/batch_scheduler_stats.jsonl | tail -20

# Identify affected batch IDs
tail -100 /var/log/batch_scheduler_stats.jsonl | jq '.batch_id' | sort -u

# Export operation state before restart
tail -1000 /var/log/batch_scheduler_stats.jsonl > /tmp/batch_scheduler_backup_$(date +%Y%m%d_%H%M%S).jsonl
```

**Data Corruption Recovery:**
1. **Identify Corrupted Data**
   ```bash
   # Check for checksum errors in logs
   grep "corruption\|checksum\|invalid" /var/log/*.log
   ```

2. **Isolate Affected Components**
   ```bash
   # Disable specific instances if needed
   export BLACKLISTED_INSTANCES="0,5,12"  # Add problematic instance IDs
   ```

3. **Data Validation**
   ```bash
   # Run integrity checks on completed operations
   # (Specific to your application's data validation needs)
   ```

### Emergency Shutdown Procedure

**Graceful Shutdown:**
```bash
# 1. Stop accepting new requests
export BATCH_SCHEDULER_ACCEPTING_REQUESTS=0

# 2. Wait for current operations to complete (60 second timeout)
timeout 60 bash -c 'while pgrep -f batch_scheduler > /dev/null; do sleep 1; done'

# 3. Force shutdown if timeout exceeded
if pgrep -f batch_scheduler > /dev/null; then
    echo "WARNING: Forcing shutdown, pending operations may be lost"
    pkill -9 -f batch_scheduler
fi

# 4. Cleanup resources
rm -f /tmp/batch_scheduler_* 2>/dev/null

# 5. Export final state
cp /var/log/batch_scheduler_stats.jsonl /tmp/final_state_$(date +%Y%m%d_%H%M%S).jsonl
```

**Emergency Power-Off Safety:**
```bash
# If system must be powered down immediately:
# 1. Disable batch scheduler
unset BATCH_SCHEDULER_ENABLED

# 2. Sync filesystems
sync

# 3. Unmount CXL devices safely
sudo umount /dev/cxl_tmatmul* 2>/dev/null || true

# 4. System can now be safely powered off
```

### Critical Failure Scenarios

**Scenario 1: Mass Instance Failure (>50% instances unhealthy)**
```bash
# Detection
bench/batch_scheduler/health_check.sh | grep "Instance health"

# Response
export DEGRADED_MODE=1
export INSTANCE_COUNT=8  # Reduce to healthy instances
export BATCH_SIZE=32   # Reduce batch size accordingly

# Verify remaining instances
ls -la /dev/cxl_tmatmul*
```

**Scenario 2: Memory Exhaustion**
```bash
# Detection
free -h  # Check available memory
dmesg | grep -i "out of memory\|oom"

# Response
export BATCH_SIZE=16  # Reduce batch size
export PIPELINE_ENABLE_PREFETCH=0  # Disable prefetching
export PIPELINE_ENABLE_DOUBLE_BUFFER=0  # Disable double buffering

# Clear caches if needed
sync; echo 3 > /proc/sys/vm/drop_caches
```

**Scenario 3: GPU Fallback Saturation**
```bash
# Detection
tail -20 /var/log/batch_scheduler_stats.jsonl | jq '.fallback_count'
# If > 10% fallback rate

# Response
export BATCH_SCHEDULER_ENABLED=0  # Disable batch scheduler temporarily
# Restart batch scheduler after issue resolution
```

## Common Issues

### Low TPS Performance
**Symptoms:** TPS below target (8 for Kimi, 2000 for matmulfreellm)

**Solutions:**
1. Check batch size: `echo $BATCH_SIZE`
2. Verify instance count: `nvidia-smi`
3. Monitor utilization: Check scheduler stats
4. Enable prefetching: `PIPELINE_ENABLE_PREFETCH=1`

### High Fallback Rate
**Symptoms:** Many operations falling back to GPU

**Solutions:**
1. Check FPGA device status: `ls /dev/cxl_tmatmul*`
2. Monitor instance health: Check health monitor stats
3. Verify CXL device permissions: `ls -la /dev/cxl_tmatmul*`
4. Check system logs: `dmesg | grep -i tmatmul`

### Memory Issues
**Symptoms:** OOM errors or high memory usage

**Solutions:**
1. Reduce batch size: `BATCH_SIZE=32`
2. Enable memory pooling: `PIPELINE_ENABLE_POOLING=1`
3. Check GPU memory: `nvidia-smi`

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

## Monitoring and Statistics

### Instance Utilization
```bash
# Check utilization across all 16 instances
# Target: >85% average utilization
```

### Average Latency
```bash
# Monitor average operation latency
# Target: <1000us for optimal performance
```

### Fallback Count
```bash
# Track GPU fallback operations
# Target: <1% fallback rate
```

### Post-Recovery Verification

**System Health Validation:**
```bash
# 1. Run comprehensive health check
bench/batch_scheduler/health_check.sh

# 2. Verify all 16 instances are accessible
INSTANCES=$(ls /dev/cxl_tmatmul* | wc -l)
echo "Available instances: $INSTANCES"
if [ "$INSTANCES" -lt 16 ]; then
    echo "WARNING: Not all instances are available"
fi

# 3. Test basic batch scheduler functionality
export BATCH_SCHEDULER_ENABLED=1
export BATCH_SIZE=16  # Start with conservative batch size
export INSTANCE_COUNT=$INSTANCES

# 4. Monitor first batch for errors
tail -f /var/log/batch_scheduler_stats.jsonl | jq '.errors, .fallback_count'
```

**Performance Recovery Validation:**
```bash
# Test with small workload first
export BATCH_SIZE=16
export TEST_ITERATIONS=100

# Monitor performance metrics
watch -n 5 'tail -1 /var/log/batch_scheduler_stats.jsonl | jq ".average_latency_us, .instance_utilizations"'

# Gradually increase to target configuration
export BATCH_SIZE=32
# Monitor for 5 minutes, then increase to 64
```

**Data Integrity Check:**
```bash
# Verify no data corruption during recovery
grep -i "corruption\|checksum" /var/log/batch_scheduler_stats.jsonl | tail -20

# Check for consistent operation counts
tail -100 /var/log/batch_scheduler_stats.jsonl | jq '.batch_id' | wc -l
```

## Disaster Recovery

**Complete System Restoration:**
```bash
# 1. Backup current state (if possible)
cp /var/log/batch_scheduler_stats.jsonl /tmp/disaster_backup_$(date +%Y%m%d_%H%M%S).jsonl

# 2. Rebuild from scratch
cd /home/victoryang00/hetGPU
cargo clean
cargo build --release

# 3. Reset all devices
sudo modprobe -r cxl_tmatmul_driver nvidia  # Order matters
sudo modprobe nvidia cxl_tmatmul_driver

# 4. Verify device access
ls -la /dev/cxl_tmatmul*
nvidia-smi

# 5. Start with minimal configuration
export BATCH_SCHEDULER_ENABLED=1
export BATCH_SIZE=8  # Conservative start
export INSTANCE_COUNT=4  # Use subset of instances

# 6. Gradually restore to full capacity
# Increase batch size and instance count incrementally
```

**Rollback to Previous Working State:**
```bash
# If you have git history of working configurations:
cd /home/victoryang00/hetGPU
git log --oneline | head -10  # Find working commit
git checkout <working-commit-hash>
cargo build --release

# Or restore from configuration backup if available
```

## Getting Help
1. Check this troubleshooting guide first
2. Enable debug mode and review logs
3. Run health check script
4. Check documentation: `/docs/batch_scheduler_user_guide.md`