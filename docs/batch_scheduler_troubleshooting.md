# Batch Scheduler Troubleshooting Guide

**Version:** 1.0.0  
**Last Updated:** 2026-08-19  
**Status:** Production Ready

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

