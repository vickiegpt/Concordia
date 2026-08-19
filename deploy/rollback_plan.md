# Rollback Plan

## Pre-Deployment Checklist
- [ ] Backup current ZLUDA build
- [ ] Document current performance baselines
- [ ] Prepare monitoring tools
- [ ] Test rollback procedure
- [ ] Notify stakeholders of deployment

## Deployment Steps

### 1. Preparation
```bash
# Backup current version
cd /home/victoryang00/hetGPU
git stash
git checkout main
git pull origin main

# Document current state
echo "Pre-deployment baseline:" > deployment.log
cargo test --lib batch_scheduler 2>&1 | tee -a deployment.log
```

### 2. Deploy Batch Scheduler
```bash
# Source production configuration
source deploy/production_config.sh

# Build with batch scheduler
cargo build --release --lib

# Run integration tests
./tests/final_integration_test.sh
```

### 3. Monitor Deployment
- Watch for error spikes in logs
- Monitor TPS metrics
- Check instance health
- Track fallback rate

### 4. Post-Deployment Verification
```bash
# Verify TPS targets met
# Check error rates < 1%
# Confirm instance utilization > 85%
# Validate fallback rate < 0.1%
```

## Rollback Triggers
1. **Performance degradation**: TPS < 50% of target for > 5 minutes
2. **High error rate**: Error rate > 1% for > 2 minutes  
3. **System instability**: crashes or hangs observed
4. **Instance failures**: > 50% instances unhealthy
5. **User complaints**: significant increase in user-reported issues

## Rollback Procedure

### Immediate Rollback (Critical Issues)
```bash
# Step 1: Graceful shutdown of existing processes
echo "Initiating graceful shutdown..."
timeout 30 bash -c 'while pgrep -f zluda > /dev/null; do sleep 1; done' || {
    echo "⚠️  Graceful shutdown timeout, proceeding with forceful termination"
    pkill -9 -f zluda
    sleep 5
}

# Step 2: Disable batch scheduler
unset BATCH_SCHEDULER_ENABLED
export BATCH_SCHEDULER_ENABLED=0

# Step 3: Clear any batch-related state
rm -f /var/log/batch_scheduler_stats.jsonl 2>/dev/null || true
rm -f $HOME/.hetgpu_logs/batch_scheduler_stats.jsonl 2>/dev/null || true

# Step 4: Restart services without batch scheduler
echo "Starting baseline configuration..."
cd /home/victoryang00/hetGPU
cargo run --release --bin zluda &

# Step 5: Verify processes started
sleep 10
if pgrep -f zluda > /dev/null; then
    echo "✅ Services restarted successfully"
else
    echo "❌ Failed to restart services - manual intervention required"
    exit 1
fi

# Step 6: Verify baseline restored
echo "Rollback complete. Verifying baseline performance..."
```

### Graceful Rollback (Less Critical)
```bash
# Complete current batch
# Stop accepting new requests
# Drain existing operations
# Disable batch scheduler
unset BATCH_SCHEDULER_ENABLED

# Restart with baseline configuration
systemctl restart zluda-service
```

## Rollback Verification
After rollback, verify:
- Kimi TPS returns to baseline (0.62)
- matmulfreellm TPS returns to baseline (12.85)
- No error spikes in logs
- System stability restored
- User complaints resolved

## Post-Rollback Analysis
1. Collect logs from failed deployment
2. Analyze failure patterns
3. Identify root cause
4. Document lessons learned
5. Plan fixes for future deployment

## Emergency Contacts
- Development Team: dev-team@company.com | Slack: #hetgpu-urgent | PagerDuty: hetgpu-dev
- Operations Team: ops-team@company.com | Slack: #ops-urgent | PagerDuty: hetgpu-ops
- Management: manager@company.com | Slack: #management-escalation | Cell: +1-XXX-XXX-XXXX
- On-Call Engineer: on-call@company.com | Cell: +1-XXX-XXX-XXXX (24/7)

## Monitoring During Deployment
```bash
# Terminal 1: Monitor TPS (from stats file)
watch -n 5 'tail -5 /var/log/batch_scheduler_stats.jsonl | jq -r "select(.tps) | .tps" 2>/dev/null || echo "No TPS data yet"'

# Terminal 2: Monitor errors
tail -f /var/log/batch_scheduler_stats.jsonl | grep -i error || tail -f $HOME/.hetgpu_logs/batch_scheduler_stats.jsonl | grep -i error

# Terminal 3: Monitor instance health
watch -n 10 'ps aux | grep zluda | grep -v grep | wc -l | xargs echo "ZLUDA processes:"'

# Alternative: Simple process monitoring
watch -n 5 'echo "=== System Status ==="; echo "ZLUDA processes: $(pgrep -f zluda | wc -l)"; echo "Memory usage: $(free -h | grep Mem)"; echo "Load average: $(uptime)"'
```

## Deployment Timeline
- **T-1 hour**: Begin pre-deployment checks
- **T-30 min**: Notify stakeholders
- **T-15 min**: Start monitoring
- **T-0**: Deploy batch scheduler
- **T+15 min**: Post-deployment verification
- **T+30 min**: Full monitoring for 30 minutes
- **T+1 hour**: Extended monitoring if issues detected

## Success Criteria
Deployment considered successful if:
- All integration tests pass
- TPS targets met or exceeded
- Error rate < 0.1%
- Instance utilization > 85%
- No rollback triggered for 1 hour