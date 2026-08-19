# Rollback Plan

## Pre-Deployment Checklist
- [ ] Backup current ZLUDA build
- [ ] Document current performance baselines
- [ ] Prepare monitoring tools
- [ ] Test rollback procedure in staging environment
- [ ] Notify stakeholders of deployment
- [ ] Verify deployment environment and dependencies
- [ ] Confirm audit logging functionality
- [ ] Validate rollback script permissions and paths

## Dependency Requirements

### System Dependencies
Before deployment or rollback, ensure these dependencies are available:

```bash
# Required command-line tools
REQUIRED_TOOLS=(cargo git jq timeout ps grep pgrep)

check_dependency() {
    local tool=$1
    if command -v "$tool" >/dev/null 2>&1; then
        echo "✅ $tool: $(command -v $tool)"
        return 0
    else
        echo "❌ $tool: NOT FOUND"
        return 1
    fi
}

echo "=== Dependency Verification ==="
MISSING=0
for tool in "${REQUIRED_TOOLS[@]}"; do
    check_dependency "$tool" || MISSING=$((MISSING + 1))
done

if [ $MISSING -gt 0 ]; then
    echo "ERROR: $MISSING required dependencies missing"
    echo "Install missing tools before deployment"
    exit 1
fi

# Optional but recommended tools
OPTIONAL_TOOLS=(bc systemd systemctl)
for tool in "${OPTIONAL_TOOLS[@]}"; do
    check_dependency "$tool" || echo "⚠️  Optional tool $tool not available"
done
```

### Hardware Requirements
- FPGA hardware available and accessible
- Sufficient memory for batch operations
- Network connectivity for distributed operations
- Appropriate permissions for hardware access

### Software Requirements
- Rust toolchain (rustc, cargo)
- Git for version control
- Build tools and dependencies
- Sufficient disk space for builds

## Pre-Deployment Dependency Check Script
```bash
#!/usr/bin/env bash
# Comprehensive pre-deployment dependency check

echo "=== PRE-DEPLOYMENT DEPENDENCY CHECK ==="

# System requirements
check_mem() {
    local total_mem=$(free -g | grep Mem | awk '{print $2}')
    if [ "$total_mem" -ge 8 ]; then
        echo "✅ Memory: ${total_mem}GB (minimum 8GB required)"
        return 0
    else
        echo "❌ Memory: ${total_mem}GB (minimum 8GB required)"
        return 1
    fi
}

check_disk() {
    local available_gb=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$available_gb" -ge 10 ]; then
        echo "✅ Disk space: ${available_gb}GB available"
        return 0
    else
        echo "❌ Disk space: ${available_gb}GB available (minimum 10GB required)"
        return 1
    fi
}

# Run all checks
FAILED=0
check_mem || FAILED=$((FAILED + 1))
check_disk || FAILED=$((FAILED + 1))

# Check Rust toolchain
if rustc --version >/dev/null 2>&1; then
    echo "✅ Rust toolchain: $(rustc --version)"
else
    echo "❌ Rust toolchain not found"
    FAILED=$((FAILED + 1))
fi

# Check Git
if git --version >/dev/null 2>&1; then
    echo "✅ Git: $(git --version)"
else
    echo "❌ Git not found"
    FAILED=$((FAILED + 1))
fi

# Check project structure
if [ -f "Cargo.toml" ] && [ -d "zluda" ]; then
    echo "✅ Project structure valid"
else
    echo "❌ Project structure invalid - not in hetGPU root directory"
    FAILED=$((FAILED + 1))
fi

if [ $FAILED -eq 0 ]; then
    echo "=== ALL CHECKS PASSED ==="
    exit 0
else
    echo "=== $FAILED CHECKS FAILED ==="
    exit 1
fi
```

## Deployment Steps

### 1. Preparation
```bash
# Navigate to project root
cd /home/victoryang00/hetGPU || { echo "ERROR: hetGPU directory not found"; exit 1; }

# Verify we're in the right directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "zluda" ]]; then
    echo "ERROR: Not in hetGPU root directory"
    exit 1
fi

# Backup current version with verification
echo "Creating backup..."
BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).tar.gz"
git stash push -u -m "pre-deployment-backup" || {
    echo "ERROR: Git stash failed - check for uncommitted changes"
    exit 1
}

# Verify backup was created successfully
if git stash list | grep -q "pre-deployment-backup"; then
    echo "✅ Backup verified: git stash created successfully"
    BACKUP_STASH_REF=$(git stash list | grep "pre-deployment-backup" | head -1 | cut -d: -f1)
    echo "Backup reference: $BACKUP_STASH_REF"
else
    echo "ERROR: Backup verification failed"
    exit 1
fi

# Document current state with audit logging
echo "=== PRE-DEPLOYMENT BASELINE ===" | tee deployment.log
echo "Timestamp: $(date)" | tee -a deployment.log
echo "Git commit: $(git rev-parse HEAD)" | tee -a deployment.log
echo "Directory: $(pwd)" | tee -a deployment.log

# Run pre-deployment tests with timeout
timeout 300 cargo test --lib batch_scheduler 2>&1 | tee -a deployment.log || {
    echo "WARNING: Pre-deployment tests failed or timed out"
    echo "Consider investigating before deployment"
}
```

### 2. Deploy Batch Scheduler
```bash
# Navigate to project root and verify environment
cd /home/victoryang00/hetGPU || { echo "ERROR: hetGPU directory not found"; exit 1; }

# Source production configuration with error handling
if ! source deploy/production_config.sh; then
    echo "ERROR: Failed to load production configuration"
    echo "Check deploy/production_config.sh for syntax or dependency issues"
    exit 1
fi

# Build with batch scheduler and timeout
echo "Building with batch scheduler..."
timeout 600 cargo build --release --lib || {
    echo "ERROR: Build failed or timed out"
    echo "Check build logs and consider rollback"
    exit 1
}

# Run integration tests with comprehensive validation
echo "Running integration tests..."
if [[ -f "./tests/final_integration_test.sh" ]]; then
    timeout 300 ./tests/final_integration_test.sh || {
        echo "WARNING: Integration tests failed - consider before proceeding"
        read -p "Continue deployment despite test failures? (y/N): " CONTINUE
        if [[ "$CONTINUE" != "y" && "$CONTINUE" != "Y" ]]; then
            echo "Deployment cancelled by operator"
            exit 1
        fi
    }
else
    echo "WARNING: Integration test script not found"
    echo "Consider manual testing before production deployment"
fi
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
#!/usr/bin/env bash
# Immediate rollback script with verification and audit logging

set -euo pipefail

# Navigate to project root with verification
cd /home/victoryang00/hetGPU || { echo "ERROR: hetGPU directory not found"; exit 1; }
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "zluda" ]]; then
    echo "ERROR: Not in hetGPU root directory"
    exit 1
fi

# Setup audit logging
ROLLBACK_LOG="$HOME/.hetgpu_rollback_$(date +%Y%m%d_%H%M%S).log"
echo "=== ROLLBACK STARTED $(date) ===" | tee "$ROLLBACK_LOG"

# Step 1: Graceful shutdown of existing processes with verification
echo "Step 1: Initiating graceful shutdown..." | tee -a "$ROLLBACK_LOG"
ZLUDA_PROCS_BEFORE=$(pgrep -f zluda | wc -l)
echo "ZLUDA processes before shutdown: $ZLUDA_PROCS_BEFORE" | tee -a "$ROLLBACK_LOG"

if timeout 30 bash -c 'while pgrep -f zluda > /dev/null; do sleep 1; done' 2>/dev/null; then
    echo "✅ Graceful shutdown completed" | tee -a "$ROLLBACK_LOG"
else
    echo "⚠️  Graceful shutdown timeout, proceeding with forceful termination" | tee -a "$ROLLBACK_LOG"
    pkill -9 -f zluda || true
    sleep 5
fi

# Verify shutdown
ZLUDA_PROCS_AFTER=$(pgrep -f zluda | wc -l)
if [ "$ZLUDA_PROCS_AFTER" -eq 0 ]; then
    echo "✅ All ZLUDA processes terminated" | tee -a "$ROLLBACK_LOG"
else
    echo "⚠️  Warning: $ZLUDA_PROCS_AFTER ZLUDA processes still running" | tee -a "$ROLLBACK_LOG"
fi

# Step 2: Disable batch scheduler and restore environment
echo "Step 2: Disabling batch scheduler..." | tee -a "$ROLLBACK_LOG"
unset BATCH_SCHEDULER_ENABLED
export BATCH_SCHEDULER_ENABLED=0
unset BATCH_SIZE INSTANCE_COUNT PIPELINE_ENABLE_DOUBLE_BUFFER PIPELINE_ENABLE_PREFETCH
unset KIMI_TARGET_TPS MATMULFREELLM_TARGET_TPS HEALTH_CHECK_INTERVAL FALLBACK_THRESHOLD

# Step 3: Clear batch-related state with configurable paths
echo "Step 3: Clearing batch scheduler state..." | tee -a "$ROLLBACK_LOG"
STATS_LOCATIONS=(
    "${BATCH_SCHEDULER_STATS_FILE:-/var/log/batch_scheduler_stats.jsonl}"
    "$HOME/.hetgpu_logs/batch_scheduler_stats.jsonl"
    "/var/log/batch_scheduler_stats.jsonl"
)

CLEANED=0
for location in "${STATS_LOCATIONS[@]}"; do
    if [[ -f "$location" ]]; then
        echo "Removing stats file: $location" | tee -a "$ROLLBACK_LOG"
        rm -f "$location" && CLEANED=$((CLEANED + 1))
    fi
done

if [ $CLEANED -gt 0 ]; then
    echo "✅ Cleared $CLEANED stats file(s)" | tee -a "$ROLLBACK_LOG"
else
    echo "ℹ️  No stats files found to clean" | tee -a "$ROLLBACK_LOG"
fi

# Step 4: Restore backup with verification
echo "Step 4: Restoring backup..." | tee -a "$ROLLBACK_LOG"
if git stash list | grep -q "pre-deployment-backup"; then
    BACKUP_STASH=$(git stash list | grep "pre-deployment-backup" | head -1 | cut -d: -f1)
    echo "Found backup: $BACKUP_STASH" | tee -a "$ROLLBACK_LOG"

    if git stash pop "$BACKUP_STASH" --index; then
        echo "✅ Backup restored successfully" | tee -a "$ROLLBACK_LOG"
    else
        echo "ERROR: Failed to restore backup" | tee -a "$ROLLBACK_LOG"
        exit 1
    fi
else
    echo "WARNING: No backup found, attempting rebuild from main branch" | tee -a "$ROLLBACK_LOG"
    git checkout main || git checkout - || {
        echo "ERROR: Cannot restore from backup or main branch" | tee -a "$ROLLBACK_LOG"
        exit 1
    }
fi

# Step 5: Rebuild and restart services
echo "Step 5: Rebuilding baseline configuration..." | tee -a "$ROLLBACK_LOG"
timeout 600 cargo build --release --bin zluda || {
    echo "ERROR: Build failed during rollback" | tee -a "$ROLLBACK_LOG"
    exit 1
}

echo "Starting baseline ZLUDA services..." | tee -a "$ROLLBACK_LOG"
cargo run --release --bin zluda &
ZLUDA_PID=$!

# Step 6: Verification and monitoring
echo "Step 6: Verifying rollback..." | tee -a "$ROLLBACK_LOG"
sleep 10

if pgrep -f zluda > /dev/null; then
    echo "✅ ZLUDA services restarted successfully" | tee -a "$ROLLBACK_LOG"
    echo "=== ROLLBACK COMPLETED SUCCESSFULLY $(date) ===" | tee -a "$ROLLBACK_LOG"
else
    echo "❌ Failed to restart ZLUDA services - manual intervention required" | tee -a "$ROLLBACK_LOG"
    echo "=== ROLLBACK FAILED - MANUAL INTERVENTION REQUIRED $(date) ===" | tee -a "$ROLLBACK_LOG"
    exit 1
fi

echo "Monitor system for baseline performance restoration"
echo "Check $ROLLBACK_LOG for detailed rollback information"
```

### Graceful Rollback (Less Critical)
```bash
#!/usr/bin/env bash
# Graceful rollback with idempotency and verification

set -euo pipefail

# Navigate to project root with verification
cd /home/victoryang00/hetGPU || { echo "ERROR: hetGPU directory not found"; exit 1; }
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "zluda" ]]; then
    echo "ERROR: Not in hetGPU root directory"
    exit 1
fi

# Setup audit logging
ROLLBACK_LOG="$HOME/.hetgpu_graceful_rollback_$(date +%Y%m%d_%H%M%S).log"
echo "=== GRACEFUL ROLLBACK STARTED $(date) ===" | tee "$ROLLBACK_LOG"

# Step 1: Dependency verification
echo "Step 1: Verifying dependencies..." | tee -a "$ROLLBACK_LOG"
REQUIRED_DEPS=(cargo git jq timeout)
MISSING_DEPS=()

for cmd in "${REQUIRED_DEPS[@]}"; do
    command -v "$cmd" >/dev/null 2>&1 || MISSING_DEPS+=("$cmd")
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo "ERROR: Missing required dependencies: ${MISSING_DEPS[*]}" | tee -a "$ROLLBACK_LOG"
    exit 1
fi

echo "✅ All dependencies verified" | tee -a "$ROLLBACK_LOG"

# Step 2: Check current batch scheduler state
echo "Step 2: Checking current state..." | tee -a "$ROLLBACK_LOG"
if [ "${BATCH_SCHEDULER_ENABLED:-0}" = "1" ]; then
    echo "Batch scheduler is currently ENABLED" | tee -a "$ROLLBACK_LOG"
else
    echo "ℹ️  Batch scheduler not enabled - nothing to rollback" | tee -a "$ROLLBACK_LOG"
    exit 0
fi

# Step 3: Graceful shutdown - allow current batch to complete
echo "Step 3: Initiating graceful shutdown..." | tee -a "$ROLLBACK_LOG"
echo "Waiting for current batch to complete (max 60 seconds)..." | tee -a "$ROLLBACK_LOG"

# Check for running processes
ZLUDA_PROCS=$(pgrep -f zluda | wc -l)
echo "Active ZLUDA processes: $ZLUDA_PROCS" | tee -a "$ROLLBACK_LOG"

if timeout 60 bash -c 'while pgrep -f zluda > /dev/null; do sleep 1; done' 2>/dev/null; then
    echo "✅ Graceful shutdown completed" | tee -a "$ROLLBACK_LOG"
else
    echo "⚠️  Graceful shutdown timeout - proceeding with termination" | tee -a "$ROLLBACK_LOG"
    pkill -TERM -f zluda || true
    sleep 10
    # Force kill if still running
    pgrep -f zluda >/dev/null && pkill -9 -f zluda || true
    sleep 5
fi

# Verify shutdown
ZLUDA_PROCS_AFTER=$(pgrep -f zluda | wc -l)
if [ "$ZLUDA_PROCS_AFTER" -eq 0 ]; then
    echo "✅ All ZLUDA processes terminated" | tee -a "$ROLLBACK_LOG"
else
    echo "⚠️  Warning: $ZLUDA_PROCS_AFTER processes still running" | tee -a "$ROLLBACK_LOG"
fi

# Step 4: Disable batch scheduler and clean environment
echo "Step 4: Disabling batch scheduler..." | tee -a "$ROLLBACK_LOG"
unset BATCH_SCHEDULER_ENABLED
export BATCH_SCHEDULER_ENABLED=0
unset BATCH_SIZE INSTANCE_COUNT PIPELINE_ENABLE_DOUBLE_BUFFER PIPELINE_ENABLE_PREFETCH
unset KIMI_TARGET_TPS MATMULFREELLM_TARGET_TPS HEALTH_CHECK_INTERVAL FALLBACK_THRESHOLD
unset BATCH_SCHEDULER_STATS_FILE BATCH_SCHEDULER_AUDIT_LOG

echo "✅ Batch scheduler disabled" | tee -a "$ROLLBACK_LOG"

# Step 5: Restore backup with verification
echo "Step 5: Restoring backup..." | tee -a "$ROLLBACK_LOG"
if git stash list | grep -q "pre-deployment-backup"; then
    BACKUP_STASH=$(git stash list | grep "pre-deployment-backup" | head -1 | cut -d: -f1)
    echo "Found backup: $BACKUP_STASH" | tee -a "$ROLLBACK_LOG"

    if git stash pop "$BACKUP_STASH" --index; then
        echo "✅ Backup restored successfully" | tee -a "$ROLLBACK_LOG"
    else
        echo "ERROR: Failed to restore backup" | tee -a "$ROLLBACK_LOG"
        exit 1
    fi
else
    echo "ℹ️  No backup found, using current code" | tee -a "$ROLLBACK_LOG"
fi

# Step 6: Restart services (idempotent)
echo "Step 6: Restarting services..." | tee -a "$ROLLBACK_LOG"
timeout 600 cargo build --release --bin zluda || {
    echo "ERROR: Build failed during rollback" | tee -a "$ROLLBACK_LOG"
    exit 1
}

# Stop any existing instances (idempotent)
pkill -f zluda || true
sleep 5

# Start fresh instances
cargo run --release --bin zluda &
ZLUDA_PID=$!

# Step 7: Verification
echo "Step 7: Verifying rollback..." | tee -a "$ROLLBACK_LOG"
sleep 10

if pgrep -f zluda > /dev/null; then
    RUNNING_PROCS=$(pgrep -f zluda | wc -l)
    echo "✅ ZLUDA services restarted ($RUNNING_PROCS processes)" | tee -a "$ROLLBACK_LOG"
    echo "=== GRACEFUL ROLLBACK COMPLETED SUCCESSFULLY $(date) ===" | tee -a "$ROLLBACK_LOG"
else
    echo "❌ Failed to restart services - manual intervention required" | tee -a "$ROLLBACK_LOG"
    echo "=== GRACEFUL ROLLBACK FAILED $(date) ===" | tee -a "$ROLLBACK_LOG"
    exit 1
fi

echo "Monitor system for baseline performance restoration"
echo "Check $ROLLBACK_LOG for detailed rollback information"
```

## Rollback Verification
After rollback, verify baseline performance with comprehensive checks:

### Performance Verification
```bash
# Automated verification script
echo "=== Rollback Performance Verification ==="

# Source the audit log location
if [ -f "$HOME/.hetgpu_audit.log" ]; then
    echo "Recent audit log entries:"
    tail -20 "$HOME/.hetgpu_audit.log"
fi

# Check if batch scheduler is disabled
if [ "${BATCH_SCHEDULER_ENABLED:-0}" = "0" ]; then
    echo "✅ Batch scheduler confirmed disabled"
else
    echo "⚠️  Warning: Batch scheduler may still be enabled"
fi

# Verify process state
ZLUDA_PROCS=$(pgrep -f zluda | wc -l)
echo "Active ZLUDA processes: $ZLUDA_PROCS"

if [ "$ZLUDA_PROCS" -gt 0 ]; then
    echo "✅ ZLUDA processes running"
else
    echo "❌ ERROR: No ZLUDA processes detected"
fi

# Check for errors in recent logs
echo "Checking for errors in recent logs..."
find /var/log $HOME/.hetgpu_logs -name "*batch*" -type f -mmin -10 2>/dev/null | while read log; do
    ERRORS=$(grep -i error "$log" | wc -l)
    if [ "$ERRORS" -gt 0 ]; then
        echo "⚠️  Found $ERRORS errors in $log"
    fi
done

echo "=== Manual verification required ==="
echo "Confirm the following metrics return to baseline:"
```

### Baseline Metrics
- **Kimi TPS returns to baseline**: 0.62 TPS
- **matmulfreellm TPS returns to baseline**: 12.85 TPS
- **Error rate**: Returns to normal (< 0.1%)
- **System stability**: No crashes or hangs
- **Resource usage**: CPU/memory return to normal levels
- **User feedback**: Complaints resolved

### Automated Verification Commands
```bash
# Quick health check
watch -n 5 'echo "=== Baseline Verification ==="; echo "ZLUDA processes: $(pgrep -f zluda | wc -l)"; echo "Batch scheduler: $([ "${BATCH_SCHEDULER_ENABLED:-0}" = "0" ] && echo "DISABLED ✅" || echo "ENABLED ⚠️")"; echo "Memory: $(free -h | grep Mem)"; uptime'

# Detailed log analysis
tail -100 /var/log/syslog | grep -i zluda || tail -100 $HOME/.hetgpu_logs/*.log | grep -i error

# Performance baseline check (if monitoring available)
# Compare current TPS against known baseline values
```

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
# Load configuration to get correct paths
source deploy/production_config.sh

# Terminal 1: Monitor TPS (from configurable stats file)
watch -n 5 'tail -5 "${BATCH_SCHEDULER_STATS_FILE}" | jq -r "select(.tps) | .tps" 2>/dev/null || echo "No TPS data yet"'

# Terminal 2: Monitor errors (using configurable paths)
tail -f "${BATCH_SCHEDULER_STATS_FILE}" | grep -i error

# Terminal 3: Monitor audit logs
tail -f "${BATCH_SCHEDULER_AUDIT_LOG}"

# Terminal 4: Monitor instance health
watch -n 10 'ps aux | grep zluda | grep -v grep | wc -l | xargs echo "ZLUDA processes:"'

# Alternative: Comprehensive system monitoring
watch -n 5 'echo "=== System Status ==="; echo "ZLUDA processes: $(pgrep -f zluda | wc -l)"; echo "Memory usage: $(free -h | grep Mem)"; echo "Load average: $(uptime)"; echo "Batch scheduler: $([ "${BATCH_SCHEDULER_ENABLED:-0}" = "1" ] && echo "ENABLED" || echo "DISABLED")"'

# Check for recent errors in all log locations
echo "Checking for recent errors..."
find /var/log $HOME/.hetgpu_logs -name "*batch*" -type f -mmin -10 2>/dev/null | while read log; do
    echo "=== $log ==="
    tail -10 "$log" | grep -i error || echo "No recent errors"
done
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
Deployment considered successful if all criteria are met:

### Automated Criteria
- All integration tests pass
- TPS targets met or exceeded:
  - Kimi IQ1S hybrid: ≥ 8 TPS
  - matmulfreellm 2.7B: ≥ 2000 TPS
- Error rate < 0.1%
- Instance utilization > 85%
- No rollback triggered for 1 hour

### Manual Verification Required
- System stability confirmed (no crashes/hangs)
- Resource usage within acceptable limits
- User acceptance testing passes
- Monitoring dashboards show expected improvements
- Audit logs show clean deployment with no critical errors

### Rollback Success Criteria
Rollback considered successful if:
- All batch scheduler processes terminated
- Backup restored without errors
- Baseline ZLUDA services restarted
- Performance returns to baseline levels
- No new error patterns introduced
- Audit trail shows clean rollback procedure