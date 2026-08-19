#!/usr/bin/env bash
# /home/victoryang00/hetGPU/deploy/production_config.sh

set -euo pipefail

# Dry-run mode - validate configuration without making changes
DRY_RUN=${DRY_RUN:-false}
if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN MODE - No changes will be made ==="
fi

echo "=== Production Deployment Configuration ==="

# Audit logging setup
AUDIT_LOG=${AUDIT_LOG:-"$HOME/.hetgpu_audit.log"}
echo "$(date '+%Y-%m-%d %H:%M:%S') - START: production_config.sh sourced by USER=$(whoami) DRY_RUN=$DRY_RUN" >> "$AUDIT_LOG" 2>/dev/null || true

# Environment validation and dependency checking
echo "Validating deployment environment..."

# Check if we're in the correct directory
if [[ ! -f "Cargo.toml" ]] || [[ ! -d "zluda" ]]; then
    echo "❌ ERROR: Not in hetGPU root directory"
    echo "   Current directory: $(pwd)"
    echo "   Required: Run from /home/victoryang00/hetGPU directory"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Invalid deployment directory $(pwd)" >> "$AUDIT_LOG"
    exit 1
fi

# Check for required dependencies
MISSING_DEPS=()
command -v cargo >/dev/null 2>&1 || MISSING_DEPS+=("cargo")
command -v rustc >/dev/null 2>&1 || MISSING_DEPS+=("rustc")
command -v jq >/dev/null 2>&1 || MISSING_DEPS+=("jq")

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    echo "❌ ERROR: Missing required dependencies: ${MISSING_DEPS[*]}"
    echo "   Install missing tools before deployment"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Missing dependencies ${MISSING_DEPS[*]}" >> "$AUDIT_LOG"
    exit 1
fi

# Check for FPGA hardware availability
if [[ ! -d "/dev/dri" ]] && [[ ! -d "/dev/xilinx" ]] && [[ ! -d "/dev/intel" ]]; then
    echo "⚠️  WARNING: No standard FPGA device directories found"
    echo "   This may indicate missing FPGA hardware or drivers"
    echo "   Continuing deployment but hardware availability not confirmed"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: No FPGA hardware detected" >> "$AUDIT_LOG"
fi

echo "✅ Environment validation passed"
echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: Environment validation successful" >> "$AUDIT_LOG"

# Environment setup
export BATCH_SCHEDULER_ENABLED=1
export BATCH_SIZE=64
export INSTANCE_COUNT=16
export PIPELINE_ENABLE_DOUBLE_BUFFER=1
export PIPELINE_ENABLE_PREFETCH=1

# Performance targets
export KIMI_TARGET_TPS=8
export MATMULFREELLM_TARGET_TPS=2000

# Health monitoring
export HEALTH_CHECK_INTERVAL=100
export FALLBACK_THRESHOLD=0.1

# Logging
export BATCH_SCHEDULER_LOG_LEVEL=info
export BATCH_SCHEDULER_STATS_FILE=${BATCH_SCHEDULER_STATS_FILE:-"/var/log/batch_scheduler_stats.jsonl"}
export BATCH_SCHEDULER_AUDIT_LOG="$AUDIT_LOG"

# Create log directory if it doesn't exist
LOG_DIR=$(dirname "$BATCH_SCHEDULER_STATS_FILE")
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: Creating log directory $LOG_DIR" >> "$AUDIT_LOG"

    # Check sudo availability and handle gracefully with security fixes
    if command -v sudo &> /dev/null && sudo -n true 2> /dev/null; then
        # Sudo is available and has no-password privilege
        CURRENT_USER=$(whoami)
        CURRENT_GROUP=$(id -gn)
        sudo mkdir -p "$LOG_DIR" 2>/dev/null || {
            echo "⚠️  Warning: Could not create log directory with sudo"
            echo "Attempting to create in user directory..."
            LOG_DIR="$HOME/.hetgpu_logs"
            mkdir -p "$LOG_DIR"
            export BATCH_SCHEDULER_STATS_FILE="$LOG_DIR/batch_scheduler_stats.jsonl"
            echo "Using alternative location: $BATCH_SCHEDULER_STATS_FILE"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Using alternative log location $LOG_DIR" >> "$AUDIT_LOG"
        }
        # Use safer chown with explicit user and group
        sudo chown "$CURRENT_USER:$CURRENT_GROUP" "$LOG_DIR" 2>/dev/null || true
        echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: Log directory permissions set for $CURRENT_USER:$CURRENT_GROUP" >> "$AUDIT_LOG"
    elif [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: Would create log directory: $LOG_DIR"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - DRY_RUN: Would create log directory $LOG_DIR" >> "$AUDIT_LOG"
    else
        # No sudo available, try user directory
        echo "⚠️  Sudo not available - using user directory for logs"
        LOG_DIR="$HOME/.hetgpu_logs"
        mkdir -p "$LOG_DIR" || {
            echo "❌ Failed to create log directory: $LOG_DIR"
            echo "Suggested actions:"
            echo "  1. Run 'sudo mkdir -p $LOG_DIR && sudo chown $(whoami):$(id -gn) $LOG_DIR'"
            echo "  2. Or set BATCH_SCHEDULER_STATS_FILE to a writable location"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Failed to create log directory $LOG_DIR" >> "$AUDIT_LOG"
            exit 1
        }
        export BATCH_SCHEDULER_STATS_FILE="$LOG_DIR/batch_scheduler_stats.jsonl"
        echo "Using alternative location: $BATCH_SCHEDULER_STATS_FILE"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Using alternative log location $LOG_DIR" >> "$AUDIT_LOG"
    fi
fi

echo "Production configuration loaded successfully"
echo "Batch Size: $BATCH_SIZE"
echo "Instance Count: $INSTANCE_COUNT"
echo "Kimi Target TPS: $KIMI_TARGET_TPS"
echo "matmulfreellm Target TPS: $MATMULFREELLM_TARGET_TPS"
echo "Stats File: $BATCH_SCHEDULER_STATS_FILE"
echo "Audit Log: $BATCH_SCHEDULER_AUDIT_LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: Configuration loaded - BATCH_SIZE=$BATCH_SIZE INSTANCE_COUNT=$INSTANCE_COUNT KIMI_TPS=$KIMI_TARGET_TPS MATMUL_TPS=$MATMULFREELLM_TARGET_TPS" >> "$AUDIT_LOG"

# Comprehensive verification of ALL configuration parameters
echo ""
echo "Verifying comprehensive configuration..."
VALIDATION_FAILED=false
VALIDATION_WARNINGS=()

# Core configuration validation
if [ "$BATCH_SCHEDULER_ENABLED" != "1" ]; then
    echo "❌ BATCH_SCHEDULER_ENABLED not set to 1"
    echo "   Current value: $BATCH_SCHEDULER_ENABLED"
    echo "   Fix: export BATCH_SCHEDULER_ENABLED=1"
    VALIDATION_FAILED=true
fi

if [ -z "$BATCH_SIZE" ] || [ "$BATCH_SIZE" -lt 1 ]; then
    echo "❌ Invalid BATCH_SIZE: $BATCH_SIZE"
    echo "   Fix: export BATCH_SIZE=64"
    VALIDATION_FAILED=true
elif [ "$BATCH_SIZE" -gt 128 ]; then
    echo "⚠️  WARNING: BATCH_SIZE=$BATCH_SIZE exceeds recommended maximum (128)"
    VALIDATION_WARNINGS+=("Large batch size may impact memory usage")
fi

if [ -z "$INSTANCE_COUNT" ] || [ "$INSTANCE_COUNT" -lt 1 ]; then
    echo "❌ Invalid INSTANCE_COUNT: $INSTANCE_COUNT"
    echo "   Fix: export INSTANCE_COUNT=16"
    VALIDATION_FAILED=true
elif [ "$INSTANCE_COUNT" -gt 32 ]; then
    echo "⚠️  WARNING: INSTANCE_COUNT=$INSTANCE_COUNT exceeds tested maximum (32)"
    VALIDATION_WARNINGS+=("High instance count may require additional FPGA resources")
fi

# Performance target validation
if [ -z "$KIMI_TARGET_TPS" ] || [ "$KIMI_TARGET_TPS" -lt 1 ]; then
    echo "❌ Invalid KIMI_TARGET_TPS: $KIMI_TARGET_TPS"
    echo "   Fix: export KIMI_TARGET_TPS=8"
    VALIDATION_FAILED=true
fi

if [ -z "$MATMULFREELLM_TARGET_TPS" ] || [ "$MATMULFREELLM_TARGET_TPS" -lt 1 ]; then
    echo "❌ Invalid MATMULFREELLM_TARGET_TPS: $MATMULFREELLM_TARGET_TPS"
    echo "   Fix: export MATMULFREELLM_TARGET_TPS=2000"
    VALIDATION_FAILED=true
fi

# Health monitoring validation
if [ -z "$HEALTH_CHECK_INTERVAL" ] || [ "$HEALTH_CHECK_INTERVAL" -lt 10 ]; then
    echo "⚠️  WARNING: HEALTH_CHECK_INTERVAL=$HEALTH_CHECK_INTERVAL below recommended minimum (10)"
    VALIDATION_WARNINGS+=("Health check interval too frequent may impact performance")
fi

if [ -z "$FALLBACK_THRESHOLD" ] || [ "$(echo "$FALLBACK_THRESHOLD > 0.5" | bc -l 2>/dev/null || echo 1)" -eq 1 ]; then
    echo "⚠️  WARNING: FALLBACK_THRESHOLD=$FALLBACK_THRESHOLD above recommended maximum (0.5)"
    VALIDATION_WARNINGS+=("High fallback threshold may reduce system availability")
fi

# Pipeline configuration validation
if [ "${PIPELINE_ENABLE_DOUBLE_BUFFER:-0}" != "1" ] && [ "${PIPELINE_ENABLE_DOUBLE_BUFFER:-0}" != "0" ]; then
    echo "❌ Invalid PIPELINE_ENABLE_DOUBLE_BUFFER: ${PIPELINE_ENABLE_DOUBLE_BUFFER:-0}"
    echo "   Fix: export PIPELINE_ENABLE_DOUBLE_BUFFER=1"
    VALIDATION_FAILED=true
fi

if [ "${PIPELINE_ENABLE_PREFETCH:-0}" != "1" ] && [ "${PIPELINE_ENABLE_PREFETCH:-0}" != "0" ]; then
    echo "❌ Invalid PIPELINE_ENABLE_PREFETCH: ${PIPELINE_ENABLE_PREFETCH:-0}"
    echo "   Fix: export PIPELINE_ENABLE_PREFETCH=1"
    VALIDATION_FAILED=true
fi

# Logging configuration validation
if [ -z "$BATCH_SCHEDULER_LOG_LEVEL" ]; then
    echo "❌ Missing BATCH_SCHEDULER_LOG_LEVEL"
    echo "   Fix: export BATCH_SCHEDULER_LOG_LEVEL=info"
    VALIDATION_FAILED=true
elif [[ ! "$BATCH_SCHEDULER_LOG_LEVEL" =~ ^(debug|info|warn|error)$ ]]; then
    echo "⚠️  WARNING: BATCH_SCHEDULER_LOG_LEVEL=$BATCH_SCHEDULER_LOG_LEVEL not standard"
    VALIDATION_WARNINGS+=("Non-standard log level may not work as expected")
fi

# Log file path validation
if [ -z "$BATCH_SCHEDULER_STATS_FILE" ]; then
    echo "❌ Missing BATCH_SCHEDULER_STATS_FILE"
    echo "   Fix: export BATCH_SCHEDULER_STATS_FILE=/var/log/batch_scheduler_stats.jsonl"
    VALIDATION_FAILED=true
fi

if [ "$VALIDATION_FAILED" = true ]; then
    echo ""
    echo "=== Configuration validation FAILED ==="
    echo "Required actions:"
    echo "  1. Fix the environment variables listed above"
    echo "  2. Run: source deploy/production_config.sh"
    echo "  3. Or use rollback procedure from deploy/rollback_plan.md"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: Configuration validation failed" >> "$AUDIT_LOG"
    exit 1
fi

echo "✅ Configuration verification passed"
if [ ${#VALIDATION_WARNINGS[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Configuration warnings detected:"
    for warning in "${VALIDATION_WARNINGS[@]}"; do
        echo "   - $warning"
    done
    echo "$(date '+%Y-%m-%d %H:%M:%S') - WARNING: Configuration validation completed with warnings" >> "$AUDIT_LOG"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') - INFO: Configuration validation successful" >> "$AUDIT_LOG"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN COMPLETE - Configuration valid ==="
    echo "To apply changes, run: source deploy/production_config.sh"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - DRY_RUN: Configuration validation completed successfully" >> "$AUDIT_LOG"
    exit 0
fi

echo "=== Ready for production deployment ==="
echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: Production configuration loaded successfully" >> "$AUDIT_LOG"

# Display configuration summary for operator verification
echo ""
echo "Configuration Summary:"
echo "  - Batch Scheduler: ENABLED"
echo "  - Processing: ${BATCH_SIZE} requests per batch"
echo "  - Instances: ${INSTANCE_COUNT} FPGA instances"
echo "  - Double Buffering: $([ "${PIPELINE_ENABLE_DOUBLE_BUFFER:-0}" = "1" ] && echo "ENABLED" || echo "DISABLED")"
echo "  - Prefetching: $([ "${PIPELINE_ENABLE_PREFETCH:-0}" = "1" ] && echo "ENABLED" || echo "DISABLED")"
echo "  - Health Check Interval: ${HEALTH_CHECK_INTERVAL}ms"
echo "  - Fallback Threshold: ${FALLBACK_THRESHOLD}"
echo ""
echo "Performance Targets:"
echo "  - Kimi IQ1S hybrid: ${KIMI_TARGET_TPS} TPS"
echo "  - matmulfreellm 2.7B: ${MATMULFREELLM_TARGET_TPS} TPS"
echo ""
echo "Monitoring and Logging:"
echo "  - Stats File: ${BATCH_SCHEDULER_STATS_FILE}"
echo "  - Audit Log: ${BATCH_SCHEDULER_AUDIT_LOG}"
echo "  - Log Level: ${BATCH_SCHEDULER_LOG_LEVEL}"
echo ""