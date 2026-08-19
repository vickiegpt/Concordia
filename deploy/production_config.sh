#!/usr/bin/env bash
# /home/victoryang00/hetGPU/deploy/production_config.sh

set -euo pipefail

# Dry-run mode - validate configuration without making changes
DRY_RUN=${DRY_RUN:-false}
if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN MODE - No changes will be made ==="
fi

echo "=== Production Deployment Configuration ==="

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

# Create log directory if it doesn't exist
LOG_DIR=$(dirname "$BATCH_SCHEDULER_STATS_FILE")
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"

    # Check sudo availability and handle gracefully
    if command -v sudo &> /dev/null && sudo -n true 2> /dev/null; then
        # Sudo is available and has no-password privilege
        sudo mkdir -p "$LOG_DIR" 2>/dev/null || {
            echo "⚠️  Warning: Could not create log directory with sudo"
            echo "Attempting to create in user directory..."
            LOG_DIR="$HOME/.hetgpu_logs"
            mkdir -p "$LOG_DIR"
            export BATCH_SCHEDULER_STATS_FILE="$LOG_DIR/batch_scheduler_stats.jsonl"
            echo "Using alternative location: $BATCH_SCHEDULER_STATS_FILE"
        }
        sudo chown $USER:$USER "$LOG_DIR" 2>/dev/null || true
    elif [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY RUN: Would create log directory: $LOG_DIR"
    else
        # No sudo available, try user directory
        echo "⚠️  Sudo not available - using user directory for logs"
        LOG_DIR="$HOME/.hetgpu_logs"
        mkdir -p "$LOG_DIR" || {
            echo "❌ Failed to create log directory: $LOG_DIR"
            echo "Suggested actions:"
            echo "  1. Run 'sudo mkdir -p $LOG_DIR && sudo chown $USER:$USER $LOG_DIR'"
            echo "  2. Or set BATCH_SCHEDULER_STATS_FILE to a writable location"
            exit 1
        }
        export BATCH_SCHEDULER_STATS_FILE="$LOG_DIR/batch_scheduler_stats.jsonl"
        echo "Using alternative location: $BATCH_SCHEDULER_STATS_FILE"
    fi
fi

echo "Production configuration loaded successfully"
echo "Batch Size: $BATCH_SIZE"
echo "Instance Count: $INSTANCE_COUNT"
echo "Kimi Target TPS: $KIMI_TARGET_TPS"
echo "matmulfreellm Target TPS: $MATMULFREELLM_TARGET_TPS"
echo "Stats File: $BATCH_SCHEDULER_STATS_FILE"

# Verify configuration
echo ""
echo "Verifying configuration..."
VALIDATION_FAILED=false

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
fi

if [ -z "$INSTANCE_COUNT" ] || [ "$INSTANCE_COUNT" -lt 1 ]; then
    echo "❌ Invalid INSTANCE_COUNT: $INSTANCE_COUNT"
    echo "   Fix: export INSTANCE_COUNT=16"
    VALIDATION_FAILED=true
fi

if [ "$VALIDATION_FAILED" = true ]; then
    echo ""
    echo "=== Configuration validation FAILED ==="
    echo "Required actions:"
    echo "  1. Fix the environment variables listed above"
    echo "  2. Run: source deploy/production_config.sh"
    echo "  3. Or use rollback procedure from deploy/rollback_plan.md"
    exit 1
fi

echo "✅ Configuration verification passed"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "=== DRY RUN COMPLETE - Configuration valid ==="
    echo "To apply changes, run: source deploy/production_config.sh"
    exit 0
fi

echo "=== Ready for production deployment ==="