#!/usr/bin/env bash
# Batch Scheduler Health Check Script
# Version: 1.0.0
# Last Updated: 2026-08-19

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Health check counters
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

print_header() {
    echo "================================================"
    echo "$1"
    echo "================================================"
}

print_header "Batch Scheduler Health Check"
echo "Repository: $REPO_ROOT"
echo "Date: $(date)"
echo ""

# Check 1: Batch Scheduler Enabled
print_header "Check 1: Batch Scheduler Configuration"
if [[ "${BATCH_SCHEDULER_ENABLED:-0}" == "1" ]]; then
    check_pass "Batch scheduler is enabled (BATCH_SCHEDULER_ENABLED=1)"
else
    log_warning "Batch scheduler is not enabled (BATCH_SCHEDULER_ENABLED=${BATCH_SCHEDULER_ENABLED:-0})"
fi

# Check batch size
BATCH_SIZE=${BATCH_SIZE:-64}
if [[ "$BATCH_SIZE" -ge 16 && "$BATCH_SIZE" -le 128 ]]; then
    check_pass "Batch size is reasonable: $BATCH_SIZE"
else
    log_warning "Batch size may be suboptimal: $BATCH_SIZE (recommended: 16-128)"
fi

# Check instance count
INSTANCE_COUNT=${INSTANCE_COUNT:-16}
if [[ "$INSTANCE_COUNT" == "16" ]]; then
    check_pass "Instance count set to 16"
else
    log_warning "Instance count is $INSTANCE_COUNT (expected 16)"
fi

echo ""

# Check 2: Build Status
print_header "Check 2: Build Status"
if [[ -f "$REPO_ROOT/target/release/libzluda.so" ]]; then
    check_pass "Release build exists: libzluda.so"
else
    log_warning "Release build not found. Run: cargo build --release"
fi

if [[ -f "$REPO_ROOT/Cargo.toml" ]]; then
    check_pass "Project structure valid (Cargo.toml exists)"
else
    check_fail "Project structure invalid (Cargo.toml missing)"
fi

echo ""

# Check 3: Device Access
print_header "Check 3: Device Access"
CXL_DEVICES=$(ls /dev/cxl_tmatmul* 2>/dev/null | wc -l)
if [[ "$CXL_DEVICES" -gt 0 ]]; then
    check_pass "CXL devices found: $CXL_DEVICES devices"
    for device in /dev/cxl_tmatmul*; do
        if [[ -r "$device" && -w "$device" ]]; then
            check_pass "Device accessible: $(basename $device)"
        else
            log_error "Device permissions issue: $(basename $device)"
        fi
    done
else
    log_error "No CXL devices found at /dev/cxl_tmatmul*"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    check_pass "nvidia-smi available"
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)
    log_info "GPU free memory: ${GPU_MEMORY} MB"
else
    log_warning "nvidia-smi not available"
fi

echo ""

# Check 4: Memory and Resources
print_header "Check 4: System Resources"
# Check available memory
TOTAL_MEM=$(free -g | awk '/^Mem:/ {print $2}')
AVAIL_MEM=$(free -g | awk '/^Mem:/ {print $7}')
log_info "Available memory: ${AVAIL_MEM}GB / ${TOTAL_MEM}GB"

if [[ "$AVAIL_MEM" -lt 4 ]]; then
    log_warning "Low memory available (${AVAIL_MEM}GB < 4GB)"
else
    check_pass "Sufficient memory available"
fi

# Check disk space
DISK_AVAIL=$(df -h "$REPO_ROOT" | awk 'NR==2 {print $4}')
DISK_AVAIL_GB=$(df -BG "$REPO_ROOT" | awk 'NR==2 {print $4}' | tr -d 'G')
log_info "Available disk space: ${DISK_AVAIL}"

if [[ "$DISK_AVAIL_GB" -lt 10 ]]; then
    log_warning "Low disk space (${DISK_AVAIL} < 10GB)"
else
    check_pass "Sufficient disk space"
fi

echo ""

# Check 5: Log Directory and Files
print_header "Check 5: Logging Configuration"
if [[ -d "/var/log" ]]; then
    check_pass "Log directory exists"
    STATS_FILE="${BATCH_SCHEDULER_STATS_FILE:-/var/log/batch_scheduler_stats.jsonl}"
    STATS_DIR=$(dirname "$STATS_FILE")

    if [[ -d "$STATS_DIR" ]]; then
        check_pass "Stats log directory exists: $STATS_DIR"
    else
        log_warning "Stats log directory doesn't exist: $STATS_DIR"
    fi

    if [[ -f "$STATS_FILE" ]]; then
        check_pass "Stats log file exists: $STATS_FILE"
        RECENT_ENTRIES=$(tail -5 "$STATS_FILE" 2>/dev/null | wc -l)
        log_info "Recent log entries: $RECENT_ENTRIES"
    else
        log_warning "Stats log file doesn't exist: $STATS_FILE"
    fi
else
    log_warning "Log directory not accessible"
fi

echo ""

# Check 6: Benchmark Scripts
print_header "Check 6: Benchmark Scripts Availability"
if [[ -f "$REPO_ROOT/bench/kimi_k26_tps/run_kimi_k26_tps.sh" ]]; then
    check_pass "Kimi benchmark script exists"
else
    log_warning "Kimi benchmark script not found"
fi

if [[ -f "$REPO_ROOT/bench/batch_scheduler/performance_test.sh" ]]; then
    check_pass "Performance test script exists"
else
    log_warning "Performance test script not found"
fi

echo ""

# Check 7: Performance Targets
print_header "Check 7: Performance Configuration"
KIMI_TARGET=${KIMI_TARGET_TPS:-8}
MATMUL_TARGET=${MATMULFREELLM_TARGET_TPS:-2000}
log_info "Kimi target TPS: $KIMI_TARGET"
log_info "matmulfreellm target TPS: $MATMUL_TARGET"

# Health score calculation
HEALTH_SCORE=100
if [[ "$WARNINGS" -gt 0 ]]; then
    HEALTH_SCORE=$((HEALTH_SCORE - WARNINGS * 5))
fi
if [[ "$CHECKS_FAILED" -gt 0 ]]; then
    HEALTH_SCORE=$((HEALTH_SCORE - CHECKS_FAILED * 20))
fi

echo ""
print_header "Health Check Summary"
echo "Total Checks: $((CHECKS_PASSED + CHECKS_FAILED + WARNINGS))"
echo -e "${GREEN}Passed:${NC} $CHECKS_PASSED"
echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
echo -e "${RED}Failed:${NC} $CHECKS_FAILED"
echo ""

if [[ "$CHECKS_FAILED" -eq 0 ]]; then
    if [[ "$HEALTH_SCORE" -ge 80 ]]; then
        echo -e "${GREEN}Overall Health: GOOD (${HEALTH_SCORE}%)${NC}"
        exit 0
    else
        echo -e "${YELLOW}Overall Health: FAIR (${HEALTH_SCORE}%)${NC}"
        exit 0
    fi
else
    echo -e "${RED}Overall Health: POOR (${HEALTH_SCORE}%)${NC}"
    echo ""
    echo "Recommended actions:"
    if [[ "$CXL_DEVICES" -eq 0 ]]; then
        echo "  - Check FPGA device connectivity and drivers"
    fi
    if [[ ! -f "$REPO_ROOT/target/release/libzluda.so" ]]; then
        echo "  - Build the project: cd $REPO_ROOT && cargo build --release"
    fi
    if [[ "$WARNINGS" -gt 2 ]]; then
        echo "  - Review warnings and address configuration issues"
    fi
    exit 1
fi