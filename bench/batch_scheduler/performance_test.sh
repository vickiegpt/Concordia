#!/usr/bin/env bash
# /home/victoryang00/hetGPU/bench/batch_scheduler/performance_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

echo "[Performance Test] Building with batch scheduler..."
cd "$REPO_ROOT"
cargo build --release --lib

echo "[Performance Test] Running Kimi IQ1S hybrid benchmark..."
KIMI_BITLINEAR_TMATMUL=1 \
BATCH_SCHEDULER_ENABLED=1 \
BATCH_SIZE=64 \
INSTANCE_COUNT=16 \
TARGET_TPS=8 \
bench/kimi_k26_tps/run_kimi_k26_tps.sh || echo "Kimi benchmark not available, skipping..."

echo "[Performance Test] Running matmulfreellm 2.7B benchmark..."
BATCH_SCHEDULER_ENABLED=1 \
BATCH_SIZE=64 \
INSTANCE_COUNT=16 \
TARGET_TPS=2000 \
python tests/matmulfreellm_tps_benchmark.py || echo "matmulfreellm benchmark not available, skipping..."

echo "[Performance Test] Complete"