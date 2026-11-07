#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
PREFIX=${1:?unique log prefix required}
BUILD_ID=${2:?verified firmware build marker required}
PARALLEL=${3:-4}
WORKLOAD=${4:-count16}
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$PARALLEL" =~ ^[1-8]$ ]] || exit 2
[[ "$BUILD_ID" =~ ^0x[0-9a-fA-F]+$ ]] || exit 2
case "$WORKLOAD" in
    count16) SUBMISSIONS=8245 ;;
    count64) SUBMISSIONS=31525 ;;
    *) echo 'supported workloads: count16, count64' >&2; exit 2 ;;
esac
test ! -e "logs/phases-$PREFIX.json"
test ! -e "logs/$PREFIX.json"
[[ $(</proc/sys/kernel/random/boot_id) == 89a6c374-059a-40da-8c4f-c00179fd48d4 ]] || {
    echo 'Different boot: requalify owned DDR and firmware before benchmarking' >&2; exit 2;
}
python3 bin/pacc-preflight.py
source bin/pacc-env.sh
export HETGPU_SIFIVE_SHARED_DDR_BASE=0x2075d000000 HETGPU_SIFIVE_SHARED_DDR_BYTES=0x400000
export HETGPU_SIFIVE_JOBD_BOOTSTRAP=0 HETGPU_SIFIVE_JOB_TIMEOUT_MS=10000
export HETGPU_SIFIVE_LOG_MEMORY=0 HETGPU_SIFIVE_ZLUDA_IRQ_TRACE=0 HETGPU_SIFIVE_MMVQ_PROFILE=1
export HETGPU_SIFIVE_MMVF_ASYNC_NO_WAIT=0 HETGPU_SIFIVE_SKIP_JOB_WAIT=0
# Negative polling experiments must not silently alter the default comparison.
unset HETGPU_SIFIVE_RESPONSE_POLL HETGPU_SIFIVE_STATUS_POLL_SLEEP_US
python3 bin/profile-jobd-phases.py --output "logs/phases-$PREFIX.json" \
    --timeout 900 --sample-ms 1 --expected-submissions "$SUBMISSIONS" --require-build "$BUILD_ID" \
    -- python3 bin/bench-qwen-requests.py --mode pacc --parallel "$PARALLEL" \
    --workload "$WORKLOAD" --prefix "/srv/lanxin/qwen35-a3b/logs/$PREFIX"
