#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
PREFIX=${1:?capture prefix required}
BUILD_ID=${2:?firmware marker required}
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ && "$BUILD_ID" =~ ^0x[0-9a-fA-F]+$ ]] || exit 2
test ! -e "logs/$PREFIX.json"
[[ $(</proc/sys/kernel/random/boot_id) == 89a6c374-059a-40da-8c4f-c00179fd48d4 ]] || {
    echo 'Different boot: requalify owned DDR and firmware' >&2; exit 2;
}
python3 bin/pacc-preflight.py
source bin/pacc-env.sh
: "${Q8_TEST_LIBRARY:?candidate libnvcuda.so required}"
[[ "$Q8_TEST_LIBRARY" == "$CONCORDIA_RELEASE/libnvcuda.so" ]] || {
    echo 'Candidate library must match the preloaded runtime directory' >&2; exit 2;
}
test -r "$Q8_TEST_LIBRARY"
export HETGPU_SIFIVE_SHARED_DDR_BASE=0x2075d000000 HETGPU_SIFIVE_SHARED_DDR_BYTES=0x400000
export HETGPU_SIFIVE_JOBD_BOOTSTRAP=0 HETGPU_SIFIVE_JOB_TIMEOUT_MS=10000
export HETGPU_SIFIVE_LOG_MEMORY=0 HETGPU_SIFIVE_ZLUDA_IRQ_TRACE=0
export HETGPU_SIFIVE_MMVQ_PROFILE=1 HETGPU_SIFIVE_MMVQ_TILE_ROWS=512
export HETGPU_SIFIVE_MMVF_ASYNC_NO_WAIT=0 HETGPU_SIFIVE_SKIP_JOB_WAIT=0
python3 bin/profile-jobd-phases.py --output "logs/$PREFIX.json" --timeout 180 \
    --sample-ms 1 --expected-submissions 21 --require-build "$BUILD_ID" \
    -- bin/test_jobd_q8 --hardware
