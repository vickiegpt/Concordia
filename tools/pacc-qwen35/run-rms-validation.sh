#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
[[ $(</proc/sys/kernel/random/boot_id) == 89a6c374-059a-40da-8c4f-c00179fd48d4 ]] || exit 2
python3 bin/pacc-preflight.py
source bin/pacc-env.sh
export HETGPU_SIFIVE_SHARED_DDR_BASE=0x2075d000000 HETGPU_SIFIVE_SHARED_DDR_BYTES=0x400000
export HETGPU_SIFIVE_JOBD_BOOTSTRAP=0 HETGPU_SIFIVE_JOB_TIMEOUT_MS=10000
export HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK=0 HETGPU_SIFIVE_SKIP_JOB_WAIT=0
export HETGPU_SIFIVE_RMSNORM_WAIT_OUTPUT=1 HETGPU_SIFIVE_RMSNORM_ASSUME_SUCCESS_ON_OUTPUT_TIMEOUT=0
export RMS_TEST_LIBRARY="$CONCORDIA_RELEASE/libnvcuda.so"
timeout 180 bin/test-rms-hardware
