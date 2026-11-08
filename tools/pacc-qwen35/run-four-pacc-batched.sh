#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
PREFIX=${1:?unique result prefix required}
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
export CONCORDIA_RELEASE=${CONCORDIA_RELEASE:-$PWD/runtime-xm-local-batch-v5}
export HETGPU_SIFIVE_QK_LOCAL_BATCH=1 HETGPU_SIFIVE_QK_VECTOR_PACK=1
export HETGPU_SIFIVE_QK_EXPERT_BATCH=1
export HETGPU_SIFIVE_QK_PROFILE=0 HETGPU_SIFIVE_ELF_PROFILE=0
export HETGPU_SIFIVE_KERNEL_POST_DOORBELL_IRQ_SLEEP_US=1000
export HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES=1
QUALIFICATION="logs/$PREFIX-qualification"
test ! -e "logs/$PREFIX"
mkdir "$QUALIFICATION"
export Q8_TEST_LIBRARY="$CONCORDIA_RELEASE/libnvcuda.so"
for device in 0 1 2 3; do
    bash bin/run-pacc-isolated.sh "$device" -- python3 bin/profile-jobd-phases-four.py \
        --device "$device" --require-build 0x20036 --sample-ms 1 --timeout 180 \
        --output "$QUALIFICATION/pacc$device.json" -- bin/test_jobd_q8 --hardware "$device" \
        > "$QUALIFICATION/pacc$device-driver.log" 2>&1
done
exec python3 bin/run-four-pacc-smoke.py "$PREFIX" --runtime "$CONCORDIA_RELEASE" \
    --reference "${QWEN_CPU_REFERENCE:-logs/ngl99-direct-cpu-reference.json}" \
    --qualification "$QUALIFICATION" --timeout 5400
