#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
PREFIX=${1:?unique log prefix required}
BUILD_ID=${2:?verified firmware marker required}
MODE=${3:-full}
[[ "$MODE" == full || "$MODE" == kernels ]] || exit 2
: "${CONCORDIA_RELEASE:?set the candidate runtime directory}"
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ && "$BUILD_ID" =~ ^0x[0-9a-fA-F]+$ ]] || exit 2
test ! -e "logs/$PREFIX-validation.log"
python3 bin/pacc-preflight.py
export Q8_TEST_LIBRARY="$CONCORDIA_RELEASE/libnvcuda.so"
bash bin/run-q8-validation.sh "$PREFIX-identity" "$BUILD_ID" > "logs/$PREFIX-identity-driver.log" 2>&1
source bin/pacc-env.sh
export HETGPU_SIFIVE_SHARED_DDR_BASE=0x2075d000000 HETGPU_SIFIVE_SHARED_DDR_BYTES=0x400000
export HETGPU_SIFIVE_JOBD_BOOTSTRAP=0 HETGPU_SIFIVE_JOB_TIMEOUT_MS=10000
export HETGPU_SIFIVE_ZLUDA_IRQ_TRACE=0 HETGPU_SIFIVE_LOG_MEMORY=0
export HETGPU_SIFIVE_GEMM_SHARED_SLOTS=1 HETGPU_SIFIVE_GEMM_DEVICES=0
for test in test-concat-cuda-abi test-gdn-hardware test-gdn-cuda-abi \
    test-topk-hardware test-topk-cuda-abi test-mmvf-tiles test-mmvf-heads \
    test-kv-scatter test-softmax-cuda; do
    timeout 180 python3 "bin/$test.py" > "logs/$PREFIX-$test.log" 2>&1
    printf 'PASS %s\n' "$test" >> "logs/$PREFIX-validation.log"
done
for qtype in 4 5 8; do
    QK_TEST_TYPE=$qtype QK_TEST_BIAS=1 timeout 180 python3 bin/test-q4k-cuda-abi.py \
        > "logs/$PREFIX-q$qtype-bias.log" 2>&1
    printf 'PASS quantized type %s and fusion bias\n' "$qtype" >> "logs/$PREFIX-validation.log"
done
SOFTMAX_REAL_PTX=1 timeout 180 python3 bin/test-softmax-cuda.py \
    > "logs/$PREFIX-softmax-real-ptx.log" 2>&1
QK_TEST_TYPE=8 QK_TEST_SMALL=1 timeout 180 python3 bin/test-q4k-cuda-abi.py \
    > "logs/$PREFIX-q8-small.log" 2>&1
printf 'PASS real-PTX softmax and small-K Q8\n' >> "logs/$PREFIX-validation.log"
if [[ "$MODE" == kernels ]]; then
    printf 'KERNELS_ONLY: no end-to-end generation or TPS result\n' >> "logs/$PREFIX-validation.log"
    exit 0
fi
bash bin/run-xm-qwen-benchmark.sh "$PREFIX-rawmath" "$BUILD_ID" 1 rawmath 99 \
    > "logs/$PREFIX-rawmath-driver.log" 2>&1
printf 'PASS ngl99 rawmath; see JSON for token verification and timing scope\n' \
    >> "logs/$PREFIX-validation.log"
