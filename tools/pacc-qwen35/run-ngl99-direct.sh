#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
PREFIX=${1:?unique log prefix required}
BUILD_ID=${2:?qualified firmware marker required}
COUNT=${3:-2}
: "${CONCORDIA_RELEASE:?set the qualified runtime directory}"
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ && "$BUILD_ID" =~ ^0x[0-9a-fA-F]+$ ]] || exit 2
[[ "$COUNT" =~ ^([1-9]|1[0-6])$ ]] || exit 2
for suffix in -cpu.json .json -hashes.log; do test ! -e "logs/$PREFIX$suffix"; done
python3 bin/pacc-preflight.py
export Q8_TEST_LIBRARY="$CONCORDIA_RELEASE/libnvcuda.so"
bash bin/run-q8-validation.sh "$PREFIX-identity" "$BUILD_ID" > "logs/$PREFIX-identity-driver.log" 2>&1
c++ -O2 -std=c++17 -I llama.cpp/include -I llama.cpp/ggml/include \
    -I llama.cpp/vendor/nlohmann bin/qwen-ngl-smoke.cpp \
    -L llama.cpp/build-pacc-cuda-gcc14/bin \
    -Wl,-rpath,/srv/lanxin/qwen35-a3b/llama.cpp/build-pacc-cuda-gcc14/bin \
    -L "$CONCORDIA_RELEASE" -Wl,-rpath,"$CONCORDIA_RELEASE" \
    -lllama -lggml -lggml-base -lcuda -lhetgpu_cuda_shim \
    -o bin/qwen-ngl-smoke > "logs/$PREFIX-build.log" 2>&1
sha256sum bin/qwen-ngl-smoke.cpp bin/qwen-ngl-smoke \
    "$CONCORDIA_RELEASE/libnvcuda.so" "$CONCORDIA_RELEASE/libhetgpu_cuda_shim.so" \
    > "logs/$PREFIX-hashes.log"
source bin/pacc-env.sh
export HETGPU_SIFIVE_SHARED_DDR_BASE=0x2075d000000 HETGPU_SIFIVE_SHARED_DDR_BYTES=0x400000
export HETGPU_SIFIVE_JOBD_BOOTSTRAP=0 HETGPU_SIFIVE_JOB_TIMEOUT_MS=10000
export HETGPU_SIFIVE_ZLUDA_IRQ_TRACE=0 HETGPU_SIFIVE_LOG_MEMORY=0
export HETGPU_SIFIVE_GEMM_SHARED_SLOTS=1 HETGPU_SIFIVE_GEMM_DEVICES=0
trap 'python3 bin/snapshot-pacc0.py > "logs/$PREFIX-final-snapshot.log" 2>&1 || true' EXIT
timeout 120 bin/qwen-ngl-smoke models/Qwen3.5-35B-A3B-Q4_K_M.gguf 0 "$COUNT" \
    '1+1=' "logs/$PREFIX-cpu.json" - > "logs/$PREFIX-cpu.log" 2>&1
python3 bin/profile-jobd-phases.py --output "logs/phases-$PREFIX.json" \
    --timeout 3600 --sample-ms 1 --require-build "$BUILD_ID" -- \
    bin/qwen-ngl-smoke models/Qwen3.5-35B-A3B-Q4_K_M.gguf 99 "$COUNT" \
    '1+1=' "logs/$PREFIX.json" "logs/$PREFIX-cpu.json"
jq -e '.all_correct == true and (.generated_tokens | length) == .requested_tokens' \
    "logs/$PREFIX.json"
