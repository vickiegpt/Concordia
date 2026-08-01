#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-/home/ubuntu/Documents/llama.cpp}"
LLAMA_BUILD="${LLAMA_BUILD:-${LLAMA_ROOT}/build-lanxin-pacc-cpu-clang}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mnt/usb/hetgpu_build_target/releasefix/release}"
CC="${CC:-/usr/bin/clang-20}"
OUT="${OUT:-/tmp/pacc_iq1s_bf16_batch_probe}"
PACC_DEVICES="${PACC_DEVICES:-0,1,2,3}"
PACC_PROBE_WORKERS="${PACC_PROBE_WORKERS:-4}"

"$CC" -O2 -fuse-ld=lld \
    -I"${LLAMA_ROOT}/ggml/include" \
    -I"${LLAMA_ROOT}/ggml/src" \
    "${ROOT}/tools/pacc_iq1s_bf16_batch_probe.c" \
    -L"${LLAMA_BUILD}/bin" \
    -Wl,-rpath,"${LLAMA_BUILD}/bin" \
    -lggml-base -ldl -lm -pthread -o "$OUT"

export LD_LIBRARY_PATH="${LLAMA_BUILD}/bin:${RUNTIME_ROOT}:${RUNTIME_ROOT}/deps${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="${RUNTIME_ROOT}/libpacc_runtime_sys.so${LD_PRELOAD:+ ${LD_PRELOAD}}"
export HETGPU_PACC_VISIBLE_DEVICES="$PACC_DEVICES"
export HETGPU_PACC_GEMM_DEVICES="$PACC_DEVICES"
export HETGPU_PACC_MBOX_DEVICE='/dev/hetgpu_pacc_mbox_ddr_coh{}'
export HETGPU_PACC_MAILBOX_DEVICE='/dev/hetgpu_pacc_mbox_live{}'
export HETGPU_PACC_MBOX_BACKEND=helper
export HETGPU_PACC_MBOX_IRQ=1
export HETGPU_PACC_ZLUDA_IRQ_MBOX_HELPER=1
export HETGPU_PACC_JOBD_BOOTSTRAP=0
export HETGPU_PACC_SHARED_DDR_BASE=0x20110600000
export HETGPU_PACC_SHARED_DDR_PACC_BASE=0x20110600000
export HETGPU_PACC_SHARED_DDR_USER_OFF=0x100000
export HETGPU_PACC_SHARED_DDR_PAYLOAD_BASE_OFF=0x100000
export HETGPU_PACC_SHARED_DDR_BYTES=0x100000000
export HETGPU_PACC_SHARED_DDR_MMAP=1
export HETGPU_PACC_SHARED_DDR_CONTROL_MMAP=1
export HETGPU_PACC_IQ1S_COH_DEV=/dev/hetgpu_pacc_mbox_ddr_coh0
export HETGPU_PACC_ALLOW_HOST_GEMM_FALLBACK=0
export HETGPU_PACC_GEMM_TRACE="${HETGPU_PACC_GEMM_TRACE:-1}"
export PACC_PROBE_WORKERS

exec "$OUT"
