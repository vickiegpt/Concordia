#!/usr/bin/env bash

# Shared environment for Qwen3.5-35B-A3B on the validated pacc0 path.
set -euo pipefail

export QWEN_ROOT="${QWEN_ROOT:-/srv/lanxin/qwen35-a3b}"
export QWEN_MODEL="${QWEN_MODEL:-${QWEN_ROOT}/models/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
export LLAMA_BUILD="${LLAMA_BUILD:-${QWEN_ROOT}/llama.cpp/build-pacc-cuda-gcc14}"
export CONCORDIA_RELEASE="${CONCORDIA_RELEASE:-/home/ubuntu/Concordia/target-pacc/release}"
export FAKE_CUDA="${FAKE_CUDA:-/home/ubuntu/fake_cuda}"

export LD_LIBRARY_PATH="${CONCORDIA_RELEASE}:${FAKE_CUDA}/lib64:${LLAMA_BUILD}/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PACC_DEVICE_REDIRECT="${PACC_DEVICE_REDIRECT:-${QWEN_ROOT}/bin/libpacc-device-redirect.so}"
export LD_PRELOAD="${PACC_DEVICE_REDIRECT} ${CONCORDIA_RELEASE}/libnvcuda.so ${CONCORDIA_RELEASE}/libhetgpu_cuda_shim.so${LD_PRELOAD:+ ${LD_PRELOAD}}"

# Keep launch routing deterministic while only pacc0 has the matching jobd image.
export HETGPU_SIFIVE_VISIBLE_DEVICES="${HETGPU_SIFIVE_VISIBLE_DEVICES:-0}"
export HETGPU_SIFIVE_KERNEL_SLOT_COUNT="${HETGPU_SIFIVE_KERNEL_SLOT_COUNT:-1}"
export HETGPU_SIFIVE_KERNEL_TOTAL_SLOTS="${HETGPU_SIFIVE_KERNEL_TOTAL_SLOTS:-1}"
export HETGPU_SIFIVE_DEVICE_PATH="${HETGPU_SIFIVE_DEVICE_PATH:-/dev/pacc%d}"
export HETGPU_SIFIVE_MBOX_DEVICE="${HETGPU_SIFIVE_MBOX_DEVICE:-/dev/hetgpu_sifive_mbox_ddr_coh%d}"
export HETGPU_SIFIVE_MAILBOX_DEVICE="${HETGPU_SIFIVE_MAILBOX_DEVICE:-/dev/hetgpu_sifive_mbox_ddr_coh%d}"
export HETGPU_SIFIVE_KERNEL_MBOX_SUBMIT=1
export HETGPU_SIFIVE_CONTROL_BACKEND=shared_ddr
export HETGPU_SIFIVE_ZLUDA_IRQ_TRACE="${HETGPU_SIFIVE_ZLUDA_IRQ_TRACE:-1}"
export HETGPU_SIFIVE_WAIT_KERNEL_LAUNCH=1

# Keep the 22 GB model in host memory and stage working sets through shared DDR.
# The staging aperture is the helper's actual allocation, not the VRAM limit below.
export HETGPU_SIFIVE_ALLOW_HOST_DEVICE_MEM=1
export HETGPU_SIFIVE_REAL_DEVICE_MEM=0
export HETGPU_SIFIVE_SHARED_DEVICE_MEM=0
export HETGPU_SIFIVE_VRAM_BYTES="${HETGPU_SIFIVE_VRAM_BYTES:-68719476736}"

# Hardware-only completion policy: no emulator and no success-on-error paths.
export HETGPU_SIFIVE_STRICT=1
export HETGPU_SIFIVE_REAL_DEVICE_REQUIRED=1
export HETGPU_SIFIVE_JOBD_EMULATOR=0
export HETGPU_SIFIVE_EMULATE_JOBD=0
export HETGPU_SIFIVE_DISABLE_JOBD_EMULATOR_AUTO=1
export HETGPU_CUDART_FAIL_OPEN=0
export HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR=0
export HETGPU_SIFIVE_SKIP_JOB_WAIT=0
export HETGPU_SIFIVE_RMSNORM_HOST_FALLBACK=0
export HETGPU_SIFIVE_MMVF_HOST_FALLBACK=0
export HETGPU_SIFIVE_ALLOW_NAMED_HOST_FALLBACK=0
export HETGPU_SIFIVE_MMVF_ASSUME_SUCCESS_ON_OUTPUT_TIMEOUT=0
export HETGPU_SIFIVE_RMSNORM_ASSUME_SUCCESS_ON_OUTPUT_TIMEOUT=0

export GGML_CUDA_DISABLE_GRAPHS=1
export HETGPU_CUDART_PTX_CACHE_DIR="${HETGPU_CUDART_PTX_CACHE_DIR:-${QWEN_ROOT}/ptx-cache}"
export HETGPU_SIFIVE_ELF_CACHE_DIR="${HETGPU_SIFIVE_ELF_CACHE_DIR:-${QWEN_ROOT}/elf-cache}"
export HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES="${HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES:-1}"
export HETGPU_SIFIVE_LOG_MEMORY="${HETGPU_SIFIVE_LOG_MEMORY:-1}"

mkdir -p "${QWEN_ROOT}/logs" "${HETGPU_CUDART_PTX_CACHE_DIR}" "${HETGPU_SIFIVE_ELF_CACHE_DIR}"
[[ -r "${QWEN_MODEL}" ]] || { echo "model not readable: ${QWEN_MODEL}" >&2; exit 2; }
[[ -x "${LLAMA_BUILD}/bin/llama-cli" ]] || { echo "llama-cli not built: ${LLAMA_BUILD}/bin/llama-cli" >&2; exit 2; }
