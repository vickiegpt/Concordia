#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CSV_HEADER="case,status,total_ms,exit_code,stdout_bytes,stderr_bytes,lifter_markers,lifted_ptx_files,lifted_ptx_bytes,message"
CASE_NAME="kimi_k26_iq1m"
CARGO="${CARGO:-cargo}"
if [[ -x /usr/local/cuda-12.8/bin/cuobjdump ]]; then
    CUOBJDUMP="${CUOBJDUMP:-/usr/local/cuda-12.8/bin/cuobjdump}"
else
    CUOBJDUMP="${CUOBJDUMP:-cuobjdump}"
fi

WORK_DIR="${HETGPU_KIMI_E2E_WORKDIR:-$(mktemp -d /tmp/hetgpu-kimi-k26-e2e.XXXXXX)}"
if [[ "${HETGPU_KIMI_E2E_KEEP:-0}" != "1" && -z "${HETGPU_KIMI_E2E_WORKDIR:-}" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
else
    echo "[kimi-k26-e2e] keeping work dir: ${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}/logs"

csv="${HETGPU_KIMI_E2E_CSV:-${WORK_DIR}/bench_kimi_k26_e2e.csv}"
printf '%s\n' "${CSV_HEADER}" >"${csv}"

append_row() {
    local status="$1"
    local total_ms="$2"
    local exit_code="$3"
    local stdout_bytes="$4"
    local stderr_bytes="$5"
    local lifter_markers="$6"
    local lifted_ptx_files="$7"
    local lifted_ptx_bytes="$8"
    local message="$9"

    message="${message//$'\r'/ }"
    message="${message//$'\n'/ }"
    message="${message//,/;}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${CASE_NAME}" "${status}" "${total_ms}" "${exit_code}" \
        "${stdout_bytes}" "${stderr_bytes}" "${lifter_markers}" \
        "${lifted_ptx_files}" "${lifted_ptx_bytes}" "${message}" >>"${csv}"
}

finish_non_pass() {
    local status="$1"
    local stderr_log="${2:-}"

    if [[ "${status}" != "pass" && "${status}" != "pass_ptx_only" && "${HETGPU_KIMI_E2E_ALLOW_FAILURES:-0}" != "1" ]]; then
        if [[ -n "${stderr_log}" && -f "${stderr_log}" ]]; then
            tail -n 200 "${stderr_log}" >&2
        fi
        exit 1
    fi
}

runner="${BITNET_LLAMA_CLI:-/root/hetGPU/BitNet-work/build/bin/llama-cli}"
model_dir="${MODEL_DIR:-/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M}"
model_prefix="${MODEL_PREFIX:-$(basename "${model_dir}")}"

required_shards=()
for shard_index in 1 2 3 4 5 6; do
    required_shards+=("$(printf '%s-%05d-of-00006.gguf' "${model_prefix}" "${shard_index}")")
done

if [[ ! -x "${runner}" ]]; then
    append_row "skipped_missing_runner" 0 0 0 0 0 0 0 "missing_runner:${runner}"
    echo "[kimi-k26-e2e] CSV: ${csv}"
    exit 0
fi

if [[ ! -d "${model_dir}" ]]; then
    append_row "skipped_missing_model" 0 0 0 0 0 0 0 "missing_model_dir:${model_dir}"
    echo "[kimi-k26-e2e] CSV: ${csv}"
    exit 0
fi

for shard in "${required_shards[@]}"; do
    if [[ ! -f "${model_dir}/${shard}" ]]; then
        append_row "skipped_missing_model" 0 0 0 0 0 0 0 "missing_shard:${shard}"
        echo "[kimi-k26-e2e] CSV: ${csv}"
        exit 0
    fi
done

cargo_features="nvidia"
if [[ "${HETGPU_KIMI_E2E_USE_CUDART_SHIM:-0}" == "1" ||
      "${HETGPU_KIMI_E2E_LD_PRELOAD:-}" == *"libhetgpu_cuda_shim.so"* ]]; then
    cargo_features="nvidia,embed_cudart"
fi

echo "[kimi-k26-e2e] building libnvcuda.so with features ${cargo_features}"
if ! "${CARGO}" build -p zluda --no-default-features --features "${cargo_features}" >"${WORK_DIR}/logs/cargo-build.log" 2>&1; then
    cargo_log_bytes="$(stat -c%s "${WORK_DIR}/logs/cargo-build.log")"
    append_row "run_failed" 0 1 0 "${cargo_log_bytes}" 0 0 0 "cargo_build_failed"
    echo "[kimi-k26-e2e] CSV: ${csv}" >&2
    finish_non_pass "run_failed" "${WORK_DIR}/logs/cargo-build.log"
    exit 0
fi

if [[ -n "${HETGPU_KIMI_E2E_LD_PRELOAD:-}" ]]; then
    kimi_ld_preload="${HETGPU_KIMI_E2E_LD_PRELOAD}"
elif [[ "${HETGPU_KIMI_E2E_USE_CUDART_SHIM:-0}" == "1" ]]; then
    kimi_ld_preload="${REPO_ROOT}/target/debug/libhetgpu_cuda_shim.so:${REPO_ROOT}/target/debug/libnvcuda.so"
else
    kimi_ld_preload="${REPO_ROOT}/target/debug/libnvcuda.so"
fi
kimi_cudart_defer_module_load="${HETGPU_KIMI_E2E_CUDART_DEFER_MODULE_LOAD:-${HETGPU_CUDART_DEFER_MODULE_LOAD:-0}}"
kimi_cudart_compute_capability="${HETGPU_KIMI_E2E_CUDART_COMPUTE_CAPABILITY:-${HETGPU_CUDART_COMPUTE_CAPABILITY:-}}"
kimi_cudart_prefer_fatbin_cubin_for_sass="${HETGPU_KIMI_E2E_PREFER_FATBIN_CUBIN_FOR_SASS:-${HETGPU_CUDART_PREFER_FATBIN_CUBIN_FOR_SASS:-1}}"
kimi_gpu_layers="${HETGPU_KIMI_E2E_N_GPU_LAYERS:-${LLAMA_ARG_N_GPU_LAYERS:-1}}"
kimi_extra_llama_args="${HETGPU_KIMI_E2E_EXTRA_LLAMA_ARGS:-${KIMI_EXTRA_LLAMA_ARGS:-}}"

stdout_log="${WORK_DIR}/logs/kimi.stdout"
stderr_log="${WORK_DIR}/logs/kimi.stderr"
ptx_dump="${WORK_DIR}/lifted_kimi_k26.ptx"
prompt="${KIMI_PROMPT:-Say that you have started in one short sentence.}"

start_ms="$(date +%s%3N)"
set +e
env \
    LD_PRELOAD="${kimi_ld_preload}" \
    HETGPU_KIMI_E2E_EFFECTIVE_LD_PRELOAD="${kimi_ld_preload}" \
    HETGPU_SASS_LIFTER_LOG=1 \
    HETGPU_SASS_LIFTER_DUMP="${ptx_dump}" \
    HETGPU_SASS_LIFTER_CUOBJDUMP="${CUOBJDUMP}" \
    HETGPU_CUDART_DEFER_MODULE_LOAD="${kimi_cudart_defer_module_load}" \
    HETGPU_CUDART_COMPUTE_CAPABILITY="${kimi_cudart_compute_capability}" \
    HETGPU_CUDART_PREFER_FATBIN_CUBIN_FOR_SASS="${kimi_cudart_prefer_fatbin_cubin_for_sass}" \
    BITNET_LLAMA_CLI="${runner}" \
    MODEL_DIR="${model_dir}" \
    MODEL_PREFIX="${model_prefix}" \
    LLAMA_ARG_N_GPU_LAYERS="${kimi_gpu_layers}" \
    KIMI_EXTRA_LLAMA_ARGS="${kimi_extra_llama_args}" \
    "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh" "${prompt}" \
    >"${stdout_log}" 2>"${stderr_log}"
exit_code="$?"
set -e
end_ms="$(date +%s%3N)"
total_ms="$((end_ms - start_ms))"

stdout_bytes="$(stat -c%s "${stdout_log}")"
stderr_bytes="$(stat -c%s "${stderr_log}")"
lifter_markers="$(grep -c "\\[hetGPU SASS\\] lifted" "${stderr_log}" || true)"
ptx_source_markers="$(grep -c "\\[NVIDIA Backend\\] Detected PTX source" "${stderr_log}" || true)"
if [[ -s "${ptx_dump}" ]]; then
    lifted_ptx_files=1
    lifted_ptx_bytes="$(stat -c%s "${ptx_dump}")"
else
    lifted_ptx_files=0
    lifted_ptx_bytes=0
fi

status="pass"
message="ok"
if [[ "${exit_code}" != "0" ]]; then
    status="run_failed"
    message="runner_exit_${exit_code}"
elif [[ "${stdout_bytes}" == "0" ]]; then
    status="empty_output"
    message="empty_stdout"
elif grep -Eq 'ggml_cuda_init: failed to initialize CUDA|not compiled with GPU offload support|cuDeviceGetCount returned .*count=0|offloading 0 repeating layers|offloaded 0/' "${stderr_log}"; then
    status="skipped_no_cuda_offload"
    message="no_cuda_offload"
elif [[ "${lifter_markers}" == "0" ]]; then
    if [[ "${ptx_source_markers}" != "0" ]]; then
        status="pass_ptx_only"
        message="ptx_modules_${ptx_source_markers}_no_sass_lift"
    else
        status="missing_lifter_marker"
        message="no_lifter_marker"
    fi
elif [[ "${lifted_ptx_files}" == "0" || "${lifted_ptx_bytes}" == "0" ]]; then
    status="missing_lifter_dump_marker"
    message="no_lifted_ptx_dump"
fi

append_row "${status}" "${total_ms}" "${exit_code}" "${stdout_bytes}" "${stderr_bytes}" \
    "${lifter_markers}" "${lifted_ptx_files}" "${lifted_ptx_bytes}" "${message}"

echo "[kimi-k26-e2e] ${CASE_NAME}: ${status} lifter_markers=${lifter_markers} lifted_ptx_bytes=${lifted_ptx_bytes}"
echo "[kimi-k26-e2e] CSV: ${csv}"

finish_non_pass "${status}" "${stderr_log}"
