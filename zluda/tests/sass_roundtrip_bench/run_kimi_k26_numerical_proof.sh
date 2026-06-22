#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CASE_NAME="kimi_k26_numerical"
CSV_HEADER="case,status,total_ms,baseline_exit_code,hooked_exit_code,baseline_stdout_sha256,hooked_stdout_sha256,baseline_stdout_bytes,hooked_stdout_bytes,ptx_source_markers,sass_lifter_markers,offloaded_layers,message"
CARGO="${CARGO:-cargo}"

WORK_DIR="${HETGPU_KIMI_NUMERICAL_WORKDIR:-$(mktemp -d /tmp/hetgpu-kimi-k26-numerical.XXXXXX)}"
if [[ "${HETGPU_KIMI_NUMERICAL_KEEP:-0}" != "1" && -z "${HETGPU_KIMI_NUMERICAL_WORKDIR:-}" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
else
    echo "[kimi-k26-numerical] keeping work dir: ${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}/logs"

csv="${HETGPU_KIMI_NUMERICAL_CSV:-${WORK_DIR}/bench_kimi_k26_numerical.csv}"
printf '%s\n' "${CSV_HEADER}" >"${csv}"

csv_escape() {
    local value="$1"
    value="${value//$'\r'/ }"
    value="${value//$'\n'/ }"
    value="${value//,/;}"
    printf '%s' "${value}"
}

append_row() {
    local status="$1"
    local total_ms="$2"
    local baseline_exit="$3"
    local hooked_exit="$4"
    local baseline_sha="$5"
    local hooked_sha="$6"
    local baseline_bytes="$7"
    local hooked_bytes="$8"
    local ptx_source_markers="$9"
    local sass_lifter_markers="${10}"
    local offloaded_layers="${11}"
    local message="${12}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "${CASE_NAME}")" \
        "$(csv_escape "${status}")" \
        "$(csv_escape "${total_ms}")" \
        "$(csv_escape "${baseline_exit}")" \
        "$(csv_escape "${hooked_exit}")" \
        "$(csv_escape "${baseline_sha}")" \
        "$(csv_escape "${hooked_sha}")" \
        "$(csv_escape "${baseline_bytes}")" \
        "$(csv_escape "${hooked_bytes}")" \
        "$(csv_escape "${ptx_source_markers}")" \
        "$(csv_escape "${sass_lifter_markers}")" \
        "$(csv_escape "${offloaded_layers}")" \
        "$(csv_escape "${message}")" >>"${csv}"
}

finish_status() {
    local status="$1"
    local stderr_log="${2:-}"
    if [[ "${status}" != "pass" && "${HETGPU_KIMI_NUMERICAL_ALLOW_FAILURES:-0}" != "1" ]]; then
        if [[ -n "${stderr_log}" && -f "${stderr_log}" ]]; then
            tail -n 160 "${stderr_log}" >&2
        fi
        exit 1
    fi
}

sha_file() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        sha256sum "${path}" | awk '{ print $1 }'
    else
        printf 'missing'
    fi
}

file_size() {
    local path="$1"
    if [[ -f "${path}" ]]; then
        stat -c%s "${path}"
    else
        printf '0'
    fi
}

runner="${BITNET_LLAMA_CLI:-/root/hetGPU/BitNet-work/build/bin/llama-cli}"
model_dir="${MODEL_DIR:-/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M}"
model_prefix="${MODEL_PREFIX:-$(basename "${model_dir}")}"

required_shards=()
for shard_index in 1 2 3 4 5 6; do
    required_shards+=("$(printf '%s-%05d-of-00006.gguf' "${model_prefix}" "${shard_index}")")
done

if [[ "${1:-}" == "--dry-run" ]]; then
    append_row dry_run 0 0 0 none none 0 0 0 0 planned planned
    echo "[kimi-k26-numerical] dry-run CSV: ${csv}"
    exit 0
fi

if [[ ! -x "${runner}" ]]; then
    append_row skipped_missing_runner 0 0 0 none none 0 0 0 0 unknown "missing_runner:${runner}"
    echo "[kimi-k26-numerical] CSV: ${csv}"
    exit 0
fi

if [[ ! -d "${model_dir}" ]]; then
    append_row skipped_missing_model 0 0 0 none none 0 0 0 0 unknown "missing_model_dir:${model_dir}"
    echo "[kimi-k26-numerical] CSV: ${csv}"
    exit 0
fi

for shard in "${required_shards[@]}"; do
    if [[ ! -f "${model_dir}/${shard}" ]]; then
        append_row skipped_missing_model 0 0 0 none none 0 0 0 0 unknown "missing_shard:${shard}"
        echo "[kimi-k26-numerical] CSV: ${csv}"
        exit 0
    fi
done

cargo_features="nvidia"
if [[ "${HETGPU_KIMI_NUMERICAL_USE_CUDART_SHIM:-0}" == "1" ||
      "${HETGPU_KIMI_NUMERICAL_LD_PRELOAD:-}" == *"libhetgpu_cuda_shim.so"* ]]; then
    cargo_features="nvidia,embed_cudart"
fi

echo "[kimi-k26-numerical] building libnvcuda.so with features ${cargo_features}"
if ! "${CARGO}" build -p zluda --no-default-features --features "${cargo_features}" >"${WORK_DIR}/logs/cargo-build.log" 2>&1; then
    cargo_log_bytes="$(stat -c%s "${WORK_DIR}/logs/cargo-build.log")"
    append_row build_failed 0 0 0 none none 0 0 0 0 unknown "cargo_build_failed_log_bytes:${cargo_log_bytes}"
    echo "[kimi-k26-numerical] CSV: ${csv}" >&2
    finish_status build_failed "${WORK_DIR}/logs/cargo-build.log"
    exit 0
fi

if [[ -n "${HETGPU_KIMI_NUMERICAL_LD_PRELOAD:-}" ]]; then
    hooked_ld_preload="${HETGPU_KIMI_NUMERICAL_LD_PRELOAD}"
elif [[ "${HETGPU_KIMI_NUMERICAL_USE_CUDART_SHIM:-0}" == "1" ]]; then
    hooked_ld_preload="${REPO_ROOT}/target/debug/libhetgpu_cuda_shim.so:${REPO_ROOT}/target/debug/libnvcuda.so"
else
    hooked_ld_preload="${REPO_ROOT}/target/debug/libnvcuda.so"
fi

baseline_gpu_layers="${HETGPU_KIMI_NUMERICAL_BASELINE_GPU_LAYERS:-${LLAMA_ARG_N_GPU_LAYERS:-1}}"
hooked_gpu_layers="${HETGPU_KIMI_NUMERICAL_HOOKED_GPU_LAYERS:-${LLAMA_ARG_N_GPU_LAYERS:-1}}"
prompt="${KIMI_PROMPT:-SayOK.}"
deterministic_args="${KIMI_NUMERICAL_DETERMINISTIC_ARGS:---seed 42 --temp 0 --top-k 1 --top-p 1.0 --min-p 0.0 --repeat-penalty 1.0 --no-display-prompt --simple-io}"
extra_args="${KIMI_EXTRA_LLAMA_ARGS:-}"
combined_extra_args="${deterministic_args}"
if [[ -n "${extra_args}" ]]; then
    combined_extra_args+=" ${extra_args}"
fi

run_role() {
    local role="$1"
    local stdout_log="$2"
    local stderr_log="$3"
    local ld_preload="$4"
    local gpu_layers="$5"

    local env_args=(
        HETGPU_KIMI_NUMERICAL_RUN_ROLE="${role}"
        BITNET_LLAMA_CLI="${runner}"
        MODEL_DIR="${model_dir}"
        MODEL_PREFIX="${model_prefix}"
        THREADS="${THREADS:-16}"
        CTX_SIZE="${CTX_SIZE:-512}"
        N_PREDICT="${N_PREDICT:-8}"
        NO_WARMUP="${NO_WARMUP:-1}"
        TEMP=0
        LLAMA_ARG_N_GPU_LAYERS="${gpu_layers}"
        KIMI_EXTRA_LLAMA_ARGS="${combined_extra_args}"
    )

    if [[ "${role}" == "hooked" ]]; then
        env_args+=(
            HETGPU_KIMI_NUMERICAL_EFFECTIVE_LD_PRELOAD="${ld_preload}"
            HETGPU_SASS_LIFTER_LOG=1
            HETGPU_CUDART_DEFER_MODULE_LOAD="${HETGPU_KIMI_NUMERICAL_CUDART_DEFER_MODULE_LOAD:-${HETGPU_CUDART_DEFER_MODULE_LOAD:-0}}"
            HETGPU_CUDART_COMPUTE_CAPABILITY="${HETGPU_KIMI_NUMERICAL_CUDART_COMPUTE_CAPABILITY:-${HETGPU_CUDART_COMPUTE_CAPABILITY:-}}"
            HETGPU_CUDART_PREFER_FATBIN_CUBIN_FOR_SASS="${HETGPU_KIMI_NUMERICAL_PREFER_FATBIN_CUBIN_FOR_SASS:-${HETGPU_CUDART_PREFER_FATBIN_CUBIN_FOR_SASS:-0}}"
        )
    fi

    local status=0
    if [[ -n "${ld_preload}" ]]; then
        env "${env_args[@]}" LD_PRELOAD="${ld_preload}" \
            "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh" "${prompt}" \
            >"${stdout_log}" 2>"${stderr_log}" || status="$?"
    else
        env "${env_args[@]}" \
            "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh" "${prompt}" \
            >"${stdout_log}" 2>"${stderr_log}" || status="$?"
    fi
    return "${status}"
}

baseline_stdout="${WORK_DIR}/logs/baseline.stdout"
baseline_stderr="${WORK_DIR}/logs/baseline.stderr"
hooked_stdout="${WORK_DIR}/logs/hooked.stdout"
hooked_stderr="${WORK_DIR}/logs/hooked.stderr"

start_ms="$(date +%s%3N)"
set +e
run_role baseline "${baseline_stdout}" "${baseline_stderr}" "" "${baseline_gpu_layers}"
baseline_exit="$?"
run_role hooked "${hooked_stdout}" "${hooked_stderr}" "${hooked_ld_preload}" "${hooked_gpu_layers}"
hooked_exit="$?"
set -e
end_ms="$(date +%s%3N)"
total_ms="$((end_ms - start_ms))"

baseline_sha="$(sha_file "${baseline_stdout}")"
hooked_sha="$(sha_file "${hooked_stdout}")"
baseline_bytes="$(file_size "${baseline_stdout}")"
hooked_bytes="$(file_size "${hooked_stdout}")"
ptx_source_markers="$(grep -c "\\[NVIDIA Backend\\] Detected PTX source" "${hooked_stderr}" || true)"
sass_lifter_markers="$(grep -c "\\[hetGPU SASS\\] lifted" "${hooked_stderr}" || true)"
offloaded_layers="$(
    {
        grep -oE 'offloaded [0-9]+/[0-9]+ layers' "${hooked_stderr}" \
            | tail -n 1 \
            | awk '{ print $2 }'
    } || true
)"
offloaded_layers="${offloaded_layers:-unknown}"

status="pass"
message="stdout_sha_match"
if [[ "${baseline_exit}" != "0" ]]; then
    status="baseline_failed"
    message="baseline_exit_${baseline_exit}"
elif [[ "${hooked_exit}" != "0" ]]; then
    status="hooked_failed"
    message="hooked_exit_${hooked_exit}"
elif [[ "${baseline_bytes}" == "0" || "${hooked_bytes}" == "0" ]]; then
    status="empty_output"
    message="baseline_bytes_${baseline_bytes}_hooked_bytes_${hooked_bytes}"
elif [[ "${baseline_sha}" != "${hooked_sha}" ]]; then
    status="output_mismatch"
    message="baseline_sha_${baseline_sha}_hooked_sha_${hooked_sha}"
elif grep -Eq 'ggml_cuda_init: failed to initialize CUDA|not compiled with GPU offload support|offloading 0 repeating layers|offloaded 0/' "${hooked_stderr}"; then
    status="skipped_no_cuda_offload"
    message="no_cuda_offload"
elif [[ "${offloaded_layers}" == "unknown" ]]; then
    status="skipped_no_cuda_offload"
    message="missing_offload_marker"
elif [[ "$((ptx_source_markers + sass_lifter_markers))" == "0" && "${HETGPU_KIMI_NUMERICAL_REQUIRE_PTX_CAPTURE:-1}" == "1" ]]; then
    status="missing_ptx_capture"
    message="no_ptx_or_sass_capture_markers"
fi

append_row "${status}" "${total_ms}" "${baseline_exit}" "${hooked_exit}" \
    "${baseline_sha}" "${hooked_sha}" "${baseline_bytes}" "${hooked_bytes}" \
    "${ptx_source_markers}" "${sass_lifter_markers}" "${offloaded_layers}" "${message}"

echo "[kimi-k26-numerical] ${CASE_NAME}: ${status} baseline_sha=${baseline_sha} hooked_sha=${hooked_sha} ptx_modules=${ptx_source_markers} sass_lifts=${sass_lifter_markers} offloaded=${offloaded_layers}"
echo "[kimi-k26-numerical] CSV: ${csv}"

finish_status "${status}" "${hooked_stderr}"
