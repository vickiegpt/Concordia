#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

runner="${BITNET_LLAMA_CLI:-/root/hetGPU/BitNet-work/build/bin/llama-cli}"
model_dir="${MODEL_DIR:-/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M}"
model_prefix="${MODEL_PREFIX:-$(basename "${model_dir}")}"
model="${MODEL:-${model_dir}/${model_prefix}-00001-of-00006.gguf}"
work_dir="${KIMI_TPS_WORKDIR:-$(mktemp -d /tmp/kimi-k26-tps.XXXXXX)}"
cases_csv="${KIMI_TPS_CASES:-baseline,concordia}"
csv="${KIMI_TPS_CSV:-${work_dir}/kimi_k26_tps.csv}"
jsonl="${KIMI_TPS_JSONL:-${work_dir}/kimi_k26_tps.jsonl}"
prompt="${KIMI_TPS_PROMPT:-Say that you have started in one short sentence.}"
require_run="${KIMI_TPS_REQUIRE_RUN:-0}"

mkdir -p "${work_dir}/logs" "${work_dir}/aof"

if [[ "${KIMI_TPS_KEEP:-0}" != "1" && -z "${KIMI_TPS_WORKDIR:-}" ]]; then
    trap 'rm -rf "${work_dir}"' EXIT
else
    echo "[kimi-tps] keeping work dir: ${work_dir}"
fi

record_skip() {
    local case_name="$1"
    local status="$2"
    local stdout_log="${work_dir}/logs/${case_name}.stdout"
    local stderr_log="${work_dir}/logs/${case_name}.stderr"
    : >"${stdout_log}"
    printf '%s\n' "${status}" >"${stderr_log}"
    python3 "${SCRIPT_DIR}/parse_kimi_tps.py" \
        --case "${case_name}" \
        --stdout "${stdout_log}" \
        --stderr "${stderr_log}" \
        --exit-code 0 \
        --total-ms 0 \
        --runner "${runner}" \
        --model "${model}" \
        --gpu "${gpu}" \
        --commit "${commit}" \
        --status "${status}" \
        --csv "${csv}" \
        --jsonl "${jsonl}" >/dev/null
}

finish_if_required() {
    local status="$1"
    if [[ "${require_run}" == "1" && "${status}" != "pass" ]]; then
        exit 1
    fi
}

detect_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local name
        name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed -n '1p' || true)"
        if [[ -n "${name}" && "${name}" != NVIDIA-SMI*failed* ]]; then
            printf '%s\n' "${name}"
        else
            printf 'unknown'
        fi
    else
        printf 'unknown'
    fi
}

env_for_case() {
    local case_name="$1"
    local aof="$2"
    shift 2

    local ld_preload="${KIMI_TPS_LD_PRELOAD:-}"
    if [[ "${case_name}" == "concordia" ]]; then
        if [[ -z "${ld_preload}" ]]; then
            if [[ "${KIMI_TPS_USE_CUDART_SHIM:-0}" == "1" ]]; then
                ld_preload="${REPO_ROOT}/target/debug/libhetgpu_cuda_shim.so:${REPO_ROOT}/target/debug/libnvcuda.so"
            else
                ld_preload="${REPO_ROOT}/target/debug/libnvcuda.so"
            fi
        fi
        env \
            LD_PRELOAD="${ld_preload}" \
            HETGPU_KIMI_CONCORDIA=1 \
            HETGPU_KIMI_CONCORDIA_AOF_PATH="${aof}" \
            CONCORDIA_AOF_PATH="${aof}" \
            HETGPU_KIMI_CONCORDIA_LOGS="${HETGPU_KIMI_CONCORDIA_LOGS:-1}" \
            BITNET_LLAMA_CLI="${runner}" \
            MODEL_DIR="${model_dir}" \
            MODEL_PREFIX="${model_prefix}" \
            "$@"
    else
        if [[ "${KIMI_TPS_BASELINE_WITH_SHIM:-1}" == "1" ]]; then
            if [[ -z "${ld_preload}" ]]; then
                ld_preload="${REPO_ROOT}/target/debug/libnvcuda.so"
            fi
            env \
                LD_PRELOAD="${ld_preload}" \
                HETGPU_KIMI_CONCORDIA=0 \
                BITNET_LLAMA_CLI="${runner}" \
                MODEL_DIR="${model_dir}" \
                MODEL_PREFIX="${model_prefix}" \
                "$@"
        else
            env \
                HETGPU_KIMI_CONCORDIA=0 \
                BITNET_LLAMA_CLI="${runner}" \
                MODEL_DIR="${model_dir}" \
                MODEL_PREFIX="${model_prefix}" \
                "$@"
        fi
    fi
}

run_case() {
    local case_name="$1"
    local stdout_log="${work_dir}/logs/${case_name}.stdout"
    local stderr_log="${work_dir}/logs/${case_name}.stderr"
    local aof="${work_dir}/aof/${case_name}.aof"
    local exit_code
    local start_ms
    local end_ms
    local total_ms

    rm -f "${aof}"

    start_ms="$(date +%s%3N)"
    set +e
    case "${case_name}" in
        baseline)
            env_for_case baseline "${aof}" \
                "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh" "${prompt}" \
                >"${stdout_log}" 2>"${stderr_log}"
            exit_code="$?"
            ;;
        concordia)
            env_for_case concordia "${aof}" \
                "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh" "${prompt}" \
                >"${stdout_log}" 2>"${stderr_log}"
            exit_code="$?"
            ;;
        *)
            printf 'unknown case: %s\n' "${case_name}" >"${stderr_log}"
            : >"${stdout_log}"
            exit_code=64
            ;;
    esac
    set -e
    end_ms="$(date +%s%3N)"
    total_ms="$((end_ms - start_ms))"

    python3 "${SCRIPT_DIR}/parse_kimi_tps.py" \
        --case "${case_name}" \
        --stdout "${stdout_log}" \
        --stderr "${stderr_log}" \
        --exit-code "${exit_code}" \
        --total-ms "${total_ms}" \
        --aof "${aof}" \
        --runner "${runner}" \
        --model "${model}" \
        --gpu "${gpu}" \
        --commit "${commit}" \
        --csv "${csv}" \
        --jsonl "${jsonl}" >/dev/null

    if [[ "${require_run}" == "1" && "${exit_code}" != "0" ]]; then
        tail -n 120 "${stderr_log}" >&2
        exit "${exit_code}"
    fi
}

gpu="$(detect_gpu)"
commit="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'unknown')"

if [[ ! -x "${runner}" ]]; then
    for case_name in ${cases_csv//,/ }; do
        record_skip "${case_name}" "skipped_missing_runner"
    done
    echo "[kimi-tps] CSV: ${csv}"
    echo "[kimi-tps] JSONL: ${jsonl}"
    finish_if_required "skipped_missing_runner"
    exit 0
fi

if [[ ! -d "${model_dir}" ]]; then
    for case_name in ${cases_csv//,/ }; do
        record_skip "${case_name}" "skipped_missing_model"
    done
    echo "[kimi-tps] CSV: ${csv}"
    echo "[kimi-tps] JSONL: ${jsonl}"
    finish_if_required "skipped_missing_model"
    exit 0
fi

for shard_index in 1 2 3 4 5 6; do
    shard="$(printf '%s-%05d-of-00006.gguf' "${model_prefix}" "${shard_index}")"
    if [[ ! -f "${model_dir}/${shard}" ]]; then
        for case_name in ${cases_csv//,/ }; do
            record_skip "${case_name}" "skipped_missing_model"
        done
        echo "[kimi-tps] CSV: ${csv}"
        echo "[kimi-tps] JSONL: ${jsonl}"
        finish_if_required "skipped_missing_model"
        exit 0
    fi
done

if [[ "${KIMI_TPS_BUILD_ZLUDA:-1}" == "1" &&
      ( "${cases_csv}" == *"concordia"* || "${KIMI_TPS_BASELINE_WITH_SHIM:-1}" == "1" ) ]]; then
    cargo_features="nvidia"
    if [[ "${KIMI_TPS_USE_CUDART_SHIM:-0}" == "1" ]]; then
        cargo_features="nvidia,embed_cudart"
    fi
    echo "[kimi-tps] building ZLUDA with features ${cargo_features}"
    cargo build -p zluda --no-default-features --features "${cargo_features}" \
        >"${work_dir}/logs/cargo-build.log" 2>&1
fi

for case_name in ${cases_csv//,/ }; do
    run_case "${case_name}"
done

echo "[kimi-tps] CSV: ${csv}"
echo "[kimi-tps] JSONL: ${jsonl}"
