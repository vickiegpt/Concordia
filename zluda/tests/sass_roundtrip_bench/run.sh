#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CASES=(int_add pred_select fma_bits shared_reverse)
CSV_HEADER="case,sm,status,cubin_bytes,lifted_ptx_bytes,lift_diagnostics,load_cubin_us,load_ptx_us,kernel_cubin_us,kernel_ptx_us,total_us,message"

if [[ "${1:-}" == "--list-cases" ]]; then
    printf '%s\n' "${CASES[@]}"
    exit 0
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

if [[ -x /usr/local/cuda-12.8/bin/ptxas ]]; then
    PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"
else
    PTXAS="${PTXAS:-ptxas}"
fi
CC="${CC:-cc}"
CARGO="${CARGO:-cargo}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

WORK_DIR="${HETGPU_ROUNDTRIP_WORKDIR:-$(mktemp -d /tmp/hetgpu-sass-roundtrip.XXXXXX)}"
if [[ "${HETGPU_ROUNDTRIP_KEEP:-0}" != "1" && -z "${HETGPU_ROUNDTRIP_WORKDIR:-}" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
else
    echo "[sass-roundtrip] keeping work dir: ${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}/cubin" "${WORK_DIR}/lifted" "${WORK_DIR}/logs" "${WORK_DIR}/ptx"

cap="${HETGPU_ROUNDTRIP_SM:-}"
if [[ -z "${cap}" ]]; then
    if ! raw_cap="$("${NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader 2>&1)"; then
        echo "[sass-roundtrip] failed to query GPU compute capability with ${NVIDIA_SMI}" >&2
        echo "${raw_cap}" >&2
        echo "[sass-roundtrip] rerun with HETGPU_ROUNDTRIP_SM=120 on this Blackwell host" >&2
        exit 1
    fi
    raw_cap="${raw_cap%%$'\n'*}"
    raw_cap="${raw_cap//$'\r'/}"
    raw_cap="${raw_cap//[[:space:]]/}"
    cap="${raw_cap//./}"
fi
if [[ -z "${cap}" ]]; then
    echo "[sass-roundtrip] failed to determine GPU compute capability" >&2
    exit 1
fi

sm="sm_${cap}"
csv="${WORK_DIR}/bench.csv"
printf '%s\n' "${CSV_HEADER}" >"${csv}"

selected_cases=("${CASES[@]}")
if [[ -n "${HETGPU_ROUNDTRIP_CASES:-}" ]]; then
    IFS=', ' read -r -a selected_cases <<<"${HETGPU_ROUNDTRIP_CASES}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    for case_name in "${selected_cases[@]}"; do
        printf '%s,%s,dry_run,0,0,0,0,0,0,0,0,dry_run\n' "${case_name}" "${sm}" >>"${csv}"
    done
    echo "[sass-roundtrip] dry-run CSV: ${csv}"
    exit 0
fi

echo "[sass-roundtrip] building libnvcuda.so with NVIDIA passthrough"
if ! "${CARGO}" build -p zluda --no-default-features --features nvidia >"${WORK_DIR}/logs/cargo-build.log" 2>&1; then
    tail -n 120 "${WORK_DIR}/logs/cargo-build.log" >&2
    exit 1
fi

echo "[sass-roundtrip] compiling Driver API round-trip runner"
"${CC}" -std=c11 -Wall -Wextra -O2 "${SCRIPT_DIR}/roundtrip_runner.c" -ldl \
    -o "${WORK_DIR}/roundtrip_runner"

n="${HETGPU_ROUNDTRIP_N:-1024}"
warmups="${HETGPU_ROUNDTRIP_WARMUPS:-2}"
iters="${HETGPU_ROUNDTRIP_ITERS:-10}"
failures=0

append_result() {
    local case_name="$1"
    local result_line="$2"
    local stderr_log="$3"

    local raw_case status cubin_bytes lifted_ptx_bytes load_cubin_us load_ptx_us
    local kernel_cubin_us kernel_ptx_us total_us message diagnostics
    IFS=$'\t' read -r raw_case status cubin_bytes lifted_ptx_bytes load_cubin_us \
        load_ptx_us kernel_cubin_us kernel_ptx_us total_us message <<<"${result_line}"

    diagnostics="0"
    if grep -q "\\[hetGPU SASS\\] lifted CUBIN via Rust lifter" "${stderr_log}"; then
        diagnostics="$(grep -o 'diagnostics=[0-9]*' "${stderr_log}" | head -n 1 | cut -d= -f2)"
        diagnostics="${diagnostics:-0}"
        if ! grep -q "\\[hetGPU SASS\\] wrote lifted PTX dump" "${stderr_log}"; then
            message="missing_lifter_dump_marker"
            status="missing_lifter_dump_marker"
        fi
    else
        message="missing_lifter_marker"
        status="missing_lifter_marker"
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${case_name}" "${sm}" "${status}" "${cubin_bytes}" "${lifted_ptx_bytes}" \
        "${diagnostics}" "${load_cubin_us}" "${load_ptx_us}" "${kernel_cubin_us}" \
        "${kernel_ptx_us}" "${total_us}" "${message}" >>"${csv}"

    echo "[sass-roundtrip] ${case_name}: ${status} diagnostics=${diagnostics}"
    if [[ "${status}" != "pass" ]]; then
        failures=$((failures + 1))
    fi
}

for case_name in "${selected_cases[@]}"; do
    ptx_template="${SCRIPT_DIR}/ptx/${case_name}.ptx"
    ptx="${WORK_DIR}/ptx/${case_name}.ptx"
    cubin="${WORK_DIR}/cubin/${case_name}.cubin"
    lifted="${WORK_DIR}/lifted/${case_name}.ptx"
    stdout_log="${WORK_DIR}/logs/${case_name}.stdout"
    stderr_log="${WORK_DIR}/logs/${case_name}.stderr"

    if [[ ! -f "${ptx_template}" ]]; then
        echo "[sass-roundtrip] unknown case or missing PTX: ${case_name}" >&2
        exit 1
    fi
    sed "s/^.target sm_.*/.target ${sm}/" "${ptx_template}" >"${ptx}"

    echo "[sass-roundtrip] assembling ${case_name} for ${sm}"
    if ! "${PTXAS}" -arch="${sm}" "${ptx}" -o "${cubin}" >"${WORK_DIR}/logs/${case_name}.ptxas.log" 2>&1; then
        cat "${WORK_DIR}/logs/${case_name}.ptxas.log" >&2
        exit 1
    fi

    echo "[sass-roundtrip] running ${case_name} through LD_PRELOAD hook"
    rm -f "${lifted}"
    if ! env \
        LD_PRELOAD="${REPO_ROOT}/target/debug/libnvcuda.so" \
        HETGPU_SASS_LIFTER_LOG=1 \
        HETGPU_SASS_LIFTER_DUMP="${lifted}" \
        "${WORK_DIR}/roundtrip_runner" "${case_name}" "${cubin}" "${lifted}" "${sm}" "${n}" "${warmups}" "${iters}" \
        >"${stdout_log}" 2>"${stderr_log}"; then
        cat "${stdout_log}" >&2
        tail -n 200 "${stderr_log}" >&2
        exit 1
    fi

    result_line="$(tail -n 1 "${stdout_log}")"
    append_result "${case_name}" "${result_line}" "${stderr_log}"
done

echo "[sass-roundtrip] CSV: ${csv}"
if [[ "${failures}" != "0" && "${HETGPU_ROUNDTRIP_ALLOW_FAILURES:-0}" != "1" ]]; then
    echo "[sass-roundtrip] ${failures} round-trip case(s) failed; set HETGPU_ROUNDTRIP_ALLOW_FAILURES=1 to collect CSV without failing" >&2
    exit 1
fi
