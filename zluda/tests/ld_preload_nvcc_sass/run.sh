#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

if [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    NVCC="${NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
else
    NVCC="${NVCC:-nvcc}"
fi
CC="${CC:-cc}"
CARGO="${CARGO:-cargo}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"

WORK_DIR="${HETGPU_E2E_WORKDIR:-$(mktemp -d /tmp/hetgpu-ldpreload-sass.XXXXXX)}"
if [[ "${HETGPU_E2E_KEEP:-0}" != "1" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
else
    echo "[ld-preload-e2e] keeping work dir: ${WORK_DIR}"
fi

cap="${HETGPU_E2E_SM:-}"
if [[ -z "${cap}" ]]; then
    if ! raw_cap="$("${NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader 2>&1)"; then
        echo "[ld-preload-e2e] failed to query GPU compute capability with ${NVIDIA_SMI}" >&2
        echo "${raw_cap}" >&2
        echo "[ld-preload-e2e] rerun with HETGPU_E2E_SM=120 on this Blackwell host" >&2
        exit 1
    fi
    raw_cap="${raw_cap%%$'\n'*}"
    raw_cap="${raw_cap//$'\r'/}"
    raw_cap="${raw_cap//[[:space:]]/}"
    cap="${raw_cap//./}"
fi
if [[ -z "${cap}" ]]; then
    echo "[ld-preload-e2e] failed to determine GPU compute capability" >&2
    exit 1
fi

sm="sm_${cap}"
if ! "${NVCC}" --list-gpu-code | grep -qx "${sm}"; then
    echo "[ld-preload-e2e] ${NVCC} cannot emit ${sm}; set NVCC to a newer toolkit" >&2
    exit 1
fi

echo "[ld-preload-e2e] building libnvcuda.so with NVIDIA passthrough"
if ! "${CARGO}" build -p zluda --no-default-features --features nvidia >"${WORK_DIR}/cargo-build.log" 2>&1; then
    tail -n 120 "${WORK_DIR}/cargo-build.log" >&2
    exit 1
fi

echo "[ld-preload-e2e] compiling nvcc CUBIN for ${sm}"
"${NVCC}" -cubin -arch="${sm}" "${SCRIPT_DIR}/add_one_kernel.cu" -o "${WORK_DIR}/add_one.cubin" \
    >"${WORK_DIR}/nvcc.log" 2>&1

echo "[ld-preload-e2e] compiling driver-api host harness"
"${CC}" -std=c11 -Wall -Wextra -O2 "${SCRIPT_DIR}/driver_api_add_one.c" -ldl \
    -o "${WORK_DIR}/driver_api_add_one"

stdout_log="${WORK_DIR}/stdout.log"
stderr_log="${WORK_DIR}/stderr.log"
ptx_dump="${WORK_DIR}/lifted.ptx"

echo "[ld-preload-e2e] running through LD_PRELOAD=${REPO_ROOT}/target/debug/libnvcuda.so"
if ! env \
    LD_PRELOAD="${REPO_ROOT}/target/debug/libnvcuda.so" \
    HETGPU_SASS_LIFTER_LOG=1 \
    HETGPU_SASS_LIFTER_DUMP="${ptx_dump}" \
    "${WORK_DIR}/driver_api_add_one" "${WORK_DIR}/add_one.cubin" \
    >"${stdout_log}" 2>"${stderr_log}"; then
    cat "${stdout_log}" >&2
    tail -n 200 "${stderr_log}" >&2
    exit 1
fi

if ! grep -q "PASS nvcc cubin ld_preload sass e2e" "${stdout_log}"; then
    cat "${stdout_log}" >&2
    echo "[ld-preload-e2e] missing PASS marker" >&2
    exit 1
fi

if ! grep -q "\\[hetGPU SASS\\] lifted CUBIN via Rust lifter" "${stderr_log}"; then
    tail -n 200 "${stderr_log}" >&2
    echo "[ld-preload-e2e] missing Rust SASS lifter hook marker" >&2
    exit 1
fi

if [[ ! -s "${ptx_dump}" ]]; then
    tail -n 200 "${stderr_log}" >&2
    echo "[ld-preload-e2e] missing lifted PTX dump at ${ptx_dump}" >&2
    exit 1
fi

if ! grep -q "\\.entry add_one" "${ptx_dump}"; then
    cat "${ptx_dump}" >&2
    echo "[ld-preload-e2e] lifted PTX dump does not contain add_one entry" >&2
    exit 1
fi

if ! grep -q "\\.target sm_${cap}" "${ptx_dump}"; then
    head -n 20 "${ptx_dump}" >&2
    echo "[ld-preload-e2e] lifted PTX dump does not target sm_${cap}" >&2
    exit 1
fi

cat "${stdout_log}"
echo "[ld-preload-e2e] lifted PTX: ${ptx_dump}"
