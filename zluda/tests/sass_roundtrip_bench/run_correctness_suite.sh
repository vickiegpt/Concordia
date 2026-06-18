#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
CARGO="${CARGO:-cargo}"

WORK_DIR="${HETGPU_SASS_PROOF_WORKDIR:-$(mktemp -d /tmp/hetgpu-sass-proof.XXXXXX)}"
if [[ "${HETGPU_SASS_PROOF_KEEP:-0}" != "1" && -z "${HETGPU_SASS_PROOF_WORKDIR:-}" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
else
    echo "[sass-proof] keeping work dir: ${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}/logs"

csv="${HETGPU_SASS_PROOF_CSV:-${WORK_DIR}/sass_lifter_correctness.csv}"
printf 'step,status,elapsed_ms,message\n' >"${csv}"

csv_escape() {
    local value="$1"
    value="${value//$'\r'/ }"
    value="${value//$'\n'/ }"
    value="${value//,/;}"
    printf '%s' "${value}"
}

append_row() {
    local step="$1"
    local status="$2"
    local elapsed_ms="$3"
    local message="$4"
    printf '%s,%s,%s,%s\n' \
        "$(csv_escape "${step}")" \
        "$(csv_escape "${status}")" \
        "$(csv_escape "${elapsed_ms}")" \
        "$(csv_escape "${message}")" >>"${csv}"
}

if [[ "${1:-}" == "--dry-run" ]]; then
    append_row rust_fuzzer dry_run 0 planned
    append_row roundtrip_harness dry_run 0 planned
    append_row ld_preload_roundtrip dry_run 0 planned
    append_row kimi_e2e dry_run 0 planned
    echo "[sass-proof] dry-run CSV: ${csv}"
    exit 0
fi

run_step() {
    local step="$1"
    shift
    local log="${WORK_DIR}/logs/${step}.log"
    local start_ms end_ms elapsed_ms
    start_ms="$(date +%s%3N)"
    set +e
    "$@" >"${log}" 2>&1
    local status_code="$?"
    set -e
    end_ms="$(date +%s%3N)"
    elapsed_ms="$((end_ms - start_ms))"

    if [[ "${status_code}" == "0" ]]; then
        append_row "${step}" pass "${elapsed_ms}" "log:${log}"
        echo "[sass-proof] ${step}: pass (${elapsed_ms} ms)"
    else
        append_row "${step}" fail "${elapsed_ms}" "exit_${status_code}_log:${log}"
        echo "[sass-proof] ${step}: fail (${elapsed_ms} ms)" >&2
        tail -n 120 "${log}" >&2
        return "${status_code}"
    fi
}

run_step rust_fuzzer \
    "${CARGO}" run -p ptx --bin sass_lifter_fuzz -- \
        --seed "${HETGPU_SASS_FUZZ_SEED:-1515524608}" \
        --cases "${HETGPU_SASS_FUZZ_CASES:-1024}" \
        --max-instructions "${HETGPU_SASS_FUZZ_MAX_INSTRUCTIONS:-32}" \
        --sm-version "${HETGPU_SASS_FUZZ_SM:-120}"

run_step roundtrip_harness "${SCRIPT_DIR}/test_roundtrip_harness.sh"

if [[ "${HETGPU_SASS_PROOF_REAL:-0}" == "1" ]]; then
    run_step ld_preload_roundtrip "${SCRIPT_DIR}/run.sh"
else
    append_row ld_preload_roundtrip skipped 0 "set HETGPU_SASS_PROOF_REAL=1 to run NVIDIA driver benchmark"
    echo "[sass-proof] ld_preload_roundtrip: skipped"
fi

if [[ "${HETGPU_SASS_PROOF_KIMI:-0}" == "1" ]]; then
    run_step kimi_e2e "${SCRIPT_DIR}/run_kimi_k26_e2e.sh"
else
    append_row kimi_e2e skipped 0 "set HETGPU_SASS_PROOF_KIMI=1 to run slow Kimi capture"
    echo "[sass-proof] kimi_e2e: skipped"
fi

echo "[sass-proof] CSV: ${csv}"
