#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "${tmp_dir}"' EXIT
log="${tmp_dir}/calls.log"
results="${tmp_dir}/results"

make_fake() {
    local path="$1"
    shift
    {
        printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail'
        printf '%s\n' "$@"
    } >"${path}"
    chmod +x "${path}"
}

make_fake "${tmp_dir}/cargo" \
    'printf "cargo %s\n" "$*" >>"${NVINT4_FAKE_LOG}"'

make_fake "${tmp_dir}/nvcc" \
    'printf "nvcc %s\n" "$*" >>"${NVINT4_FAKE_LOG}"' \
    'exit 0'

make_fake "${tmp_dir}/probe" \
    'printf "probe %s\n" "$*" >>"${NVINT4_FAKE_LOG}"' \
    'printf "env route=%s fallback=%s device=%s dax=%s bdf=%s csr=%s\n" \
        "${HETGPU_NVINT4_TMATMUL-unset}" \
        "${HETGPU_NVINT4_GPU_FALLBACK-unset}" \
        "${HETGPU_CXL_TMATMUL_DEVICE-unset}" \
        "${HETGPU_CXL_TMATMUL_DAX-unset}" \
        "${HETGPU_CXL_TMATMUL_PCI_ADDR-unset}" \
        "${HETGPU_CXL_TMATMUL_CSR_BASE-unset}" >>"${NVINT4_FAKE_LOG}"' \
    'artifact=""' \
    'while (($#)); do' \
    '    if [[ "$1" == "--artifact" ]]; then artifact="$2"; shift 2; else shift; fi' \
    'done' \
    'if [[ -n "${artifact}" ]]; then printf "{}\n" >"${artifact}"; fi' \
    'exit "${FAKE_PROBE_EXIT:-0}"'

{
    printf '%s\n' '#!/usr/bin/env python3'
    printf '%s\n' 'import os'
    printf '%s\n' 'import sys'
    printf '%s\n' 'with open(os.environ["NVINT4_FAKE_LOG"], "a", encoding="utf-8") as output:'
    printf '%s\n' '    output.write("checker " + " ".join(sys.argv[1:]) + "\n")'
    printf '%s\n' 'raise SystemExit(int(os.environ.get("FAKE_CHECKER_EXIT", "0")))'
} >"${tmp_dir}/checker"
chmod +x "${tmp_dir}/checker"

export NVINT4_FAKE_LOG="${log}"
export CARGO_BIN="${tmp_dir}/cargo"
export NVCC_BIN="${tmp_dir}/nvcc"
export NVINT4_PROBE_BIN="${tmp_dir}/probe"
export NVINT4_CHECKER_BIN="${tmp_dir}/checker"
export NVINT4_RESULTS_ROOT="${results}"

bash "${script_dir}/run.sh" \
    --hardware \
    --delta 3 \
    --seed 0x4e564934 \
    --stream nondefault \
    --device /dev/cxl_tmatmulTEST \
    --dax /dev/daxTEST \
    --bdf 0000:aa:00.0 \
    --csr-base 0x1c0000 \
    >"${tmp_dir}/success.out"

grep -Fq "cargo build -p zluda --no-default-features --features nvidia" "${log}"
grep -Fq "nvcc -std=c++17 -O2 -lineinfo" "${log}"
grep -Fq "probe --hardware --delta 3 --seed 0x4e564934 --stream nondefault --artifact" "${log}"
grep -Fq "env route=1 fallback=unset device=/dev/cxl_tmatmulTEST dax=/dev/daxTEST bdf=0000:aa:00.0 csr=0x1c0000" "${log}"
grep -Fq "checker " "${log}"

set +e
FAKE_PROBE_EXIT=7 bash "${script_dir}/run.sh" \
    --hardware \
    --delta 1 \
    --stream default \
    >"${tmp_dir}/failure.out" 2>"${tmp_dir}/failure.err"
status=$?
set -e
if [[ "${status}" -ne 7 ]]; then
    echo "probe failure was not propagated: status=${status}" >&2
    exit 1
fi

echo "PASS: NVINT4 proof harness strict env, forwarding, and failure propagation"
