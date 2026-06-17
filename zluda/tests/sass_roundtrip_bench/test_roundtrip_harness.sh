#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cases="$("${SCRIPT_DIR}/run.sh" --list-cases)"
grep -Fxq "int_add" <<<"${cases}"
grep -Fxq "pred_select" <<<"${cases}"
grep -Fxq "fma_bits" <<<"${cases}"
grep -Fxq "shared_reverse" <<<"${cases}"
grep -Fxq "kimi_iq1m_unpack" <<<"${cases}"
grep -Fxq "kimi_rmsnorm_bits" <<<"${cases}"
grep -Fxq "kimi_swiglu_mix" <<<"${cases}"
grep -Fxq "kimi_rope_mix" <<<"${cases}"
grep -Fxq "kimi_attention_mask" <<<"${cases}"

work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-test.XXXXXX)"
trap 'rm -rf "${work_dir}"' EXIT

HETGPU_ROUNDTRIP_WORKDIR="${work_dir}" \
HETGPU_ROUNDTRIP_SM=120 \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null

csv="${work_dir}/bench.csv"
test -s "${csv}"
head -n 1 "${csv}" | grep -Fxq "case,sm,status,cubin_bytes,lifted_ptx_bytes,lift_diagnostics,load_cubin_us,load_ptx_us,kernel_cubin_us,kernel_ptx_us,total_us,message"
grep -Fq "int_add,sm_120,dry_run" "${csv}"

custom_work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-custom-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}"' EXIT
HETGPU_ROUNDTRIP_WORKDIR="${custom_work_dir}" \
HETGPU_ROUNDTRIP_SM=90 \
HETGPU_ROUNDTRIP_CASES=pred_select \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null
custom_csv="${custom_work_dir}/bench.csv"
grep -Fq "pred_select,sm_90,dry_run" "${custom_csv}"
if grep -Fq "int_add,sm_90,dry_run" "${custom_csv}"; then
    echo "round-trip dry-run ignored HETGPU_ROUNDTRIP_CASES" >&2
    exit 1
fi

kimi_work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-kimi-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}"' EXIT
HETGPU_ROUNDTRIP_WORKDIR="${kimi_work_dir}" \
HETGPU_ROUNDTRIP_SM=120 \
HETGPU_ROUNDTRIP_CASES=kimi_iq1m_unpack,kimi_attention_mask \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null
kimi_csv="${kimi_work_dir}/bench.csv"
grep -Fq "kimi_iq1m_unpack,sm_120,dry_run" "${kimi_csv}"
grep -Fq "kimi_attention_mask,sm_120,dry_run" "${kimi_csv}"
if grep -Fq "kimi_rope_mix,sm_120,dry_run" "${kimi_csv}"; then
    echo "round-trip dry-run ignored Kimi HETGPU_ROUNDTRIP_CASES selection" >&2
    exit 1
fi

for kimi_case in \
    kimi_iq1m_unpack \
    kimi_rmsnorm_bits \
    kimi_swiglu_mix \
    kimi_rope_mix \
    kimi_attention_mask
do
    ptx_file="${SCRIPT_DIR}/ptx/${kimi_case}.ptx"
    test -s "${ptx_file}"
    grep -Fq ".visible .entry ${kimi_case}(" "${ptx_file}"
    grep -Fq ".param .u64 out" "${ptx_file}"
    grep -Fq ".param .u64 in" "${ptx_file}"
    grep -Fq ".param .u32 n" "${ptx_file}"
done

if rg -q "list-gpu-code" "${SCRIPT_DIR}/run.sh"; then
    echo "round-trip bench should validate PTX support through ptxas, not nvcc --list-gpu-code" >&2
    exit 1
fi

rg -Fq '.target ${sm}' "${SCRIPT_DIR}/run.sh"
rg -q 'rm -f "\$\{lifted\}"' "${SCRIPT_DIR}/run.sh"
rg -q 'wrote lifted PTX dump' "${SCRIPT_DIR}/run.sh"

bar_line="$(rg -n 'bar\.sync 0' "${SCRIPT_DIR}/ptx/shared_reverse.ptx" | cut -d: -f1)"
early_done_branch="$(
    (rg -n '@%p[0-9]+ bra DONE' "${SCRIPT_DIR}/ptx/shared_reverse.ptx" || true) \
        | cut -d: -f1 \
        | awk -v bar="${bar_line}" '$1 < bar { print; exit }'
)"
if [[ -n "${early_done_branch}" ]]; then
    echo "shared_reverse branches to DONE before bar.sync at line ${early_done_branch}" >&2
    exit 1
fi
