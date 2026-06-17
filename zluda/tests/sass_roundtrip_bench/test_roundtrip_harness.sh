#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cases="$("${SCRIPT_DIR}/run.sh" --list-cases)"
grep -Fxq "int_add" <<<"${cases}"
grep -Fxq "pred_select" <<<"${cases}"
grep -Fxq "fma_bits" <<<"${cases}"
grep -Fxq "shared_reverse" <<<"${cases}"

work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-test.XXXXXX)"
trap 'rm -rf "${work_dir}"' EXIT

HETGPU_ROUNDTRIP_WORKDIR="${work_dir}" \
HETGPU_ROUNDTRIP_SM=120 \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null

csv="${work_dir}/bench.csv"
test -s "${csv}"
head -n 1 "${csv}" | grep -Fxq "case,sm,status,cubin_bytes,lifted_ptx_bytes,lift_diagnostics,load_cubin_us,load_ptx_us,kernel_cubin_us,kernel_ptx_us,total_us,message"
grep -Fq "int_add,sm_120,dry_run" "${csv}"
