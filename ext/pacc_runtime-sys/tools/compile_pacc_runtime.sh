#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
objcopy="${PACC_OBJCOPY:-riscv64-linux-gnu-objcopy}"
src="${repo_root}/pacc_kernels/hetgpu_pacc_runtime.c"
out="${1:-${repo_root}/target/hetgpu_pacc_runtime.elf}"
bin_out="${2:-${out%.elf}.bin}"
linker_script="${repo_root}/tools/pacc_runtime.ld"
header_root="${repo_root}/../llvm-project/clang/lib/Headers"

if [[ -n "${PACC_CC:-}" ]]; then
  cc="${PACC_CC}"
elif command -v clang-20 >/dev/null 2>&1; then
  cc="clang-20"
elif command -v clang >/dev/null 2>&1; then
  cc="clang"
else
  cc="riscv64-linux-gnu-gcc"
fi

mkdir -p "$(dirname "${out}")"
common_flags=(
  -nostdlib
  -static
  -ffreestanding
  -fno-pic
  -fno-asynchronous-unwind-tables
  -fno-unwind-tables
  -mabi=lp64d
  -Wl,--build-id=none
  "-Wl,-T,${linker_script}"
  -Wl,-e,_start
)

if [[ "${cc##*/}" == clang* ]]; then
  "${cc}" \
    -target riscv64-unknown-elf \
    --ld-path=riscv64-linux-gnu-ld \
    -menable-experimental-extensions \
    -march=rv64gcv_zvfbfmin_xsfvcp_xsfvfnrclipxfqf_xsfvfwmaccqqq_xsfvqmaccqoq \
    -I"${header_root}" \
    "${common_flags[@]}" \
    -o "${out}" \
    "${src}"
else
  "${cc}" \
    -march=rv64gcv \
    "${common_flags[@]}" \
    -o "${out}" \
    "${src}"
fi
"${objcopy}" -O binary "${out}" "${bin_out}"

printf '%s\n' "${out}"
printf '%s\n' "${bin_out}"
