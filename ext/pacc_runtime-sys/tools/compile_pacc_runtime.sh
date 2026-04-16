#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${PACC_CC:-riscv64-linux-gnu-gcc}"
objcopy="${PACC_OBJCOPY:-riscv64-linux-gnu-objcopy}"
src="${repo_root}/pacc_kernels/hetgpu_pacc_runtime.c"
out="${1:-${repo_root}/target/hetgpu_pacc_runtime.elf}"
bin_out="${2:-${out%.elf}.bin}"
linker_script="${repo_root}/tools/pacc_runtime.ld"

mkdir -p "$(dirname "${out}")"
"${cc}" \
  -nostdlib \
  -static \
  -ffreestanding \
  -fno-pic \
  -fno-asynchronous-unwind-tables \
  -fno-unwind-tables \
  -march=rv64gcv \
  -mabi=lp64d \
  -Wl,--build-id=none \
  -Wl,-T,"${linker_script}" \
  -Wl,-e,_start \
  -o "${out}" \
  "${src}"
"${objcopy}" -O binary "${out}" "${bin_out}"

printf '%s\n' "${out}"
printf '%s\n' "${bin_out}"
