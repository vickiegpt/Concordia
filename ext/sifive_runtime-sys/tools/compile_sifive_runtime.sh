#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="${repo_root}/sifive_kernels/hetgpu_sifive_runtime.c"
out="${1:-${repo_root}/target/hetgpu_sifive_runtime.elf}"

if [[ -n "${SIFIVE_CC:-}" ]]; then
  cc="${SIFIVE_CC}"
elif command -v riscv64-linux-gnu-gcc >/dev/null 2>&1; then
  cc="riscv64-linux-gnu-gcc"
elif command -v clang-20 >/dev/null 2>&1; then
  cc="clang-20"
elif command -v clang >/dev/null 2>&1; then
  cc="clang"
else
  cc="riscv64-linux-gnu-gcc"
fi

mkdir -p "$(dirname "${out}")"

common_flags=(
  -O2
  -Wall
  -Wextra
  -fno-tree-vectorize
  -fno-asynchronous-unwind-tables
  -fno-unwind-tables
  -march=rv64gcv
  -mabi=lp64d
)
clang_target_flags=()
if [[ "${cc##*/}" == clang* ]]; then
  clang_target_flags=(
    -target
    riscv64-linux-gnu
    -menable-experimental-extensions
  )
  if [[ -n "${SIFIVE_SYSROOT:-}" ]]; then
    clang_target_flags+=(--sysroot="${SIFIVE_SYSROOT}")
  fi
  if [[ -n "${SIFIVE_GCC_TOOLCHAIN:-}" ]]; then
    clang_target_flags+=(--gcc-toolchain="${SIFIVE_GCC_TOOLCHAIN}")
  fi
fi

if ! "${cc}" "${clang_target_flags[@]}" "${common_flags[@]}" -static -o "${out}" "${src}"; then
  "${cc}" "${clang_target_flags[@]}" "${common_flags[@]}" -o "${out}" "${src}"
fi
printf '%s\n' "${out}"
