#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PACC_LINUX_CC:-}" ]]; then
  cc="${PACC_LINUX_CC}"
elif command -v clang-20 >/dev/null 2>&1; then
  cc="clang-20"
else
  cc="gcc"
fi
src="${PACC_LINUX_JOBD_SRC:-${repo_root}/pacc_linux_jobd/hetgpu_pacc_jobd.c}"
out="${1:-${repo_root}/target/hetgpu_pacc_jobd}"
extra_cflags=()
extra_ldflags=()

if [[ -n "${PACC_LINUX_CFLAGS_EXTRA:-}" ]]; then
  read -r -a extra_cflags <<< "${PACC_LINUX_CFLAGS_EXTRA}"
fi
if [[ -n "${PACC_LINUX_LDFLAGS_EXTRA:-}" ]]; then
  read -r -a extra_ldflags <<< "${PACC_LINUX_LDFLAGS_EXTRA}"
fi

mkdir -p "$(dirname "${out}")"
cc_base="$(basename "${cc}")"
if [[ "${cc_base}" == clang* ]]; then
  common_flags=(-O3 -Wall -Wextra -funroll-loops -menable-experimental-extensions
    -march=rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvl1024b
    -mabi=lp64d -pthread)
else
  common_flags=(-O2 -Wall -Wextra -fno-tree-vectorize
    -march=rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvl1024b
    -mabi=lp64d -pthread)
fi
common_flags+=("${extra_cflags[@]}")
link_flags=(-fuse-ld=bfd -lm -ldl -rdynamic)
link_flags+=("${extra_ldflags[@]}")
if [[ "${PACC_LINUX_STATIC:-1}" == "0" ]]; then
  "${cc}" "${common_flags[@]}" -o "${out}" "${src}" "${link_flags[@]}"
elif ! "${cc}" "${common_flags[@]}" -static -o "${out}" "${src}" "${link_flags[@]}"; then
  if ! "${cc}" "${common_flags[@]}" -o "${out}" "${src}" "${link_flags[@]}"; then
    "${cc}" -O2 -Wall -Wextra -pthread -o "${out}" "${src}" "${link_flags[@]}"
  fi
fi

printf '%s\n' "${out}"
