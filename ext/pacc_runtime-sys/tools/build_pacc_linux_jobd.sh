#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${PACC_LINUX_CC:-gcc}"
src="${repo_root}/pacc_linux_jobd/hetgpu_pacc_jobd.c"
out="${1:-${repo_root}/target/hetgpu_pacc_jobd}"

mkdir -p "$(dirname "${out}")"
common_flags=(-O2 -Wall -Wextra -fno-tree-vectorize -march=rv64gcv -mabi=lp64d -pthread)
if ! "${cc}" "${common_flags[@]}" -static -o "${out}" "${src}" -lm; then
  if ! "${cc}" "${common_flags[@]}" -o "${out}" "${src}" -lm; then
    "${cc}" -O2 -Wall -Wextra -pthread -o "${out}" "${src}" -lm
  fi
fi

printf '%s\n' "${out}"
