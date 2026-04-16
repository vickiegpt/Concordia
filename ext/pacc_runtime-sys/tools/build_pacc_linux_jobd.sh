#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cc="${PACC_LINUX_CC:-gcc}"
src="${repo_root}/pacc_linux_jobd/hetgpu_pacc_jobd.c"
out="${1:-${repo_root}/target/hetgpu_pacc_jobd}"

mkdir -p "$(dirname "${out}")"
if ! "${cc}" -O2 -Wall -Wextra -static -o "${out}" "${src}" -lm; then
  "${cc}" -O2 -Wall -Wextra -o "${out}" "${src}" -lm
fi

printf '%s\n' "${out}"
