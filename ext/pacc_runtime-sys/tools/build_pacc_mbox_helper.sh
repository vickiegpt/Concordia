#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dir="${repo_root}/mailbox_helper"
out_dir="${repo_root}/target"
kernel_release="${KERNELRELEASE:-$(uname -r)}"
kernel_build="${KERNEL_BUILD:-/lib/modules/${kernel_release}/build}"
out="${out_dir}/hetgpu_pacc_mbox.ko"

mkdir -p "${out_dir}"

if [[ ! -d "${kernel_build}" ]]; then
  cat >&2 <<EOF
Missing kernel build tree: ${kernel_build}

The running kernel is ${kernel_release}.  Install or provide the matching
prepared kernel headers/source, then rerun with:
  KERNEL_BUILD=/path/to/linux-6.7.9-build ${0}

This board has CONFIG_MODVERSIONS=n and CONFIG_MODULE_SIG=n, but the module
still needs a matching prepared build tree for struct module/vermagic.
EOF
  exit 1
fi

make -C "${kernel_build}" M="${src_dir}" modules
cp "${src_dir}/hetgpu_pacc_mbox.ko" "${out}"
modinfo "${out}" || true
printf 'output=%s\n' "${out}"
