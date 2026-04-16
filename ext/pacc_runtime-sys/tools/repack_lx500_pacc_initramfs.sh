#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  repack_lx500_pacc_initramfs.sh INPUT_BIN INJECT_DIR OUTPUT_BIN

INPUT_BIN is the vendor lanxin/lx500_pacc.bin image.
INJECT_DIR contents are copied into the root of the embedded initramfs.
OUTPUT_BIN is written as a new image; INPUT_BIN is never modified.
USAGE
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

input_bin="$1"
inject_dir="$2"
output_bin="$3"

if [[ ! -f "${input_bin}" ]]; then
  echo "missing input image: ${input_bin}" >&2
  exit 1
fi
if [[ ! -d "${inject_dir}" ]]; then
  echo "missing inject dir: ${inject_dir}" >&2
  exit 1
fi
command -v cpio >/dev/null

if [[ "$(id -u)" -ne 0 ]]; then
  echo "this repack needs root so cpio can preserve device nodes; run with sudo" >&2
  exit 1
fi

cpio_offset="$(
  set +o pipefail
  grep -abo '070701' "${input_bin}" | head -1 | cut -d: -f1
)"
if [[ -z "${cpio_offset}" ]]; then
  echo "could not find newc initramfs magic 070701 in ${input_bin}" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

prefix="${tmp}/prefix.bin"
old_cpio="${tmp}/initramfs.cpio"
root="${tmp}/root"
new_cpio="${tmp}/new_initramfs.cpio"
pad="${tmp}/pad.bin"
input_size="$(stat -c '%s' "${input_bin}")"
old_cpio_size="$((input_size - cpio_offset))"

head -c "${cpio_offset}" "${input_bin}" > "${prefix}"
tail -c +"$((cpio_offset + 1))" "${input_bin}" > "${old_cpio}"

mkdir -p "${root}"
(
  cd "${root}"
  cpio -idmu --quiet < "${old_cpio}"
)

cp -a "${inject_dir}/." "${root}/"

(
  cd "${root}"
  find . -print0 | LC_ALL=C sort -z | cpio --null -o -H newc --quiet > "${new_cpio}"
)

new_cpio_size="$(stat -c '%s' "${new_cpio}")"
if (( new_cpio_size > old_cpio_size )); then
  echo "new initramfs (${new_cpio_size} bytes) is larger than embedded space (${old_cpio_size} bytes)" >&2
  echo "refusing to change the raw Linux image length; rebuild the PACC Linux image instead" >&2
  exit 1
fi

pad_size="$((old_cpio_size - new_cpio_size))"
if (( pad_size > 0 )); then
  dd if=/dev/zero of="${pad}" bs=1 count="${pad_size}" status=none
else
  : > "${pad}"
fi

cat "${prefix}" "${new_cpio}" "${pad}" > "${output_bin}"

printf 'input=%s\n' "${input_bin}"
printf 'cpio_offset=%s\n' "${cpio_offset}"
printf 'old_size=%s\n' "${input_size}"
printf 'old_cpio_size=%s\n' "${old_cpio_size}"
printf 'new_cpio_size=%s\n' "${new_cpio_size}"
printf 'pad_size=%s\n' "${pad_size}"
printf 'new_size=%s\n' "$(stat -c '%s' "${output_bin}")"
printf 'output=%s\n' "${output_bin}"
