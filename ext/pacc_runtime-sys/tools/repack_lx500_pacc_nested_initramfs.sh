#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  repack_lx500_pacc_nested_initramfs.sh INPUT_BIN INJECT_DIR OUTPUT_BIN

The LX500 PACC Linux firmware image has an outer newc initramfs that contains
core-image-minimal-qemuriscv64.cpio.  This tool preserves the outer filesystem
layout and injects files into that inner cpio rootfs.
USAGE
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

input_bin="$1"
inject_dir="$2"
output_bin="$3"
inner_name="core-image-minimal-qemuriscv64.cpio"

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

parse_u64() {
  local value="$1"
  if [[ "${value}" == 0x* || "${value}" == 0X* ]]; then
    printf '%d\n' "$((value))"
  else
    printf '%d\n' "${value}"
  fi
}

cpio_offset="$(parse_u64 "${LX500_PACC_INITRAMFS_OFFSET:-0x4C4578}")"
outer_space_size="$(parse_u64 "${LX500_PACC_INITRAMFS_LENGTH:-0x17d1e00}")"

if [[ "${LX500_PACC_INITRAMFS_AUTODETECT:-0}" == "1" ]]; then
  cpio_offset="$(
    set +o pipefail
    grep -abo '070701' "${input_bin}" | head -1 | cut -d: -f1
  )"
  if [[ -z "${cpio_offset}" ]]; then
    echo "could not find outer newc initramfs magic 070701 in ${input_bin}" >&2
    exit 1
  fi
  outer_space_size=0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

prefix="${tmp}/prefix.bin"
outer_cpio="${tmp}/outer.cpio"
outer_root="${tmp}/outer"
inner_root="${tmp}/inner"
new_inner="${tmp}/new-inner.cpio"
new_outer="${tmp}/new-outer.cpio"
outer_list="${tmp}/outer-list.txt"
pad="${tmp}/pad.bin"

input_size="$(stat -c '%s' "${input_bin}")"
if (( cpio_offset < 0 || cpio_offset > input_size )); then
  echo "initramfs offset ${cpio_offset} is outside ${input_bin} (${input_size} bytes)" >&2
  exit 1
fi
if (( outer_space_size == 0 )); then
  outer_space_size="$((input_size - cpio_offset))"
fi
if (( outer_space_size < 0 || cpio_offset + outer_space_size > input_size )); then
  echo "initramfs window offset=${cpio_offset} length=${outer_space_size} exceeds ${input_bin} (${input_size} bytes)" >&2
  exit 1
fi

head -c "${cpio_offset}" "${input_bin}" > "${prefix}"
dd if="${input_bin}" of="${outer_cpio}" bs=1 skip="${cpio_offset}" count="${outer_space_size}" status=none

mkdir -p "${outer_root}" "${inner_root}"
(
  cd "${outer_root}"
  cpio -idmu --quiet < "${outer_cpio}"
)

if [[ ! -f "${outer_root}/${inner_name}" ]]; then
  echo "outer initramfs does not contain ${inner_name}; refusing unsafe flat repack" >&2
  exit 1
fi

old_inner_size="$(stat -c '%s' "${outer_root}/${inner_name}")"
(
  cd "${inner_root}"
  cpio -idmu --quiet < "${outer_root}/${inner_name}"
)

cp -a "${inject_dir}/." "${inner_root}/"

(
  cd "${inner_root}"
  find . -print0 | LC_ALL=C sort -z | cpio --null -o -H newc --quiet > "${new_inner}"
)

cp "${new_inner}" "${outer_root}/${inner_name}"

(
  cd "${outer_root}"
  # The PACC loader path for the LX500 image is sensitive to the nested
  # initramfs appearing immediately after the outer root entry.  Keep that
  # placement instead of purely sorting the archive, otherwise even an empty
  # repack can boot far enough for pacc->id but never complete probe.
  {
    printf '.\0'
    printf './%s\0' "${inner_name}"
    find . -print0 | LC_ALL=C sort -z | \
      awk -v RS='\0' -v ORS='\0' -v inner="./${inner_name}" '$0 != "." && $0 != inner { print }'
  } > "${outer_list}"
  cpio --null -o -H newc --quiet < "${outer_list}" > "${new_outer}"
)

new_outer_size="$(stat -c '%s' "${new_outer}")"
if (( new_outer_size > outer_space_size )); then
  echo "new outer initramfs (${new_outer_size} bytes) exceeds embedded space (${outer_space_size} bytes)" >&2
  echo "rebuild the PACC Linux image instead of in-place repacking" >&2
  exit 1
fi

pad_size="$((outer_space_size - new_outer_size))"
if (( pad_size > 0 )); then
  dd if=/dev/zero of="${pad}" bs=1 count="${pad_size}" status=none
else
  : > "${pad}"
fi

cat "${prefix}" "${new_outer}" "${pad}" > "${output_bin}"
tail -c +"$((cpio_offset + outer_space_size + 1))" "${input_bin}" >> "${output_bin}"

printf 'input=%s\n' "${input_bin}"
printf 'cpio_offset=%s\n' "${cpio_offset}"
printf 'old_size=%s\n' "${input_size}"
printf 'outer_space_size=%s\n' "${outer_space_size}"
printf 'old_inner_size=%s\n' "${old_inner_size}"
printf 'new_inner_size=%s\n' "$(stat -c '%s' "${new_inner}")"
printf 'new_outer_size=%s\n' "${new_outer_size}"
printf 'pad_size=%s\n' "${pad_size}"
printf 'new_size=%s\n' "$(stat -c '%s' "${output_bin}")"
printf 'output=%s\n' "${output_bin}"
