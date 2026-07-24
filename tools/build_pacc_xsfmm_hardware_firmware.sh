#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACC_SYS="${ROOT}/ext/pacc_runtime-sys"
JOBD_SRC="${PACC_SYS}/pacc_linux_jobd"
CC="${PACC_XSFMM_CC:-clang-20}"
BUILD_DIR="${PACC_XSFMM_BUILD_DIR:-/mnt/probe_nvme0n1p4/models/.lanxin-build/xsfmm-hardware}"
KERNEL_SRC="${PACC_XSFMM_KERNEL_SRC:-/mnt/probe_nvme0n1p4/linux-6.7.9-liveprep}"
SOURCE_FIRMWARE="${PACC_XSFMM_SOURCE_FIRMWARE:-/lib/firmware/lanxin/lx500_pacc_jobd_hostbase_idmarker.bin}"
INSTALL_FIRMWARE="${PACC_XSFMM_INSTALL_FIRMWARE:-/lib/firmware/lanxin/lx500_pacc_jobd_xsfmm32_hardware_only.bin}"
MARCH="rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvl1024b"
MODULE_BUILD="${BUILD_DIR}/xsfmm_ctx"
NATIVE_OBJ="${BUILD_DIR}/xsfmm_native_bf16.o"
JOBD="${BUILD_DIR}/hetgpu_pacc_jobd"
FIRMWARE="${BUILD_DIR}/lx500_pacc_jobd_xsfmm32_hardware_only.bin"

mkdir -p "${MODULE_BUILD}"
install -m 0644 "${JOBD_SRC}/xsfmm_ctx/Makefile" "${MODULE_BUILD}/Makefile"
install -m 0644 "${JOBD_SRC}/xsfmm_ctx/xsfmm_ctx.c" "${MODULE_BUILD}/xsfmm_ctx.c"
install -m 0644 "${JOBD_SRC}/xsfmm_ctx/xsfmm_pacc_kconfig.h" \
    "${MODULE_BUILD}/xsfmm_pacc_kconfig.h"
make -C "${KERNEL_SRC}" M="${MODULE_BUILD}" LOCALVERSION=+ \
    KCFLAGS="-include ${MODULE_BUILD}/xsfmm_pacc_kconfig.h" \
    LD=/usr/bin/ld.bfd clean modules
cp "${MODULE_BUILD}/xsfmm_ctx.ko" "${MODULE_BUILD}/xsfmm_ctx_stripped.ko"
riscv64-linux-gnu-strip --strip-debug "${MODULE_BUILD}/xsfmm_ctx_stripped.ko"
MODULE_LAYOUT_SIZE="$(
    riscv64-linux-gnu-readelf -S --wide "${MODULE_BUILD}/xsfmm_ctx_stripped.ko" |
        awk '$2 == ".gnu.linkonce.this_module" { print $6; exit }'
)"
MODULE_EXIT_OFFSET="$(
    riscv64-linux-gnu-readelf -r "${MODULE_BUILD}/xsfmm_ctx_stripped.ko" |
        awk '/cleanup_module/ && $1 != "000000000000" { print $1; exit }'
)"
MODULE_UNDEFINED="$(riscv64-linux-gnu-nm -u "${MODULE_BUILD}/xsfmm_ctx_stripped.ko")"
if [[ "${MODULE_LAYOUT_SIZE}" != "0004c0" ||
      "${MODULE_EXIT_OFFSET}" != "000000000478" ||
      -n "${MODULE_UNDEFINED}" ]]; then
    echo "PACC module ABI mismatch: size=${MODULE_LAYOUT_SIZE:-missing} " \
         "exit=${MODULE_EXIT_OFFSET:-missing} undefined=${MODULE_UNDEFINED:-none}" >&2
    exit 1
fi

"${CC}" -target riscv64-linux-gnu -menable-experimental-extensions \
    --gcc-toolchain=/usr -O3 -march="${MARCH}" -mabi=lp64d \
    -c "${JOBD_SRC}/xsfmm_native_bf16.c" -o "${NATIVE_OBJ}"

"${CC}" -target riscv64-linux-gnu -menable-experimental-extensions \
    --gcc-toolchain=/usr -O3 -Wall -Wextra \
    -fno-asynchronous-unwind-tables -fno-unwind-tables \
    -DHETGPU_PACC_HAVE_XSFMM32A16F=1 \
    -march="${MARCH}" -mabi=lp64d -static \
    -o "${JOBD}" "${JOBD_SRC}/hetgpu_pacc_jobd.c" "${NATIVE_OBJ}" \
    -pthread -ldl -lm

PACC_JOBD_XSFMM_CTX_MODULE="${MODULE_BUILD}/xsfmm_ctx_stripped.ko" \
PACC_JOBD_PACC_ID=0 \
PACC_JOBD_BEACON=1 \
PACC_JOBD_SHARED_DDR_BASE=0x20110600000 \
PACC_JOBD_XSFMM_MAX_N=32 \
    "${PACC_SYS}/patch_lx500_pacc_inplace_jobd.sh" \
    "${SOURCE_FIRMWARE}" "${JOBD}" "${FIRMWARE}"

file "${JOBD}"
modinfo "${MODULE_BUILD}/xsfmm_ctx_stripped.ko" | grep -E 'description|vermagic'
sha256sum "${JOBD}" "${MODULE_BUILD}/xsfmm_ctx_stripped.ko" "${FIRMWARE}"

if [[ "${PACC_XSFMM_INSTALL:-0}" == "1" ]]; then
    sudo install -m 0644 "${FIRMWARE}" "${INSTALL_FIRMWARE}"
    echo "installed ${INSTALL_FIRMWARE}"
fi
