#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACC_SYS="${ROOT}/ext/pacc_runtime-sys"
CC="${PACC_XSFMM_CC:-clang-20}"
BUILD_DIR="${PACC_XSFMM_BUILD_DIR:-/mnt/probe_nvme0n1p4/models/.lanxin-build/xsfmm-failclosed}"
SOURCE_FIRMWARE="${PACC_XSFMM_SOURCE_FIRMWARE:-/lib/firmware/lanxin/lx500_pacc_jobd_hostbase_idmarker.bin}"
INSTALL_FIRMWARE="${PACC_XSFMM_INSTALL_FIRMWARE:-/lib/firmware/lanxin/lx500_pacc_jobd_xsfmm32_failclosed.bin}"
JOBD="${BUILD_DIR}/hetgpu_pacc_jobd"
FIRMWARE="${BUILD_DIR}/lx500_pacc_jobd_xsfmm32_failclosed.bin"

mkdir -p "${BUILD_DIR}"

"${CC}" \
    -target riscv64-linux-gnu \
    -menable-experimental-extensions \
    --gcc-toolchain=/usr \
    -O3 -Wall -Wextra \
    -fno-asynchronous-unwind-tables -fno-unwind-tables \
    -march=rv64gcv_zbb_zfh_zvfh_zfbfmin_zvfbfmin_zvfbfwma_zvl1024b \
    -mabi=lp64d -static \
    -o "${JOBD}" \
    "${PACC_SYS}/pacc_linux_jobd/hetgpu_pacc_jobd.c" \
    -pthread -ldl -lm

PACC_JOBD_PACC_ID=0 \
PACC_JOBD_BEACON=1 \
PACC_JOBD_SHARED_DDR_BASE=0x20110600000 \
PACC_JOBD_XSFMM_MAX_N=32 \
    "${PACC_SYS}/patch_lx500_pacc_inplace_jobd.sh" \
    "${SOURCE_FIRMWARE}" "${JOBD}" "${FIRMWARE}"

file "${JOBD}"
sha256sum "${JOBD}" "${FIRMWARE}"

if [[ "${PACC_XSFMM_INSTALL:-0}" == "1" ]]; then
    sudo install -m 0644 "${FIRMWARE}" "${INSTALL_FIRMWARE}"
    echo "installed ${INSTALL_FIRMWARE}"
fi
