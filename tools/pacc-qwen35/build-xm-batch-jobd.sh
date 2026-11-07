#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
VERSION=v38_q6k_xm
grep -q 'uint32_t result\[12\] = {0}' bin/hetgpu_pacc_jobd.c
grep -q '0x5330, 0x20026' bin/hetgpu_pacc_jobd.c
clang-20 -O3 -ffp-contract=off -march=rv64gcv -c bin/q6k-batch-xm.c -o firmware/q6k-batch-xm.o
clang-20 -O2 -ffp-contract=off -pthread -DHETGPU_PACC_Q6K_XM_BATCH \
    bin/test_jobd_q6k.c firmware/q6k-batch-xm.o bin/test-xm-pack-reference.c \
    -ldl -lm -o bin/test_q6k_xm_pack_reference
bin/test_q6k_xm_pack_reference
clang-20 -march=rv64gc -O2 -c bin/xm-config-probe.c -o firmware/xm-config-probe.o
clang-20 -march=rv64gcv -O2 -ffp-contract=off -c bin/xm-user-probe.c -o firmware/xm-user-probe.o
clang-20 -march=rv64gcv -O2 -c bin/xsfmm_native_bf16.c -o firmware/xsfmm_native_bf16.o
clang-20 -O2 -static -pthread -DHETGPU_PACC_DIAGNOSTIC_BUILD -DHETGPU_PACC_PHASE_PROFILE \
    -DHETGPU_PACC_USER_XM_PROBE -DHETGPU_PACC_XM_CONFIG_PROBE -DHETGPU_PACC_Q6K_XM_BATCH \
    bin/hetgpu_pacc_jobd.c firmware/q6k-batch-xm.o firmware/xm-user-probe.o \
    firmware/xm-config-probe.o firmware/xsfmm_native_bf16.o -ldl -lm \
    -o "firmware/hetgpu_pacc_jobd_$VERSION"
PACC_JOBD_PACC_ID=0 PACC_JOBD_KERNEL_THREADS=1 PACC_JOBD_BEACON=1 \
PACC_JOBD_FORCE_DEVMEM=1 PACC_JOBD_STATUS_MMAP_FALLBACK=0 \
PACC_JOBD_EARLY_DEVMEM_MARKER=1 PACC_JOBD_SHARED_DDR_BASE=0x2075d000000 \
PACC_JOBD_SHARED_DDR_SIZE=0x400000 PACC_JOBD_SHARED_DDR_PACC_BASE= \
    bash pacc-runtime-head/ext/pacc_runtime-sys/patch_lx500_pacc_inplace_jobd.sh \
    firmware/lx500_pacc.vendor-backup.bin "firmware/hetgpu_pacc_jobd_$VERSION" \
    "firmware/lx500_pacc_qwen35_pacc0_${VERSION}_raw.bin"
SHA=$(sha256sum "firmware/lx500_pacc_qwen35_pacc0_${VERSION}_raw.bin" | cut -d' ' -f1)
python3 bin/patch-xm-isa-alias.py "firmware/lx500_pacc_qwen35_pacc0_${VERSION}_raw.bin" \
    "firmware/lx500_pacc_qwen35_pacc0_$VERSION.bin" --sha256 "$SHA"
sha256sum "firmware/lx500_pacc_qwen35_pacc0_$VERSION.bin" bin/hetgpu_pacc_jobd.c
