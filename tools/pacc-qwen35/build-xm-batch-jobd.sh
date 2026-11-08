#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
VERSION=${VERSION:-v46_xm_reduce}
BUILD_ID=${BUILD_ID:-0x2002e}
COMPILER_PROFILE=${COMPILER_PROFILE:-baseline}
HOT_FLAGS=(-O3 -march=rv64gcv)
JOBD_FLAGS=(-O2)
WORKER_FLAGS=(-O2)
case "$COMPILER_PROFILE" in
    baseline) ;;
    o3-rvv)
        JOBD_FLAGS=(-O3 -march=rv64gcv)
        WORKER_FLAGS=(-O3 -march=rv64gcv)
        ;;
    x280-tune)
        HOT_FLAGS+=(-mtune=sifive-x280)
        JOBD_FLAGS=(-O3 -march=rv64gcv -mtune=sifive-x280)
        WORKER_FLAGS=(-O3 -march=rv64gcv -mtune=sifive-x280)
        ;;
    vlen1024)
        HOT_FLAGS=(-O3 -march=rv64gcv_zvl1024b -mrvv-vector-bits=1024)
        JOBD_FLAGS=("${HOT_FLAGS[@]}")
        WORKER_FLAGS=("${HOT_FLAGS[@]}")
        ;;
    *) echo "unknown compiler profile: $COMPILER_PROFILE" >&2; exit 2 ;;
esac
test ! -e "firmware/lx500_pacc_qwen35_pacc0_$VERSION.bin"
test ! -e "firmware/hetgpu_pacc_jobd_$VERSION"
grep -q 'uint32_t result\[12\] = {0}' bin/hetgpu_pacc_jobd.c
grep -q '0x5330, HETGPU_PACC_XM_BUILD_ID' bin/hetgpu_pacc_jobd.c
printf 'compiler_profile=%s hot=%s jobd=%s workers=%s\n' "$COMPILER_PROFILE" "${HOT_FLAGS[*]}" "${JOBD_FLAGS[*]}" "${WORKER_FLAGS[*]}"
clang-20 "${HOT_FLAGS[@]}" -ffp-contract=off -DHETGPU_PACC_Q6K_VECTOR_PACK \
    -DHETGPU_PACC_Q6K_VECTOR_FLOAT_PACK -DHETGPU_PACC_Q6K_VECTOR_REDUCE \
    -c bin/q6k-batch-xm.c -o firmware/q6k-batch-xm.o
clang-20 "${WORKER_FLAGS[@]}" -pthread -c bin/q6k-xm-workers.c -o firmware/q6k-xm-workers.o
TEST_Q6_OBJECT=firmware/q6k-batch-xm.o
TEST_WORKER_OBJECT=firmware/q6k-xm-workers.o
HOST_VLEN=0
if [[ "$COMPILER_PROFILE" == vlen1024 ]]; then
    clang-20 -O2 -march=rv64gcv bin/read-rvv-vlen.c -o bin/read-rvv-vlen
    HOST_VLEN=$(bin/read-rvv-vlen)
    if [[ "$HOST_VLEN" != 1024 ]]; then
        printf 'Host VLEN=%s: run VLA layout tests only; VLS1024 requires PACC validation\n' "$HOST_VLEN"
        TEST_Q6_OBJECT=firmware/q6k-batch-xm-hosttest.o
        TEST_WORKER_OBJECT=firmware/q6k-xm-workers-hosttest.o
        clang-20 -O3 -ffp-contract=off -march=rv64gcv \
            -DHETGPU_PACC_Q6K_VECTOR_PACK -DHETGPU_PACC_Q6K_VECTOR_FLOAT_PACK \
            -DHETGPU_PACC_Q6K_VECTOR_REDUCE -c bin/q6k-batch-xm.c -o "$TEST_Q6_OBJECT"
        clang-20 -O2 -pthread -c bin/q6k-xm-workers.c -o "$TEST_WORKER_OBJECT"
    fi
fi
clang-20 -O2 -ffp-contract=off -pthread -DHETGPU_PACC_Q6K_XM_BATCH \
    bin/test_jobd_q6k.c "$TEST_Q6_OBJECT" "$TEST_WORKER_OBJECT" bin/test-xm-pack-reference.c \
    -ldl -lm -o bin/test_q6k_xm_pack_reference
bin/test_q6k_xm_pack_reference
clang-20 -O2 -ffp-contract=off -pthread -DHETGPU_PACC_Q6K_XM_BATCH \
    bin/test_jobd_q8.c "$TEST_Q6_OBJECT" "$TEST_WORKER_OBJECT" bin/test-xm-pack-reference.c \
    -ldl -lm -o bin/test_q8_xm_pack_reference
bin/test_q8_xm_pack_reference
clang-20 -O2 bin/test-q8-xm-faults.c "$TEST_Q6_OBJECT" -lm -o bin/test_q8_xm_faults
bin/test_q8_xm_faults
clang-20 -O2 -pthread bin/test_jobd_q8.c -ldl -lm -o bin/test_jobd_q8
clang-20 -O2 -pthread bin/test-xm-reject.c "$TEST_Q6_OBJECT" \
    "$TEST_WORKER_OBJECT" -lm -o bin/test_xm_reject
bin/test_xm_reject
clang-20 -O2 -pthread bin/test_jobd_q6k.c -ldl -lm -o bin/test_jobd_q6k
clang-20 "${HOT_FLAGS[@]}" -c bin/payload-copy-rvv.c -o firmware/payload-copy-rvv.o
TEST_COPY_OBJECT=firmware/payload-copy-rvv.o
if [[ "$COMPILER_PROFILE" == vlen1024 && "$HOST_VLEN" != 1024 ]]; then
    TEST_COPY_OBJECT=firmware/payload-copy-hosttest.o
    clang-20 -O3 -march=rv64gcv -c bin/payload-copy-rvv.c -o "$TEST_COPY_OBJECT"
fi
clang-20 -O2 bin/test-payload-copy.c "$TEST_COPY_OBJECT" -o bin/test_payload_copy
bin/test_payload_copy
clang-20 -O2 -pthread bin/test_jobd_elf_cache.c -ldl -lm -o bin/test_jobd_elf_cache
bin/test_jobd_elf_cache
clang-20 -march=rv64gc -O2 -c bin/xm-config-probe.c -o firmware/xm-config-probe.o
clang-20 -march=rv64gcv -O2 -ffp-contract=off -c bin/xm-user-probe.c -o firmware/xm-user-probe.o
clang-20 -march=rv64gcv -O2 -c bin/xsfmm_native_bf16.c -o firmware/xsfmm_native_bf16.o
clang-20 "${JOBD_FLAGS[@]}" -static -pthread -DHETGPU_PACC_DIAGNOSTIC_BUILD -DHETGPU_PACC_PHASE_PROFILE \
    -DHETGPU_PACC_XM_BUILD_ID="$BUILD_ID" -DHETGPU_PACC_RVV_PAYLOAD_COPY \
    -DHETGPU_PACC_HOT_SCAN_US=2000 -DHETGPU_PACC_QUIET_DISPATCH \
    -DHETGPU_PACC_USER_XM_PROBE -DHETGPU_PACC_XM_CONFIG_PROBE -DHETGPU_PACC_Q6K_XM_BATCH \
    bin/hetgpu_pacc_jobd.c firmware/q6k-batch-xm.o firmware/q6k-xm-workers.o firmware/xm-user-probe.o \
    firmware/xm-config-probe.o firmware/xsfmm_native_bf16.o firmware/payload-copy-rvv.o -ldl -lm \
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
