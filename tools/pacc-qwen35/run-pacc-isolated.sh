#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
ID=${1:?physical PACC id required}
shift
[[ "$ID" =~ ^[0-3]$ && $# -gt 0 ]] || exit 2
[[ "$1" != -- ]] || shift
[[ $# -gt 0 ]] || exit 2
: "${CONCORDIA_RELEASE:?qualified partition-aware runtime required}"
[[ $(</proc/sys/kernel/random/boot_id) == 89a6c374-059a-40da-8c4f-c00179fd48d4 ]] || {
    echo 'Different boot: requalify allocation before using these partitions' >&2; exit 2;
}
mkdir -p locks
exec 9>"locks/pacc$ID.lock"
flock -n 9 || { echo "PACC$ID already has an isolated worker" >&2; exit 2; }
set +e
fuser "/dev/pacc$ID" >/dev/null 2>&1
busy=$?
set -e
[[ "$busy" == 1 ]] || { echo "PACC$ID has another client or client check failed" >&2; exit 2; }
base=$(od -An -tx8 -j33570816 -N8 "/dev/hetgpu_sifive_mbox_ddr_coh$ID" | tr -d '[:space:]')
[[ "$base" == 000002075d000000 ]] || { echo 'Owned DDR base changed' >&2; exit 2; }
[[ $(</sys/module/hetgpu_sifive_mbox_ddr_coh/parameters/shared_ddr_size) == 4194304 ]] || {
    echo 'Owned DDR size changed' >&2; exit 2;
}
export HETGPU_SIFIVE_VISIBLE_DEVICES="$ID,"
export HETGPU_SIFIVE_FORCE_DEVICE="$ID"
export HETGPU_SIFIVE_GEMM_DEVICE="$ID" HETGPU_SIFIVE_GEMM_DEVICES="$ID"
export HETGPU_SIFIVE_GEMM_MULTI_N_DEVICES="$ID"
export HETGPU_SIFIVE_KERNEL_SLOT_COUNT=1 HETGPU_SIFIVE_KERNEL_TOTAL_SLOTS=1
export HETGPU_SIFIVE_KERNEL_SLOT_BASE=0 HETGPU_SIFIVE_KERNEL_SLOT_BYTES=524288
export HETGPU_SIFIVE_KERNEL_SLOT_LOW=1 HETGPU_SIFIVE_GEMM_SHARED_SLOTS=1
export HETGPU_SIFIVE_GEMM_SLOT_BYTES=524288
export HETGPU_SIFIVE_SHARED_DDR_BASE=0x2075d000000
export HETGPU_SIFIVE_SHARED_DDR_PAYLOAD_BASE_OFF=$((0x200000 + ID * 0x80000))
# The allocator owns 4 MiB; this worker's logical upper bound ends at its partition.
export HETGPU_SIFIVE_SHARED_DDR_BYTES=$((HETGPU_SIFIVE_SHARED_DDR_PAYLOAD_BASE_OFF + 0x80000))
export HETGPU_SIFIVE_JOBD_BOOTSTRAP=0 HETGPU_SIFIVE_JOB_TIMEOUT_MS=10000
export HETGPU_SIFIVE_ZLUDA_IRQ_TRACE=0 HETGPU_SIFIVE_LOG_MEMORY=0
export HETGPU_SIFIVE_MMVF_ASYNC_NO_WAIT=0 HETGPU_SIFIVE_SKIP_JOB_WAIT=0
source bin/pacc-env.sh
printf 'isolated_worker physical=%s payload=[0x%x,0x%x) runtime=%s\n' "$ID" \
    "$HETGPU_SIFIVE_SHARED_DDR_PAYLOAD_BASE_OFF" "$HETGPU_SIFIVE_SHARED_DDR_BYTES" "$CONCORDIA_RELEASE" >&2
exec "$@"
