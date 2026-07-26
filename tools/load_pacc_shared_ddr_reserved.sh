#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE_DIR="${MODULE_DIR:-$ROOT/ext/pacc_runtime-sys/mailbox_helper}"
MODULE="${MODULE:-$MODULE_DIR/hetgpu_pacc_mbox_ddr_coh.ko}"
KDIR="${KDIR:-/mnt/probe_nvme0n1p4/linux-6.7.9-liveprep}"
SHARED_DDR_BASE="${SHARED_DDR_BASE:-0x20110600000}"
SHARED_DDR_SIZE="${SHARED_DDR_SIZE:-0x100000000}"
BOOT_ID_BEFORE="$(cat /proc/sys/kernel/random/boot_id)"

if [[ -r /sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_reserve_system_ram ]] &&
   [[ "$(< /sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_reserve_system_ram)" == "Y" ]] &&
   (( $(< /sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_base_override) == SHARED_DDR_BASE )) &&
   (( $(< /sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_size) >= SHARED_DDR_SIZE )); then
    echo "PACC shared DDR is already reserved."
    exit 0
fi

[[ -d "$KDIR" ]] || { echo "missing kernel build tree: $KDIR" >&2; exit 2; }
[[ -d "$MODULE_DIR" ]] || { echo "missing mailbox helper directory: $MODULE_DIR" >&2; exit 2; }

make -C "$KDIR" M="$MODULE_DIR" LD=ld.bfd hetgpu_pacc_mbox_ddr_coh.ko
modinfo "$MODULE" | grep -q '^parm:.*shared_ddr_reserve_system_ram:' || {
    echo "mailbox module does not support System RAM reservation: $MODULE" >&2
    exit 2
}

if lsmod | grep -q '^hetgpu_pacc_mbox_ddr_coh '; then
    sudo -n rmmod hetgpu_pacc_mbox_ddr_coh
fi

sudo -n sync
if [[ "${DROP_CACHES:-1}" == "1" ]]; then
    echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
fi

sudo -n insmod "$MODULE" \
    shared_ddr_base_override="$SHARED_DDR_BASE" \
    shared_ddr_size="$SHARED_DDR_SIZE" \
    shared_ddr_memremap_mode=4 \
    shared_ddr_dma_sync=1 \
    local_doorbell_bit=1 \
    shared_ddr_reserve_system_ram=1

[[ "$(< /sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_reserve_system_ram)" == "Y" ]]
[[ "$(cat /proc/sys/kernel/random/boot_id)" == "$BOOT_ID_BEFORE" ]] || {
    echo "main-host boot ID changed unexpectedly" >&2
    exit 99
}

echo "PACC shared DDR reserved: base=$SHARED_DDR_BASE size=$SHARED_DDR_SIZE"
grep -E 'MemAvailable|CmaFree' /proc/meminfo
