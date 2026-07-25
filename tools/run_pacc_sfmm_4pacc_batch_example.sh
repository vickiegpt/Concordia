#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICES="${PACC_SFMM_DEVICES:-0,1,2,3}"
RECOVER_MASK="${PACC_SFMM_RECOVER_MASK:-0xf}"
RECOVER_FIRMWARE="${PACC_SFMM_RECOVER_FIRMWARE:-lanxin/lx500_pacc_jobd_xsfmm_kernel_exec.bin}"
STABLE_FIRMWARE="${PACC_SFMM_STABLE_FIRMWARE:-lanxin/lx500_pacc_jobd_hostbase_idmarker.bin}"
RECOVER_SETTLE_S="${PACC_SFMM_RECOVER_SETTLE_S:-15}"
BOOT_PACC="${PACC_SFMM_BOOT:-1}"

M="${PACC_GEMM_M:-2048}"
N="${PACC_GEMM_N:-16}"
K="${PACC_GEMM_K:-2048}"
ITERS="${PACC_GEMM_ITERS:-1}"
WARMUP="${PACC_GEMM_WARMUP:-0}"
XSFMM_REPEATS="${PACC_SFMM_XSFMM_REPEATS:-}"
if [[ -z "$XSFMM_REPEATS" &&
      -r "/lib/firmware/${RECOVER_FIRMWARE}.meta" ]]; then
    XSFMM_REPEATS="$(
        awk -F= '$1 == "repeats" { print $2; exit }' \
            "/lib/firmware/${RECOVER_FIRMWARE}.meta"
    )"
fi
XSFMM_REPEATS="${XSFMM_REPEATS:-1}"

ZLUDA_BUILD_DIR="${ZLUDA_BUILD_DIR:-/mnt/usb/hetgpu_build_target/releasefix/release}"
SUBMIT_SHIM="${PACC_SFMM_SUBMIT_SHIM_SO:-/tmp/libpacc_sfmm_submit_shim.so}"
PROBE="${PACC_BF16_GEMM_PROBE:-${ROOT}/tools/pacc_gemm_bf16_probe}"
LOG_ROOT="${PACC_SFMM_LOG_ROOT:-/tmp/lanxin_disagg_eval}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PACC_SFMM_LOG_DIR:-${LOG_ROOT}/sfmm_4pacc_batch_${STAMP}}"

mkdir -p "$LOG_DIR"

HOST_BOOT_ID_BEFORE="$(cat /proc/sys/kernel/random/boot_id)"
BOOT_ATTEMPTED=0

clear_control_window() {
    local path="${PACC_SFMM_CLEAR_DDR_DEVICE:-/dev/hetgpu_pacc_mbox_ddr_coh0}"
    local off="${PACC_SFMM_CONTROL_BASE_OFF:-0x100000}"
    local bytes="${PACC_SFMM_CONTROL_BYTES:-0x8000}"
    python3 - "$path" "$off" "$bytes" <<'PY'
import os
import sys

path, off_s, size_s = sys.argv[1:4]
off = int(off_s, 0)
size = int(size_s, 0)
fd = os.open(path, os.O_RDWR)
try:
    written = os.pwrite(fd, b"\0" * size, off)
finally:
    os.close(fd)
if written != size:
    raise SystemExit(f"short control clear: {written}/{size}")
print(f"cleared shared-DDR control path={path} off=0x{off:x} bytes=0x{size:x}")
PY
}

cleanup() {
    local rc=$?
    trap - EXIT
    if [[ "$rc" != "0" && "$BOOT_ATTEMPTED" == "1" &&
          "${PACC_SFMM_RESTORE_ON_FAILURE:-1}" != "0" ]]; then
        echo "restoring stable PACC firmware mask=${RECOVER_MASK} firmware=${STABLE_FIRMWARE}" >&2
        clear_control_window || true
        HETGPU_PACC_RECOVER_FIRMWARE="$STABLE_FIRMWARE" \
        HETGPU_PACC_RECOVER_PATCH_PACC_ID_ENV="${PACC_SFMM_PATCH_PACC_ID_ENV:-1}" \
        HETGPU_PACC_RECOVER_PRE_RESET_MASK="$RECOVER_MASK" \
        HETGPU_PACC_RECOVER_PRE_RESET_SETTLE_S="${PACC_SFMM_PRE_RESET_SETTLE_S:-1}" \
        HETGPU_PACC_RECOVER_SETTLE_S="$RECOVER_SETTLE_S" \
            ext/pacc_runtime-sys/tools/pacc_recover_no_reboot.sh "$RECOVER_MASK" || true
    fi
    local host_boot_id_after
    host_boot_id_after="$(cat /proc/sys/kernel/random/boot_id)"
    {
        echo "host_boot_id_before=${HOST_BOOT_ID_BEFORE}"
        echo "host_boot_id_after=${host_boot_id_after}"
    } | tee "${LOG_DIR}/host_boot_id.txt"
    if [[ "$host_boot_id_after" != "$HOST_BOOT_ID_BEFORE" ]]; then
        echo "main-host boot ID changed during PACC-only run" >&2
        rc=99
    fi
    exit "$rc"
}
trap cleanup EXIT

if [[ ! -x "$PROBE" ]]; then
    echo "missing probe: $PROBE" >&2
    exit 2
fi
if [[ ! -r "$SUBMIT_SHIM" ]]; then
    submit_src="${ROOT}/tmp/mbox_readfix/pacc_sfmm_submit_shim.c"
    if [[ ! -r "$submit_src" ]]; then
        echo "missing submit shim source: $submit_src" >&2
        exit 2
    fi
    "${CC:-cc}" -O3 -fPIC -shared -Wall -Wextra \
        -o "$SUBMIT_SHIM" "$submit_src" -ldl -pthread
fi
if [[ ! -r "${ZLUDA_BUILD_DIR}/libnvcuda.so" || ! -r "${ZLUDA_BUILD_DIR}/libhetgpu_cuda_shim.so" ]]; then
    echo "missing ZLUDA/PACC user-space libs under $ZLUDA_BUILD_DIR" >&2
    exit 2
fi

if [[ "$BOOT_PACC" != "0" ]]; then
    if [[ "${PACC_SFMM_CLEAR_CONTROL:-1}" != "0" ]]; then
        if ! clear_control_window; then
            echo "warning: failed to clear shared-DDR control window before boot" >&2
        fi
    fi
    echo "booting PACC mask=${RECOVER_MASK} firmware=${RECOVER_FIRMWARE}"
    BOOT_ATTEMPTED=1
    HETGPU_PACC_RECOVER_FIRMWARE="$RECOVER_FIRMWARE" \
    HETGPU_PACC_RECOVER_PATCH_PACC_ID_ENV="${PACC_SFMM_PATCH_PACC_ID_ENV:-1}" \
    HETGPU_PACC_RECOVER_PRE_RESET_MASK="${PACC_SFMM_PRE_RESET_MASK:-$RECOVER_MASK}" \
    HETGPU_PACC_RECOVER_PRE_RESET_SETTLE_S="${PACC_SFMM_PRE_RESET_SETTLE_S:-1}" \
    HETGPU_PACC_RECOVER_SETTLE_S="$RECOVER_SETTLE_S" \
        ext/pacc_runtime-sys/tools/pacc_recover_no_reboot.sh "$RECOVER_MASK"
    if [[ "${PACC_SFMM_WAIT_READY:-1}" != "0" ]]; then
        ready_dev="${PACC_SFMM_READY_DDR_DEVICE:-/dev/hetgpu_pacc_mbox_ddr_coh0}"
        ready_off="${PACC_SFMM_CONTROL_BASE_OFF:-0x100000}"
        ready_timeout="${PACC_SFMM_READY_TIMEOUT_S:-45}"
        python3 - "$ready_dev" "$ready_off" "$ready_timeout" "$DEVICES" <<'PY'
import os
import struct
import sys
import time

path, off_s, timeout_s, devices_s = sys.argv[1:5]
base = int(off_s, 0)
deadline = time.time() + float(timeout_s)
devices = [int(x, 0) for x in devices_s.replace(",", " ").split() if x]
fd = os.open(path, os.O_RDONLY | os.O_SYNC)
try:
    pending = set(devices)
    while pending and time.time() < deadline:
        ready = set()
        for dev in list(pending):
            slot = base + dev * 0x2000
            beacon = os.pread(fd, 32, slot + 0x1f40)
            bmagic, bver, _bjob, bphase, _bdetail, _bseq = struct.unpack("<QIIIIQ", beacon)
            if bmagic == 0x4847505542434e31 and bver == 1 and bphase == 0x7002:
                ready.add(dev)
        pending -= ready
        if pending:
            time.sleep(0.1)
    if pending:
        raise SystemExit(f"PACC ready timeout for devices {sorted(pending)}")
    print(f"PACC ready devices={devices} path={path}")
finally:
    os.close(fd)
PY
    fi
    post_ready_sleep="${PACC_SFMM_POST_READY_SLEEP_S:-5}"
    if [[ "$post_ready_sleep" != "0" ]]; then
        echo "post-ready settle ${post_ready_sleep}s"
        sleep "$post_ready_sleep"
    fi
fi

export LD_LIBRARY_PATH="${ZLUDA_BUILD_DIR}:${ZLUDA_BUILD_DIR}/deps:/home/ubuntu/fake_cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HETGPU_PACC_VISIBLE_DEVICES="${HETGPU_PACC_VISIBLE_DEVICES:-$DEVICES}"

export HETGPU_PACC_SHARED_DDR_BASE="${HETGPU_PACC_SHARED_DDR_BASE:-0x20110600000}"
export HETGPU_PACC_SHARED_DDR_PACC_BASE="${HETGPU_PACC_SHARED_DDR_PACC_BASE:-0x20110600000}"
export HETGPU_PACC_SHARED_DDR_BYTES="${HETGPU_PACC_SHARED_DDR_BYTES:-0x100000000}"
export HETGPU_PACC_SHARED_DDR_PAYLOAD_BASE_OFF="${HETGPU_PACC_SHARED_DDR_PAYLOAD_BASE_OFF:-0x200000}"
export HETGPU_PACC_GEMM_SHARED_SLOTS="${HETGPU_PACC_GEMM_SHARED_SLOTS:-4}"
export HETGPU_PACC_GEMM_SLOT_BYTES="${HETGPU_PACC_GEMM_SLOT_BYTES:-0x01000000}"

export HETGPU_PACC_MAILBOX_DEVICE="${HETGPU_PACC_MAILBOX_DEVICE:-/dev/hetgpu_pacc_mbox_live0}"
export HETGPU_PACC_MBOX_DEVICE="${HETGPU_PACC_MBOX_DEVICE:-/dev/hetgpu_pacc_mbox_live0}"
export HETGPU_PACC_SHARED_DDR_DEVICE="${HETGPU_PACC_SHARED_DDR_DEVICE:-/dev/hetgpu_pacc_mbox_ddr_coh0}"

export HETGPU_PACC_SFMM_SUBMIT_SHIM=1
export HETGPU_PACC_SFMM_ALIAS_MIRROR="${HETGPU_PACC_SFMM_ALIAS_MIRROR:-0}"
export HETGPU_PACC_CONTROL_LEGACY_MIRROR="${HETGPU_PACC_CONTROL_LEGACY_MIRROR:-0}"
export HETGPU_PACC_TOP_MBOX_RING="${HETGPU_PACC_TOP_MBOX_RING:-0}"
export HETGPU_PACC_NOTIFY_IOCTL="${HETGPU_PACC_NOTIFY_IOCTL:-1}"
export HETGPU_PACC_JOB_TIMEOUT_MS="${HETGPU_PACC_JOB_TIMEOUT_MS:-180000}"

export HETGPU_PACC_ALLOW_HOST_DEVICE_MEM="${HETGPU_PACC_ALLOW_HOST_DEVICE_MEM:-1}"
export HETGPU_PACC_ALLOW_HOST_GEMM_FALLBACK="${HETGPU_PACC_ALLOW_HOST_GEMM_FALLBACK:-0}"
export HETGPU_PACC_GEMM_MMVF_ROUTE_MAX_N="${HETGPU_PACC_GEMM_MMVF_ROUTE_MAX_N:-0}"
export HETGPU_PACC_GEMM_MMVF_MAX_N="${HETGPU_PACC_GEMM_MMVF_MAX_N:-0}"
export HETGPU_PACC_GEMM_TRACE="${HETGPU_PACC_GEMM_TRACE:-1}"

IFS=',' read -r -a DEVICE_ARRAY <<< "$DEVICES"
if [[ "${#DEVICE_ARRAY[@]}" -lt 1 ]]; then
    echo "empty PACC_SFMM_DEVICES" >&2
    exit 2
fi

echo "log_dir=${LOG_DIR}"
echo "shape m=${M} n=${N} k=${K} warmup=${WARMUP} iters=${ITERS} devices=${DEVICES} xsfmm_repeats=${XSFMM_REPEATS}"
echo "ld_preload=${SUBMIT_SHIM} ${ZLUDA_BUILD_DIR}/libnvcuda.so ${ZLUDA_BUILD_DIR}/libhetgpu_cuda_shim.so"

pids=()
devs=()
start_ns="$(date +%s%N)"
for dev in "${DEVICE_ARRAY[@]}"; do
    dev="${dev//[[:space:]]/}"
    [[ -n "$dev" ]] || continue
    log="${LOG_DIR}/pacc${dev}.log"
    echo "launch pacc${dev} -> ${log}"
    (
        export LD_PRELOAD="${SUBMIT_SHIM} ${ZLUDA_BUILD_DIR}/libnvcuda.so ${ZLUDA_BUILD_DIR}/libhetgpu_cuda_shim.so${LD_PRELOAD:+ ${LD_PRELOAD}}"
        export HETGPU_PACC_GEMM_DEVICE="$dev"
        export HETGPU_PACC_VISIBLE_DEVICES="$dev"
        export HETGPU_PACC_GEMM_DEVICES="$dev"
        export HETGPU_PACC_GEMM_MULTI_N_DEVICES="$dev"
        export HETGPU_PACC_REDUCE_DEVICE="$dev"
        export HETGPU_PACC_MAILBOX_DEVICE="/dev/hetgpu_pacc_mbox_live${dev}"
        export HETGPU_PACC_MBOX_DEVICE="/dev/hetgpu_pacc_mbox_live${dev}"
        export HETGPU_PACC_SHARED_DDR_DEVICE="/dev/hetgpu_pacc_mbox_ddr_coh${dev}"
        export PACC_GEMM_M="$M"
        export PACC_GEMM_N="$N"
        export PACC_GEMM_K="$K"
        export PACC_GEMM_ITERS="$ITERS"
        export PACC_GEMM_WARMUP="$WARMUP"
        exec timeout --signal=TERM --kill-after=5 \
            "${PACC_SFMM_PROBE_TIMEOUT_S:-240}" "$PROBE"
    ) >"$log" 2>&1 &
    pids+=("$!")
    devs+=("$dev")
done

rc=0
for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    dev="${devs[$idx]}"
    if wait "$pid"; then
        echo "pacc${dev}: ok"
    else
        child_rc=$?
        echo "pacc${dev}: failed rc=${child_rc}; see ${LOG_DIR}/pacc${dev}.log" >&2
        rc=1
    fi
done
end_ns="$(date +%s%N)"

wall_s="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.9f", (end - start) / 1000000000.0 }')"
if ! python3 - "$LOG_DIR" "$wall_s" "$DEVICES" "$XSFMM_REPEATS" <<'PY'
import glob
import os
import re
import sys

log_dir = sys.argv[1]
wall_s = float(sys.argv[2])
expected_devices = {
    f"pacc{int(value, 0)}"
    for value in sys.argv[3].replace(",", " ").split()
    if value
}
repeats = int(sys.argv[4], 0)
if repeats < 1 or repeats > 4096:
    raise SystemExit(f"invalid Xsfmm repeat count: {repeats}")
line_re = re.compile(
    r"pacc_gemm_bf16_probe m=(\d+) n=(\d+) k=(\d+) .*?iters=(\d+) "
    r"gemm_s_avg=([0-9.eE+-]+).*?mismatches=(\d+)"
)

rows = []
seen_devices = set()
for path in sorted(glob.glob(os.path.join(log_dir, "pacc*.log"))):
    text = open(path, "r", errors="replace").read()
    m = line_re.search(text)
    dev = os.path.splitext(os.path.basename(path))[0]
    seen_devices.add(dev)
    if not m:
        rows.append((dev, None, None, None, None, None, None, "no-result"))
        continue
    mm, nn, kk, iters, avg_s, mismatches = m.groups()
    mm, nn, kk, iters = map(int, (mm, nn, kk, iters))
    avg_s = float(avg_s)
    mismatches = int(mismatches)
    ops = 2.0 * mm * nn * kk * repeats
    tops = ops / avg_s / 1.0e12 if avg_s > 0 else 0.0
    rows.append((dev, mm, nn, kk, iters, avg_s, tops, f"mismatches={mismatches}"))

print("")
print("summary:")
sum_tops = 0.0
sum_ops_total = 0.0
for dev, mm, nn, kk, iters, avg_s, tops, status in rows:
    if mm is None:
        print(f"  {dev}: {status}")
        continue
    sum_tops += tops
    sum_ops_total += 2.0 * mm * nn * kk * iters * repeats
    print(
        f"  {dev}: {mm}x{nn}x{kk} avg={avg_s*1e3:.3f} ms "
        f"batched_request_throughput={tops:.6f} TOPS-equivalent "
        f"xsfmm_repeats={repeats} {status}"
    )

wall_tops = sum_ops_total / wall_s / 1.0e12 if wall_s > 0 else 0.0
print(f"  aggregate_kernel_window={sum_tops:.6f} TOPS-equivalent")
print(f"  aggregate_end_to_end_wall={wall_tops:.6f} TOPS-equivalent wall_s={wall_s:.6f}")
print(f"  logs={log_dir}")
valid = (
    seen_devices == expected_devices and
    len(rows) == len(expected_devices) and
    all(row[1] is not None and row[7] == "mismatches=0" for row in rows)
)
if not valid:
    print("  acceptance=failed")
    raise SystemExit(1)
print("  acceptance=passed")
PY
then
    rc=1
fi

exit "$rc"
