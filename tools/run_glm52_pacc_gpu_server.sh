#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="${BUILD_ROOT:-/mnt/probe_nvme0n1p4/models/.lanxin-build}"
LLAMA_BIN="${LLAMA_BIN:-/home/ubuntu/Documents/llama.cpp/build-lanxin-nvidia/bin/llama-server}"
MODEL="${MODEL:-/mnt/probe_nvme0n1p4/models/GLM-5.2-UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
PARALLEL="${PARALLEL:-2}"
CTX="${CTX:-1024}"
BATCH_SIZE="${BATCH_SIZE:-128}"
UBATCH_SIZE="${UBATCH_SIZE:-32}"
GPU_LAYERS="${GPU_LAYERS:-3}"
LOG_DIR="${LOG_DIR:-${B}/logs/glm52-pacc-gpu-server-$(date +%Y%m%d-%H%M%S)}"
PACC_MASK="${PACC_MASK:-0xf}"
XSFMM_FIRMWARE="${XSFMM_FIRMWARE:-lanxin/lx500_pacc_jobd_xsfmm_kernel_exec.bin}"
STABLE_FIRMWARE="${STABLE_FIRMWARE:-lanxin/lx500_pacc_jobd_hostbase_idmarker.bin}"
HOST_BOOT_ID_BEFORE="$(cat /proc/sys/kernel/random/boot_id)"
PACC_BOOTED=0

cleanup() {
    local rc=$?
    trap - EXIT
    if [[ "$PACC_BOOTED" == "1" && "${PACC_RESTORE:-1}" != "0" ]]; then
        HETGPU_PACC_RECOVER_FIRMWARE="$STABLE_FIRMWARE" \
        HETGPU_PACC_RECOVER_PATCH_PACC_ID_ENV=1 \
        HETGPU_PACC_RECOVER_PRE_RESET_MASK="$PACC_MASK" \
        HETGPU_PACC_RECOVER_PRE_RESET_SETTLE_S=1 \
        HETGPU_PACC_RECOVER_SETTLE_S="${PACC_RECOVER_SETTLE_S:-12}" \
            "$ROOT/ext/pacc_runtime-sys/tools/pacc_recover_no_reboot.sh" \
            "$PACC_MASK" || rc=1
    fi
    local host_boot_id_after
    host_boot_id_after="$(cat /proc/sys/kernel/random/boot_id)"
    if [[ "$host_boot_id_after" != "$HOST_BOOT_ID_BEFORE" ]]; then
        echo "main-host boot ID changed during PACC-only run" >&2
        rc=99
    fi
    exit "$rc"
}
trap cleanup EXIT

for dev in /dev/hetgpu_pacc_mbox_ddr_coh{0..3} /dev/hetgpu_pacc_mbox_live{0..3}; do
    [[ -e "$dev" ]] || { echo "missing PACC device: $dev" >&2; exit 2; }
done
RESERVE_PARAM=/sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_reserve_system_ram
BASE_PARAM=/sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_base_override
SIZE_PARAM=/sys/module/hetgpu_pacc_mbox_ddr_coh/parameters/shared_ddr_size
[[ -r "$RESERVE_PARAM" && "$(<"$RESERVE_PARAM")" == "Y" ]] || {
    echo "unsafe PACC shared DDR: System RAM range is not reserved" >&2
    echo "run: sudo $ROOT/tools/load_pacc_shared_ddr_reserved.sh" >&2
    exit 2
}
(( $(<"$BASE_PARAM") == 0x20110600000 )) || {
    echo "unexpected PACC shared DDR base: $(<"$BASE_PARAM")" >&2
    exit 2
}
(( $(<"$SIZE_PARAM") >= 0x100000000 )) || {
    echo "PACC shared DDR reservation is smaller than 4 GiB: $(<"$SIZE_PARAM")" >&2
    exit 2
}
[[ -x "$LLAMA_BIN" ]] || { echo "missing llama-server: $LLAMA_BIN" >&2; exit 2; }
[[ -r "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }
for shard in "${MODEL%-00001-of-00006.gguf}"-0000{1..6}-of-00006.gguf; do
    [[ -r "$shard" ]] || { echo "missing model shard: $shard" >&2; exit 2; }
done
mkdir -p "$LOG_DIR"

if [[ "${PACC_BOOT:-1}" != "0" ]]; then
    python3 - <<'PY'
import os
fd = os.open("/dev/hetgpu_pacc_mbox_ddr_coh0", os.O_RDWR | os.O_SYNC)
try:
    if os.pwrite(fd, b"\0" * 0x8000, 0x100000) != 0x8000:
        raise SystemExit("short shared-DDR control clear")
finally:
    os.close(fd)
PY
    PACC_BOOTED=1
    HETGPU_PACC_RECOVER_FIRMWARE="$XSFMM_FIRMWARE" \
    HETGPU_PACC_RECOVER_PATCH_PACC_ID_ENV=1 \
    HETGPU_PACC_RECOVER_PRE_RESET_MASK="$PACC_MASK" \
    HETGPU_PACC_RECOVER_PRE_RESET_SETTLE_S=1 \
    HETGPU_PACC_RECOVER_SETTLE_S="${PACC_RECOVER_SETTLE_S:-12}" \
        "$ROOT/ext/pacc_runtime-sys/tools/pacc_recover_no_reboot.sh" "$PACC_MASK"
    python3 - <<'PY'
import os
import struct
import time

fd = os.open("/dev/hetgpu_pacc_mbox_ddr_coh0", os.O_RDONLY | os.O_SYNC)
deadline = time.time() + 45
pending = set(range(4))
try:
    while pending and time.time() < deadline:
        for dev in list(pending):
            data = os.pread(fd, 32, 0x100000 + dev * 0x2000 + 0x1f40)
            magic, version, _job, phase, _detail, _seq = struct.unpack("<QIIIIQ", data)
            if magic == 0x4847505542434e31 and version == 1 and phase == 0x7002:
                pending.remove(dev)
        if pending:
            time.sleep(0.1)
finally:
    os.close(fd)
if pending:
    raise SystemExit(f"PACC ready timeout: {sorted(pending)}")
PY
fi

export TMPDIR="${TMPDIR:-${B}/tmp}"
export LD_LIBRARY_PATH="/home/ubuntu/Documents/llama.cpp/build-lanxin-nvidia/bin:${B}/llama-sm120/bin:/home/ubuntu/fake_cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/mnt/usb/hetgpu_build_target/releasefix/release/libpacc_runtime_sys.so${LD_PRELOAD:+ ${LD_PRELOAD}}"
export CUDA_VISIBLE_DEVICES=0
export GGML_CUDA_DISABLE_GRAPHS=1
export LANXIN_NVIDIA_CUDA_COMPLETION_TIMEOUT_MS="${LANXIN_NVIDIA_CUDA_COMPLETION_TIMEOUT_MS:-30000}"
export LANXIN_NVIDIA_CUBLAS_BATCH_GPU="${LANXIN_NVIDIA_CUBLAS_BATCH_GPU:-0}"

export HETGPU_PACC_IQ1S_HOOK=1
export HETGPU_PACC_IQ1S_XSFMM_BF16=1
export HETGPU_PACC_IQ1S_XSFMM_SPLIT3="${HETGPU_PACC_IQ1S_XSFMM_SPLIT3:-1}"
export HETGPU_PACC_IQ1S_ONLY=0
export HETGPU_PACC_Q8_0_NATIVE_MMVF=0
export HETGPU_LLAMA_PACC_Q8_0_DIRECT_MMVF=0
export HETGPU_PACC_IQ1S_FILTER_TYPE="${HETGPU_PACC_IQ1S_FILTER_TYPE:-8}"
export HETGPU_PACC_IQ1S_FILTER_M="${HETGPU_PACC_IQ1S_FILTER_M:-576}"
export HETGPU_PACC_IQ1S_FILTER_K="${HETGPU_PACC_IQ1S_FILTER_K:-6144}"
export HETGPU_PACC_IQ1S_CPU_FALLBACK=0
export HETGPU_PACC_IQ1S_CPU_THREADS="${HETGPU_PACC_IQ1S_CPU_THREADS:-32}"

export HETGPU_PACC_VISIBLE_DEVICES=0
export HETGPU_PACC_GEMM_DEVICES=0,1,2,3
export HETGPU_PACC_MBOX_DEVICE='/dev/hetgpu_pacc_mbox_ddr_coh{}'
export HETGPU_PACC_MAILBOX_DEVICE='/dev/hetgpu_pacc_mbox_live{}'
export HETGPU_PACC_MBOX_BACKEND=helper
export HETGPU_PACC_MBOX_IRQ=1
export HETGPU_PACC_ZLUDA_IRQ_MBOX_HELPER=1
export HETGPU_PACC_JOBD_BOOTSTRAP=0
export HETGPU_PACC_SHARED_DDR_BASE=0x20110600000
export HETGPU_PACC_SHARED_DDR_PACC_BASE=0x20110600000
export HETGPU_PACC_SHARED_DDR_USER_OFF=0x100000
export HETGPU_PACC_SHARED_DDR_BYTES=0x100000000
export HETGPU_PACC_IQ1S_COH_DEV=/dev/hetgpu_pacc_mbox_ddr_coh0
export HETGPU_PACC_IQ1S_WEIGHT_OFF=0x01000000
export HETGPU_PACC_IQ1S_SCRATCH_OFF=0xf0000000
export HETGPU_PACC_IQ1S_MAX_WEIGHT_BYTES=536870912
export HETGPU_PACC_IQ1S_MAX_TILE_M="${HETGPU_PACC_IQ1S_MAX_TILE_M:-32}"
export HETGPU_PACC_IQ1S_MAX_TILE_K="${HETGPU_PACC_IQ1S_MAX_TILE_K:-6144}"
export HETGPU_PACC_IQ1S_CONTIGUOUS_BATCH="${HETGPU_PACC_IQ1S_CONTIGUOUS_BATCH:-1}"
export HETGPU_PACC_IQ1S_WORKERS=4
export HETGPU_PACC_ALLOW_HOST_GEMM_FALLBACK=0
export HETGPU_PACC_GEMM_TRACE=0

echo "starting GLM-5.2 server at http://${HOST}:${PORT}"
echo "parallel=${PARALLEL} gpu_layers=${GPU_LAYERS} batch=${BATCH_SIZE} ubatch=${UBATCH_SIZE}"
echo "logs=${LOG_DIR}"

set +e
"$LLAMA_BIN" \
    -m "$MODEL" --gpu-layers "$GPU_LAYERS" \
    --host "$HOST" --port "$PORT" \
    --ctx-size "$CTX" --parallel "$PARALLEL" \
    --cont-batching --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE" \
    --checkpoint-min-step "${CHECKPOINT_MIN_STEP:-16}" \
    --cache-ram "${CACHE_RAM_MIB:-8192}" --kv-unified \
    2>&1 | tee "$LOG_DIR/server.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$LOG_DIR/server.rc"
exit "$rc"
