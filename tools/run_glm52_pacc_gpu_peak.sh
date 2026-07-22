#!/usr/bin/env bash
set -euo pipefail

B="${BUILD_ROOT:-/mnt/probe_nvme0n1p4/models/.lanxin-build}"
LLAMA_BIN="${LLAMA_BIN:-/home/ubuntu/Documents/llama.cpp/build-lanxin-nvidia/bin/llama-cli}"
MODEL="${MODEL:-/mnt/probe_nvme0n1p4/models/GLM-5.2-UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf}"
PROMPT="${PROMPT:-你好}"
TOKENS="${TOKENS:-8}"
CTX="${CTX:-64}"
LOG_DIR="${LOG_DIR:-${B}/logs/glm52-pacc-gpu-peak-$(date +%Y%m%d-%H%M%S)}"

for dev in /dev/hetgpu_pacc_mbox_ddr_coh{0..3} /dev/hetgpu_pacc_mbox_live{0..3}; do
    [[ -e "$dev" ]] || { echo "missing PACC device: $dev" >&2; exit 2; }
done
[[ -x "$LLAMA_BIN" ]] || { echo "missing llama-cli: $LLAMA_BIN" >&2; exit 2; }
[[ -r "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }
mkdir -p "$LOG_DIR"

export TMPDIR="${TMPDIR:-${B}/tmp}"
export LD_LIBRARY_PATH="${B}/llama-sm120/bin:/home/ubuntu/Documents/llama.cpp/build-lanxin-nvidia/bin:/home/ubuntu/fake_cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="/mnt/usb/hetgpu_build_target/releasefix/release/libpacc_runtime_sys.so${LD_PRELOAD:+ ${LD_PRELOAD}}"
export CUDA_VISIBLE_DEVICES=0

export HETGPU_PACC_IQ1S_HOOK=1
export HETGPU_PACC_IQ1S_ONLY=0
export HETGPU_PACC_Q8_0_NATIVE_MMVF=0
export HETGPU_LLAMA_PACC_Q8_0_DIRECT_MMVF=0
export HETGPU_PACC_IQ1S_FILTER_TYPE=8
export HETGPU_PACC_IQ1S_FILTER_M=576
export HETGPU_PACC_IQ1S_FILTER_K=6144
export HETGPU_PACC_IQ1S_CPU_FALLBACK=1
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
export HETGPU_PACC_IQ1S_MAX_TILE_M=144
export HETGPU_PACC_IQ1S_MAX_TILE_K=6144
export HETGPU_PACC_IQ1S_WORKERS=4
export HETGPU_PACC_ALLOW_HOST_GEMM_FALLBACK=0
export HETGPU_PACC_GEMM_TRACE=0

set +e
printf '%s\n/exit\n' "$PROMPT" | timeout "${TIMEOUT_S:-600}" "$LLAMA_BIN" \
    -m "$MODEL" --gpu-layers 3 -n "$TOKENS" -c "$CTX" \
    --seed 1 --temp 0 --single-turn 2>&1 | tee "$LOG_DIR/run.log"
rc=${PIPESTATUS[1]}
set -e
printf '%s\n' "$rc" > "$LOG_DIR/run.rc"

pacc_calls=$(grep -c '\[pacc-iq1s\] call' "$LOG_DIR/run.log" || true)
errors=$(grep -Eci 'CUDA error|timeout|submit rc|failed|mismatch' "$LOG_DIR/run.log" || true)
timing=$(grep -o '\[ Prompt:.*\]' "$LOG_DIR/run.log" | tail -1 || true)
printf 'rc=%s pacc_calls=%s errors=%s\n%s\nlogs=%s\n' "$rc" "$pacc_calls" "$errors" "$timing" "$LOG_DIR"

[[ "$rc" -eq 0 && "$pacc_calls" -gt 0 && "$errors" -eq 0 && -n "$timing" ]]
