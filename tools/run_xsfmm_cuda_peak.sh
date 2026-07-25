#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-/tmp/lanxin_disagg_eval}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/xsfmm_cuda_peak_$(date +%Y%m%d_%H%M%S)}"
MODEL_RUNNER="${XSFMM_CUDA_MODEL_RUNNER:-}"
BOOT_ID_BEFORE="$(cat /proc/sys/kernel/random/boot_id)"
HARDWARE_ACTIVE=0

mkdir -p "$LOG_DIR"

restore_stable_firmware() {
    python3 - "${PACC_SFMM_CLEAR_DDR_DEVICE:-/dev/hetgpu_pacc_mbox_ddr_coh0}" <<'PY' || true
import os
import sys

fd = os.open(sys.argv[1], os.O_RDWR)
try:
    os.pwrite(fd, b"\0" * 0x8000, 0x100000)
finally:
    os.close(fd)
PY
    HETGPU_PACC_RECOVER_FIRMWARE="${PACC_SFMM_STABLE_FIRMWARE:-lanxin/lx500_pacc_jobd_hostbase_idmarker.bin}" \
    HETGPU_PACC_RECOVER_PATCH_PACC_ID_ENV=1 \
    HETGPU_PACC_RECOVER_PRE_RESET_MASK=0xf \
    HETGPU_PACC_RECOVER_PRE_RESET_SETTLE_S=1 \
    HETGPU_PACC_RECOVER_SETTLE_S="${PACC_SFMM_RECOVER_SETTLE_S:-15}" \
        "$ROOT/ext/pacc_runtime-sys/tools/pacc_recover_no_reboot.sh" 0xf || true
}

cleanup() {
    local rc=$?
    trap - EXIT
    if [[ "$rc" != "0" && "$HARDWARE_ACTIVE" == "1" ]]; then
        echo "restoring stable PACC firmware after failed peak run" >&2
        restore_stable_firmware
    fi
    local boot_id_after
    boot_id_after="$(cat /proc/sys/kernel/random/boot_id)"
    {
        echo "host_boot_id_before=${BOOT_ID_BEFORE}"
        echo "host_boot_id_after=${boot_id_after}"
    } | tee "${LOG_DIR}/host_boot_id.txt"
    if [[ "$boot_id_after" != "$BOOT_ID_BEFORE" ]]; then
        rc=99
    fi
    exit "$rc"
}
trap cleanup EXIT

run_gate() {
    local name=$1
    local boot=$2
    local m=$3
    local n=$4
    local k=$5
    local gate_dir="${LOG_DIR}/${name}"

    PACC_SFMM_DEVICES=0,1,2,3 \
    PACC_SFMM_RECOVER_MASK=0xf \
    PACC_SFMM_BOOT="$boot" \
    PACC_SFMM_RESTORE_ON_FAILURE=1 \
    PACC_SFMM_LOG_DIR="$gate_dir" \
    PACC_GEMM_M="$m" \
    PACC_GEMM_N="$n" \
    PACC_GEMM_K="$k" \
    PACC_GEMM_WARMUP=0 \
    PACC_GEMM_ITERS=1 \
    HETGPU_PACC_JOB_TIMEOUT_MS="${XSFMM_GATE_TIMEOUT_MS:-5000}" \
        "$ROOT/tools/run_pacc_sfmm_4pacc_batch_example.sh"
}

echo "log_dir=${LOG_DIR}"
echo "gate=4xPACC XSFMM v0.6.6 hardware-only"
if ! run_gate gate_4x4x4 1 4 4 4 >"${LOG_DIR}/gate_4x4x4.console" 2>&1; then
    echo "xsfmm_cuda_acceptance=blocked reason=4x4x4_hardware_gate_failed"
    exit 20
fi
HARDWARE_ACTIVE=1

if ! run_gate gate_32x32x32 0 32 32 32 >"${LOG_DIR}/gate_32x32x32.console" 2>&1; then
    echo "xsfmm_cuda_acceptance=blocked reason=32x32x32_hardware_gate_failed"
    exit 21
fi

if [[ -z "$MODEL_RUNNER" || ! -x "$MODEL_RUNNER" ]]; then
    echo "xsfmm_cuda_acceptance=blocked reason=missing_integrated_model_runner" >&2
    echo "set XSFMM_CUDA_MODEL_RUNNER to an executable that emits the required result line" >&2
    exit 22
fi

unset HETGPU_LLAMA_FAST_FAKE_DECODE
unset HETGPU_PACC_DELIVERY_SKIP_GEMM
unset HETGPU_PACC_SKIP_JOB_WAIT
export HETGPU_PACC_ALLOW_HOST_GEMM_FALLBACK=0
export HETGPU_PACC_IQ1S_CPU_FALLBACK=0

set +e
XSFMM_CUDA_LOG_DIR="${LOG_DIR}/model" "$MODEL_RUNNER" \
    >"${LOG_DIR}/model.out" 2>"${LOG_DIR}/model.err"
model_rc=$?
set -e
if [[ "$model_rc" != "0" ]]; then
    echo "xsfmm_cuda_acceptance=failed reason=model_runner_rc_${model_rc}"
    exit 23
fi

python3 - "${LOG_DIR}/model.out" <<'PY'
import re
import sys

text = open(sys.argv[1], "r", errors="replace").read()
match = re.search(
    r"^xsfmm_cuda_result xsfmm_calls=(\d+) generated_tokens=(\d+) "
    r"generation_s=([0-9.eE+-]+) generation_tps=([0-9.eE+-]+) "
    r"mismatches=(\d+)$",
    text,
    re.MULTILINE,
)
if match is None:
    raise SystemExit("missing canonical xsfmm_cuda_result line")
calls, tokens, seconds, tps, mismatches = match.groups()
calls = int(calls)
tokens = int(tokens)
seconds = float(seconds)
tps = float(tps)
mismatches = int(mismatches)
if calls <= 0 or tokens <= 0 or seconds <= 0 or tps <= 0 or mismatches != 0:
    raise SystemExit("invalid integrated XSFMM+CUDA result")
print(match.group(0))
print("xsfmm_cuda_acceptance=passed")
PY

HARDWARE_ACTIVE=0
