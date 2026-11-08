#!/usr/bin/env bash
set -euo pipefail
cd /srv/lanxin/qwen35-a3b
PREFIX=${1:?unique prefix}
MODE=${2:?cpu or pacc}
REFERENCE=${3:-}
[[ "$PREFIX" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$MODE" == cpu || "$MODE" == pacc ]] || exit 2
test ! -e "logs/$PREFIX.json"
export CONCORDIA_RELEASE=${CONCORDIA_RELEASE:-$PWD/runtime-ngl99-release}
source bin/pacc-env.sh
export HETGPU_SIFIVE_SHARED_DDR_BASE=0x2075d000000 HETGPU_SIFIVE_SHARED_DDR_BYTES=0x400000
export HETGPU_SIFIVE_JOBD_BOOTSTRAP=0 HETGPU_SIFIVE_JOB_TIMEOUT_MS=10000
export HETGPU_SIFIVE_ZLUDA_IRQ_TRACE=0 HETGPU_SIFIVE_LOG_MEMORY=0
export HETGPU_SIFIVE_GEMM_SHARED_SLOTS=1 HETGPU_SIFIVE_GEMM_DEVICES=0
export QWEN_TRACE_DIR="$PWD/logs/$PREFIX-tensors"
if [[ -n "$REFERENCE" ]]; then export QWEN_TRACE_REFERENCE="$PWD/logs/$REFERENCE-tensors"; fi
ngl=0
if [[ "$MODE" == pacc ]]; then ngl=99; fi
# Trace callbacks intentionally abort at the boundary. Never emit TPS from this mode.
exec timeout 1800 "${QWEN_TRACE_BIN:-bin/qwen-ngl-trace}" "$QWEN_MODEL" "$ngl" 2 "${QWEN_TRACE_PROMPT:-1}" "logs/$PREFIX.json" -
