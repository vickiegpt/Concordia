#!/usr/bin/env bash
set -euo pipefail

PACC_ROOT="${PACC_ROOT:-/home/ubuntu/Documents/hetGPU_pacc}"
LLAMA_ROOT="${LLAMA_ROOT:-/home/ubuntu/Documents/llama.cpp}"
LLAMA_BIN="${LLAMA_BIN:-${LLAMA_ROOT}/build-lanxin-nvidia/bin/llama-cli}"
MODEL="${MODEL:-/mnt/probe_nvme0n1p4/models/GLM-5.2-UD-IQ1_S/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf}"

PROMPT="${PROMPT:-你好，请用一句话介绍你自己。}"
GLM_TOKENS="${GLM_TOKENS:-2}"
GLM_CTX="${GLM_CTX:-64}"
GLM_GPU_LAYERS="${GLM_GPU_LAYERS:-4}"
GLM_KV_OFFLOAD="${GLM_KV_OFFLOAD:-1}"
GLM_TIMEOUT_S="${GLM_TIMEOUT_S:-240}"
SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-0.2}"

PACC_ENABLE="${PACC_ENABLE:-1}"
PACC_DEVICES="${PACC_DEVICES:-0,1,2,3}"
PACC_GEMM_M="${PACC_GEMM_M:-32}"
PACC_GEMM_N="${PACC_GEMM_N:-4}"
PACC_GEMM_K="${PACC_GEMM_K:-2048}"
PACC_GEMM_ITERS="${PACC_GEMM_ITERS:-32}"
PACC_GEMM_WARMUP="${PACC_GEMM_WARMUP:-0}"

LOG_ROOT="${LOG_ROOT:-/tmp/lanxin_disagg_eval}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/glm52_npu_gpu_disagg_${STAMP}}"
mkdir -p "$LOG_DIR"

die() {
    echo "error: $*" >&2
    exit 2
}

try_mount_model_backing_fs() {
    if [[ -r "$MODEL" ]]; then
        return 0
    fi
    if [[ -b /dev/nvme0n1p1 && -d /mnt/probe_nvme0n1p1 ]] && command -v sudo >/dev/null 2>&1; then
        sudo -n mount /dev/nvme0n1p1 /mnt/probe_nvme0n1p1 2>/dev/null || true
    fi
}

ensure_nvidia_nodes() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    if command -v sudo >/dev/null 2>&1; then
        if [[ ! -e /dev/nvidia0 ]]; then
            sudo -n mknod -m 666 /dev/nvidia0 c 195 0 2>/dev/null || true
        fi
        if [[ ! -e /dev/nvidiactl ]]; then
            sudo -n mknod -m 666 /dev/nvidiactl c 195 255 2>/dev/null || true
        fi
    fi
}

try_mount_model_backing_fs
ensure_nvidia_nodes

[[ -x "$LLAMA_BIN" ]] || die "missing llama binary: $LLAMA_BIN"
[[ -r "$MODEL" ]] || die "missing model split: $MODEL"
[[ -d "$PACC_ROOT" ]] || die "missing PACC repo: $PACC_ROOT"

GPU_SAMPLES="${LOG_DIR}/gpu_samples.csv"
CPU_SAMPLES="${LOG_DIR}/cpu_samples.tsv"
echo "ts,index,name,memory_used_mib,memory_free_mib,gpu_util_pct" > "$GPU_SAMPLES"
echo -e "ts\tpid\tcomm\tpcpu\tpmem\trss_kb\tstat\tetime\targs" > "$CPU_SAMPLES"

STOP_FILE="${LOG_DIR}/stop_sampling"
sample_loop() {
    while [[ ! -e "$STOP_FILE" ]]; do
        ts="$(date +%s.%N)"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
                --format=csv,noheader,nounits 2>/dev/null | awk -v ts="$ts" '{ print ts "," $0 }' >> "$GPU_SAMPLES" || true
        fi
        ps -eo pid=,comm=,pcpu=,pmem=,rss=,stat=,etime=,args= |
            awk -v ts="$ts" '($0 ~ /llama-cli/ || $0 ~ /pacc_gemm_bf16_probe/) && $0 !~ /awk/ {
                pid=$1; comm=$2; pcpu=$3; pmem=$4; rss=$5; stat=$6; etime=$7;
                sub(/^[[:space:]]*[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+/, "", $0);
                print ts "\t" pid "\t" comm "\t" pcpu "\t" pmem "\t" rss "\t" stat "\t" etime "\t" $0
            }' >> "$CPU_SAMPLES" || true
        sleep "$SAMPLE_INTERVAL_S"
    done
}

sample_loop &
SAMPLER_PID="$!"

PACC_PID=""
if [[ "$PACC_ENABLE" != "0" ]]; then
    (
        set +e
        cd "$PACC_ROOT" || exit 2
        export PACC_SFMM_LOG_DIR="${LOG_DIR}/pacc_sfmm"
        export PACC_SFMM_DEVICES="$PACC_DEVICES"
        export PACC_GEMM_M="$PACC_GEMM_M"
        export PACC_GEMM_N="$PACC_GEMM_N"
        export PACC_GEMM_K="$PACC_GEMM_K"
        export PACC_GEMM_ITERS="$PACC_GEMM_ITERS"
        export PACC_GEMM_WARMUP="$PACC_GEMM_WARMUP"
        ./tools/run_pacc_sfmm_4pacc_batch_example.sh
        echo "$?" > "${LOG_DIR}/pacc_sfmm.rc"
    ) > "${LOG_DIR}/pacc_sfmm.runner.out" 2>&1 &
    PACC_PID="$!"
fi

set +e
(
    cd "$LLAMA_ROOT" || exit 2
    GLM_KV_ARGS=()
    if [[ "$GLM_KV_OFFLOAD" == "0" ]]; then
        GLM_KV_ARGS+=(--no-kv-offload)
    fi
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/riscv64-linux-gnu:${LLAMA_ROOT}/build-lanxin-nvidia/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    unset LD_PRELOAD
    export HETGPU_PACC_IQ1S_HOOK="${HETGPU_PACC_IQ1S_HOOK:-0}"
    export HETGPU_PACC_IQ1S_CACHE_RESET="${HETGPU_PACC_IQ1S_CACHE_RESET:-1}"
    export HETGPU_PACC_IQ1S_WEIGHT_OFF="${HETGPU_PACC_IQ1S_WEIGHT_OFF:-0x01000000}"
    export HETGPU_PACC_IQ1S_SCRATCH_OFF="${HETGPU_PACC_IQ1S_SCRATCH_OFF:-0xf0000000}"
    export HETGPU_PACC_SHARED_DDR_BASE="${HETGPU_PACC_SHARED_DDR_BASE:-0x20100600000}"
    export HETGPU_PACC_SHARED_DDR_USER_OFF="${HETGPU_PACC_SHARED_DDR_USER_OFF:-0x100000}"
    export HETGPU_PACC_IQ1S_COH_DEV="${HETGPU_PACC_IQ1S_COH_DEV:-/dev/hetgpu_pacc_mbox_ddr_coh0}"
    /usr/bin/timeout "${GLM_TIMEOUT_S}s" "$LLAMA_BIN" \
        -m "$MODEL" \
        --gpu-layers "$GLM_GPU_LAYERS" --cpu-moe -c "$GLM_CTX" -n "$GLM_TOKENS" \
        -p "$PROMPT" \
        --no-warmup -st --no-display-prompt --split-mode layer --simple-io "${GLM_KV_ARGS[@]}"
) > "${LOG_DIR}/llama.out" 2> "${LOG_DIR}/llama.err"
LLAMA_RC="$?"
echo "$LLAMA_RC" > "${LOG_DIR}/llama.rc"

PACC_RC=0
if [[ -n "$PACC_PID" ]]; then
    wait "$PACC_PID"
    PACC_RC="$?"
fi
set -e

touch "$STOP_FILE"
wait "$SAMPLER_PID" 2>/dev/null || true

{
    echo "log_dir=${LOG_DIR}"
    echo "llama_rc=${LLAMA_RC}"
    echo "pacc_rc=${PACC_RC}"
    echo "model=${MODEL}"
    echo "glm_tokens=${GLM_TOKENS} glm_ctx=${GLM_CTX} glm_gpu_layers=${GLM_GPU_LAYERS} glm_kv_offload=${GLM_KV_OFFLOAD}"
    echo "pacc_shape=${PACC_GEMM_M}x${PACC_GEMM_N}x${PACC_GEMM_K} iters=${PACC_GEMM_ITERS} devices=${PACC_DEVICES}"
    echo ""
    echo "llama_timing:"
    grep -E "Prompt:|Generation:" "${LOG_DIR}/llama.out" || true
    echo ""
    echo "gpu_sample_summary:"
    awk -F, 'NR > 1 {
        gsub(/^[ ]+|[ ]+$/, "", $4); gsub(/^[ ]+|[ ]+$/, "", $5); gsub(/^[ ]+|[ ]+$/, "", $6);
        mem=$4+0; util=$6+0;
        if (mem > max_mem) max_mem=mem;
        if (util > max_util) max_util=util;
        sum_util += util; n += 1;
    } END {
        if (n > 0) printf "samples=%d max_mem_mib=%.0f max_gpu_util_pct=%.0f avg_gpu_util_pct=%.1f\n", n, max_mem, max_util, sum_util/n;
        else print "samples=0";
    }' "$GPU_SAMPLES"
    echo ""
    echo "cpu_sample_summary:"
    awk -F '\t' 'NR > 1 {
        if ($3 == "llama-cli") {
            llama_n += 1; llama_cpu += $4; if ($4 > llama_max_cpu) llama_max_cpu = $4; if ($6 > llama_max_rss) llama_max_rss = $6;
        } else if ($3 ~ /^pacc_gemm_bf16_/ || $9 ~ /pacc_gemm_bf16_probe/) {
            pacc_n += 1; pacc_cpu += $4; if ($4 > pacc_max_cpu) pacc_max_cpu = $4; if ($6 > pacc_max_rss) pacc_max_rss = $6;
        }
    } END {
        if (llama_n > 0) printf "llama-cli samples=%d avg_cpu_pct=%.1f max_cpu_pct=%.1f max_rss_mib=%.1f\n", llama_n, llama_cpu/llama_n, llama_max_cpu, llama_max_rss/1024.0;
        else print "llama-cli samples=0";
        if (pacc_n > 0) printf "pacc_gemm_bf16_probe samples=%d avg_cpu_pct=%.1f max_cpu_pct=%.1f max_rss_mib=%.1f\n", pacc_n, pacc_cpu/pacc_n, pacc_max_cpu, pacc_max_rss/1024.0;
        else print "pacc_gemm_bf16_probe samples=0";
    }' "$CPU_SAMPLES"
    echo ""
    echo "pacc_summary:"
    if [[ -d "${LOG_DIR}/pacc_sfmm" ]]; then
        grep -R -E "summary:|pacc[0-9]+:|aggregate_|mismatches|TOPS" "${LOG_DIR}/pacc_sfmm" "${LOG_DIR}/pacc_sfmm.runner.out" 2>/dev/null || true
    fi
    echo ""
    echo "llama_output_tail:"
    tail -n 80 "${LOG_DIR}/llama.out" || true
    echo ""
    echo "llama_error_key_lines:"
    grep -E "pacc-iq1s|HETGPU CPU PACC|CUDA error|error|fatal" "${LOG_DIR}/llama.err" || true
} | tee "${LOG_DIR}/summary.txt"

if [[ "$LLAMA_RC" != "0" || "$PACC_RC" != "0" ]]; then
    exit 1
fi
