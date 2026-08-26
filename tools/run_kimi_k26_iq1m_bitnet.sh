#!/usr/bin/env bash
set -euo pipefail

default_model_dir="/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M"
for candidate in \
    "/root/models/kimi-k2.6-iq1_s/moonshotai_Kimi-K2.6-IQ1_S" \
    "/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M"; do
    if [[ -f "${candidate}/$(basename "${candidate}")-00001-of-00006.gguf" ]]; then
        default_model_dir="${candidate}"
        break
    fi
done
model_dir="${MODEL_DIR:-${default_model_dir}}"
model_prefix="${MODEL_PREFIX:-$(basename "${model_dir}")}"
model="${MODEL:-${model_dir}/${model_prefix}-00001-of-00006.gguf}"
default_runner="/root/hetGPU/BitNet-work/build/bin/llama-cli"
for candidate in \
    "/home/victoryang00/DX100/benchmarks/llama.cpp/build/bin/llama-cli" \
    "/home/victoryang00/hetGPU_new/CXLMemSim/workloads/llama.cpp/main" \
    "/home/victoryang00/hetGPU_new/CXLMemSim/workloads/llama.cpp/llama-cli" \
    "/root/hetGPU/BitNet-work/build/bin/llama-cli"; do
    if [[ -x "${candidate}" ]]; then
        default_runner="${candidate}"
        break
    fi
done
runner="${BITNET_LLAMA_CLI:-${default_runner}}"
threads="${THREADS:-$(nproc)}"
ctx_size="${CTX_SIZE:-4096}"
predict="${N_PREDICT:-64}"
temp="${TEMP:-0.6}"
gpu_layers="${LLAMA_ARG_N_GPU_LAYERS:-${N_GPU_LAYERS:-}}"
system_prompt="${SYSTEM_PROMPT:-You are Kimi, an AI assistant created by Moonshot AI.}"
user_prompt="${1:-用一句中文说明你已经启动。}"

truthy() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

required=()
for shard_index in 1 2 3 4 5 6; do
    required+=("$(printf '%s-%05d-of-00006.gguf' "${model_prefix}" "${shard_index}")")
done

if [[ ! -x "$runner" ]]; then
    echo "missing Kimi llama runner: ${runner}" >&2
    exit 1
fi

for shard in "${required[@]}"; do
    if [[ ! -f "${model_dir}/${shard}" ]]; then
        echo "missing shard: ${model_dir}/${shard}" >&2
        exit 1
    fi
done

prompt="[BOS]<|im_system|>system<|im_middle|>${system_prompt}<|im_end|><|im_user|>user<|im_middle|>${user_prompt}<|im_end|><|im_assistant|>assistant<|im_middle|>"

extra_args=()
if [[ "${NO_WARMUP:-0}" == "1" ]]; then
    extra_args+=(--no-warmup)
fi
if [[ -n "${gpu_layers}" ]]; then
    extra_args+=(--n-gpu-layers "${gpu_layers}")
fi
kimi_extra_llama_args="${KIMI_EXTRA_LLAMA_ARGS---no-display-prompt}"
if [[ -n "${kimi_extra_llama_args}" ]]; then
    read -r -a kimi_extra_args <<<"${kimi_extra_llama_args}"
    extra_args+=("${kimi_extra_args[@]}")
fi

bitlinear_tmatmul="${KIMI_BITLINEAR_TMATMUL:-${HETGPU_KIMI_BITLINEAR_TMATMUL:-0}}"
if truthy "${bitlinear_tmatmul}"; then
    export HETGPU_NVINT4_TMATMUL="${HETGPU_NVINT4_TMATMUL:-1}"
    export HETGPU_NVINT4_BITLINEAR_HOOK="${HETGPU_NVINT4_BITLINEAR_HOOK:-1}"
    if ! truthy "${KIMI_BITLINEAR_TMATMUL_STRICT:-${HETGPU_KIMI_BITLINEAR_TMATMUL_STRICT:-0}}"; then
        export HETGPU_NVINT4_GPU_FALLBACK="${HETGPU_NVINT4_GPU_FALLBACK:-1}"
    fi
    export HETGPU_NVINT4_ROUTE_LOG="${HETGPU_NVINT4_ROUTE_LOG:-${KIMI_BITLINEAR_TMATMUL_ROUTE_LOG:-/tmp/kimi-bitlinear-nvint4-route.jsonl}}"
fi

cocotb_tmatmul="${KIMI_TMATMUL_COCOTB:-${HETGPU_KIMI_TMATMUL_COCOTB:-0}}"
if truthy "${cocotb_tmatmul}"; then
    export HETGPU_TMATMUL_COCOTB="${HETGPU_TMATMUL_COCOTB:-1}"
    export HETGPU_TMATMUL_ASM_DIR="${HETGPU_TMATMUL_ASM_DIR:-/tmp/tmatmul-asm}"
    export HETGPU_CXL_TMATMUL_STAGING="${HETGPU_CXL_TMATMUL_STAGING:-mmap}"
    export HETGPU_TMATMUL_MATRIX_STAGE="${HETGPU_TMATMUL_MATRIX_STAGE:-host}"
    export HETGPU_TMATMUL_IO_STAGE="${HETGPU_TMATMUL_IO_STAGE:-host}"
    export HETGPU_TMATMUL_OUTPUT_DTYPE="${HETGPU_TMATMUL_OUTPUT_DTYPE:-f32}"
    export HETGPU_BITNET_DISAGGREGATE="${HETGPU_BITNET_DISAGGREGATE:-1}"
    export HETGPU_TMATMUL_BITNET_DISAGGREGATE="${HETGPU_TMATMUL_BITNET_DISAGGREGATE:-1}"
    export HETGPU_BITNET_FFN_CXL="${HETGPU_BITNET_FFN_CXL:-1}"
    export HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK="${HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK:-1}"
    export HETGPU_TMATMUL_NAMED_FALLBACK="${HETGPU_TMATMUL_NAMED_FALLBACK:-1}"
    export HETGPU_TMATMUL_HARDWARE_MATMUL="${HETGPU_TMATMUL_HARDWARE_MATMUL:-1}"
    if ! truthy "${KIMI_TMATMUL_COCOTB_ALLOW_CXL:-${HETGPU_KIMI_TMATMUL_COCOTB_ALLOW_CXL:-0}}"; then
        export HETGPU_CXL_TMATMUL=0
        export HETGPU_TMATMUL_CXL=0
    fi
    export HETGPU_BITNET_ROUTE_LOG="${HETGPU_BITNET_ROUTE_LOG:-${KIMI_TMATMUL_ROUTE_LOG:-/tmp/kimi-bitnet-disagg-routes.jsonl}}"
fi

fpga_tmatmul="${KIMI_TMATMUL_FPGA:-${HETGPU_KIMI_TMATMUL_FPGA:-0}}"
if truthy "${fpga_tmatmul}"; then
    export HETGPU_TMATMUL_ASM_DIR="${HETGPU_TMATMUL_ASM_DIR:-/tmp/tmatmul-asm}"
    inline_tmatmul="${KIMI_TMATMUL_INLINE:-${HETGPU_KIMI_TMATMUL_INLINE:-0}}"
    if truthy "${inline_tmatmul}"; then
        export HETGPU_CXL_TMATMUL_STAGING="${HETGPU_CXL_TMATMUL_STAGING:-ioctl}"
        export HETGPU_TMATMUL_MATRIX_STAGE="${HETGPU_TMATMUL_MATRIX_STAGE:-cuda_host}"
        export HETGPU_TMATMUL_IO_STAGE="${HETGPU_TMATMUL_IO_STAGE:-cuda_host}"
        export HETGPU_CXL_TMATMUL_V3_MEMORY="${HETGPU_CXL_TMATMUL_V3_MEMORY:-ioctl}"
    else
        export HETGPU_CXL_TMATMUL_STAGING="${HETGPU_CXL_TMATMUL_STAGING:-mmap}"
        export HETGPU_TMATMUL_MATRIX_STAGE="${HETGPU_TMATMUL_MATRIX_STAGE:-cuda_dax}"
        export HETGPU_TMATMUL_IO_STAGE="${HETGPU_TMATMUL_IO_STAGE:-cuda_dax}"
    fi
    export HETGPU_TMATMUL_OUTPUT_DTYPE="${HETGPU_TMATMUL_OUTPUT_DTYPE:-f32}"
    export HETGPU_BITNET_DISAGGREGATE="${HETGPU_BITNET_DISAGGREGATE:-1}"
    export HETGPU_TMATMUL_BITNET_DISAGGREGATE="${HETGPU_TMATMUL_BITNET_DISAGGREGATE:-1}"
    export HETGPU_BITNET_FFN_CXL="${HETGPU_BITNET_FFN_CXL:-1}"
    export HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK="${HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK:-1}"
    export HETGPU_TMATMUL_NAMED_FALLBACK="${HETGPU_TMATMUL_NAMED_FALLBACK:-1}"
    export HETGPU_TMATMUL_HARDWARE_MATMUL="${HETGPU_TMATMUL_HARDWARE_MATMUL:-1}"
    export HETGPU_CXL_TMATMUL="${HETGPU_CXL_TMATMUL:-1}"
    export HETGPU_TMATMUL_CXL="${HETGPU_TMATMUL_CXL:-1}"
    export HETGPU_BITNET_ROUTE_LOG="${HETGPU_BITNET_ROUTE_LOG:-${KIMI_TMATMUL_ROUTE_LOG:-/tmp/kimi-bitnet-disagg-routes.jsonl}}"
fi

exec "$runner" \
    -m "$model" \
    -t "$threads" \
    -c "$ctx_size" \
    -n "$predict" \
    --temp "$temp" \
    "${extra_args[@]}" \
    -p "$prompt"
