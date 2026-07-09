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

exec "$runner" \
    -m "$model" \
    -t "$threads" \
    -c "$ctx_size" \
    -n "$predict" \
    --temp "$temp" \
    "${extra_args[@]}" \
    -p "$prompt"
