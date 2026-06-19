#!/usr/bin/env bash
set -euo pipefail

model_dir="${MODEL_DIR:-/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M}"
model_prefix="${MODEL_PREFIX:-$(basename "${model_dir}")}"
model="${MODEL:-${model_dir}/${model_prefix}-00001-of-00006.gguf}"
runner="${BITNET_LLAMA_CLI:-/root/hetGPU/BitNet-work/build/bin/llama-cli}"
threads="${THREADS:-$(nproc)}"
ctx_size="${CTX_SIZE:-4096}"
predict="${N_PREDICT:-64}"
temp="${TEMP:-0.6}"
system_prompt="${SYSTEM_PROMPT:-You are Kimi, an AI assistant created by Moonshot AI.}"
user_prompt="${1:-用一句中文说明你已经启动。}"

required=()
for shard_index in 1 2 3 4 5 6; do
    required+=("$(printf '%s-%05d-of-00006.gguf' "${model_prefix}" "${shard_index}")")
done

if [[ ! -x "$runner" ]]; then
    echo "missing BitNet llama-cli: ${runner}" >&2
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

exec "$runner" \
    -m "$model" \
    -t "$threads" \
    -c "$ctx_size" \
    -n "$predict" \
    --temp "$temp" \
    "${extra_args[@]}" \
    -p "$prompt"
