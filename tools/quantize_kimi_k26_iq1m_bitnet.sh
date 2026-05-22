#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    cat >&2 <<'USAGE'
Usage:
  tools/quantize_kimi_k26_iq1m_bitnet.sh INPUT_GGUF_OR_FIRST_SHARD IMATRIX_DAT OUTPUT_GGUF_OR_PREFIX [THREADS]

This uses the local BitNet-derived llama-quantize binary built at:
  /root/hetGPU/BitNet-work/build/bin/llama-quantize

Important:
  - This produces standard llama.cpp IQ1_M.
  - It does not produce Unsloth Dynamic UD-IQ1_M.
  - IQ1_M requires an importance matrix; the BitNet/llama.cpp quantizer refuses to run without one.

Example:
  tools/quantize_kimi_k26_iq1m_bitnet.sh \
    /models/Kimi-K2.6-Q4_0.gguf \
    /models/imatrix-Kimi-K2.6-Q4_X.dat \
    /models/Kimi-K2.6-IQ1_M.gguf \
    64
USAGE
    exit 2
fi

quantize_bin="/root/hetGPU/BitNet-work/build/bin/llama-quantize"
input_model="$1"
imatrix_file="$2"
output_model="$3"
threads="${4:-$(nproc)}"

if [[ ! -x "$quantize_bin" ]]; then
    echo "Missing quantizer: $quantize_bin" >&2
    echo "Build it with: cmake --build /root/hetGPU/BitNet-work/build --target llama-quantize -j" >&2
    exit 1
fi

if [[ ! -f "$input_model" ]]; then
    echo "Input GGUF not found: $input_model" >&2
    exit 1
fi

if [[ ! -f "$imatrix_file" ]]; then
    echo "Importance matrix not found: $imatrix_file" >&2
    exit 1
fi

exec "$quantize_bin" \
    --allow-requantize \
    --keep-split \
    --imatrix "$imatrix_file" \
    "$input_model" \
    "$output_model" \
    IQ1_M \
    "$threads"
