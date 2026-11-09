#!/usr/bin/env bash
set -euo pipefail
ROOT=${QWEN_ROOT:-/srv/lanxin/qwen35-a3b}
TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT=${1:?new harness output path}
[[ ! -e "$OUTPUT" ]]
g++-14 -O2 -std=c++17 "$TOOLS/sweep-qwen-multiseq.cpp" \
  -I"$ROOT/seq512-memory-source/include" \
  -I"$ROOT/llama.cpp/ggml/include" -I"$ROOT/llama.cpp/vendor/nlohmann" \
  -L"$ROOT/build-seq512-memory/bin" -L"$ROOT/llama.cpp/build-pacc-cuda-gcc14/bin" \
  -Wl,-rpath,"$ROOT/build-seq512-memory/bin" \
  -Wl,-rpath,"$ROOT/llama.cpp/build-pacc-cuda-gcc14/bin" \
  -Wl,--allow-shlib-undefined -lllama -lggml -lggml-base -o "$OUTPUT"
