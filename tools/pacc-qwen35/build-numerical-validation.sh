#!/usr/bin/env bash
set -euo pipefail
ROOT=${QWEN_ROOT:-/srv/lanxin/qwen35-a3b}
TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:?new output directory}
[[ ! -e "$OUT" ]]
mkdir "$OUT"
COMMON=(-O3 -std=c++17 -ffp-contract=off
  -I"$ROOT/seq512-memory-source/include" -I"$ROOT/llama.cpp/ggml/include"
  -I"$ROOT/llama.cpp/ggml/src" -I"$ROOT/llama.cpp/vendor/nlohmann"
  -L"$ROOT/build-seq512-memory/bin" -L"$ROOT/llama.cpp/build-pacc-cuda-gcc14/bin"
  -Wl,-rpath,"$ROOT/build-seq512-memory/bin"
  -Wl,-rpath,"$ROOT/llama.cpp/build-pacc-cuda-gcc14/bin" -Wl,--allow-shlib-undefined)
for NAME in qwen-q81-validation trace-multiseq head-quant-oracle; do
  g++-14 "${COMMON[@]}" "$TOOLS/$NAME.cpp" -lllama -lggml -lggml-base -o "$OUT/$NAME"
done
"${RUSTC:-/home/ubuntu/.cargo/bin/rustc}" --test \
  "$TOOLS/test-sifive-half.rs" -o "$OUT/test-sifive-half"
"$OUT/test-sifive-half"
