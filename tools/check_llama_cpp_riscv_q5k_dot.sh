#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-/home/ubuntu/Documents/llama.cpp}"
BUILD_ROOT="${BUILD_ROOT:-${LLAMA_ROOT}/build-lanxin-nvidia}"
CC="${CC:-/usr/bin/clang-20}"
OUT="${TMPDIR:-/tmp}/check_llama_cpp_riscv_q5k_dot"

"$CC" -fuse-ld=lld -O2 \
    -march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d \
    -I"${LLAMA_ROOT}/ggml/include" \
    -I"${LLAMA_ROOT}/ggml/src" \
    -I"${LLAMA_ROOT}/ggml/src/ggml-cpu" \
    "${ROOT}/tools/check_llama_cpp_riscv_q5k_dot.c" \
    -L"${BUILD_ROOT}/bin" -Wl,-rpath,"${BUILD_ROOT}/bin" \
    -lggml-cpu -lggml-base -lm -o "$OUT"

"$OUT"
