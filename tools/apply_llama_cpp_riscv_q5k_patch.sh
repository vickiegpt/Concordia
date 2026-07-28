#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${1:-/home/ubuntu/Documents/llama.cpp}"
TARGET="${LLAMA_ROOT}/ggml/src/ggml-cpu/arch/riscv/quants.c"
PATCH="${ROOT}/patches/llama-cpp-riscv-q5k-dot.patch"
BASE_SHA="96820420ed8d0d49b64f064d4fed708e3112f1946ea260f33ff51976d9f04e7d"
RESULT_SHA="ab9a75fcc7166fcbea40a28c6e07f34cd1d802be30e15db164c24ca9f4e016de"

if [[ ! -f "$TARGET" ]]; then
    echo "missing llama.cpp RISC-V quant source: $TARGET" >&2
    exit 2
fi

current_sha="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$current_sha" == "$RESULT_SHA" ]]; then
    echo "RISC-V Q5_K patch already applied: $TARGET"
    exit 0
fi
if [[ "$current_sha" != "$BASE_SHA" ]]; then
    echo "unexpected RISC-V quant baseline: $current_sha" >&2
    echo "expected $BASE_SHA; refusing a fuzzy source rewrite" >&2
    exit 3
fi

patch -d "$LLAMA_ROOT" -p1 --forward --batch < "$PATCH"
current_sha="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$current_sha" != "$RESULT_SHA" ]]; then
    echo "patched source hash mismatch: $current_sha" >&2
    exit 4
fi
echo "applied RISC-V Q5_K patch: $TARGET"
