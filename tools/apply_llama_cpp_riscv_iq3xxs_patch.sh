#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${1:-/home/ubuntu/Documents/llama.cpp}"
TARGET="${LLAMA_ROOT}/ggml/src/ggml-cpu/arch/riscv/quants.c"
PATCH="${ROOT}/patches/llama-cpp-riscv-iq3xxs-metadata.patch"
BASE_SHA="ff3c7ef1197106a99d7ef780973912b08f123a83c9080d1b518b5223e8375d30"
RESULT_SHA="d533c03e8e5cd683e63cb812b46b974fc21dacc6e5ff26dc3f538e3b072a8f2f"

if [[ ! -f "$TARGET" ]]; then
    echo "missing llama.cpp RISC-V quant source: $TARGET" >&2
    exit 2
fi

current_sha="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$current_sha" == "$RESULT_SHA" ]]; then
    echo "RISC-V IQ3_XXS patch already applied: $TARGET"
    exit 0
fi
if [[ "$current_sha" != "$BASE_SHA" ]]; then
    echo "unexpected RISC-V quant baseline: $current_sha" >&2
    echo "expected $BASE_SHA; apply the Q5_K and IQ1_S patches first" >&2
    exit 3
fi

patch -d "$LLAMA_ROOT" -p1 --forward --batch < "$PATCH"
current_sha="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$current_sha" != "$RESULT_SHA" ]]; then
    echo "patched source hash mismatch: $current_sha" >&2
    exit 4
fi
echo "applied RISC-V IQ3_XXS patch: $TARGET"
