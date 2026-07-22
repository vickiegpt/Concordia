#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${1:-/home/ubuntu/Documents/llama.cpp}"
TARGET="${LLAMA_ROOT}/ggml/src/ggml-cuda/ggml-cuda.cu"
PATCH="${ROOT}/patches/llama-cpp-ggml-cuda-pacc-peak.patch"
BASE_SHA="c54d685c6000dd7765203557f5451a15811c6e67a009dad93e422d4c18dc4f78"
RESULT_SHA="64a845244dc52575d5a41ad6c71c5521912cc9ad9c4114aefa5dea8388f2cf8b"

if [[ ! -f "${TARGET}" ]]; then
    echo "missing llama.cpp CUDA source: ${TARGET}" >&2
    exit 2
fi

current_sha="$(sha256sum "${TARGET}" | awk '{print $1}')"
if [[ "${current_sha}" == "${RESULT_SHA}" ]]; then
    echo "PACC peak patch already applied: ${TARGET}"
    exit 0
fi
if [[ "${current_sha}" != "${BASE_SHA}" ]]; then
    echo "unexpected ggml-cuda.cu baseline: ${current_sha}" >&2
    echo "expected ${BASE_SHA}; refusing a fuzzy source rewrite" >&2
    exit 3
fi

patch -d "${LLAMA_ROOT}" -p1 --forward --batch < "${PATCH}"
current_sha="$(sha256sum "${TARGET}" | awk '{print $1}')"
if [[ "${current_sha}" != "${RESULT_SHA}" ]]; then
    echo "patched source hash mismatch: ${current_sha}" >&2
    exit 4
fi
echo "applied PACC peak patch: ${TARGET}"
