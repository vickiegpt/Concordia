#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${1:-/home/ubuntu/Documents/llama.cpp}"

apply_one() {
    local label="$1"
    local target="$2"
    local patch_file="$3"
    local base_sha="$4"
    local result_sha="$5"

    if [[ ! -f "${target}" ]]; then
        echo "missing llama.cpp ${label} source: ${target}" >&2
        exit 2
    fi

    local current_sha
    current_sha="$(sha256sum "${target}" | awk '{print $1}')"
    if [[ "${current_sha}" == "${result_sha}" ]]; then
        echo "PACC ${label} patch already applied: ${target}"
        return
    fi
    if [[ "${current_sha}" != "${base_sha}" ]]; then
        echo "unexpected ${label} baseline: ${current_sha}" >&2
        echo "expected ${base_sha}; refusing a fuzzy source rewrite" >&2
        exit 3
    fi

    patch -d "${LLAMA_ROOT}" -p1 --forward --batch < "${patch_file}"
    current_sha="$(sha256sum "${target}" | awk '{print $1}')"
    if [[ "${current_sha}" != "${result_sha}" ]]; then
        echo "patched ${label} source hash mismatch: ${current_sha}" >&2
        exit 4
    fi
    echo "applied PACC ${label} patch: ${target}"
}

apply_one \
    "CUDA peak" \
    "${LLAMA_ROOT}/ggml/src/ggml-cuda/ggml-cuda.cu" \
    "${ROOT}/patches/llama-cpp-ggml-cuda-pacc-peak.patch" \
    "c54d685c6000dd7765203557f5451a15811c6e67a009dad93e422d4c18dc4f78" \
    "9962eb18b843919ccf46e480c918ce4a10a3b5c7326b11dcbbe62474696ad996"

apply_one \
    "CPU MoE" \
    "${LLAMA_ROOT}/ggml/src/ggml-cpu/ggml-cpu.c" \
    "${ROOT}/patches/llama-cpp-ggml-cpu-pacc-moe.patch" \
    "480d1535782932d9e66ca0339071d7baff491bad875df7e5201cec671171385c" \
    "412fb86821038d6793314ddb5b683916a267954bb84c4093375801c7c42fd672"
