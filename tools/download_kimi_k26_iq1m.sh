#!/usr/bin/env bash
set -euo pipefail

repo="bartowski/moonshotai_Kimi-K2.6-GGUF"
subdir="moonshotai_Kimi-K2.6-IQ1_M"
base_url="https://huggingface.co/${repo}/resolve/main/${subdir}"
out_dir="${1:-/root/hetGPU/models/${repo}/${subdir}}"

mkdir -p "$out_dir"

files=(
    "moonshotai_Kimi-K2.6-IQ1_M-00001-of-00006.gguf:39839065952"
    "moonshotai_Kimi-K2.6-IQ1_M-00002-of-00006.gguf:39228265248"
    "moonshotai_Kimi-K2.6-IQ1_M-00003-of-00006.gguf:39177210560"
    "moonshotai_Kimi-K2.6-IQ1_M-00004-of-00006.gguf:39195589440"
    "moonshotai_Kimi-K2.6-IQ1_M-00005-of-00006.gguf:39220925216"
    "moonshotai_Kimi-K2.6-IQ1_M-00006-of-00006.gguf:35764717344"
)

for entry in "${files[@]}"; do
    name="${entry%%:*}"
    expected="${entry##*:}"
    target="${out_dir}/${name}"
    current=0
    if [[ -f "$target" ]]; then
        current="$(stat -c%s "$target")"
    fi
    if [[ "$current" == "$expected" ]]; then
        echo "ok: ${name} (${expected} bytes)"
        continue
    fi

    echo "downloading: ${name}"
    curl \
        --fail \
        --location \
        --continue-at - \
        --retry 20 \
        --retry-all-errors \
        --connect-timeout 30 \
        --speed-time 120 \
        --speed-limit 1024 \
        --output "$target" \
        "${base_url}/${name}?download=true"

    current="$(stat -c%s "$target")"
    if [[ "$current" != "$expected" ]]; then
        echo "size mismatch for ${name}: got ${current}, expected ${expected}" >&2
        exit 1
    fi
done

echo "download complete: ${out_dir}"
