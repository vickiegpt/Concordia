#!/usr/bin/env bash
set -euo pipefail

repo='nohurry/Qwen3.5-397B-A17B-TQ1_0-GGUF'
name='Qwen3.5-397B-A17B-UD-TQ1_0.gguf'
url="https://huggingface.co/${repo}/resolve/main/${name}?download=true"
expected_size=94155830880
expected_sha='0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568'
destination="${1:-/root/models/qwen35-tq1}"

if [[ "${HETGPU_MODEL_FETCH_TESTING:-0}" == 1 ]]; then
    expected_size="${HETGPU_TEST_MODEL_SIZE:?HETGPU_TEST_MODEL_SIZE is required in test mode}"
fi

[[ ! -e "${destination}" || -d "${destination}" ]] || {
    echo "model destination exists and is not a directory: ${destination}" >&2
    exit 1
}
mkdir -p "${destination}"

final="${destination}/${name}"
partial="${final}.partial"

verify_file() {
    local path="$1"
    [[ -f "${path}" ]] || return 1
    [[ "$(stat -c %s "${path}")" == "${expected_size}" ]] || return 1
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected_sha}" ]]
}

if verify_file "${final}"; then
    printf 'verified_model=%s\nverified_size=%s\nverified_sha256=%s\n' \
        "${final}" "${expected_size}" "${expected_sha}"
    exit 0
fi

available="$(df -PB1 "${destination}" | awk 'NR == 2 { print $4 }')"
[[ "${available}" =~ ^[0-9]+$ ]] || {
    echo "could not determine free bytes for ${destination}" >&2
    exit 1
}
current=0
if [[ -e "${partial}" ]]; then
    [[ -f "${partial}" ]] || {
        echo "partial model path is not a regular file: ${partial}" >&2
        exit 1
    }
    current="$(stat -c %s "${partial}")"
fi
remaining=$((expected_size - current))
(( remaining >= 0 && available >= remaining + 1073741824 )) || {
    echo "insufficient free bytes for verified Qwen model" >&2
    exit 1
}

curl --fail --location --retry 8 --retry-all-errors --continue-at - \
    --output "${partial}" "${url}"

[[ "$(stat -c %s "${partial}")" == "${expected_size}" ]] || {
    echo "Qwen model size mismatch" >&2
    exit 1
}
[[ "$(sha256sum "${partial}" | awk '{print $1}')" == "${expected_sha}" ]] || {
    echo "Qwen model SHA-256 mismatch" >&2
    exit 1
}

mv -f -- "${partial}" "${final}"
printf 'verified_model=%s\nverified_size=%s\nverified_sha256=%s\n' \
    "${final}" "${expected_size}" "${expected_sha}"
