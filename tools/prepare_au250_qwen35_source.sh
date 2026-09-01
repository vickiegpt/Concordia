#!/usr/bin/env bash
set -euo pipefail

pinned_revision="925e1179947ea0c0ebfb0032df18af3a729822be"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch_file="${script_dir}/llama-qwen35-tq1-hetgpu.patch"
header_file="${script_dir}/qwen35-tq1-bridge.h"
marker_name=".hetgpu-qwen35-overlay-v1"

usage() {
    echo "usage: $0 <pristine-llama-source> <overlay-destination>" >&2
    exit 2
}

fail() {
    echo "prepare_au250_qwen35_source: $*" >&2
    exit 1
}

test "$#" -eq 2 || usage
source_input="$1"
destination_input="$2"
testing="${HETGPU_OVERLAY_TESTING:-0}"
test_revision="${HETGPU_TEST_LLAMA_REVISION:-}"

case "${testing}" in
    0|1) ;;
    *) fail "HETGPU_OVERLAY_TESTING must be exactly 1 when used" ;;
esac
if test "${testing}" != 1 && test -n "${test_revision}"; then
    fail "HETGPU_TEST_LLAMA_REVISION is forbidden outside guarded overlay tests"
fi
if test "${testing}" = 1; then
    test -n "${test_revision}" || fail "guarded overlay tests require HETGPU_TEST_LLAMA_REVISION"
    expected_revision="${test_revision}"
else
    expected_revision="${pinned_revision}"
fi

test -f "${patch_file}" || fail "missing overlay patch ${patch_file}"
test -f "${header_file}" || fail "missing bridge header ${header_file}"
test -d "${source_input}/.git" || fail "source is not a git checkout"
source_root="$(realpath -e -- "${source_input}")"
test "${source_root}" != / || fail "source may not be filesystem root"

destination_parent="$(dirname -- "${destination_input}")"
destination_name="$(basename -- "${destination_input}")"
mkdir -p -- "${destination_parent}"
destination_parent="$(realpath -e -- "${destination_parent}")"
destination_root="${destination_parent}/${destination_name}"
test "${destination_root}" != / || fail "destination may not be filesystem root"
test "${destination_root}" != "${source_root}" || fail "source and destination must differ"

source_revision="$(git -C "${source_root}" rev-parse HEAD)"
test "${source_revision}" = "${expected_revision}" || \
    fail "source revision ${source_revision} does not match required ${expected_revision}"
test -z "$(git -C "${source_root}" status --porcelain --untracked-files=normal)" || \
    fail "source checkout must be clean"

patch_sha256="$(sha256sum "${patch_file}" | awk '{print $1}')"
header_sha256="$(sha256sum "${header_file}" | awk '{print $1}')"
expected_marker="revision=${source_revision}
patch_sha256=${patch_sha256}
header_sha256=${header_sha256}"

verify_overlay() {
    local root="$1"
    grep -Fq 'hetgpu_tq1_register_tensor_v1' "${root}/src/llama-model-loader.cpp"
    grep -Fq 'hetgpu_iq1s_register_tensor_v1' "${root}/src/llama-model-loader.cpp"
    grep -Fq 'hetgpu_tq1_try_mul_mat_id_v1' "${root}/ggml/src/ggml-cuda/ggml-cuda.cu"
    grep -Fq 'hetgpu_iq1s_bind_device_v1' "${root}/ggml/src/ggml-cuda/ggml-cuda.cu"
    grep -Fq 'hetgpu_qwen35_cuda_buffer_max_size' "${root}/ggml/src/ggml-cuda/ggml-cuda.cu"
    grep -Fq 'const std::string & path() const' "${root}/src/llama-mmap.h"
    grep -Fq 'HETGPU_TQ1_ABI_VERSION' "${root}/tools/qwen35-tq1-bridge.h"
    grep -Fq 'HETGPU_IQ1S_ABI_VERSION' "${root}/tools/qwen35-tq1-bridge.h"
    test "$(cat "${root}/${marker_name}")" = "${expected_marker}"
}

if test -e "${destination_root}"; then
    test -d "${destination_root}" || fail "destination exists and is not a directory"
    if test -f "${destination_root}/${marker_name}" && \
       test "$(cat "${destination_root}/${marker_name}")" = "${expected_marker}"; then
        verify_overlay "${destination_root}" || fail "existing qualified overlay failed verification"
        echo "Qwen TQ1 overlay already prepared at ${destination_root}"
        exit 0
    fi
    if find "${destination_root}" -mindepth 1 -print -quit | grep -q .; then
        fail "destination is nonempty and is not the requested qualified overlay"
    fi
    rmdir -- "${destination_root}"
fi

staging="$(mktemp -d "${destination_parent}/.qwen35-overlay.XXXXXX")"
cleanup() {
    if test -d "${staging}"; then
        rm -rf -- "${staging}"
    fi
}
trap cleanup EXIT

tar -C "${source_root}" \
    --exclude='./.git' \
    --exclude='./build' \
    --exclude='./build-*' \
    --exclude='./.cache' \
    -cf - . | tar -C "${staging}" -xf -
mkdir -p -- "${staging}/tools"
cp -- "${header_file}" "${staging}/tools/qwen35-tq1-bridge.h"
patch --directory="${staging}" --batch --forward -p1 < "${patch_file}"
printf '%s\n' "${expected_marker}" > "${staging}/${marker_name}"
verify_overlay "${staging}" || fail "new overlay failed marker verification"

mv -- "${staging}" "${destination_root}"
trap - EXIT
echo "Prepared Qwen TQ1 llama.cpp overlay at ${destination_root}"
