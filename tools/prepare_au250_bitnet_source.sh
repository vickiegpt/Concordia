#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
    echo "usage: $0 <pristine-bitnet-source> <overlay-destination> <patch>" >&2
    exit 2
}

source_root="$(realpath "$1")"
overlay_root="$2"
patch_file="$(realpath "$3")"
mmq_dir="3rdparty/llama.cpp/ggml/src/ggml-cuda"

[[ -f "${source_root}/${mmq_dir}/mmq.cu" ]] || {
    echo "missing BitNet MMQ source under ${source_root}" >&2
    exit 1
}
[[ -f "${source_root}/${mmq_dir}/mmq.cuh" ]] || {
    echo "missing BitNet MMQ header under ${source_root}" >&2
    exit 1
}
[[ -s "${patch_file}" ]] || {
    echo "missing AU250 no-stream-k patch ${patch_file}" >&2
    exit 1
}
[[ -n "${overlay_root}" && "${overlay_root}" != "/" ]] || {
    echo "refusing unsafe overlay destination ${overlay_root@Q}" >&2
    exit 1
}
install -d "${overlay_root}"
overlay_root="$(realpath "${overlay_root}")"
[[ "${overlay_root}" != "${source_root}" ]] || {
    echo "overlay destination must differ from pristine source" >&2
    exit 1
}

if grep -Fqx '    const bool use_stream_k = false;' "${overlay_root}/${mmq_dir}/mmq.cu" 2>/dev/null \
    && grep -Fqx '    const bool use_stream_k = args.use_stream_k;' "${overlay_root}/${mmq_dir}/mmq.cuh" 2>/dev/null; then
    exit 0
fi
if find "${overlay_root}" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to overwrite nonempty, unqualified overlay ${overlay_root}" >&2
    exit 1
fi

tar -C "${source_root}" \
    --exclude='./.git' \
    --exclude='./build' \
    --exclude='./build-*' \
    -cf - . | tar -C "${overlay_root}" -xf -
patch --batch --silent -p1 -d "${overlay_root}" <"${patch_file}"

grep -Fqx '    const bool use_stream_k = false;' "${overlay_root}/${mmq_dir}/mmq.cu"
grep -Fqx '    const bool use_stream_k = args.use_stream_k;' "${overlay_root}/${mmq_dir}/mmq.cuh"
