#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launcher="${repo_root}/tools/run_kimi_k26_iq1m_bitnet.sh"
runtime="${repo_root}/zluda/src/impl/iq1s_tmatmul.rs"

fixture="$(mktemp -d)"
trap 'rm -rf "${fixture}"' EXIT
prefix="fixture"
for shard_index in 1 2 3 4 5 6; do
    touch "${fixture}/$(printf '%s-%05d-of-00006.gguf' "${prefix}" "${shard_index}")"
done

trace="$({
    env -i \
        PATH="${PATH}" \
        MODEL_DIR="${fixture}" \
        MODEL_PREFIX="${prefix}" \
        BITNET_LLAMA_CLI=/bin/true \
        KIMI_TMATMUL_FPGA=1 \
        HETGPU_BITNET_DISAGG_STRICT=1 \
        bash -x "${launcher}"
} 2>&1)"

for expected in \
    'HETGPU_CXL_TMATMUL_STAGING=mmap' \
    'HETGPU_TMATMUL_MATRIX_STAGE=cuda_dax' \
    'HETGPU_TMATMUL_IO_STAGE=cuda_dax' \
    'HETGPU_CXL_TMATMUL_V3_MEMORY=dax' \
    'HETGPU_CXL_TMATMUL=1' \
    'HETGPU_TMATMUL_CXL=1' \
    'HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK=0' \
    'HETGPU_TMATMUL_NAMED_FALLBACK=0' \
    'HETGPU_NVINT4_GPU_FALLBACK=0'; do
    grep -q "export ${expected}" <<<"${trace}"
done

if grep -qE 'V3_MEMORY=.*ioctl|STAGING=.*ioctl|KIMI_TMATMUL_INLINE' "${launcher}"; then
    echo "launcher still exposes inline/ioctl staging" >&2
    exit 1
fi

if grep -qE 'open_without_dax|IoctlDaxAccess|CXL_TYPE2_MEM_IO' "${runtime}"; then
    echo "IQ1_S strict runtime still contains software memory transport" >&2
    exit 1
fi
