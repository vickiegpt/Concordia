#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
work_root="$(mktemp -d)"
trap 'rm -rf "${work_root}"' EXIT
source_root="${work_root}/source"
overlay_root="${work_root}/overlay"
mmq_dir="3rdparty/llama.cpp/ggml/src/ggml-cuda"
install -d "${source_root}/${mmq_dir}"

cat >"${source_root}/${mmq_dir}/mmq.cu" <<'EOF'
    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = compute_capability >= CC_VOLTA && compute_capability < CC_OFFSET_AMD && src1_ncols == ne11;
    const mmq_args args = {src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff, stride00, src1_padded_row_size, src1_ncols, ne11, nrows_dst, use_stream_k};
EOF

cat >"${source_root}/${mmq_dir}/mmq.cuh" <<'EOF'
    const int mmq_x_max = get_mmq_x_max_host(cc);
    const int mmq_y = get_mmq_y_host(cc);
    const int block_num_y = (args.ne01 + mmq_y - 1) / mmq_y;
    const bool use_stream_k = cc >= CC_VOLTA && cc < CC_OFFSET_AMD;

    int mmq_x_best  = 0;
EOF

"${repo_root}/tools/prepare_au250_bitnet_source.sh" \
    "${source_root}" \
    "${overlay_root}" \
    "${repo_root}/tools/bitnet-disable-stream-k-for-au250.patch"

grep -Fqx '    const bool use_stream_k = false;' "${overlay_root}/${mmq_dir}/mmq.cu"
grep -Fqx '    const bool use_stream_k = args.use_stream_k;' "${overlay_root}/${mmq_dir}/mmq.cuh"
grep -Fq 'compute_capability >= CC_VOLTA' "${source_root}/${mmq_dir}/mmq.cu"
grep -Fq 'cc >= CC_VOLTA' "${source_root}/${mmq_dir}/mmq.cuh"

# Re-preparing the same qualified overlay must be idempotent.
"${repo_root}/tools/prepare_au250_bitnet_source.sh" \
    "${source_root}" \
    "${overlay_root}" \
    "${repo_root}/tools/bitnet-disable-stream-k-for-au250.patch"
grep -Fqx '    const bool use_stream_k = false;' "${overlay_root}/${mmq_dir}/mmq.cu"
