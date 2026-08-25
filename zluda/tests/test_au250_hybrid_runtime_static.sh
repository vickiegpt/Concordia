#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/tools/au250_hybrid_run.sh"
builder="${repo_root}/tools/build_au250_kimi_runtime.sh"

bash -n "${runner}"
bash -n "${builder}"
rendered="$(${runner} --print-docker true)"
grep -Fq -- '--gpus all' <<<"${rendered}"
grep -Fq -- '-v /au250_xrt:/au250_xrt:ro' <<<"${rendered}"
grep -Fq -- '/usr/local/cuda-13.0:/usr/local/cuda-13.0:ro' <<<"${rendered}"
grep -Fq -- '/home/eabban/BitNet:/bitnet:ro' <<<"${rendered}"
grep -Fq -- '/root/models/kimi-k2.6-iq1_s/moonshotai_Kimi-K2.6-IQ1_S:/models/kimi:ro' <<<"${rendered}"
grep -Fq -- 'app215' <<<"${rendered}"
grep -Fq -- 'CMAKE_CUDA_ARCHITECTURES=120' "${builder}"
grep -Fq -- 'GGML_CUDA=ON' "${builder}"
