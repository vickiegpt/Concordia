#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
bitnet_root="${AU250_BITNET_ROOT:-/home/eabban/BitNet}"
model_root="${AU250_KIMI_MODEL_ROOT:-/root/models/kimi-k2.6-iq1_s/moonshotai_Kimi-K2.6-IQ1_S}"
cuda_root="${AU250_CUDA_ROOT:-/usr/local/cuda-13.0}"
source /au250_xrt/env.sh >/dev/null

# The stock helper builds its flag string from an intentionally uninitialized
# local, so call it with nounset disabled in a command-substitution subshell.
_au250_hybrid_devflags() {
    set +u
    _au250_devflags
}

if [[ "${1:-}" == "--print-docker" ]]; then
    shift
    printf 'docker run --rm --gpus all --privileged %s -v /sys:/sys -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro -v /au250_xrt:/au250_xrt:ro -v %q:/work -v %q:/bitnet:ro -v %q:/models/kimi:ro -v %q:/usr/local/cuda-13.0:ro app215 %q\n' \
        "$(_au250_hybrid_devflags)" "${repo_root}" "${bitnet_root}" "${model_root}" "${cuda_root}" "$*"
    exit 0
fi

[[ $# -gt 0 ]] || { echo "usage: $0 <command...>" >&2; exit 2; }
temperature="$(_au250_fpga_temp)"
[[ -z "${temperature}" || "${temperature}" -lt "${AU250_TEMP_LIMIT:-85}" ]] || {
    echo "AU250 temperature ${temperature}C exceeds guard" >&2
    exit 1
}

docker run --rm --gpus all --privileged $(_au250_hybrid_devflags) \
    -v /sys:/sys \
    -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
    -v /au250_xrt:/au250_xrt:ro \
    -v "${repo_root}":/work -w /work \
    -v "${bitnet_root}":/bitnet:ro \
    -v "${model_root}":/models/kimi:ro \
    -v "${cuda_root}":/usr/local/cuda-13.0:ro \
    app215 bash -lc 'source /XRT/build/Release/opt/xilinx/xrt/setup.sh >/dev/null 2>&1; export PATH=/usr/local/cuda-13.0/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}; exec "$@"' _ "$@"
