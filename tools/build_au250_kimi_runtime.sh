#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

"${repo_root}/tools/au250_hybrid_run.sh" bash -lc '
set -euo pipefail
cmake -S /bitnet -B /work/target/au250-bitnet-cuda130 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build /work/target/au250-bitnet-cuda130 --target llama-cli llama-tokenize -j"$(nproc)"

export RUSTUP_HOME=/work/target/au250-runtime/rustup
export CARGO_HOME=/work/target/au250-runtime/cargo
export CARGO_TARGET_DIR=/work/target/au250-app215
export PATH=/work/target/au250-runtime/bin:${CARGO_HOME}/bin:${PATH}
install -d /work/target/au250-runtime/bin
test -x /work/target/au250-runtime/bin/ninja || install -m 0755 /opt/miniconda3/bin/ninja /work/target/au250-runtime/bin/ninja
test -x "${CARGO_HOME}/bin/cargo" || {
  curl --proto "=https" --tlsv1.2 -fsS https://sh.rustup.rs -o /work/target/au250-runtime/rustup-init.sh
  sh /work/target/au250-runtime/rustup-init.sh -y --profile minimal --default-toolchain 1.92.0
}
cargo build -p zluda --features nvidia --no-default-features

/work/target/au250-bitnet-cuda130/bin/llama-cli --version
ldd /work/target/au250-app215/debug/libnvcuda.so | grep -F libxrt_coreutil || true
nvidia-smi -L
xbutil examine -d 0000:64:00.1 -r platform
'
