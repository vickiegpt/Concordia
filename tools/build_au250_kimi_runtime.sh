#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

"${repo_root}/tools/au250_hybrid_run.sh" bash -lc '
set -euo pipefail

if [[ ! -x /usr/bin/g++-12 ]]; then
  dpkg -i \
    /work/target/au250-runtime/debs/libstdc++-12-dev_12.3.0-1ubuntu1~22.04.3_amd64.deb \
    /work/target/au250-runtime/debs/g++-12_12.3.0-1ubuntu1~22.04.3_amd64.deb
fi

compat_root=/work/target/au250-cuda13-compat/include
cuda_include=/usr/local/cuda-13.0/targets/x86_64-linux/include
compat_math=${compat_root}/crt/math_functions.h
install -d "${compat_root}"
cp -a "${cuda_include}/crt" "${compat_root}/"
patch --batch --silent "${compat_math}" /work/tools/cuda13-rsqrt-noexcept.patch
grep -Fqx "__func__(double rsqrt(double a) noexcept(true));" "${compat_math}"
grep -Fqx "__func__(float rsqrtf(float a) noexcept(true));" "${compat_math}"
export NVCC_PREPEND_FLAGS=-I/work/target/au250-cuda13-compat/include

bitnet_overlay=/work/target/au250-bitnet-source-au250-no-stream-k
bitnet_build=/work/target/au250-bitnet-cuda130-au250-no-stream-k
/work/tools/prepare_au250_bitnet_source.sh \
  /bitnet \
  "${bitnet_overlay}" \
  /work/tools/bitnet-disable-stream-k-for-au250.patch

cmake -S "${bitnet_overlay}" -B "${bitnet_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build "${bitnet_build}" --target llama-cli llama-tokenize -j"$(nproc)"

export RUSTUP_HOME=/work/target/au250-runtime/rustup
export CARGO_HOME=/work/target/au250-runtime/cargo
export CARGO_TARGET_DIR=/work/target/au250-app215
export PATH=/work/target/au250-runtime/bin:${CARGO_HOME}/bin:${PATH}
export HETGPU_CUDART_SYMBOL_VERSION=13
install -d /work/target/au250-runtime/bin
test -x /work/target/au250-runtime/bin/ninja || install -m 0755 /opt/miniconda3/bin/ninja /work/target/au250-runtime/bin/ninja
test -x "${CARGO_HOME}/bin/cargo" || {
  curl --proto "=https" --tlsv1.2 -fsS https://sh.rustup.rs -o /work/target/au250-runtime/rustup-init.sh
  sh /work/target/au250-runtime/rustup-init.sh -y --profile minimal --default-toolchain 1.92.0
}
cargo build -p zluda --features nvidia,embed_cudart --no-default-features

"${bitnet_build}/bin/llama-cli" --version
test -s /work/target/au250-app215/debug/libhetgpu_cuda_shim.so
ldd /work/target/au250-app215/debug/libnvcuda.so | grep -F libxrt_coreutil || true
nvidia-smi -L
xbutil examine -d 0000:64:00.1 -r platform
'
