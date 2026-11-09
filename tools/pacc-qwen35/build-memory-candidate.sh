#!/usr/bin/env bash
set -euo pipefail
ROOT=/srv/lanxin/qwen35-a3b
REPO="$ROOT/Concordia-xm-release"
TOOLS="$REPO/tools/pacc-qwen35"
SRC=${1:?new source directory}
BUILD=${2:?new build directory}
RUNTIME=${3:?new runtime directory}
[[ ! -e "$SRC" && ! -e "$BUILD" && ! -e "$RUNTIME" ]]
cp -a "$ROOT/seq512-source" "$SRC"
git -C "$SRC" apply "$TOOLS/seq512-mmap-weights.patch"
cp "$TOOLS/pacc-mmap-weights.cpp" "$TOOLS/pacc-mmap-weights.h" "$SRC/src/"
cp "$TOOLS/seq512-CMakeLists.txt" "$SRC/CMakeLists.txt"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 \
  -DLLAMA_SOURCE="$SRC" -DGGML_SOURCE="$ROOT/llama.cpp/ggml" \
  -DGGML_BIN="$ROOT/llama.cpp/build-pacc-cuda-gcc14/bin" \
  -DPROBE_SOURCE="$TOOLS/qwen-memory-probe.cpp" \
  -DHARNESS_SOURCE="$TOOLS/seq512-qwen-multiseq.cpp" \
  -DJSON_INCLUDE="$ROOT/llama.cpp/vendor/nlohmann"
cmake --build "$BUILD" -j8
env PATH=/home/ubuntu/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
  LLVM_ZLUDA_PREBUILT=/usr/lib/llvm-20 CARGO_TARGET_DIR=/home/ubuntu/Concordia/target-pacc-compile \
  cargo build --manifest-path "$REPO/Cargo.toml" --locked -j4 --release -p zluda \
    --no-default-features --features sifive,embed_cudart
cp -a "$ROOT/runtime-xm-request-batch-v18" "$RUNTIME"
cp /home/ubuntu/Concordia/target-pacc-compile/release/libnvcuda.so "$RUNTIME/libnvcuda.so"
# The CMake capacity target uses the chosen probe source; give it an explicit name.
cp "$BUILD/bin/qwen-capacity-probe" "$BUILD/bin/qwen-memory-probe"
printf 'Candidate only; no driver, firmware, or default runtime was changed.\n'
