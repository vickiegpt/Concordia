#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

require_grep() {
  local pattern="$1"
  local file="$2"
  local label="$3"
  if ! grep -q "${pattern}" "${file}"; then
    echo "missing ${label}: expected pattern '${pattern}' in ${file}" >&2
    exit 1
  fi
}

require_grep "github.com/NVlabs/cuda-oxide" Cargo.toml "NVlabs cuda-oxide dependency"
require_grep "cuda-device" Cargo.toml "cuda-device dependency"
require_grep "cuda-core" Cargo.toml "cuda-core dependency"
require_grep "#\\[kernel\\]" src/main.rs "cuda-oxide kernel macro"
require_grep "#\\[cuda_module\\]" src/main.rs "cuda-oxide module macro"
if grep -q "DIFF_PTX\\|cuda_types\\|nvidia_runtime_sys" src/main.rs Cargo.toml; then
  echo "benchmark must use NVlabs/cuda-oxide instead of embedded PTX or local CUDA bindings" >&2
  exit 1
fi

if [[ "${CONCORDIA_BENCH_STATIC_ONLY:-0}" == "1" ]]; then
  echo "static smoke passed: benchmark is wired to NVlabs cuda-oxide"
  exit 0
fi

set +e
if command -v cargo-oxide >/dev/null 2>&1 || cargo oxide --help >/dev/null 2>&1; then
  output="$(CONCORDIA_BENCH_SELF_TEST=1 cargo oxide run --arch "${CUDA_OXIDE_ARCH:-sm_120}" 2>&1)"
else
  oxide_repo="${CUDA_OXIDE_REPO:-/tmp/cuda-oxide}"
  if [[ ! -f "${oxide_repo}/crates/cargo-oxide/Cargo.toml" ]]; then
    echo "cargo-oxide not found. Install it from https://github.com/NVlabs/cuda-oxide or set CUDA_OXIDE_REPO." >&2
    exit 127
  fi
  output="$(CONCORDIA_BENCH_SELF_TEST=1 cargo run --manifest-path "${oxide_repo}/crates/cargo-oxide/Cargo.toml" -- run --arch "${CUDA_OXIDE_ARCH:-sm_120}" 2>&1)"
fi
status=$?
set -e
printf '%s\n' "${output}"
if [[ ${status} -ne 0 ]]; then
  exit "${status}"
fi

grep -q "self-test passed" <<<"${output}"
grep -q "dirty_pages=2" <<<"${output}"
