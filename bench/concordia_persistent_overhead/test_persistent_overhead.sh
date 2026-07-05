#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${root}/../.." && pwd)"
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
require_grep "worker_blocks,theoretical_sm_pct" src/main.rs "CSV schema"

if grep -q "DIFF_PTX\\|extern \"C\" __global__\\|nvcc" src/main.rs Cargo.toml; then
  echo "persistent overhead benchmark must use NVlabs/cuda-oxide, not embedded PTX/CUDA C" >&2
  exit 1
fi

workdir="${CONCORDIA_PERSISTENT_OVERHEAD_TEST_WORKDIR:-/tmp/concordia-persistent-overhead-test}"
mkdir -p "${workdir}"

python3 "${root}/plot_persistent_overhead.py" \
  --overhead-csv "${root}/fixtures/persistent_overhead_sample.csv" \
  --delta-log "${repo_root}/bench/concordia_persistent_overhead/fixtures/delta_sample.log" \
  --out-dir "${workdir}" \
  --formats pdf,png

test -s "${workdir}/concordia_persistent_overhead_ablation.pdf"
test -s "${workdir}/concordia_artifact_delta_rerun.pdf"

if [[ "${CONCORDIA_PERSISTENT_OVERHEAD_STATIC_ONLY:-0}" == "1" ]]; then
  echo "static smoke passed: persistent overhead benchmark is wired to NVlabs cuda-oxide"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "no visible NVIDIA GPU; live persistent-overhead run skipped" >&2
  exit 0
fi

set +e
oxide_toolchain="${CONCORDIA_PERSISTENT_OVERHEAD_RUST_TOOLCHAIN:-nightly-2026-04-03}"
if command -v cargo-oxide >/dev/null 2>&1 || cargo oxide --help >/dev/null 2>&1; then
  output="$(RUSTUP_TOOLCHAIN="${oxide_toolchain}" CONCORDIA_PERSISTENT_OVERHEAD_SELF_TEST=1 cargo oxide run --arch "${CUDA_OXIDE_ARCH:-sm_120}" 2>&1)"
else
  oxide_repo="${CUDA_OXIDE_REPO:-/tmp/cuda-oxide}"
  if [[ ! -f "${oxide_repo}/crates/cargo-oxide/Cargo.toml" ]]; then
    discovered="$(find "${HOME}/.cargo/git/checkouts" -path '*/crates/cargo-oxide/Cargo.toml' -print -quit 2>/dev/null || true)"
    if [[ -n "${discovered}" ]]; then
      oxide_repo="$(cd "$(dirname "${discovered}")/../.." && pwd)"
    fi
  fi
  if [[ ! -f "${oxide_repo}/crates/cargo-oxide/Cargo.toml" ]]; then
    echo "cargo-oxide not found. Install it from https://github.com/NVlabs/cuda-oxide or set CUDA_OXIDE_REPO." >&2
    exit 127
  fi
  output="$(RUSTUP_TOOLCHAIN="${oxide_toolchain}" CONCORDIA_PERSISTENT_OVERHEAD_SELF_TEST=1 cargo run --manifest-path "${oxide_repo}/crates/cargo-oxide/Cargo.toml" -- run --arch "${CUDA_OXIDE_ARCH:-sm_120}" 2>&1)"
fi
status=$?
set -e
printf '%s\n' "${output}"
if [[ ${status} -ne 0 ]]; then
  exit "${status}"
fi

grep -q "self-test passed" <<<"${output}"
grep -q "worker_blocks" <<<"${output}"
