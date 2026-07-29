#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
tmatmul_root="${TMATMUL_REPO_ROOT:-/root/ternary_matmul}"
results_root="${NVINT4_RESULTS_ROOT:-${tmatmul_root}/results/nvint4_ptx_tmatmul}"
cargo_bin="${CARGO_BIN:-cargo}"
nvcc_bin="${NVCC_BIN:-/usr/local/cuda-12.8/bin/nvcc}"
checker_bin="${NVINT4_CHECKER_BIN:-${tmatmul_root}/synth/intel_ia780i/sw/check_nvint4_route_artifact.py}"

device="${HETGPU_CXL_TMATMUL_DEVICE:-/dev/cxl_tmatmul3b000}"
dax="${HETGPU_CXL_TMATMUL_DAX:-/dev/dax0.0}"
bdf="${HETGPU_CXL_TMATMUL_PCI_ADDR:-0000:3b:00.0}"
csr_base="${HETGPU_CXL_TMATMUL_CSR_BASE:-0x1c0000}"
mode="hardware"
probe_args=()

while (($#)); do
    case "$1" in
        --converter-only)
            mode="converter-only"
            probe_args+=("$1")
            shift
            ;;
        --hardware)
            mode="hardware"
            probe_args+=("$1")
            shift
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --dax)
            dax="$2"
            shift 2
            ;;
        --bdf)
            bdf="$2"
            shift 2
            ;;
        --csr-base)
            csr_base="$2"
            shift 2
            ;;
        --delta|--seed|--stream)
            probe_args+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if ((${#probe_args[@]} == 0)); then
    probe_args+=("--hardware")
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
run_dir="${results_root}/${run_id}"
mkdir -p "${run_dir}"
artifact="${run_dir}/${mode}.json"
route_log="${run_dir}/runtime-routes.jsonl"
probe_bin="${NVINT4_PROBE_BIN:-${run_dir}/nvint4_tmatmul_probe}"

(
    cd "${repo_root}"
    "${cargo_bin}" build -p zluda --no-default-features --features nvidia
)

"${nvcc_bin}" \
    -std=c++17 \
    -O2 \
    -lineinfo \
    -ccbin /usr/bin/g++-14 \
    -include "${script_dir}/cuda_glibc_compat.h" \
    -Xcompiler="-include,${script_dir}/cuda_glibc_compat.h" \
    "${script_dir}/probe.cu" \
    -ldl \
    -o "${probe_bin}"

export HETGPU_NVINT4_TMATMUL=1
unset HETGPU_NVINT4_GPU_FALLBACK
export HETGPU_NVINT4_ROUTE_LOG="${route_log}"
export HETGPU_CXL_TMATMUL_DEVICE="${device}"
export HETGPU_CXL_TMATMUL_DAX="${dax}"
export HETGPU_CXL_TMATMUL_PCI_ADDR="${bdf}"
export HETGPU_CXL_TMATMUL_CSR_BASE="${csr_base}"
export LD_PRELOAD="${repo_root}/target/debug/libnvcuda.so"

"${probe_bin}" "${probe_args[@]}" --artifact "${artifact}"

if [[ "${mode}" == "hardware" ]]; then
    python3 "${checker_bin}" "${artifact}"
fi

echo "artifact=${artifact}"
