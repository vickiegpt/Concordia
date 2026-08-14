#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
tmatmul_root="${TMATMUL_REPO_ROOT:-/root/ternary_matmul}"
results_root="${NVINT4_RESULTS_ROOT:-${tmatmul_root}/results/nvint4_ptx_tmatmul}"
cargo_bin="${CARGO_BIN:-cargo}"
nvcc_bin="${NVCC_BIN:-/usr/local/cuda-12.8/bin/nvcc}"
checker_bin="${NVINT4_CHECKER_BIN:-${tmatmul_root}/synth/intel_ia780i/sw/check_nvint4_route_artifact.py}"

device="${HETGPU_CXL_TMATMUL_DEVICE:-/dev/cxl_tmatmul3b001}"
dax="${HETGPU_CXL_TMATMUL_DAX:-/dev/dax0.0}"
bdf="${HETGPU_CXL_TMATMUL_PCI_ADDR:-0000:3b:00.0}"
csr_base="${HETGPU_CXL_TMATMUL_CSR_BASE:-0x1c0000}"
staging="${HETGPU_CXL_TMATMUL_STAGING:-mmap}"
numa_node="${HETGPU_TMATMUL_NUMA_NODE:-1}"
numa_hpa_base="${HETGPU_TMATMUL_NUMA_HPA_BASE:-0xc0f00000000}"
numa_hpa_size="${HETGPU_TMATMUL_NUMA_HPA_SIZE:-0x100000000}"
numa_max_dpa="${HETGPU_TMATMUL_NUMA_MAX_DPA:-0x80000000}"
numa_scan_pages="${HETGPU_TMATMUL_NUMA_SCAN_PAGES:-64}"
mode="hardware"
build=1
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
        --no-build)
            build=0
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
        --numa-node)
            staging="numa_memcpy"
            numa_node="$2"
            shift 2
            ;;
        --numa-max-dpa)
            staging="numa_memcpy"
            numa_max_dpa="$2"
            shift 2
            ;;
        --numa-scan-pages)
            staging="numa_memcpy"
            numa_scan_pages="$2"
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
driver_shim="${repo_root}/target/debug/libnvcuda.so"
phase_log="${run_dir}/phases.log"

phase() {
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"${phase_log}"
    sync -d "${phase_log}"
}

phase "runner_start build=${build} mode=${mode}"
if ((build)); then
    phase "cargo_build_start"
    (
        cd "${repo_root}"
        CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}" \
            "${cargo_bin}" build -p zluda --no-default-features --features nvidia
    )
    phase "cargo_build_done"

    phase "nvcc_build_start"
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
    phase "nvcc_build_done"
else
    if [[ -z "${NVINT4_PROBE_BIN:-}" ]]; then
        echo "--no-build requires NVINT4_PROBE_BIN to name an existing probe" >&2
        exit 2
    fi
fi

[[ -x "${probe_bin}" ]] || {
    echo "probe is missing or not executable: ${probe_bin}" >&2
    exit 2
}
[[ -s "${driver_shim}" ]] || {
    echo "ZLUDA driver shim is missing or empty: ${driver_shim}" >&2
    exit 2
}

export HETGPU_NVINT4_TMATMUL=1
unset HETGPU_NVINT4_GPU_FALLBACK
export HETGPU_NVINT4_ROUTE_LOG="${route_log}"
export HETGPU_CXL_TMATMUL_DEVICE="${device}"
export HETGPU_CXL_TMATMUL_DAX="${dax}"
export HETGPU_CXL_TMATMUL_PCI_ADDR="${bdf}"
export HETGPU_CXL_TMATMUL_CSR_BASE="${csr_base}"
export HETGPU_CXL_TMATMUL_STAGING="${staging}"
export HETGPU_TMATMUL_NUMA_NODE="${numa_node}"
export HETGPU_TMATMUL_NUMA_HPA_BASE="${numa_hpa_base}"
export HETGPU_TMATMUL_NUMA_HPA_SIZE="${numa_hpa_size}"
export HETGPU_TMATMUL_NUMA_MAX_DPA="${numa_max_dpa}"
export HETGPU_TMATMUL_NUMA_SCAN_PAGES="${numa_scan_pages}"
export HETGPU_TMATMUL_PHASE_LOG="${phase_log}"
ln -s "${driver_shim}" "${run_dir}/libcuda.so.1"
export LD_LIBRARY_PATH="${run_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_PRELOAD="${driver_shim}"

phase "probe_start"
"${probe_bin}" "${probe_args[@]}" --artifact "${artifact}"
phase "probe_done"

if [[ "${mode}" == "hardware" ]]; then
    phase "checker_start"
    python3 "${checker_bin}" "${artifact}"
    phase "checker_done"
fi

echo "artifact=${artifact}"
