#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

cases="$("${SCRIPT_DIR}/run.sh" --list-cases)"
grep -Fxq "int_add" <<<"${cases}"
grep -Fxq "pred_select" <<<"${cases}"
grep -Fxq "fma_bits" <<<"${cases}"
grep -Fxq "popc_shf_mix" <<<"${cases}"
grep -Fxq "global_offset_pair" <<<"${cases}"
grep -Fxq "shared_reverse" <<<"${cases}"
grep -Fxq "kimi_iq1m_unpack" <<<"${cases}"
grep -Fxq "kimi_rmsnorm_bits" <<<"${cases}"
grep -Fxq "kimi_swiglu_mix" <<<"${cases}"
grep -Fxq "kimi_rope_mix" <<<"${cases}"
grep -Fxq "kimi_attention_mask" <<<"${cases}"

work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-test.XXXXXX)"
trap 'rm -rf "${work_dir}"' EXIT

HETGPU_ROUNDTRIP_WORKDIR="${work_dir}" \
HETGPU_ROUNDTRIP_SM=120 \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null

csv="${work_dir}/bench.csv"
test -s "${csv}"
head -n 1 "${csv}" | grep -Fxq "case,sm,status,cubin_bytes,lifted_ptx_bytes,lift_diagnostics,load_cubin_us,load_ptx_us,kernel_cubin_us,kernel_ptx_us,total_us,message"
grep -Fq "int_add,sm_120,dry_run" "${csv}"
grep -Fq "popc_shf_mix,sm_120,dry_run" "${csv}"
grep -Fq "global_offset_pair,sm_120,dry_run" "${csv}"

custom_work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-custom-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}"' EXIT
HETGPU_ROUNDTRIP_WORKDIR="${custom_work_dir}" \
HETGPU_ROUNDTRIP_SM=90 \
HETGPU_ROUNDTRIP_CASES=pred_select \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null
custom_csv="${custom_work_dir}/bench.csv"
grep -Fq "pred_select,sm_90,dry_run" "${custom_csv}"
if grep -Fq "int_add,sm_90,dry_run" "${custom_csv}"; then
    echo "round-trip dry-run ignored HETGPU_ROUNDTRIP_CASES" >&2
    exit 1
fi

kimi_work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-kimi-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}"' EXIT
HETGPU_ROUNDTRIP_WORKDIR="${kimi_work_dir}" \
HETGPU_ROUNDTRIP_SM=120 \
HETGPU_ROUNDTRIP_CASES=kimi_iq1m_unpack,kimi_attention_mask \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null
kimi_csv="${kimi_work_dir}/bench.csv"
grep -Fq "kimi_iq1m_unpack,sm_120,dry_run" "${kimi_csv}"
grep -Fq "kimi_attention_mask,sm_120,dry_run" "${kimi_csv}"
if grep -Fq "kimi_rope_mix,sm_120,dry_run" "${kimi_csv}"; then
    echo "round-trip dry-run ignored Kimi HETGPU_ROUNDTRIP_CASES selection" >&2
    exit 1
fi

for kimi_case in \
    kimi_iq1m_unpack \
    kimi_rmsnorm_bits \
    kimi_swiglu_mix \
    kimi_rope_mix \
    kimi_attention_mask
do
    ptx_file="${SCRIPT_DIR}/ptx/${kimi_case}.ptx"
    test -s "${ptx_file}"
    grep -Fq ".visible .entry ${kimi_case}(" "${ptx_file}"
    grep -Fq ".param .u64 out" "${ptx_file}"
    grep -Fq ".param .u64 in" "${ptx_file}"
    grep -Fq ".param .u32 n" "${ptx_file}"
done

for scalar_case in \
    popc_shf_mix \
    global_offset_pair
do
    ptx_file="${SCRIPT_DIR}/ptx/${scalar_case}.ptx"
    test -s "${ptx_file}"
    grep -Fq ".visible .entry ${scalar_case}(" "${ptx_file}"
    grep -Fq ".param .u64 out" "${ptx_file}"
    grep -Fq ".param .u64 in" "${ptx_file}"
    grep -Fq ".param .u32 n" "${ptx_file}"
done

if rg -q "list-gpu-code" "${SCRIPT_DIR}/run.sh"; then
    echo "round-trip bench should validate PTX support through ptxas, not nvcc --list-gpu-code" >&2
    exit 1
fi

rg -Fq '.target ${sm}' "${SCRIPT_DIR}/run.sh"
rg -q 'rm -f "\$\{lifted\}"' "${SCRIPT_DIR}/run.sh"
rg -q 'wrote lifted PTX dump' "${SCRIPT_DIR}/run.sh"
rg -q 'HETGPU_SASS_LIFTER_CUOBJDUMP' "${SCRIPT_DIR}/run.sh"
rg -q -- '--n-gpu-layers "\$\{gpu_layers\}"' "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh"
rg -q 'KIMI_EXTRA_LLAMA_ARGS' "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh"
rg -q 'LLAMA_ARG_N_GPU_LAYERS="\$\{kimi_gpu_layers\}"' "${SCRIPT_DIR}/run_kimi_k26_e2e.sh"
rg -q 'HETGPU_KIMI_E2E_EXTRA_LLAMA_ARGS' "${SCRIPT_DIR}/run_kimi_k26_e2e.sh"
rg -q 'HETGPU_KIMI_E2E_CUDART_COMPUTE_CAPABILITY' "${SCRIPT_DIR}/run_kimi_k26_e2e.sh"
rg -q 'skipped_no_cuda_offload' "${SCRIPT_DIR}/run_kimi_k26_e2e.sh"
rg -q 'HETGPU_ROUNDTRIP_WORKDIR="\$\{WORK_DIR\}/ld_preload_roundtrip"' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'HETGPU_KIMI_E2E_WORKDIR="\$\{WORK_DIR\}/kimi_e2e"' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'HETGPU_ROUNDTRIP_KEEP=1' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'HETGPU_KIMI_E2E_KEEP=1' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'append_roundtrip_csv_summary' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'append_kimi_e2e_csv_summary' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'kimi_e2e_child_status' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'ld_preload_roundtrip_cases' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'hetgpu_driver_stream' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'CUstream driver_stream = hetgpu_driver_stream\(stream\);' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'HETGPU_CUDART_COMPUTE_CAPABILITY' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'hetgpu_cudart_compute_capability' "${REPO_ROOT}/zluda/src/cudart_shim.c"
if rg -q '\(CUstream\)stream,' "${REPO_ROOT}/zluda/src/cudart_shim.c"; then
    echo "cudart shim must not pass managed cudaStream_t wrappers directly to driver cuLaunchKernel" >&2
    exit 1
fi
rg -q 'hetgpu_resolve_pacc_submit_gemm_mmvf_small_n_fn' "${REPO_ROOT}/zluda/src/cublas_shim.c"
rg -q 'hetgpu_pacc_submit_gemm_mmvf_small_n_checked' "${REPO_ROOT}/zluda/src/cublas_shim.c"
if rg -q '^extern int hetgpu_pacc_submit_gemm' "${REPO_ROOT}/zluda/src/cublas_shim.c"; then
    echo "cublas shim must resolve optional PACC GEMM submit symbols lazily, not require externs at load time" >&2
    exit 1
fi
rg -q 'hetgpu_is_ggml_cuda_rms_norm_f32' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'lazy PTX resolved named-only candidate' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'float eps = \(args && args\[3\]\)' "${REPO_ROOT}/zluda/src/cudart_shim.c"
if rg -q 'args\[[67]\]' "${REPO_ROOT}/zluda/src/cudart_shim.c"; then
    echo "RMSNorm host fallback must not probe beyond the known four-argument ggml CUDA signature" >&2
    exit 1
fi
rg -q 'try_extract_nvidia_cubin_from_fatbin' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'selected fatbin CUBIN for NVIDIA module' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'copy_nvidia_module_image_for_lifter' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'binary module image is neither ELF CUBIN nor CUDA fatbin' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'CUDA fatbin did not contain a raw NVIDIA CUBIN' "${REPO_ROOT}/zluda/src/impl/module.rs"

proof_work_dir="$(mktemp -d /tmp/hetgpu-sass-proof-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}"' EXIT
HETGPU_SASS_PROOF_WORKDIR="${proof_work_dir}" \
    "${SCRIPT_DIR}/run_correctness_suite.sh" --dry-run >/dev/null
proof_csv="${proof_work_dir}/sass_lifter_correctness.csv"
head -n 1 "${proof_csv}" | grep -Fxq "step,status,elapsed_ms,message"
grep -Fq "rust_fuzzer,dry_run,0,planned" "${proof_csv}"
grep -Fq "roundtrip_harness,dry_run,0,planned" "${proof_csv}"
grep -Fq "ld_preload_roundtrip,dry_run,0,planned" "${proof_csv}"
grep -Fq "kimi_e2e,dry_run,0,planned" "${proof_csv}"

e2e_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}"' EXIT
HETGPU_KIMI_E2E_WORKDIR="${e2e_work_dir}" \
BITNET_LLAMA_CLI="${e2e_work_dir}/missing-llama-cli" \
MODEL_DIR="${e2e_work_dir}/missing-model" \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null
e2e_csv="${e2e_work_dir}/bench_kimi_k26_e2e.csv"
head -n 1 "${e2e_csv}" | grep -Fxq "case,status,total_ms,exit_code,stdout_bytes,stderr_bytes,lifter_markers,lifted_ptx_files,lifted_ptx_bytes,message"
grep -Fq "kimi_k26_iq1m,skipped_missing_runner" "${e2e_csv}"

fake_runner="${e2e_work_dir}/fake-llama-cli"
printf '#!/usr/bin/env bash\nprintf "fake kimi output\\n"\n' >"${fake_runner}"
chmod +x "${fake_runner}"
e2e_model_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-model-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}"' EXIT
HETGPU_KIMI_E2E_WORKDIR="${e2e_model_work_dir}" \
BITNET_LLAMA_CLI="${fake_runner}" \
MODEL_DIR="${e2e_model_work_dir}/missing-model" \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null
e2e_model_csv="${e2e_model_work_dir}/bench_kimi_k26_e2e.csv"
grep -Fq "kimi_k26_iq1m,skipped_missing_model" "${e2e_model_csv}"

e2e_comma_work_dir="$(mktemp -d "/tmp/hetgpu-kimi-e2e,comma-test.XXXXXX")"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}"' EXIT
HETGPU_KIMI_E2E_WORKDIR="${e2e_comma_work_dir}" \
BITNET_LLAMA_CLI="${e2e_comma_work_dir}/missing,llama-cli" \
MODEL_DIR="${e2e_comma_work_dir}/missing-model" \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null
e2e_comma_csv="${e2e_comma_work_dir}/bench_kimi_k26_e2e.csv"
awk -F, 'NF != 10 { exit 1 }' "${e2e_comma_csv}"

e2e_marker_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-marker-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}"' EXIT
fake_model_dir="${e2e_marker_work_dir}/model"
mkdir -p "${fake_model_dir}"
for shard in \
    moonshotai_Kimi-K2.6-IQ1_S-00001-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00002-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00003-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00004-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00005-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00006-of-00006.gguf
do
    : >"${fake_model_dir}/${shard}"
done
fake_marker_runner="${e2e_marker_work_dir}/fake-marker-llama-cli"
printf '#!/usr/bin/env bash\nprintf "fake kimi output\\n"\nprintf "[hetGPU SASS] lifted fake marker\\n" >&2\n' >"${fake_marker_runner}"
chmod +x "${fake_marker_runner}"
set +e
HETGPU_KIMI_E2E_WORKDIR="${e2e_marker_work_dir}" \
BITNET_LLAMA_CLI="${fake_marker_runner}" \
MODEL_DIR="${fake_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO=/bin/true \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null 2>&1
marker_status="$?"
set -e
if [[ "${marker_status}" == "0" ]]; then
    echo "Kimi e2e accepted lifter marker without a lifted PTX dump" >&2
    exit 1
fi
e2e_marker_csv="${e2e_marker_work_dir}/bench_kimi_k26_e2e.csv"
grep -Fq "kimi_k26_iq1m,missing_lifter_dump_marker" "${e2e_marker_csv}"

e2e_ptx_only_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-ptx-only-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_ptx_only_work_dir}"' EXIT
fake_ptx_only_runner="${e2e_ptx_only_work_dir}/fake-ptx-only-llama-cli"
printf '#!/usr/bin/env bash\nprintf "fake kimi output\\n"\nprintf "[NVIDIA Backend] Detected PTX source (123 bytes)\\n" >&2\n' >"${fake_ptx_only_runner}"
chmod +x "${fake_ptx_only_runner}"
HETGPU_KIMI_E2E_WORKDIR="${e2e_ptx_only_work_dir}" \
BITNET_LLAMA_CLI="${fake_ptx_only_runner}" \
MODEL_DIR="${fake_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO=/bin/true \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null 2>&1
e2e_ptx_only_csv="${e2e_ptx_only_work_dir}/bench_kimi_k26_e2e.csv"
grep -Fq "kimi_k26_iq1m,pass_ptx_only" "${e2e_ptx_only_csv}"

e2e_no_cuda_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-no-cuda-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_ptx_only_work_dir}" "${e2e_no_cuda_work_dir}"' EXIT
fake_no_cuda_runner="${e2e_no_cuda_work_dir}/fake-no-cuda-llama-cli"
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'printf "fake kimi output\n"'
    printf '%s\n' 'printf "ggml_cuda_init: failed to initialize CUDA: cudaErrorUnknown\n" >&2'
    printf '%s\n' 'printf "warning: not compiled with GPU offload support, --gpu-layers option will be ignored\n" >&2'
} >"${fake_no_cuda_runner}"
chmod +x "${fake_no_cuda_runner}"
HETGPU_KIMI_E2E_WORKDIR="${e2e_no_cuda_work_dir}" \
HETGPU_KIMI_E2E_ALLOW_FAILURES=1 \
BITNET_LLAMA_CLI="${fake_no_cuda_runner}" \
MODEL_DIR="${fake_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO=/bin/true \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null 2>&1
e2e_no_cuda_csv="${e2e_no_cuda_work_dir}/bench_kimi_k26_e2e.csv"
grep -Fq "kimi_k26_iq1m,skipped_no_cuda_offload" "${e2e_no_cuda_csv}"

e2e_preload_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-preload-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}"' EXIT
fake_preload_model_dir="${e2e_preload_work_dir}/model"
mkdir -p "${fake_preload_model_dir}"
for shard in \
    moonshotai_Kimi-K2.6-IQ1_S-00001-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00002-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00003-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00004-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00005-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00006-of-00006.gguf
do
    : >"${fake_preload_model_dir}/${shard}"
done
fake_preload_runner="${e2e_preload_work_dir}/fake-preload-llama-cli"
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'printf "fake kimi output\n"'
    printf '%s\n' 'printf "preload=%s\n" "${HETGPU_KIMI_E2E_EFFECTIVE_LD_PRELOAD:-}" >&2'
    printf '%s\n' 'printf "defer=%s\n" "${HETGPU_CUDART_DEFER_MODULE_LOAD:-}" >&2'
    printf '%s\n' 'printf "[hetGPU SASS] lifted fake marker\n" >&2'
    printf '%s\n' 'printf ".version 8.8\n" >"${HETGPU_SASS_LIFTER_DUMP:?}"'
} >"${fake_preload_runner}"
chmod +x "${fake_preload_runner}"
preload_probe="/lib/x86_64-linux-gnu/libc.so.6"
if [[ ! -f "${preload_probe}" ]]; then
    preload_probe="/usr/lib/x86_64-linux-gnu/libc.so.6"
fi
test -f "${preload_probe}"
HETGPU_KIMI_E2E_WORKDIR="${e2e_preload_work_dir}" \
HETGPU_KIMI_E2E_LD_PRELOAD="${preload_probe}" \
HETGPU_KIMI_E2E_CUDART_DEFER_MODULE_LOAD=1 \
BITNET_LLAMA_CLI="${fake_preload_runner}" \
MODEL_DIR="${fake_preload_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO=/bin/true \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null 2>&1
e2e_preload_csv="${e2e_preload_work_dir}/bench_kimi_k26_e2e.csv"
grep -Fq "kimi_k26_iq1m,pass" "${e2e_preload_csv}"
grep -Fq "preload=${preload_probe}" "${e2e_preload_work_dir}/logs/kimi.stderr"
grep -Fq "defer=1" "${e2e_preload_work_dir}/logs/kimi.stderr"

e2e_cudart_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-cudart-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}" "${e2e_cudart_work_dir}"' EXIT
fake_cudart_model_dir="${e2e_cudart_work_dir}/model"
mkdir -p "${fake_cudart_model_dir}"
for shard in \
    moonshotai_Kimi-K2.6-IQ1_S-00001-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00002-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00003-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00004-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00005-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00006-of-00006.gguf
do
    : >"${fake_cudart_model_dir}/${shard}"
done
fake_cudart_runner="${e2e_cudart_work_dir}/fake-cudart-llama-cli"
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'printf "fake kimi output\n"'
    printf '%s\n' 'printf "preload=%s\n" "${HETGPU_KIMI_E2E_EFFECTIVE_LD_PRELOAD:-}" >&2'
    printf '%s\n' 'printf "[hetGPU SASS] lifted fake marker\n" >&2'
    printf '%s\n' 'printf ".version 8.8\n" >"${HETGPU_SASS_LIFTER_DUMP:?}"'
} >"${fake_cudart_runner}"
chmod +x "${fake_cudart_runner}"
fake_cargo="${e2e_cudart_work_dir}/fake-cargo"
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'printf "%s\n" "$*" >"${HETGPU_FAKE_CARGO_ARGS:?}"'
} >"${fake_cargo}"
chmod +x "${fake_cargo}"
HETGPU_KIMI_E2E_WORKDIR="${e2e_cudart_work_dir}" \
HETGPU_KIMI_E2E_USE_CUDART_SHIM=1 \
HETGPU_FAKE_CARGO_ARGS="${e2e_cudart_work_dir}/cargo.args" \
BITNET_LLAMA_CLI="${fake_cudart_runner}" \
MODEL_DIR="${fake_cudart_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO="${fake_cargo}" \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null 2>&1
grep -Fq -- "--features nvidia,embed_cudart" "${e2e_cudart_work_dir}/cargo.args"
grep -Fq "libhetgpu_cuda_shim.so" "${e2e_cudart_work_dir}/logs/kimi.stderr"

bar_line="$(rg -n 'bar\.sync 0' "${SCRIPT_DIR}/ptx/shared_reverse.ptx" | cut -d: -f1)"
early_done_branch="$(
    (rg -n '@%p[0-9]+ bra DONE' "${SCRIPT_DIR}/ptx/shared_reverse.ptx" || true) \
        | cut -d: -f1 \
        | awk -v bar="${bar_line}" '$1 < bar { print; exit }'
)"
if [[ -n "${early_done_branch}" ]]; then
    echo "shared_reverse branches to DONE before bar.sync at line ${early_done_branch}" >&2
    exit 1
fi
