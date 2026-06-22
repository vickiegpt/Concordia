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

diagnostic_log="${work_dir}/sass-diagnostics.stderr"
cat >"${diagnostic_log}" <<'LOG'
[hetGPU SASS] diagnostic addr=Some(128) opcode=TEX instruction lifting is not implemented inst=/*0080*/ TEX R4, R5, R6, 0x2 ; /* 0x4000000000600504 */
[hetGPU SASS] diagnostic addr=Some(144) opcode=TEX instruction lifting is not implemented inst=/*0090*/ TEX R7, R8, R9, 0x2 ; /* 0x4000000000900807 */
[hetGPU SASS] diagnostic addr=Some(160) opcode=JMX instruction lifting is not implemented
LOG
diagnostic_csv="${work_dir}/sass-diagnostic-buckets.csv"
"${SCRIPT_DIR}/bucket_sass_diagnostics.sh" --output "${diagnostic_csv}" "${diagnostic_log}" >/dev/null
head -n 1 "${diagnostic_csv}" | grep -Fxq "opcode,message,count,sample_instruction"
grep -Fq '"TEX","instruction lifting is not implemented",2,"/*0080*/ TEX R4, R5, R6, 0x2 ; /* 0x4000000000600504 */"' "${diagnostic_csv}"
grep -Fq '"JMX","instruction lifting is not implemented",1,""' "${diagnostic_csv}"

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
rg -q 'append_kimi_numerical_csv_summary' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'kimi_e2e_child_status' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'kimi_numerical_child_status' "${SCRIPT_DIR}/run_correctness_suite.sh"
rg -q 'ld_preload_roundtrip_cases' "${SCRIPT_DIR}/run_correctness_suite.sh"
test -s "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh"
rg -q 'KIMI_NUMERICAL_DETERMINISTIC_ARGS' "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh"
rg -q 'HETGPU_KIMI_NUMERICAL_RUN_ROLE' "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh"
rg -q 'output_mismatch' "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh"
rg -q 'missing_ptx_capture' "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh"
rg -q 'hetgpu_driver_stream' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'CUstream driver_stream = hetgpu_driver_stream\(stream\);' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'HETGPU_CUDART_COMPUTE_CAPABILITY' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'hetgpu_cudart_compute_capability' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'HETGPU_CUDART_PREFER_FATBIN_CUBIN_FOR_SASS' "${REPO_ROOT}/zluda/src/cudart_shim.c"
if rg -q '\(CUstream\)stream,' "${REPO_ROOT}/zluda/src/cudart_shim.c"; then
    echo "cudart shim must not pass managed cudaStream_t wrappers directly to driver cuLaunchKernel" >&2
    exit 1
fi
rg -q 'hetgpu_resolve_pacc_submit_gemm_mmvf_small_n_fn' "${REPO_ROOT}/zluda/src/cublas_shim.c"
rg -q 'hetgpu_pacc_submit_gemm_mmvf_small_n_checked' "${REPO_ROOT}/zluda/src/cublas_shim.c"
rg -q 'HETGPU_CUBLAS_FORWARD_REAL' "${REPO_ROOT}/zluda/src/cublas_shim.c"
rg -q 'hetgpu_cublas_real_forward_enabled' "${REPO_ROOT}/zluda/src/cublas_shim.c"
rg -q 'hetgpu_cublas_driver_stream' "${REPO_ROOT}/zluda/src/cublas_shim.c"
rg -q 'real_cublasSetStream_v2' "${REPO_ROOT}/zluda/src/cublas_shim.c"
rg -q 'HETGPU_SHIM_ENABLE_REAL_CUBLAS_BY_DEFAULT' "${REPO_ROOT}/zluda/build.rs"
if rg -q '^extern int hetgpu_pacc_submit_gemm' "${REPO_ROOT}/zluda/src/cublas_shim.c"; then
    echo "cublas shim must resolve optional PACC GEMM submit symbols lazily, not require externs at load time" >&2
    exit 1
fi
rg -q 'hetgpu_is_ggml_cuda_rms_norm_f32' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q "launch-time lazy PTX resolved '%s'" "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'HETGPU_CUDART_LAZY_PTX_FAIL_OPEN",.*0' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -Fq 'lookup_registered_function_exact(symbol' "${REPO_ROOT}/zluda/src/cudart_shim.c"
rg -q 'float eps = \(args && args\[3\]\)' "${REPO_ROOT}/zluda/src/cudart_shim.c"
if rg -q 'args\[[67]\]' "${REPO_ROOT}/zluda/src/cudart_shim.c"; then
    echo "RMSNorm host fallback must not probe beyond the known four-argument ggml CUDA signature" >&2
    exit 1
fi
rg -q 'try_extract_nvidia_cubin_from_fatbin' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'selected fatbin CUBIN for NVIDIA module' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'copy_nvidia_module_image_for_lifter' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'binary module image is neither ELF CUBIN nor CUDA fatbin' "${REPO_ROOT}/zluda/src/impl/module.rs"
rg -q 'CUDA fatbin did not contain a loadable NVIDIA CUBIN' "${REPO_ROOT}/zluda/src/impl/module.rs"

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
grep -Fq "kimi_numerical,dry_run,0,planned" "${proof_csv}"

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
    printf '%s\n' 'printf "prefer_cubin=%s\n" "${HETGPU_CUDART_PREFER_FATBIN_CUBIN_FOR_SASS:-}" >&2'
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
grep -Fq "prefer_cubin=1" "${e2e_cudart_work_dir}/logs/kimi.stderr"

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

cublas_forward_work_dir="$(mktemp -d /tmp/hetgpu-cublas-forward-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}" "${e2e_cudart_work_dir}" "${cublas_forward_work_dir}"' EXIT
fake_cublas_src="${cublas_forward_work_dir}/fake_cublas.c"
shim_forward_so="${cublas_forward_work_dir}/libhetgpu_cublas_probe.so"
probe_src="${cublas_forward_work_dir}/probe.c"
probe_bin="${cublas_forward_work_dir}/probe"
fake_cublas_log="${cublas_forward_work_dir}/fake_cublas.log"
{
    printf '%s\n' '#include <stdio.h>'
    printf '%s\n' '#include <stdlib.h>'
    printf '%s\n' 'typedef void* cublasHandle_t;'
    printf '%s\n' 'typedef int cublasStatus_t;'
    printf '%s\n' 'typedef int cublasOperation_t;'
    printf '%s\n' 'static void log_call(const char *name) {'
    printf '%s\n' '    const char *path = getenv("HETGPU_FAKE_CUBLAS_LOG");'
    printf '%s\n' '    FILE *f = path ? fopen(path, "a") : NULL;'
    printf '%s\n' '    if (f) { fprintf(f, "%s\n", name); fclose(f); }'
    printf '%s\n' '}'
    printf '%s\n' 'cublasStatus_t cublasCreate_v2(cublasHandle_t *handle) {'
    printf '%s\n' '    log_call("create");'
    printf '%s\n' '    if (handle) *handle = (void*)0xfeedbeef;'
    printf '%s\n' '    return 0;'
    printf '%s\n' '}'
    printf '%s\n' 'cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) { log_call("destroy"); return handle == (void*)0xfeedbeef ? 0 : 88; }'
    printf '%s\n' 'cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void *stream) { log_call(stream ? "set_stream_nonnull" : "set_stream_null"); return handle == (void*)0xfeedbeef ? 0 : 89; }'
    printf '%s\n' 'cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void **stream) { log_call("get_stream"); if (stream) *stream = NULL; return handle == (void*)0xfeedbeef ? 0 : 90; }'
    printf '%s\n' 'cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode) { log_call("set_math"); return handle == (void*)0xfeedbeef ? 0 : 91; }'
    printf '%s\n' 'cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int *mode) { log_call("get_math"); if (mode) *mode = 0; return handle == (void*)0xfeedbeef ? 0 : 92; }'
    printf '%s\n' 'cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, int mode) { log_call("set_pointer"); return handle == (void*)0xfeedbeef ? 0 : 93; }'
    printf '%s\n' 'cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, int *mode) { log_call("get_pointer"); if (mode) *mode = 0; return handle == (void*)0xfeedbeef ? 0 : 94; }'
    printf '%s\n' 'cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {'
    printf '%s\n' '    log_call("sgemm");'
    printf '%s\n' '    if (handle != (void*)0xfeedbeef) return 95;'
    printf '%s\n' '    if (m == 1 && n == 1 && k == 1 && C) *C = (*alpha) * (*A) * (*B) + (*beta) * (*C);'
    printf '%s\n' '    return 0;'
    printf '%s\n' '}'
} >"${fake_cublas_src}"
{
    printf '%s\n' '#include <stdint.h>'
    printf '%s\n' 'typedef void* cublasHandle_t;'
    printf '%s\n' 'typedef int cublasStatus_t;'
    printf '%s\n' 'typedef int cublasOperation_t;'
    printf '%s\n' 'typedef struct { uint64_t magic; int device; unsigned int flags; int priority; } HetgpuCudaStream;'
    printf '%s\n' 'cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);'
    printf '%s\n' 'cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void *stream);'
    printf '%s\n' 'cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode);'
    printf '%s\n' 'cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, int mode);'
    printf '%s\n' 'cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);'
    printf '%s\n' 'int main(void) {'
    printf '%s\n' '    cublasHandle_t h = 0;'
    printf '%s\n' '    HetgpuCudaStream stream = { 0x485447505354524dULL, 0, 0, 0 };'
    printf '%s\n' '    float alpha = 2.0f, beta = 1.0f, A = 3.0f, B = 5.0f, C = 7.0f;'
    printf '%s\n' '    if (cublasCreate_v2(&h) != 0) return 10;'
    printf '%s\n' '    if ((uintptr_t)h != (uintptr_t)0xfeedbeef) return 11;'
    printf '%s\n' '    if (cublasSetStream_v2(h, &stream) != 0) return 12;'
    printf '%s\n' '    if (cublasSetMathMode(h, 0) != 0) return 13;'
    printf '%s\n' '    if (cublasSetPointerMode_v2(h, 0) != 0) return 14;'
    printf '%s\n' '    if (cublasSgemm_v2(h, 0, 0, 1, 1, 1, &alpha, &A, 1, &B, 1, &beta, &C, 1) != 0) return 15;'
    printf '%s\n' '    return C == 37.0f ? 0 : 16;'
    printf '%s\n' '}'
} >"${probe_src}"
"${CC:-cc}" -shared -fPIC -Wl,-soname,libcublas.so.12 -o "${cublas_forward_work_dir}/libcublas.so.12" "${fake_cublas_src}"
"${CC:-cc}" -shared -fPIC -Wno-unused-parameter -DHETGPU_SHIM_ENABLE_REAL_CUBLAS_BY_DEFAULT \
    -o "${shim_forward_so}" "${REPO_ROOT}/zluda/src/cublas_shim.c" -ldl -lm -pthread
"${CC:-cc}" -o "${probe_bin}" "${probe_src}" "${shim_forward_so}" -Wl,--allow-shlib-undefined -Wl,-rpath,"${cublas_forward_work_dir}"
HETGPU_FAKE_CUBLAS_LOG="${fake_cublas_log}" \
HETGPU_CUBLAS_FORWARD_REAL=1 \
LD_LIBRARY_PATH="${cublas_forward_work_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${probe_bin}"
grep -Fxq create "${fake_cublas_log}"
grep -Fxq set_stream_null "${fake_cublas_log}"
grep -Fxq set_math "${fake_cublas_log}"
grep -Fxq set_pointer "${fake_cublas_log}"
grep -Fxq sgemm "${fake_cublas_log}"

cudart_fatbin_work_dir="$(mktemp -d /tmp/hetgpu-cudart-fatbin-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}" "${e2e_cudart_work_dir}" "${cublas_forward_work_dir}" "${cudart_fatbin_work_dir}"' EXIT
cudart_fatbin_so="${cudart_fatbin_work_dir}/libhetgpu_cudart_probe.so"
cudart_fatbin_probe_src="${cudart_fatbin_work_dir}/probe.c"
cudart_fatbin_probe_bin="${cudart_fatbin_work_dir}/probe"
cudart_fatbin_log="${cudart_fatbin_work_dir}/module.log"
{
    printf '%s\n' '#define _GNU_SOURCE'
    printf '%s\n' '#include <stdint.h>'
    printf '%s\n' '#include <stdio.h>'
    printf '%s\n' '#include <stdlib.h>'
    printf '%s\n' '#include <string.h>'
    printf '%s\n' 'typedef int CUresult;'
    printf '%s\n' 'typedef int CUdevice;'
    printf '%s\n' 'typedef void* CUcontext;'
    printf '%s\n' 'typedef void* CUmodule;'
    printf '%s\n' 'typedef void* CUfunction;'
    printf '%s\n' 'typedef void* CUstream;'
    printf '%s\n' 'typedef struct { uint32_t magic; uint16_t version; uint16_t header_size; uint64_t files_size; } FatbinHeader;'
    printf '%s\n' 'typedef struct { uint16_t kind; uint16_t version; uint32_t header_size; uint32_t padded_payload_size; uint32_t unknown0; uint32_t payload_size; uint32_t unknown1; uint32_t unknown2; uint32_t sm_version; uint32_t bit_width; uint32_t unknown3; uint64_t unknown4; uint64_t unknown5; uint64_t uncompressed_payload; } FatbinFileHeader;'
    printf '%s\n' 'typedef struct { uint32_t magic; uint32_t version; void *data; void *filename; } FatbinWrapper;'
    printf '%s\n' 'void **__cudaRegisterFatBinary(void *fatCubin);'
    printf '%s\n' 'int hetgpu_lz4_decompress(const char *src, char *dst, int compressedSize, int dstCapacity) { (void)src; (void)dst; (void)compressedSize; (void)dstCapacity; return -1; }'
    printf '%s\n' 'int hetgpu_zstd_decompress(const char *src, char *dst, int compressedSize, int dstCapacity) { (void)src; (void)dst; (void)compressedSize; (void)dstCapacity; return -1; }'
    printf '%s\n' 'CUresult cuDeviceGet(CUdevice *device, int ordinal) { if (device) *device = ordinal; return 0; }'
    printf '%s\n' 'CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) { (void)dev; if (pctx) *pctx = (void*)0x3333; return 0; }'
    printf '%s\n' 'CUresult cuCtxSetCurrent(CUcontext ctx) { (void)ctx; return 0; }'
    printf '%s\n' 'CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) { (void)hmod; (void)name; if (hfunc) *hfunc = (void*)0x2222; return 0; }'
    printf '%s\n' 'CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz, unsigned int sh, CUstream s, void **params, void **extra) { (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)sh; (void)s; (void)params; (void)extra; return 0; }'
    printf '%s\n' 'static void write_log(const char *value) {'
    printf '%s\n' '    const char *path = getenv("HETGPU_CUDART_FATBIN_PROBE_LOG");'
    printf '%s\n' '    FILE *f = path ? fopen(path, "a") : NULL;'
    printf '%s\n' '    if (f) { fprintf(f, "%s\n", value); fclose(f); }'
    printf '%s\n' '}'
    printf '%s\n' 'CUresult cuModuleLoadData(CUmodule *module, const void *image) {'
    printf '%s\n' '    const unsigned char *p = (const unsigned char*)image;'
    printf '%s\n' '    if (p[0] == 0x50 && p[1] == 0xed && p[2] == 0x55 && p[3] == 0xba) write_log("loaded_fatbin");'
    printf '%s\n' '    else if (p[0] == 0x7f && p[1] == 0x45 && p[2] == 0x4c && p[3] == 0x46) write_log("loaded_elf");'
    printf '%s\n' '    else if (memcmp(p, ".version", 8) == 0) write_log("loaded_ptx");'
    printf '%s\n' '    else write_log("loaded_other");'
    printf '%s\n' '    if (module) *module = (void*)0x1111;'
    printf '%s\n' '    return 0;'
    printf '%s\n' '}'
    printf '%s\n' 'static void put_entry(unsigned char *entry, uint16_t kind, const unsigned char *payload, uint32_t payload_size) {'
    printf '%s\n' '    FatbinFileHeader *fh = (FatbinFileHeader*)entry;'
    printf '%s\n' '    memset(fh, 0, sizeof(*fh));'
    printf '%s\n' '    fh->kind = kind;'
    printf '%s\n' '    fh->version = 0x101;'
    printf '%s\n' '    fh->header_size = sizeof(FatbinFileHeader);'
    printf '%s\n' '    fh->payload_size = payload_size;'
    printf '%s\n' '    fh->padded_payload_size = (payload_size + 7u) & ~7u;'
    printf '%s\n' '    fh->sm_version = 120;'
    printf '%s\n' '    fh->bit_width = 64;'
    printf '%s\n' '    memcpy(entry + sizeof(FatbinFileHeader), payload, payload_size);'
    printf '%s\n' '}'
    printf '%s\n' 'int main(void) {'
    printf '%s\n' '    static const unsigned char ptx[] = ".version 8.8\n.target sm_120\n.address_size 64\n.visible .entry fake_kernel() { ret; }\n";'
    printf '%s\n' '    unsigned char elf[96] = {0};'
    printf '%s\n' '    elf[0] = 0x7f; elf[1] = 0x45; elf[2] = 0x4c; elf[3] = 0x46; elf[4] = 2; elf[5] = 1; elf[18] = 0xbe;'
    printf '%s\n' '    size_t ptx_padded = (sizeof(ptx) + 7u) & ~7u;'
    printf '%s\n' '    size_t elf_padded = (sizeof(elf) + 7u) & ~7u;'
    printf '%s\n' '    size_t fatbin_size = sizeof(FatbinHeader) + sizeof(FatbinFileHeader) + ptx_padded + sizeof(FatbinFileHeader) + elf_padded;'
    printf '%s\n' '    unsigned char *fatbin = (unsigned char*)calloc(1, fatbin_size);'
    printf '%s\n' '    FatbinHeader *header = (FatbinHeader*)fatbin;'
    printf '%s\n' '    header->magic = 0xba55ed50u;'
    printf '%s\n' '    header->version = 1;'
    printf '%s\n' '    header->header_size = sizeof(FatbinHeader);'
    printf '%s\n' '    header->files_size = fatbin_size - sizeof(FatbinHeader);'
    printf '%s\n' '    unsigned char *entry = fatbin + sizeof(FatbinHeader);'
    printf '%s\n' '    put_entry(entry, 1, ptx, (uint32_t)sizeof(ptx));'
    printf '%s\n' '    entry += sizeof(FatbinFileHeader) + ptx_padded;'
    printf '%s\n' '    put_entry(entry, 2, elf, (uint32_t)sizeof(elf));'
    printf '%s\n' '    FatbinWrapper wrapper = { 0x466243b1u, 1, fatbin, NULL };'
    printf '%s\n' '    void **handle = __cudaRegisterFatBinary(&wrapper);'
    printf '%s\n' '    free(fatbin);'
    printf '%s\n' '    return handle ? 0 : 1;'
    printf '%s\n' '}'
} >"${cudart_fatbin_probe_src}"
"${CC:-cc}" -shared -fPIC -Wno-unused-parameter \
    -o "${cudart_fatbin_so}" "${REPO_ROOT}/zluda/src/cudart_shim.c" -ldl -lm -pthread
"${CC:-cc}" -rdynamic -o "${cudart_fatbin_probe_bin}" "${cudart_fatbin_probe_src}" \
    "${cudart_fatbin_so}" -ldl -Wl,--allow-shlib-undefined -Wl,-rpath,"${cudart_fatbin_work_dir}"
HETGPU_CUDART_PREFER_FATBIN_CUBIN_FOR_SASS=1 \
HETGPU_CUDART_EAGER_PTX=1 \
HETGPU_CUDART_FATBIN_PROBE_LOG="${cudart_fatbin_log}" \
    "${cudart_fatbin_probe_bin}"
if grep -Fxq loaded_ptx "${cudart_fatbin_log}"; then
    echo "cudart shim loaded PTX even though fatbin CUBIN exposure was requested" >&2
    exit 1
fi
if ! grep -Eq 'loaded_fatbin|loaded_elf' "${cudart_fatbin_log}"; then
    echo "cudart shim did not expose fatbin/CUBIN payload to cuModuleLoadData" >&2
    exit 1
fi

cudart_lazy_launch_work_dir="$(mktemp -d /tmp/hetgpu-cudart-lazy-launch-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}" "${e2e_cudart_work_dir}" "${cublas_forward_work_dir}" "${cudart_fatbin_work_dir}" "${cudart_lazy_launch_work_dir}"' EXIT
cudart_lazy_launch_so="${cudart_lazy_launch_work_dir}/libhetgpu_cudart_lazy_probe.so"
cudart_lazy_launch_probe_src="${cudart_lazy_launch_work_dir}/probe.c"
cudart_lazy_launch_probe_bin="${cudart_lazy_launch_work_dir}/probe"
cudart_lazy_launch_log="${cudart_lazy_launch_work_dir}/driver.log"
cudart_lazy_launch_stderr="${cudart_lazy_launch_work_dir}/stderr.log"
{
    printf '%s\n' '#define _GNU_SOURCE'
    printf '%s\n' '#include <stdint.h>'
    printf '%s\n' '#include <stdio.h>'
    printf '%s\n' '#include <stdlib.h>'
    printf '%s\n' '#include <string.h>'
    printf '%s\n' 'typedef int CUresult;'
    printf '%s\n' 'typedef int CUdevice;'
    printf '%s\n' 'typedef void* CUcontext;'
    printf '%s\n' 'typedef void* CUmodule;'
    printf '%s\n' 'typedef void* CUfunction;'
    printf '%s\n' 'typedef void* CUstream;'
    printf '%s\n' 'typedef void* cudaStream_t;'
    printf '%s\n' 'typedef struct { unsigned int x; unsigned int y; unsigned int z; } dim3;'
    printf '%s\n' 'typedef struct { uint32_t magic; uint16_t version; uint16_t header_size; uint64_t files_size; } FatbinHeader;'
    printf '%s\n' 'typedef struct { uint16_t kind; uint16_t version; uint32_t header_size; uint32_t padded_payload_size; uint32_t unknown0; uint32_t payload_size; uint32_t unknown1; uint32_t unknown2; uint32_t sm_version; uint32_t bit_width; uint32_t unknown3; uint64_t unknown4; uint64_t unknown5; uint64_t uncompressed_payload; } FatbinFileHeader;'
    printf '%s\n' 'typedef struct { uint32_t magic; uint32_t version; void *data; void *filename; } FatbinWrapper;'
    printf '%s\n' 'void **__cudaRegisterFatBinary(void *fatCubin);'
    printf '%s\n' 'void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun, const char* deviceName, int thread_limit, void* tid, void* bid, void* bDim, void* gDim, void* wSize);'
    printf '%s\n' 'int cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);'
    printf '%s\n' 'int hetgpu_lz4_decompress(const char *src, char *dst, int compressedSize, int dstCapacity) { (void)src; (void)dst; (void)compressedSize; (void)dstCapacity; return -1; }'
    printf '%s\n' 'int hetgpu_zstd_decompress(const char *src, char *dst, int compressedSize, int dstCapacity) { (void)src; (void)dst; (void)compressedSize; (void)dstCapacity; return -1; }'
    printf '%s\n' 'CUresult cuDeviceGet(CUdevice *device, int ordinal) { if (device) *device = ordinal; return 0; }'
    printf '%s\n' 'CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) { (void)dev; if (pctx) *pctx = (void*)0x3333; return 0; }'
    printf '%s\n' 'CUresult cuCtxSetCurrent(CUcontext ctx) { (void)ctx; return 0; }'
    printf '%s\n' 'static int g_in_launch = 0;'
    printf '%s\n' 'static void write_log(const char *tag, const char *value) {'
    printf '%s\n' '    const char *path = getenv("HETGPU_CUDART_LAZY_LAUNCH_PROBE_LOG");'
    printf '%s\n' '    FILE *f = path ? fopen(path, "a") : NULL;'
    printf '%s\n' '    if (f) { fprintf(f, "%s:%s\n", tag, value ? value : ""); fclose(f); }'
    printf '%s\n' '}'
    printf '%s\n' 'CUresult cuModuleLoadData(CUmodule *module, const void *image) {'
    printf '%s\n' '    const unsigned char *p = (const unsigned char*)image;'
    printf '%s\n' '    write_log("load_phase", g_in_launch ? "launch" : "register");'
    printf '%s\n' '    if (memcmp(p, ".version", 8) == 0) write_log("loaded", "ptx");'
    printf '%s\n' '    else write_log("loaded", "other");'
    printf '%s\n' '    if (module) *module = (void*)0x1111;'
    printf '%s\n' '    return 0;'
    printf '%s\n' '}'
    printf '%s\n' 'CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {'
    printf '%s\n' '    (void)hmod;'
    printf '%s\n' '    write_log("getfunc_phase", g_in_launch ? "launch" : "register");'
    printf '%s\n' '    write_log("getfunc", name);'
    printf '%s\n' '    if (getenv("HETGPU_CUDART_LAZY_LAUNCH_GETFUNC_FAIL")) return 77;'
    printf '%s\n' '    if (hfunc) *hfunc = (void*)0x2222;'
    printf '%s\n' '    return 0;'
    printf '%s\n' '}'
    printf '%s\n' 'CUresult cuLaunchKernel(CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz, unsigned int bx, unsigned int by, unsigned int bz, unsigned int sh, CUstream s, void **params, void **extra) {'
    printf '%s\n' '    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz; (void)sh; (void)s; (void)params; (void)extra;'
    printf '%s\n' '    const char *path = getenv("HETGPU_CUDART_LAZY_LAUNCH_PROBE_LOG");'
    printf '%s\n' '    FILE *log = path ? fopen(path, "a") : NULL;'
    printf '%s\n' '    if (log) { fprintf(log, "launch:%p\n", f); fclose(log); }'
    printf '%s\n' '    return f == (void*)0x2222 ? 0 : 99;'
    printf '%s\n' '}'
    printf '%s\n' 'static void fake_host_kernel(void) {}'
    printf '%s\n' 'static void put_entry(unsigned char *entry, uint16_t kind, const unsigned char *payload, uint32_t payload_size) {'
    printf '%s\n' '    FatbinFileHeader *fh = (FatbinFileHeader*)entry;'
    printf '%s\n' '    memset(fh, 0, sizeof(*fh));'
    printf '%s\n' '    fh->kind = kind;'
    printf '%s\n' '    fh->version = 0x101;'
    printf '%s\n' '    fh->header_size = sizeof(FatbinFileHeader);'
    printf '%s\n' '    fh->payload_size = payload_size;'
    printf '%s\n' '    fh->padded_payload_size = (payload_size + 7u) & ~7u;'
    printf '%s\n' '    fh->sm_version = 120;'
    printf '%s\n' '    fh->bit_width = 64;'
    printf '%s\n' '    memcpy(entry + sizeof(FatbinFileHeader), payload, payload_size);'
    printf '%s\n' '}'
    printf '%s\n' 'int main(void) {'
    printf '%s\n' '    static const unsigned char ptx[] = ".version 8.8\n.target sm_120\n.address_size 64\n.visible .entry fake_kernel() { ret; }\n";'
    printf '%s\n' '    size_t ptx_padded = (sizeof(ptx) + 7u) & ~7u;'
    printf '%s\n' '    size_t fatbin_size = sizeof(FatbinHeader) + sizeof(FatbinFileHeader) + ptx_padded;'
    printf '%s\n' '    unsigned char *fatbin = (unsigned char*)calloc(1, fatbin_size);'
    printf '%s\n' '    FatbinHeader *header = (FatbinHeader*)fatbin;'
    printf '%s\n' '    header->magic = 0xba55ed50u;'
    printf '%s\n' '    header->version = 1;'
    printf '%s\n' '    header->header_size = sizeof(FatbinHeader);'
    printf '%s\n' '    header->files_size = fatbin_size - sizeof(FatbinHeader);'
    printf '%s\n' '    put_entry(fatbin + sizeof(FatbinHeader), 1, ptx, (uint32_t)sizeof(ptx));'
    printf '%s\n' '    FatbinWrapper wrapper = { 0x466243b1u, 1, fatbin, NULL };'
    printf '%s\n' '    void **handle = __cudaRegisterFatBinary(&wrapper);'
    printf '%s\n' '    if (!handle) return 2;'
    printf '%s\n' '    __cudaRegisterFunction(handle, (const char*)fake_host_kernel, NULL, "fake_kernel", -1, NULL, NULL, NULL, NULL, NULL);'
    printf '%s\n' '    write_log("phase", "registered");'
    printf '%s\n' '    g_in_launch = 1;'
    printf '%s\n' '    dim3 one = {1, 1, 1};'
    printf '%s\n' '    int rc = cudaLaunchKernel((const void*)fake_host_kernel, one, one, NULL, 0, NULL);'
    printf '%s\n' '    g_in_launch = 0;'
    printf '%s\n' '    free(fatbin);'
    printf '%s\n' '    return rc;'
    printf '%s\n' '}'
} >"${cudart_lazy_launch_probe_src}"
"${CC:-cc}" -shared -fPIC -Wno-unused-parameter \
    -o "${cudart_lazy_launch_so}" "${REPO_ROOT}/zluda/src/cudart_shim.c" -ldl -lm -pthread
"${CC:-cc}" -rdynamic -o "${cudart_lazy_launch_probe_bin}" "${cudart_lazy_launch_probe_src}" \
    "${cudart_lazy_launch_so}" -ldl -Wl,--allow-shlib-undefined -Wl,-rpath,"${cudart_lazy_launch_work_dir}"
set +e
HETGPU_CUDART_DEFER_MODULE_LOAD=1 \
HETGPU_CUDART_LAZY_LAUNCH_PROBE_LOG="${cudart_lazy_launch_log}" \
    "${cudart_lazy_launch_probe_bin}" >"${cudart_lazy_launch_stderr}" 2>&1
cudart_lazy_launch_status="$?"
set -e
if [[ "${cudart_lazy_launch_status}" != "0" ]]; then
    cat "${cudart_lazy_launch_stderr}" >&2
    exit "${cudart_lazy_launch_status}"
fi
grep -Fxq "phase:registered" "${cudart_lazy_launch_log}"
grep -Fxq "load_phase:launch" "${cudart_lazy_launch_log}"
grep -Fxq "getfunc_phase:launch" "${cudart_lazy_launch_log}"
if grep -Fxq "load_phase:register" "${cudart_lazy_launch_log}" ||
   grep -Fxq "getfunc_phase:register" "${cudart_lazy_launch_log}"; then
    echo "cudart shim eager-loaded module/function before deferred launch" >&2
    exit 1
fi
grep -Fxq "loaded:ptx" "${cudart_lazy_launch_log}"
grep -Fxq "getfunc:fake_kernel" "${cudart_lazy_launch_log}"
grep -Fxq "launch:0x2222" "${cudart_lazy_launch_log}"
if grep -Fqi "fail-open" "${cudart_lazy_launch_log}" "${cudart_lazy_launch_stderr}"; then
    echo "cudart shim used fail-open instead of the deferred module/function launch path" >&2
    exit 1
fi

: >"${cudart_lazy_launch_log}"
: >"${cudart_lazy_launch_stderr}"
set +e
HETGPU_CUDART_DEFER_MODULE_LOAD=1 \
HETGPU_CUDART_LAZY_LAUNCH_GETFUNC_FAIL=1 \
HETGPU_CUDART_LAZY_LAUNCH_PROBE_LOG="${cudart_lazy_launch_log}" \
    "${cudart_lazy_launch_probe_bin}" >"${cudart_lazy_launch_stderr}" 2>&1
cudart_lazy_launch_fail_status="$?"
set -e
if [[ "${cudart_lazy_launch_fail_status}" == "0" ]]; then
    echo "cudart shim succeeded after lazy cuModuleGetFunction failure with fail-open disabled" >&2
    exit 1
fi
grep -Fxq "load_phase:launch" "${cudart_lazy_launch_log}"
grep -Fxq "getfunc_phase:launch" "${cudart_lazy_launch_log}"
if grep -Fq "launch:" "${cudart_lazy_launch_log}"; then
    echo "cudart shim called cuLaunchKernel after lazy cuModuleGetFunction failure" >&2
    exit 1
fi
if grep -Fqi "fail-open" "${cudart_lazy_launch_log}" "${cudart_lazy_launch_stderr}"; then
    echo "cudart shim fail-opened after lazy cuModuleGetFunction failure" >&2
    exit 1
fi

numerical_work_dir="$(mktemp -d /tmp/hetgpu-kimi-numerical-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}" "${e2e_cudart_work_dir}" "${cublas_forward_work_dir}" "${cudart_fatbin_work_dir}" "${cudart_lazy_launch_work_dir}" "${numerical_work_dir}"' EXIT
fake_numerical_model_dir="${numerical_work_dir}/model"
mkdir -p "${fake_numerical_model_dir}"
for shard in \
    moonshotai_Kimi-K2.6-IQ1_S-00001-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00002-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00003-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00004-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00005-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_S-00006-of-00006.gguf
do
    : >"${fake_numerical_model_dir}/${shard}"
done
fake_numerical_runner="${numerical_work_dir}/fake-numerical-llama-cli"
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'printf "same deterministic kimi output\n"'
    printf '%s\n' 'printf "llm_load_tensors: offloaded 1/62 layers to GPU\n" >&2'
    printf '%s\n' 'if [[ "${HETGPU_KIMI_NUMERICAL_RUN_ROLE:-}" == "hooked" ]]; then printf "[hetGPU SASS] lifted CUBIN via Rust lifter: input=64 bytes ptx=32 bytes diagnostics=0\n" >&2; fi'
} >"${fake_numerical_runner}"
chmod +x "${fake_numerical_runner}"
HETGPU_KIMI_NUMERICAL_WORKDIR="${numerical_work_dir}" \
BITNET_LLAMA_CLI="${fake_numerical_runner}" \
MODEL_DIR="${fake_numerical_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO=/bin/true \
    "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh" >/dev/null 2>&1
numerical_csv="${numerical_work_dir}/bench_kimi_k26_numerical.csv"
head -n 1 "${numerical_csv}" | grep -Fxq "case,status,total_ms,baseline_exit_code,hooked_exit_code,baseline_stdout_sha256,hooked_stdout_sha256,baseline_stdout_bytes,hooked_stdout_bytes,ptx_source_markers,sass_lifter_markers,offloaded_layers,message"
grep -Fq "kimi_k26_numerical,pass" "${numerical_csv}"

numerical_mismatch_work_dir="$(mktemp -d /tmp/hetgpu-kimi-numerical-mismatch-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}" "${e2e_cudart_work_dir}" "${cublas_forward_work_dir}" "${cudart_fatbin_work_dir}" "${cudart_lazy_launch_work_dir}" "${numerical_work_dir}" "${numerical_mismatch_work_dir}"' EXIT
fake_mismatch_runner="${numerical_mismatch_work_dir}/fake-mismatch-llama-cli"
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'printf "%s output\n" "${HETGPU_KIMI_NUMERICAL_RUN_ROLE:-unknown}"'
    printf '%s\n' 'printf "llm_load_tensors: offloaded 1/62 layers to GPU\n" >&2'
    printf '%s\n' 'if [[ "${HETGPU_KIMI_NUMERICAL_RUN_ROLE:-}" == "hooked" ]]; then printf "[NVIDIA Backend] Detected PTX source (123 bytes)\n" >&2; fi'
} >"${fake_mismatch_runner}"
chmod +x "${fake_mismatch_runner}"
set +e
HETGPU_KIMI_NUMERICAL_WORKDIR="${numerical_mismatch_work_dir}" \
BITNET_LLAMA_CLI="${fake_mismatch_runner}" \
MODEL_DIR="${fake_numerical_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO=/bin/true \
    "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh" >/dev/null 2>&1
numerical_mismatch_status="$?"
set -e
if [[ "${numerical_mismatch_status}" == "0" ]]; then
    echo "Kimi numerical proof accepted mismatched baseline/hooked output" >&2
    exit 1
fi
grep -Fq "kimi_k26_numerical,output_mismatch" "${numerical_mismatch_work_dir}/bench_kimi_k26_numerical.csv"

numerical_hook_fail_work_dir="$(mktemp -d /tmp/hetgpu-kimi-numerical-hook-fail-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${proof_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}" "${e2e_comma_work_dir}" "${e2e_marker_work_dir}" "${e2e_no_cuda_work_dir}" "${e2e_preload_work_dir}" "${e2e_cudart_work_dir}" "${cublas_forward_work_dir}" "${cudart_fatbin_work_dir}" "${cudart_lazy_launch_work_dir}" "${numerical_work_dir}" "${numerical_mismatch_work_dir}" "${numerical_hook_fail_work_dir}"' EXIT
fake_hook_fail_runner="${numerical_hook_fail_work_dir}/fake-hook-fail-llama-cli"
{
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'if [[ "${HETGPU_KIMI_NUMERICAL_RUN_ROLE:-}" == "hooked" ]]; then printf "hooked failed\n" >&2; exit 7; fi'
    printf '%s\n' 'printf "baseline output\n"'
    printf '%s\n' 'printf "llm_load_tensors: offloaded 1/62 layers to GPU\n" >&2'
} >"${fake_hook_fail_runner}"
chmod +x "${fake_hook_fail_runner}"
set +e
HETGPU_KIMI_NUMERICAL_WORKDIR="${numerical_hook_fail_work_dir}" \
BITNET_LLAMA_CLI="${fake_hook_fail_runner}" \
MODEL_DIR="${fake_numerical_model_dir}" \
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_S \
CARGO=/bin/true \
    "${SCRIPT_DIR}/run_kimi_k26_numerical_proof.sh" >/dev/null 2>&1
numerical_hook_fail_status="$?"
set -e
if [[ "${numerical_hook_fail_status}" == "0" ]]; then
    echo "Kimi numerical proof accepted a failing hooked run" >&2
    exit 1
fi
grep -Fq "kimi_k26_numerical,hooked_failed" "${numerical_hook_fail_work_dir}/bench_kimi_k26_numerical.csv"
