#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cases="$("${SCRIPT_DIR}/run.sh" --list-cases)"
grep -Fxq "int_add" <<<"${cases}"
grep -Fxq "pred_select" <<<"${cases}"
grep -Fxq "fma_bits" <<<"${cases}"
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

if rg -q "list-gpu-code" "${SCRIPT_DIR}/run.sh"; then
    echo "round-trip bench should validate PTX support through ptxas, not nvcc --list-gpu-code" >&2
    exit 1
fi

rg -Fq '.target ${sm}' "${SCRIPT_DIR}/run.sh"
rg -q 'rm -f "\$\{lifted\}"' "${SCRIPT_DIR}/run.sh"
rg -q 'wrote lifted PTX dump' "${SCRIPT_DIR}/run.sh"
rg -q 'HETGPU_SASS_LIFTER_CUOBJDUMP' "${SCRIPT_DIR}/run.sh"

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
    moonshotai_Kimi-K2.6-IQ1_M-00001-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_M-00002-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_M-00003-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_M-00004-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_M-00005-of-00006.gguf \
    moonshotai_Kimi-K2.6-IQ1_M-00006-of-00006.gguf
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
