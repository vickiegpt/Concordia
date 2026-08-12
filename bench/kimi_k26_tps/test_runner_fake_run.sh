#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d /tmp/kimi-k26-tps-fake.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

runner="${tmp}/llama-cli"
model_dir="${tmp}/moonshotai_Kimi-K2.6-IQ1_M"
model_prefix="$(basename "${model_dir}")"
mkdir -p "${model_dir}"
for shard_index in 1 2 3 4 5 6; do
  touch "$(printf '%s/%s-%05d-of-00006.gguf' "${model_dir}" "${model_prefix}" "${shard_index}")"
done

cat >"${runner}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'Kimi fake output\n'
printf 'nvint4=%s bitlinear=%s fallback=%s route=%s\n' \
  "${HETGPU_NVINT4_TMATMUL:-unset}" \
  "${HETGPU_NVINT4_BITLINEAR_HOOK:-unset}" \
  "${HETGPU_NVINT4_GPU_FALLBACK:-unset}" \
  "${HETGPU_NVINT4_ROUTE_LOG:-unset}" >&2
printf 'cocotb=%s asm_dir=%s staging=%s matrix_stage=%s io_stage=%s output_dtype=%s cxl=%s tmatmul_cxl=%s bitnet_disagg=%s pre_jit=%s hw_matmul=%s\n' \
  "${HETGPU_TMATMUL_COCOTB:-unset}" \
  "${HETGPU_TMATMUL_ASM_DIR:-unset}" \
  "${HETGPU_CXL_TMATMUL_STAGING:-unset}" \
  "${HETGPU_TMATMUL_MATRIX_STAGE:-unset}" \
  "${HETGPU_TMATMUL_IO_STAGE:-unset}" \
  "${HETGPU_TMATMUL_OUTPUT_DTYPE:-unset}" \
  "${HETGPU_CXL_TMATMUL:-unset}" \
  "${HETGPU_TMATMUL_CXL:-unset}" \
  "${HETGPU_BITNET_DISAGGREGATE:-unset}" \
  "${HETGPU_TMATMUL_PRE_JIT_NAMED_FALLBACK:-unset}" \
  "${HETGPU_TMATMUL_HARDWARE_MATMUL:-unset}" >&2
cat >&2 <<'EOF'
llama_perf_context_print:        eval time =     500.00 ms /    25 tokens (   20.00 ms per token,    50.00 tokens per second)
llama_perf_context_print:       total time =     750.00 ms /    40 tokens
EOF
SH
chmod +x "${runner}"

BITNET_LLAMA_CLI="${runner}" \
MODEL_DIR="${model_dir}" \
MODEL_PREFIX="${model_prefix}" \
KIMI_TPS_WORKDIR="${tmp}/work" \
KIMI_TPS_CASES="baseline" \
KIMI_TPS_BUILD_ZLUDA=0 \
KIMI_TPS_BASELINE_WITH_SHIM=0 \
KIMI_TPS_REQUIRE_RUN=1 \
KIMI_BITLINEAR_TMATMUL=1 \
KIMI_TMATMUL_COCOTB=1 \
bash "${root}/run_kimi_k26_tps.sh"

csv="${tmp}/work/kimi_k26_tps.csv"
jsonl="${tmp}/work/kimi_k26_tps.jsonl"
grep -q '^baseline,pass,50,25,750,0,' "${csv}"
grep -q '"status": "pass"' "${jsonl}"
grep -q '"tps": 50' "${jsonl}"
grep -q '"tokens": 25' "${jsonl}"
grep -q 'nvint4=1 bitlinear=1 fallback=1 route=' "${tmp}/work/logs/baseline.stderr"
grep -q 'cocotb=1 asm_dir=/tmp/tmatmul-asm staging=mmap matrix_stage=host io_stage=host output_dtype=f32 cxl=0 tmatmul_cxl=0 bitnet_disagg=1 pre_jit=1 hw_matmul=1' "${tmp}/work/logs/baseline.stderr"

KIMI_TPS_WORKDIR="${tmp}/work_fpga" \
BITNET_LLAMA_CLI="${runner}" \
MODEL_DIR="${model_dir}" \
MODEL_PREFIX="${model_prefix}" \
KIMI_TPS_CASES="baseline" \
KIMI_TPS_BUILD_ZLUDA=0 \
KIMI_TPS_BASELINE_WITH_SHIM=0 \
KIMI_TPS_REQUIRE_RUN=1 \
KIMI_BITLINEAR_TMATMUL=1 \
KIMI_TMATMUL_FPGA=1 \
bash "${root}/run_kimi_k26_tps.sh"

grep -q 'nvint4=1 bitlinear=1 fallback=1 route=' "${tmp}/work_fpga/logs/baseline.stderr"
grep -q 'cocotb=unset asm_dir=/tmp/tmatmul-asm staging=mmap matrix_stage=cuda_dax io_stage=cuda_dax output_dtype=f32 cxl=1 tmatmul_cxl=1 bitnet_disagg=1 pre_jit=1 hw_matmul=1' "${tmp}/work_fpga/logs/baseline.stderr"

echo "runner fake-run test passed"
