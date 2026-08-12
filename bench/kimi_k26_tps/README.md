# Kimi K2.6 TPS Evidence Benchmark

This benchmark records real Kimi K2.6 BitNet throughput evidence from llama/BitNet timing logs. It does not infer TPS from wall-clock time unless the model runtime omits its own timing lines; in that case the row is marked `missing_timing`.

## Output

The parser appends both CSV and JSONL rows with these fields:

```text
case,status,tps,tokens,total_ms,aof_bytes,runner,model,gpu,commit,checkpoint_markers,message
```

The required proof fields are first-class columns:

- `tps`: generation tokens per second parsed from the llama/BitNet eval timing line.
- `tokens`: generation tokens or eval runs parsed from the same timing line.
- `total_ms`: parsed `total time = ... ms` when available, otherwise measured runner wall time.
- `exit_code`: folded into `status` and `message`; non-zero exits become `run_failed`.
- `aof_bytes`: size of the Concordia AOF file or directory for the case.
- `checkpoint_markers`: count of Concordia checkpoint/AOF/dirty-scan marker lines in stdout/stderr.

## Run

```bash
bash bench/kimi_k26_tps/run_kimi_k26_tps.sh
```

Default cases:

- `baseline`: runs through the ZLUDA NVIDIA shim with `HETGPU_KIMI_CONCORDIA=0`.
- `concordia`: runs through the shim with `HETGPU_KIMI_CONCORDIA=1` and a case-local AOF path.

Important environment variables:

```text
BITNET_LLAMA_CLI=/path/to/llama-cli
MODEL_DIR=/path/to/moonshotai_Kimi-K2.6-IQ1_M
MODEL_PREFIX=moonshotai_Kimi-K2.6-IQ1_M
KIMI_TPS_CASES=baseline,concordia
KIMI_TPS_WORKDIR=/tmp/kimi-k26-tps
KIMI_TPS_REQUIRE_RUN=1
KIMI_TPS_BASELINE_WITH_SHIM=1
KIMI_TPS_USE_CUDART_SHIM=0
KIMI_TPS_ZLUDA_FEATURES=nvidia
```

If the runner or model is missing, the script still emits CSV/JSONL evidence with `skipped_missing_runner` or `skipped_missing_model`. Set `KIMI_TPS_REQUIRE_RUN=1` to turn those statuses into a failing command.

For local PTX/cocotb ternary disaggregation without the default CXL `/dev` path:

```bash
KIMI_TMATMUL_COCOTB=1 \
KIMI_BITLINEAR_TMATMUL=1 \
KIMI_TPS_CASES=baseline \
bash bench/kimi_k26_tps/run_kimi_k26_tps.sh
```

That profile defaults to `HETGPU_TMATMUL_COCOTB=1`, `HETGPU_TMATMUL_ASM_DIR=/tmp/tmatmul-asm`, host matrix/I/O staging, `HETGPU_TMATMUL_OUTPUT_DTYPE=f32`, BitNet disaggregation, and `HETGPU_CXL_TMATMUL=0`/`HETGPU_TMATMUL_CXL=0`. Override `KIMI_TPS_ZLUDA_FEATURES` if you need a specific ZLUDA backend feature set.

For real FPGA offload from GPU memory through `/dev/cxl_tmatmul*`:

```bash
KIMI_TMATMUL_FPGA=1 \
KIMI_BITLINEAR_TMATMUL=1 \
KIMI_TPS_CASES=baseline \
bash bench/kimi_k26_tps/run_kimi_k26_tps.sh
```

That profile defaults to `HETGPU_CXL_TMATMUL=1`, `HETGPU_TMATMUL_CXL=1`, `HETGPU_TMATMUL_MATRIX_STAGE=cuda_dax`, `HETGPU_TMATMUL_IO_STAGE=cuda_dax`, and `HETGPU_TMATMUL_HARDWARE_MATMUL=1`, so the NVIDIA launch hook can submit to the FPGA device and copy the result back.

## Parser Tests

```bash
bash bench/kimi_k26_tps/test_parser.sh
bash bench/kimi_k26_tps/test_runner_static.sh
bash bench/kimi_k26_tps/test_runner_fake_run.sh
```
