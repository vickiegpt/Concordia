# Kimi K2.6 SASS Lifter Correctness Benchmarks

**Date:** 2026-06-17
**Status:** Approved for planning
**Scope:** Extend the existing NVIDIA SASS-to-PTX round-trip harness with Kimi K2.6-style correctness benchmarks and an optional slow GGUF end-to-end capture path.

## Goals

1. Add deterministic Kimi-style microkernels that stress the SASS lifter on inference-like instruction patterns.
2. Keep the primary correctness contract strict: original CUBIN output must match lifted-PTX output on the NVIDIA Driver API path.
3. Preserve the current `LD_PRELOAD=target/debug/libnvcuda.so` flow in `zluda/tests/sass_roundtrip_bench`.
4. Add an optional actual Kimi K2.6 GGUF runner tier using `tools/run_kimi_k26_iq1m_bitnet.sh`.
5. Emit machine-readable CSV rows that distinguish pass, mismatch, lifter failure, missing runner, missing model, and skipped e2e states.

## Non-Goals

- Download Kimi K2.6 model shards automatically during the benchmark.
- Require the Kimi GGUF model or BitNet `llama-cli` for normal microkernel correctness tests.
- Prove full text-generation equivalence between native CUDA and the hook in the first pass.
- Replace the existing SASS lifter or `sass_inliner --recover-ptx` entrypoints.
- Benchmark production throughput as the main success criterion. Timing is recorded, but correctness is the gate.

## Existing Context

The repo already has a lifter-focused correctness harness:

- `zluda/tests/sass_roundtrip_bench/run.sh` assembles PTX templates with `ptxas`, runs CUBINs through `LD_PRELOAD=target/debug/libnvcuda.so`, captures lifted PTX dumps, then loads and runs the lifted PTX.
- `zluda/tests/sass_roundtrip_bench/roundtrip_runner.c` compares original CUBIN output with lifted-PTX output for kernels using the ABI `(out, in, n)`.
- Existing cases cover scalar arithmetic, predicate selection, FMA conversion, and shared-memory synchronization.
- `tools/run_kimi_k26_iq1m_bitnet.sh` is the local Kimi K2.6 IQ1_M GGUF runner wrapper.
- `tools/download_kimi_k26_iq1m.sh` documents the six expected GGUF shards and sizes.

Local probing showed the default e2e path is not currently ready on this host:

- `/root/hetGPU/BitNet-work/build/bin/llama-cli` is missing.
- The default Kimi shard directory did not list available files.

The e2e tier must therefore skip cleanly unless the caller supplies valid paths through environment variables.

## Architecture

The benchmark will have two tiers.

### Tier 1: Synthetic Kimi-Style Lifter Correctness

Add new PTX templates under:

```text
zluda/tests/sass_roundtrip_bench/ptx/
```

The initial case set:

- `kimi_iq1m_unpack`: low-bit field extraction, sign extension, scale-like integer arithmetic, and packed memory access.
- `kimi_rmsnorm_bits`: deterministic integer/fp32 normalization-style computation using multiply-add, reciprocal square-root-style operations where PTX support is practical, and conversion back to integer outputs.
- `kimi_swiglu_mix`: gate/value paths using predicate selection, FMA, multiply, and integer reinterpretation.
- `kimi_rope_mix`: pairwise rotate/mix pattern over adjacent values to stress address arithmetic and paired loads/stores.
- `kimi_attention_mask`: predicate-heavy masking and branch/select behavior similar to causal attention masking.

Each case should keep the current ABI unless a specific case needs a runner extension:

```c
kernel(uint32_t *out, const uint32_t *in, uint32_t n)
```

Keeping this ABI lets the current C runner compare original CUBIN output with lifted PTX output without adding model-specific host setup.

### Tier 2: Actual Kimi K2.6 GGUF E2E Capture

Add a separate runner script under:

```text
zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh
```

The script will:

1. Validate `BITNET_LLAMA_CLI` or the default `BitNet-work/build/bin/llama-cli`.
2. Validate `MODEL_DIR` and the six expected `moonshotai_Kimi-K2.6-IQ1_M-*.gguf` shards.
3. Build `libnvcuda.so` with NVIDIA passthrough if needed.
4. Run `tools/run_kimi_k26_iq1m_bitnet.sh` under `LD_PRELOAD=target/debug/libnvcuda.so`.
5. Capture stdout, stderr, runtime, exit code, lifter marker counts, lifted PTX dump counts, and generated output byte count.
6. Write a CSV row even when skipped.

The e2e tier is optional and slower. It is not part of the default synthetic correctness run unless explicitly requested.

## Data Flow

Synthetic tier:

```text
PTX template -> ptxas -> CUBIN
       CUBIN -> LD_PRELOAD module load -> SASS lifter -> lifted PTX dump
       CUBIN -> roundtrip_runner original launch -> output A
 lifted PTX -> roundtrip_runner recovered launch -> output B
 output A vs output B -> CSV status
```

E2E tier:

```text
Kimi GGUF shards + llama-cli
       -> tools/run_kimi_k26_iq1m_bitnet.sh
       -> LD_PRELOAD libnvcuda.so
       -> CUDA module/fatbin/launch interception
       -> lifter logs and optional PTX dumps
       -> CSV status and timing summary
```

## Result Format

The existing synthetic CSV header remains compatible:

```text
case,sm,status,cubin_bytes,lifted_ptx_bytes,lift_diagnostics,load_cubin_us,load_ptx_us,kernel_cubin_us,kernel_ptx_us,total_us,message
```

The e2e CSV should use a separate file because the fields differ:

```text
case,status,total_ms,exit_code,stdout_bytes,stderr_bytes,lifter_markers,lifted_ptx_files,lifted_ptx_bytes,message
```

## Error Handling

Synthetic cases:

- `ptxas` failure stops the run because the case is invalid for the selected SM.
- Missing lifter marker becomes `missing_lifter_marker`.
- Missing lifted PTX dump becomes `missing_lifter_dump_marker`.
- CUBIN vs lifted PTX output mismatch becomes `mismatch`.
- Lifted PTX load or launch failures keep their existing statuses.

E2E Kimi tier:

- Missing `llama-cli` becomes `skipped_missing_runner`.
- Missing model shard becomes `skipped_missing_model`.
- Nonzero process exit becomes `run_failed`.
- Zero output bytes with zero exit becomes `empty_output`.
- Zero lifter markers with zero exit becomes `missing_lifter_marker`.
- Nonzero output plus lifter evidence becomes `pass`.

## Testing

Static shell tests extend `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh`:

1. `--list-cases` includes the new Kimi-style synthetic cases.
2. `--dry-run` writes rows for selected Kimi-style cases.
3. `HETGPU_ROUNDTRIP_CASES` can select one Kimi-style case.
4. The e2e script reports `skipped_missing_runner` or `skipped_missing_model` instead of failing when defaults are absent.
5. The e2e script writes the expected CSV header.

Runtime verification on an NVIDIA host:

```bash
zluda/tests/sass_roundtrip_bench/run.sh
```

Optional e2e verification when local paths exist:

```bash
BITNET_LLAMA_CLI=/path/to/llama-cli \
MODEL_DIR=/path/to/moonshotai_Kimi-K2.6-IQ1_M \
zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh
```

## Acceptance Criteria

- Synthetic Kimi-style cases are part of `--list-cases`.
- Dry-run tests cover synthetic Kimi-style case selection.
- Runtime synthetic cases compare original CUBIN output with lifted PTX output and fail on mismatches.
- The e2e Kimi script skips cleanly when `llama-cli` or model shards are missing.
- When e2e dependencies are present, the script runs under `LD_PRELOAD`, captures lifter evidence, and writes a CSV summary.
- Generated benchmark artifacts remain outside tracked source unless explicitly requested.
