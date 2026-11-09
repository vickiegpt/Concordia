# Qwen3.5 numerical validation, runtime v23

## Scope

Model: Qwen3.5-35B-A3B-Q4_K_M, llama.cpp private seq512 build on lanxin2.
No host reboot, no default firmware change, no interruption to CPU server PID 4807.
All four PACC devices retain jobd build 0x20036 and their separate 512 KiB staging slots.

`ngl=1` offloads the output head, not a repeating transformer layer.
`ngl=2` offloads layer39 and the output head. These are hybrid CPU/PACC results,
not full-model PACC throughput and not a 2048-concurrent-request benchmark.

## First Numerical Errors

The native CPU uses Q8_K activation quantization for Q6_K matmul; the CUDA
compatibility path uses Q8_1. Native CPU token equality alone is therefore not
a valid way to identify an incorrect CUDA kernel. At sequence1/step7, native CPU
prefers token4530 over2876 by 0.0220146, while independently evaluated Q8_1 math
prefers2876. Step indices are zero-based.

Two actual implementation errors were reproduced and repaired:

1. The compatibility layer truncated FP32-to-FP16 conversion instead of using
   round-to-nearest, ties-to-even. Selected-row Q6_K/Q8_1 output-head error fell
   from 0.006955786 to 0.000003825 against independent FP64 dot products.
   The old output agreed with the deliberately truncated-scale oracle, giving
   a direct causal check. Finite half values and adjacent rounding boundaries
   have exhaustive standalone Rust tests.
2. The old jobd RMS reciprocal-square-root iteration can converge to a negative
   value. On hardware, constant sqrt(3.5) input produced -0.99999356 instead of
   +0.99999988. The new device ELF uses FP32 squares, FP64 summation, hardware
   sqrt and checked completion. Both ordinary and strided paths are wired in
   v23; all four devices match five independent reference cases bit-for-bit.
   The old firmware routine is NOT replaced: opt in with
   `HETGPU_SIFIVE_RMS_EXACT=1`. Failure does not invoke CPU arithmetic fallback.

## Remaining Automatic-Offload Difference

With automatic operation migration enabled, same-prefix tracing finds the
first nonzero layer-output difference at decode call19/layer19, before the
sequence1/step20 greedy mismatch. Earlier layer boundaries in that call match.
The later logits differ enough to flip token314 to1817. Thus this failure is
not isolated to the output-head matmul.

Disabling opportunistic operation migration (`QWEN_OP_OFFLOAD=0`) leaves the
explicitly requested NGL layers on PACC. For ngl1, all 40 layer outputs and
output-normalization tensors match the independent CPU trace across 32 tokens,
and all argmax choices match. This is a qualified execution setting, NOT proof
that the automatic-offload operator has been repaired. The automatically moved
RoPE path uses integer-power/approximate-trigonometric arithmetic different
from CPU; attention/RoPE is a candidate, not a fully established sole cause.

At sequence1/step20, the full-vocabulary maximum logit error against the
same-Q8_1 oracle is 0.243027 with automatic migration and 0.0000076294 with
explicit placement. The latter also restores the expected argmax314.

The reference harness independently dequantizes GGUF weights, quantizes
activations using CUDA Q8_1 rules, and accumulates in FP64 on CPU. For ngl2 it
also applies that contract to quantized matmuls in layer39. It does not reuse
PACC output, force benchmark tokens, or act as a device fallback. Raw native
CPU and CUDA-contract comparisons are retained separately.

## Evidence And Reproduction

`trace-multiseq.cpp` is teacher-forced DIAGNOSTICS ONLY and always reports null
TPS. `qwen-q81-validation.cpp` samples actual greedy tokens and stops at the
first mismatch. `run-four-pacc-requests.py` requires four successful workers,
exact reference token equality, checked XM batching, device qualification and
shared-clock timing before publishing aggregate TPS.

The live phase sampler sometimes rejects a duplicate phase/sequence. Those
failed runs remain invalid. `--phase-profile qualification` performs startup
hardware/build qualification and uses synchronous completion plus numerical
checks during inference; it does not claim runtime phase accounting.

Build the diagnostic tools with `build-numerical-validation.sh NEW_DIRECTORY`.
Run ordinary/strided RMS checks with `run-rms-exact-checks.py --runtime RUNTIME
--prefix NEW_PREFIX`. `run-numerical-eval.py` creates an independent reference
before launching four strict workers. Use a new prefix for every run.

## Throughput Limits

All rows below use four devices, a synchronized worker start and actual greedy
sampling. Model loading is excluded; output TPS includes prefill. The two
prompts are replicated across workers; the 16-request case also repeats each
prompt twice within a worker. Separate random-column N=4/N=8 matrix tests pass
on all four devices after the long run, with unchanged output guards.

| NGL | Requests/card | Tokens/request | Correct output tokens | Aggregate decode TPS | Output TPS |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 32 | 256/256 | 2.3791 | 1.9064 |
| 2 | 2 | 32 | 256/256 | 1.0904 | 0.8767 |
| 1 | 4 | 64 | 1024/1024 | 4.3458 | 3.8669 |

All three pass XM batching and token gates with `QWEN_OP_OFFLOAD=0` and
`HETGPU_SIFIVE_RMS_EXACT=1`. NGL2 is only qualified here through 32 tokens;
NGL1 through 64. No general-model accuracy/perplexity claim is made.

The output head alone has 248320 x 2048 Q6_K weights, approximately 398 MiB.
With the current 512 KiB payload it is restaged in 809 submissions at N=2.
At N=4 this is 820 submissions. In the 64-token test, measured host elapsed
time inside the output-head staging/submission/wait/readback function accounts
for 91.64% of summed per-worker decode-call time. This is NOT pure XM compute
time and cannot identify interrupts versus DMA versus matrix execution alone.
XM processes small integer tiles and applies quantization scales outside the
matrix unit. This is genuine cross-request matrix work, but low-N batching,
repeated staging and tiny-tile submission are far from a peak-throughput
serving configuration. The competitor's 500 aggregate TPS remains a target,
not a comparable result without matched model, precision and concurrency.

Final measurements are recorded in `evidence-numerical-v23/results.json` and
the per-run `summary.json` files; failed runs retain null TPS.

Example rerun (no firmware load or host reboot):

```sh
ROOT=/srv/lanxin/qwen35-a3b
python3 "$ROOT/bin/run-numerical-eval.py" NEW_PREFIX \
  --ngl 1 --ubatch 4 --case "$ROOT/bin/numerical-long64.json" \
  --runtime "$ROOT/runtime-rne-exact-v23" \
  --harness "$ROOT/build-seq512-memory/bin/qwen-q81-validation-v3"
```

Use `--ngl 2`, `--ubatch 2` and `ngl-sweep-long.json` to repeat the qualified
NGL2 case. Both commands create the independent CPU reference before touching
PACC and reject a reused output prefix. The boot/DDR lease wrapper and 32 GiB
available-memory floor remain enabled.
