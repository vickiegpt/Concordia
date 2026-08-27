# Qwen3.5 Mixed-Quant IQ1_S GPU-Attention / AU250-BitLinear Evaluation Design

**Date:** 2026-08-27
**Status:** Approved
**Target checkout:** `/home/victoryang00/hetGPU`
**Supersedes:** `2026-08-26-qwen35-tq1-au250-hybrid-evaluation-design.md`

## Goal

Evaluate text generation with Qwen3.5-397B-A17B while CUDA executes attention,
all non-expert work, and routed experts whose weights are not IQ1_S. The Alveo
U250 executes every qualified IQ1_S routed-expert matrix product through the
existing exact affine-ternary backend. Compare this strict hybrid mode with a
CUDA-only reference using the same verified GGUF, binary, prompt, sampling,
context, threads, and CUDA layer placement.

The evaluation is correctness-first and fail-closed. Performance is reportable
only after the model-type audit, numerical qualification, physical XRT
evidence, deterministic token equality, and timing validation all pass.

## Corrected Model Contract

The fixed model artifact remains:

- base model: `Qwen/Qwen3.5-397B-A17B`;
- checkpoint: `nohurry/Qwen3.5-397B-A17B-TQ1_0-GGUF`;
- file: `Qwen3.5-397B-A17B-UD-TQ1_0.gguf`;
- byte count: 94,155,830,880;
- SHA-256: `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`;
- architecture: `qwen35moe`, 60 layers, 512 routed experts, and 10 selected
  experts per token.

The published `UD-TQ1_0` name is not a literal tensor-type contract. Inspection
of the verified GGUF records no `GGML_TYPE_TQ1_0` tensors. Its 180 routed-expert
weight tensors are:

| GGML type | Routed-expert tensors | Hybrid route |
| --- | ---: | --- |
| `IQ1_S` | 141 | strict AU250 eligibility |
| `IQ2_XXS` | 24 | native CUDA |
| `IQ3_S` | 4 | native CUDA |
| `MXFP4` | 11 | native CUDA |

All 141 IQ1_S tensors in this artifact are routed-expert weights. Therefore an
exact type-19 CUDA-kernel boundary selects only the intended expert work for
this fixed model. The runner must reproduce and record this tensor-type audit
before it starts either benchmark mode. A different count, type distribution,
model hash, or presence of a non-expert IQ1_S tensor aborts the evaluation.

The model is not rewritten or re-quantized. The existing Kimi checkpoint and
Kimi proof artifacts remain unchanged.

## Runtime Architecture

The revised Qwen path reuses three qualified repository components and adds a
Qwen-specific proof contract:

1. **Current llama.cpp CUDA runtime.** The pinned Qwen-capable llama.cpp build
   owns model loading, graph construction, attention, sampling, and every
   native CUDA operation.
2. **Existing NVIDIA IQ1_S interception.** The pre-native CUDA launch hook
   recognizes exact `ggml_type19` MMQ/MMVQ kernel symbols, validates the live
   launch signature and pointers, and captures the packed IQ1_S matrix and
   activation without executing the selected native kernel.
3. **Existing exact IQ1_S XRT backend.** The affine-ternary decomposition,
   D=1024 tiling, persistent four-CU pool, matrix/input/output/program four-BO
   ABI, existing assembler-produced 128-bit instructions, STALL completion,
   reconstruction, and CUDA copy-back remain authoritative.
4. **Revised Qwen runner and validator.** The runner audits the GGUF, switches
   only the IQ1_S route between modes, captures route/XRT/device evidence, and
   rejects incomplete or semantically different runs.

The direct `GGML_TYPE_TQ1_0` Qwen bridge remains compiled for its qualified
unit and live numerical tests but is disabled in both benchmark modes for this
model. No type-34 production-route claim is made.

## Routing and Data Flow

CUDA-only mode disables BitLinear disaggregation and executes the untouched
native llama.cpp path.

Hybrid mode enables:

```text
HETGPU_TMATMUL_BACKEND=xrt
HETGPU_BITNET_DISAGGREGATE=1
HETGPU_BITNET_DISAGG_STRICT=1
HETGPU_TMATMUL_HARDWARE_MATMUL=1
HETGPU_BITNET_CXL_KERNELS=mul_mat_q,mul_mat_vec_q
HETGPU_QWEN_TQ1_XRT=0
HETGPU_QWEN_TQ1_STRICT=0
```

For each CUDA launch, the route is:

```text
non-IQ1_S or non-matmul kernel -> native CUDA
exact IQ1_S MMQ/MMVQ kernel   -> validate and capture
captured IQ1_S operation      -> exact affine-ternary decomposition
ternary components            -> four-BO XRT waves across all four CUs
validated STALL completions   -> exact f32 reconstruction
complete finite output        -> CUDA copy-back, native launch suppressed
```

The type-19 kernel recognizer must exclude stream-fixup helpers and must accept
only the launch layouts already validated by `iq1s_tmatmul`. A selected launch
cannot return to native CUDA after capture begins.

## Numerical and Physical Contract

IQ1_S is affine ternary rather than a plain two-bit matrix. The existing
backend splits every IQ1_S group into its grid and delta ternary components,
preserves block scales and affine metadata, quantizes/captures the activation
with the established Q8 layout, and reconstructs the exact upstream result
from validated raw integer components.

Physical execution continues to use:

- four compute units from `MaxCores_370M.xclbin`;
- per-CU DDR memory groups and lane capacities from the versioned CU JSON;
- four BOs in matrix, input, output, program order;
- 16-byte little-endian assembler output for each 128-bit instruction;
- nonzero terminal STALL codes;
- unique request IDs, exact completion ownership, zero padding, and proven raw
  bounds;
- the persistent XRT pool shared with the TQ1 qualification path.

All four CUs must have positive validated submissions and completions during
the timed hybrid workload. Fewer active CUs is a failed qualification, not a
lower-throughput result.

## Failure Rules

Hybrid inference aborts on:

- model size, hash, architecture, or tensor-type audit mismatch;
- an IQ1_S tensor outside the routed-expert name set;
- malformed or unsupported IQ1_S MMQ/MMVQ kernel signature;
- invalid matrix, activation, output, shape, stride, batch, or allocation
  identity;
- missing or unwritable route/XRT evidence;
- packing, scale, arithmetic, or finite-output failure;
- XRT configuration, BO, synchronization, timeout, request-ID, CU ownership,
  STALL, padding, or raw-bound failure;
- CUDA synchronization or copy-back failure;
- native execution or fallback after an eligible IQ1_S launch is selected.

The runner never substitutes CPU experts, fewer GPU layers, a different model,
or a different binary. It retains failed proof directories and prints no
performance summary for an invalid run.

## Correctness and Evaluation Gates

Acceptance proceeds in this order:

1. GGUF audit proves the fixed hash, 180 routed-expert tensors, the exact
   `141/24/4/11` type distribution, zero TQ1_0 tensors, and no non-expert IQ1_S.
2. Existing IQ1_S decode, capture, decomposition, XRT, assembler, and routing
   tests pass without changing the Kimi path.
3. Existing TQ1_0 unit and live AU250 numerical qualifications remain passing;
   these qualify shared hardware but do not claim Qwen type-34 routing.
4. A current-llama one-token Qwen diagnostic produces at least one qualified
   IQ1_S route, physical XRT completions, all four active CUs, clean device
   health, and no selected fallback.
5. CUDA-only and strict-hybrid semantic prompts both return exact `OK` with
   identical token IDs.
6. The timed workload uses an exact 256-token prompt, 32 generated tokens,
   greedy seed-42 sampling, one warm-up, and five measurements per mode.
7. Generated token IDs match across all repetitions and modes; every eligible
   IQ1_S operation is handled, fallback/error counts are zero, and FPGA health
   remains clean.

The report includes model load time, prompt tokens/s, time to first token,
generation tokens/s, end-to-end latency, minimum, maximum, median, population
standard deviation, coefficient of variation, eligible-route coverage, tensor
type coverage, and per-CU work. It explicitly states that only the 141 IQ1_S
routed-expert tensors are eligible; it does not describe the whole mixed model
as pure TQ1_0 or claim that all 180 expert tensors ran on the FPGA.

## Out of Scope

- Re-quantizing or rewriting the selected Qwen GGUF.
- Offloading IQ2_XXS, IQ3_S, MXFP4, attention, linear attention, router, shared
  experts, normalization, KV/state updates, or sampling.
- Changing the FPGA bitstream, tmatmul ISA, four-BO order, or completion
  protocol.
- Relaxing exact token equality, strict no-fallback behavior, full CUDA model
  placement, or four-CU activity requirements.
- Changing or deleting the existing Kimi checkpoint, route, or proof contract.
- Vision inputs, speculative decoding, multi-user serving, or throughput
  batching.
