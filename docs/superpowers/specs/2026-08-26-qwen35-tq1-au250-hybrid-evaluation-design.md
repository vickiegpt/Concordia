# Qwen3.5 TQ1_0 GPU-Attention / AU250-BitLinear Evaluation Design

**Date:** 2026-08-26
**Status:** Approved
**Target checkout:** `/home/victoryang00/hetGPU`

## Goal

Evaluate text generation with Qwen3.5-397B-A17B using the selected TQ1_0
GGUF while attention and all non-routed-expert work execute on the NVIDIA GPU
and qualified routed-expert BitLinear matrix products execute on the Alveo
U250. Compare that strict hybrid mode against a CUDA-only reference using the
same model, binary base, prompt, sampling parameters, and layer placement.

The route is correctness-first and fail-closed. Performance is reportable only
after deterministic output, numerical checks, physical XRT submissions, and
complete timing evidence all pass.

## Fixed Model Contract

The evaluation uses:

- base model: `Qwen/Qwen3.5-397B-A17B`;
- architecture: `qwen35moe` / `Qwen3_5MoeForConditionalGeneration`;
- text dimensions: hidden size 4096, MoE intermediate size 1024, 60 layers,
  512 routed experts, and 10 selected experts per token;
- checkpoint: `nohurry/Qwen3.5-397B-A17B-TQ1_0-GGUF`;
- file: `Qwen3.5-397B-A17B-UD-TQ1_0.gguf`;
- expected file size: 94,155,830,880 bytes;
- expected LFS SHA-256:
  `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.

The benchmark is text-only and does not load a vision projection. The existing
Kimi checkpoint and Kimi/IQ1_S implementation remain intact.

## Runtime Architecture

The old BitNet llama.cpp fork cannot load Qwen3.5 and is not extended for this
evaluation. A separate, pinned current llama.cpp source overlay is used. The
initial inspected upstream revision is
`925e1179947ea0c0ebfb0032df18af3a729822be`; the build records and verifies the
final pinned revision rather than following a moving branch.

The Qwen path consists of four bounded components:

1. **Current llama.cpp overlay.** A repository-owned preparation script applies
   only the bridge needed to offer qualified TQ1_0 `mul_mat_id` operations to
   HetGPU. It produces a separate source and build directory and does not alter
   the Kimi overlay.
2. **TQ1_0 adapter.** A new HetGPU module validates tensor metadata, decodes
   TQ1_0 blocks, plans D=1024 tiles, packs matrices and activations, reconstructs
   scaled results, and copies complete outputs back to the graph.
3. **Persistent XRT executor.** The existing Rust four-CU executor and four-BO
   contract remain the physical submission implementation.
4. **Qwen evaluation wrapper.** A new runner owns model verification, platform
   preflight, CUDA-only and strict-hybrid modes, evidence capture, and final
   validation.

The adapter opens a separate verified, read-only mapping of the GGUF tensor
regions, so routed expert weights remain host-addressable regardless of their
llama.cpp backend placement and never require a GPU-to-host weight copy. The
llama graph scheduler keeps attention and the remaining graph on CUDA and
transfers only the completed expert result across the established backend
boundary. The bridge has a three-state result:

- `not handled`: the operation is outside the qualified Qwen/TQ1_0 boundary and
  follows the normal llama.cpp backend;
- `handled`: the complete AU250 result has been validated and installed;
- `error`: a selected operation failed and the strict run aborts.

An eligible operation may never return `not handled` after strict selection.

## Routing Boundary

Only a routed-expert `GGML_OP_MUL_MAT_ID` is eligible, and only when all of the
following hold:

- model architecture is `qwen35moe`;
- source weights are exactly `GGML_TYPE_TQ1_0`;
- tensor role is a routed-expert gate/up or down projection;
- expert selection metadata is present and in bounds;
- dimensions, strides, and storage spans match the validated contiguous
  contract;
- the activation and destination types are supported by the adapter;
- batch and token geometry can be represented by the AU250 lane plan.

Embeddings, the router, shared experts, normalization, linear attention, full
attention, Q/K/V operations, RoPE, state/KV updates, sampling, and unqualified
matrix products remain native llama.cpp operations on CUDA as configured.

## TQ1_0 Numerical Contract

A TQ1_0 block represents 256 weights with 48 base-3 payload bytes, four
high-tail bytes, and one FP16 scale. Decoding follows the pinned upstream
`dequantize_row_tq1_0` ordering exactly. Each logical weight is one trit in
`{-1, 0, +1}` multiplied by the block scale.

The AU250 matrix encoding is:

| Trit | Two-bit code |
| ---: | ---: |
| -1 | `0b11` |
| 0 | `0b00` |
| +1 | `0b01` |

Four trits occupy each matrix byte. A physical matrix BO always contains one
zero-padded 1024x1024 tile and therefore occupies 262,144 bytes.

### Tiling and Lane Multiplexing

Output rows and K columns are tiled in chunks of 1024. One K tile contains four
independently scaled 256-value TQ1_0 blocks per output row. These scales cannot
be applied after a combined 1024-value dot product, so the adapter preserves
the four group results independently.

One full 256-element ternary-by-int8 dot can reach `+32768`, which is one past
the signed-16-bit output maximum. The adapter therefore divides every TQ1_0
block into two 128-element half-groups. Each lane input is zero except for one
half-group. Its exact raw-dot range is:

```text
-16384 <= half_dot <= 16384
```

Eight lanes cover the eight half-groups in one 1024-element K tile for one
logical token, and unused lanes remain zero. A nine-lane CU evaluates the tile
in one physical job. The six-lane CU evaluates it as two tagged physical jobs
covering six and two half-groups. All jobs share the same packed matrix tile,
and reconstruction requires both halves of all four blocks before accepting a
logical tile.

The FPGA returns one signed raw dot per output row and lane. The bound for a
256-element ternary-by-int8 dot is checked before accepting the result. The
adapter reconstructs each block contribution as:

```text
(half_dot_0 + half_dot_1) * tq1_block_scale * activation_block_scale
```

Contributions accumulate in deterministic expert, K-tile, and group order into
an f32 destination. Every planned group must be present exactly once and every
final value must be finite before the result is exposed to llama.cpp.

### Matrix Cache

Packed tiles are cached by model identity, tensor identity, allocation
generation, content hash, expert index, row tile, and K tile. The cache is
byte-bounded with LRU eviction. Activation and output BOs are refreshed for
every submission. A cached tile may not be reused when any identity field
differs.

## Four-BO ABI

Each CU retains the established BO ordering:

1. matrix BO containing a packed 1024x1024 ternary tile;
2. input BO containing signed fixed-point lane activations;
3. output BO containing signed raw dot products;
4. program BO containing assembler-produced 128-bit instructions.

The existing assembler remains the sole instruction encoder. The program is
the existing sequence:

```text
ldv v0, PARAM_INPUT
tmatmul_import v0
tmatmul_go PARAM_MATRIX
tmatmul_export v1
sv v1, PARAM_OUTPUT
stall
```

BO device addresses are bound through the existing labels. The runtime does
not introduce a second instruction format, embed instructions in a different
BO, or reorder the four arguments.

## Persistent Four-CU Execution

The application image is opened once per benchmark process. Each validated CU
owns one reusable four-BO set in its connected bank and one assembled program.
The scheduler starts compatible jobs across all four CUs, permits out-of-order
physical completion, and restores deterministic logical accumulation with
request IDs.

Every completion must have a known unique request ID, the expected CU, the
expected output byte count, a valid terminal STALL code, zero padding, and raw
values inside the proven numerical bound. A timeout or malformed completion
poisons the CU and aborts strict inference after safe quiescence.

## Failure Rules

The hybrid run fails closed on:

- model, revision, file-size, or hash mismatch;
- unsupported TQ1_0 layout, shape, stride, activation, or destination;
- invalid expert IDs or duplicated routing entries;
- trit decode, tiling, packing, scale, or arithmetic overflow;
- XRT configuration, BO, synchronization, timeout, or STALL failure;
- missing, duplicated, misrouted, or out-of-range completion data;
- non-finite reconstruction or failed CUDA/backend copy-back;
- unwritable required evidence logs;
- any native fallback of an operation already classified as eligible.

Failure never exposes a partially reconstructed destination and never reduces
silently to CUDA or fewer CUs.

## Correctness Gates

Implementation is accepted in this order:

1. **TQ1_0 unit tests:** decode payload and tail trits, FP16 scales, malformed
   blocks, and randomized rows against pinned upstream llama.cpp.
2. **Adapter tests:** matrix encoding, padding, tile planning, lane packing,
   raw bounds, deterministic reconstruction, cache identity, and overflow.
3. **Executor tests:** four-BO ordering, assembler output, multi-CU request
   ownership, reordered completion, timeout, poisoning, and strict no-fallback.
4. **Existing regression:** current XRT, assembler, IQ1_S, Kimi routing, and
   proof-validator tests remain passing.
5. **Live single-tile gate:** a TQ1_0 matrix/activation fixture matches the
   upstream CPU dot-product reference through the AU250.
6. **Live tiled gate:** row and K dimensions span multiple D=1024 tiles and use
   all four CUs; every reconstructed value matches the qualified reference.
7. **One-token Qwen gate:** deterministic CUDA-only and hybrid token IDs match
   and every eligible expert operation has validated AU250 evidence.
8. **Short semantic gate:** the prompt requesting exactly `OK` produces the
   same valid response in both modes.
9. **Timed evaluation:** runs only after every preceding gate passes.

## A/B Performance Protocol

CUDA-only and hybrid modes use the same Qwen-capable binary, verified GGUF,
prompt, context, CPU threads, full CUDA model placement, and deterministic
sampling. The same binary is used with strict AU250 routing disabled or
enabled, avoiding a comparison between unrelated llama.cpp builds. Full CUDA
placement must pass memory preflight in both modes; otherwise the A/B
evaluation stops rather than substituting a CPU-expert baseline.

### Semantic Workload

- batch size 1;
- context size 512;
- deterministic greedy sampling;
- prompt requests the exact answer `OK`;
- CUDA-only and hybrid output token IDs must match.

### Timed Workload

- fixed 256-token text prompt stored with the proof;
- 32 generated tokens;
- batch size 1 and context size 512;
- deterministic greedy sampling;
- one warm-up followed by five measured repetitions;
- a fresh process for each mode, with the model retained across repetitions;
- identical generated token IDs across modes.

Model load time is recorded separately and excluded from prompt/generation
throughput. Placement may not be tuned independently per mode.

## Evidence and Reported Metrics

Each proof directory records:

- model repository revision, file size, and SHA-256;
- llama.cpp revision and binary hash;
- HetGPU revision and dirty-tree manifest;
- xclbin path and hash;
- command line and an allowlisted environment manifest;
- prompt text, prompt token IDs, generated token IDs, and decoded output;
- process exit status and llama timing output;
- route decisions and eligible/handled/error counts;
- per-CU submissions, completions, accelerator cycles, and STALL codes;
- matrix, input, output, and program bytes transferred;
- host decode/packing, XRT, reconstruction, and copy-back times;
- GPU memory/utilization and FPGA health before and after the run.

The final A/B report includes model load time, prompt tokens/s, time to first
token, generation tokens/s, end-to-end latency, minimum, maximum, median, and
dispersion across the five repetitions. It also reports the fraction of
eligible expert operations offloaded and the per-CU work distribution.

No throughput result is valid unless the process exits cleanly, deterministic
outputs match, eligible-route coverage is 100%, all four CUs have nonzero
validated completions, device health remains clean, and every required timing
field is present. A failed qualification is reported as a failed run, not as a
TPS measurement.

## Out of Scope

- Changing or deleting the existing Kimi checkpoint.
- Re-quantizing Qwen to IQ1_S or another format.
- Offloading attention, linear attention, the router, or shared experts.
- Changing the FPGA bitstream, ternary core, four-BO order, or instruction ISA.
- Vision inputs or multimodal projection evaluation.
- Multi-user serving, speculative decoding, or throughput batching beyond the
  lane multiplexing required for the selected single-request workload.
