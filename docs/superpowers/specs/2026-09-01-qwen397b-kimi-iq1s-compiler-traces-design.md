# Qwen3.5-397B Kimi IQ1_S Compatibility and Compiler Traces

**Date:** 2026-09-01
**Status:** Approved for spec review
**Target checkout:** `/home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826`

## Goal

Make the existing Kimi 2.6 IQ1_S CUDA-launch interception path compatible with
the local Qwen3.5-397B-A17B TQ1_0 GGUF. Fix the current Qwen hybrid-run failure,
make Qwen weight loading and reuse explicit, and add a compiler trace builder
that lowers every qualified CUDA launch through the repository's
`AlgorithmTree` into the exact tmatmul program executed by the Alveo U250.

The final system keeps attention and every non-qualified operation on the
NVIDIA RTX PRO 6000, sends qualified IQ1_S routed-expert work to all four U250
compute units, and proves correctness against a CUDA-only run. The performance
gate is at least 15 aggregate generated tokens per second under the fixed
continuous-batching workload.

## Scope and fixed contract

The model and runtime contract is:

- model: `Qwen3.5-397B-A17B`;
- checkpoint: `/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`;
- expected model size: `94,155,830,880` bytes;
- expected SHA-256:
  `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`;
- inference frontend: the pinned Qwen-capable llama.cpp overlay built and run
  through the local BitNet environment;
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition;
- FPGA: Xilinx Alveo U250 with the existing `MaxCores_370M.xclbin` four-CU
  contract;
- model context limit: 262,144 tokens;
- benchmark context: 512 tokens;
- benchmark load: 64 requests, no more than 16 active sequences, and exactly
  32 generated tokens per completed request;
- decoding: greedy and deterministic;
- throughput: validated generated tokens divided by first measured enqueue to
  last measured completion; prompt tokens, load time, warm-up, and failed
  requests are excluded.

The local GGUF is a mixed-quant preset rather than a file containing literal
`GGML_TYPE_TQ1_0` tensors. Its 180 routed-expert tensors are exactly:

- 141 `IQ1_S` tensors, eligible for U250;
- 24 `IQ2_XXS` tensors, GPU-native;
- 4 `IQ3_S` tensors, GPU-native;
- 11 `MXFP4` tensors, GPU-native.

The implementation does not convert the 39 non-IQ1_S tensors. Embeddings,
linear and full attention, DeltaNet, KV and recurrent state, normalization,
expert routing, sampling, and every non-qualified matrix operation remain on
the GPU.

## Existing implementation boundary

This is a compatibility extension of the Kimi route, not a new accelerator
backend. It reuses:

- `zluda/src/impl/function.rs` for CUDA launch interception and launch-argument
  extraction;
- `zluda/src/impl/bitnet_disagg.rs` for explicit GPU/U250 routing;
- `zluda/src/impl/iq1s_tmatmul.rs` for IQ1_S decoding, Q8 activation capture,
  tiling, scale reconstruction, and component identity;
- `zluda/src/impl/iq1s_xrt.rs` for tile planning, host packed-matrix caching,
  four-CU wave construction, and completion reconstruction;
- `zluda/src/impl/xrt_tmatmul.rs` for the persistent four-CU/four-BO XRT
  executor and assembler integration;
- `ptx/src/pass/tmatmul_algorithm_tree.rs` for dependency construction,
  scheduling, register allocation, assembly generation, and duration
  calculation;
- the current Qwen runner, evaluator, model auditor, and proof validators.

Standalone U250 numerical evidence already proves a single tile and a tiled
four-CU fixture with small finite error. It does not prove a Qwen hybrid run or
throughput. The latest end-to-end attempt failed when
`iq1s_tmatmul.rs` selected the hard-coded Kimi `libggml.so` path instead of the
Qwen build's library. No end-to-end or 15 tok/s claim is inherited from that
attempt.

## Runtime architecture

The runtime remains CUDA-launch interception:

```text
Qwen llama.cpp / BitNet environment
  -> CUDA launch
  -> exact launch-signature and argument decoder
       -> GPU-native: attention, state, routing, 39 non-IQ1_S expert tensors
       -> U250 candidate: qualified IQ1_S expert MMVQ/MMQ launch
            -> handwritten planner, or compiler AlgorithmTree planner
            -> shared host/device weight cache
            -> shared assembler and encoded program cache
            -> shared persistent four-CU XRT executor
            -> deterministic IQ1_S scale reconstruction
            -> atomic CUDA output publication
```

The launch is eligible only if every one of these conditions holds:

1. Qwen strict IQ1_S mode is explicitly enabled.
2. The symbol is a supported `mul_mat_vec_q` or `mul_mat_q` specialization for
   `ggml_type19` and is not a stream-K fixup or other helper kernel.
3. The decoded ABI variant, dimensions, strides, active batch width, pointers,
   and CUDA stream match a supported Kimi IQ1_S capture contract.
4. The operation belongs to a Qwen routed-expert gate, up, or down projection.
5. Its tensor identity belongs to the verified set of 141 IQ1_S expert
   tensors.
6. Its row, K, batch, and storage extents pass checked arithmetic and bounded
   allocation validation.

GPU marker rules take precedence before eligibility. After a launch is marked
eligible, returning to native CUDA is an error.

## Qwen compatibility repair

The strict Qwen runner must remove reliance on `DEFAULT_LIBGGML`. Before model
startup it will:

1. read the Qwen build manifest;
2. resolve the manifest's `libggml.so` path;
3. require an existing regular file under the Qwen build root;
4. canonicalize the path and reject symlink or path-escape mismatches;
5. verify the recorded file SHA-256;
6. open the library and resolve the IQ1_S oracle symbols used by the existing
   Kimi decoder;
7. export the verified path to the server process;
8. record the path and hash in the proof environment.

Strict Qwen mode has no fallback library. Failure of any step aborts before
loading the 94 GB model. The default Kimi path remains available only to the
existing Kimi launcher so its behavior is preserved.

The Qwen launch decoder extends the already-tested Kimi signatures rather than
adding a generic mangled-name match. Each supported CUDA 13 MMVQ/MMQ template
variant gets an explicit decoder and shape contract. Unknown template
instantiations remain GPU-native unless a future contract qualifies them.

## Shared weight loading and cache

Weight loading uses two bounded cache levels under one identity contract.

### Host packed-tile cache

The existing `PackedMatrixCache` remains responsible for decoded and packed
IQ1_S tiles. Its key is extended or verified to bind all of:

- model SHA-256;
- tensor name, layer, projection role, and expert identity;
- CUDA matrix pointer and allocation generation;
- logical shape and physical strides;
- quantization and component kind;
- row tile, K tile, and scale-group coordinates;
- content hash.

A zero or missing content hash is not accepted for strict Qwen qualification.
The cache remains byte-bounded by `HETGPU_XRT_MATRIX_CACHE_BYTES`, uses LRU
eviction, and never evicts an entry held by an in-flight request.

### U250-resident tile cache

The current XRT pool copies each matrix into one reusable per-CU matrix BO.
That behavior cannot establish working weight reuse and adds avoidable PCIe
traffic. The shared executor will add a bank-aware resident matrix cache:

- a resident entry owns an XRT matrix BO in the target CU's connected bank;
- the entry is keyed by the same immutable packed-tile identity plus memory
  group;
- first use allocates, writes, synchronizes, and records the BO address;
- a cache hit reuses that device address without another matrix transfer;
- entries are reference-counted while requests are in flight;
- eviction is per bank, LRU, byte-bounded, and permitted only for quiescent
  entries;
- allocation, write, sync, address, or eviction failure aborts strict mode;
- all BO handles remain owned by the persistent XRT pool and are released only
  after confirmed quiescence.

Evidence distinguishes host-pack hits, resident-BO hits, bytes packed, bytes
transferred, and eviction counts. Measured throughput is valid only if the
warm-up created at least one resident entry and measured requests demonstrate
nonzero resident hits.

## Handwritten and compiler trace builders

Two trace modes are selected explicitly for evaluation:

- `handwritten`: the current Kimi-compatible planner is the correctness and
  regression reference;
- `compiler`: the qualified launch is lowered into an `AlgorithmTree` and the
  resulting program is the program executed by XRT.

Both modes use the same capture, weight identity, packed and resident caches,
assembler, executor, completion validation, and reconstruction.

### Compiler input

The compiler input is a validated semantic launch contract containing:

- normalized kernel family and ABI version;
- rows, K, active batch width, and physical strides;
- expert/projection identity;
- IQ1_S component and scale-group plan;
- row and K tile coordinates;
- matrix, input, and output address labels;
- U250 lane capacities and vector-register count.

The compiler never infers these values from pointer size or an unqualified
symbol substring.

### Algorithm-tree lowering

For each actual launch, the compiler creates abstract vectors and operations
for the real tile and component plan. It lowers each tmatmul operation into
`TMatmulImport`, one or more `TMatmulGo` operations, and the required
`TMatmulExport` and accumulation operations. It then uses the existing tree to
compute dependencies, instruction ordering, register assignments, swaps, and
assembly tokens.

The emitted program contains real address labels and terminates in `stall`.
The existing assembler remains the only encoder of the 128-bit instruction
format. The compiler does not fabricate instruction counts or maintain a
second binary encoding.

Compiler arithmetic uses checked 64-bit or wider intermediates for token,
batch, expert, tile, byte-offset, and trace-count calculations. Tests exercise
the model limit of 262,144 tokens, but the compiler emits work only for the
actual active launch. A 512-token benchmark therefore does not manufacture a
262,144-token trace.

### Program cache and execution binding

The persistent pool currently installs one fixed program per CU. Compiler mode
requires a shared program cache keyed by:

- algorithm-tree semantic hash;
- ordered assembly hash;
- bound label addresses;
- vector-register count;
- CU memory group and lane capacity.

Each cache entry owns the encoded program BO for that CU and records its device
address and byte count. The job submitted to XRT names the resident matrix and
program handles it must execute. The executor programs the MM2S address and
length from those handles, validates complete 16-byte instruction words, and
records the same program hash in its completion evidence. A trace that is only
logged but not bound to the physical program BO fails qualification.

The handwritten and compiler programs may differ in safe instruction ordering.
They are required to have the same semantic tile/component coverage and
numerically equivalent results, not byte-for-byte identical assembly.

## Completion and output publication

The existing four-CU wave scheduler remains the physical execution owner. It
may complete requests out of order, but every completion must match a unique
request ID, expected CU, resident matrix identity, program hash, byte count,
and nonzero terminal STALL code.

Reconstruction preserves the existing IQ1_S scale semantics and deterministic
component/K-tile accumulation order. Raw bounds, finite scales, finite final
values, complete component coverage, and destination size are checked before
publication. The runtime performs one complete CUDA copy for the finished
destination. Partial results are never exposed.

## Fail-closed behavior

The hybrid process aborts on any of the following:

- model path, size, hash, architecture, or tensor audit mismatch;
- Qwen build-manifest, `libggml.so`, symbol, or hash mismatch;
- ambiguous or unsupported ABI after an operation is selected as eligible;
- invalid pointer, shape, stride, batch width, expert ID, or allocation
  generation;
- missing content identity for an eligible weight;
- IQ1_S decode, scale, packing, tile, or reconstruction failure;
- host or device cache accounting, allocation, transfer, sync, or eviction
  failure;
- algorithm-tree dependency, scheduling, register allocation, assembly, or
  encoding failure;
- trace hash differing from the program bound to XRT;
- duplicate, missing, stale, misrouted, timed-out, or malformed completion;
- zero STALL, out-of-range raw result, non-finite output, or CUDA publication
  failure;
- required evidence that cannot be written or validated;
- any GPU fallback of an already eligible launch.

An unqualified operation remains GPU-native and is recorded separately. The 39
non-IQ1_S routed-expert tensors are expected GPU operations, not fallbacks.

## Evidence contract

Each eligible launch writes a route record and an execution record. Repeated
traces may store assembly and encoded words once by hash, but every launch must
reference the exact stored trace. Evidence includes:

- kernel name and decoded ABI family;
- model, tensor, layer, expert, and projection identity;
- shapes, strides, active batch, and tile/component counts;
- trace mode;
- algorithm-tree semantic hash and ordered operation summary;
- assembly hash, full ordered assembly, encoded program hash, encoded byte
  count, and program BO address;
- host and resident cache hit/miss and byte counters;
- per-CU submissions and completions, request IDs, STALL codes, and timing;
- raw result range, reconstruction timing, comparison status, and error
  metrics.

The final proof validator independently requires `handled == eligible`, zero
fallbacks, zero errors, nonzero physical XRT work, activity on all four CUs,
resident-cache reuse, complete trace-to-program binding, healthy firewall
evidence, and unchanged source/model/build identities.

## Correctness and test plan

Validation proceeds in this order:

1. Qwen runner tests prove explicit verified `libggml.so` propagation and
   rejection of the missing, wrong, symlink-escaped, unhashed, or symbol-missing
   library.
2. Launch-decoder tests cover every supported CUDA 13 IQ1_S MMVQ/MMQ template,
   the observed Qwen shapes, batch variants, attention precedence, non-IQ1_S
   GPU routing, and strict rejection after eligibility.
3. Cache tests cover every identity field, host and device hits, per-bank
   capacity, in-flight eviction protection, stale allocation generations,
   transfer accounting, and quiescent cleanup.
4. Algorithm-tree tests compare actual semantic coverage with the handwritten
   planner, verify dependency ordering and register allocation, assemble every
   emitted trace, reject incomplete 128-bit programs, bind program hashes to
   jobs, and exercise checked scheduling at 262,144 tokens plus overflow.
5. Existing Kimi 2.6 IQ1_S, assembler, XRT, routing, and proof-validator tests
   remain passing.
6. Live single-tile and tiled IQ1_S fixtures pass through all four CUs for both
   trace modes. The reference and U250 outputs are finite and satisfy
   `abs_error <= 1e-4 + 1e-3 * abs(reference)` elementwise.
7. One-token Qwen handwritten and compiler hybrid runs both complete with
   strict routing, physical XRT evidence, and the same greedy token as CUDA.
8. The fixed continuous-batch workload runs in CUDA-only, handwritten-hybrid,
   and compiler-hybrid modes. Every one of the 64 requests completes with
   exactly 32 generated token IDs. Token IDs are identical by request across
   all three modes.
9. Sampled FFN outputs from both hybrid modes are finite and satisfy
   `atol=1e-4`, `rtol=1e-3` against the CUDA TQ1 reference.
10. After one unmeasured warm-up, three measured passes are run for each hybrid
    mode. Every pass reaches at least 15 aggregate generated tok/s. Queue,
    service, first-token, end-to-end, and single-stream decode latency are
    reported separately.

The current U250 PCIe link is Gen3 x4 rather than its x16 capability. This is a
performance risk and must be recorded in the proof. The design does not claim
15 tok/s until the measured acceptance gate passes.

## Deliverables

The implementation will produce:

- the Qwen compatibility and strict `libggml.so` repair;
- Qwen launch-contract decoding over the existing Kimi IQ1_S path;
- one shared host/device weight cache;
- compiler AlgorithmTree lowering and real trace/program evidence;
- program-cache support in the four-CU executor;
- updated Qwen continuous-batch runner and fail-closed validator;
- regression, compiler, cache, and live-hardware tests;
- immutable CUDA/handwritten/compiler proof bundles;
- an evidence-backed evaluation report containing the measured throughput and
  correctness results.

## Non-goals

- Converting IQ2_XXS, IQ3_S, MXFP4, embeddings, or attention weights to
  ternary format.
- Replacing llama.cpp/BitNet with a new model runtime.
- Building a native GGML U250 backend.
- Moving attention, DeltaNet, KV/recurrent state, routing, or sampling to the
  FPGA.
- Claiming full 262,144-token runtime allocation or throughput; compiler
  arithmetic and trace planning support that limit while the fixed performance
  workload uses a 512-token context.
- Claiming 15 tok/s from projections, simulations, standalone tiles, or an
  incomplete run.
