# Qwen3.5-397B U250 Layer-Persistent IQ1_S FFN

**Date:** 2026-09-02
**Status:** Conversationally approved; pending written-spec review
**Software checkout:** `/home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826`
**RTL checkout:** `/home/victoryang00/hetGPU/ternary_matmul`

## Goal

Replace the current synchronous expert/component U250 path with a true
layer-level, multi-matrix persistent implementation for Qwen3.5-397B-A17B.
The implementation must preload all 141 IQ1_S routed-expert tensors into the
four U250 DDR banks, submit gate/up/down work in layer transactions, batch
activation and result DMA, and execute real handwritten and AlgorithmTree
compiler traces on four persistent compute units.

Attention, routing, recurrent and KV state, normalization, embeddings,
sampling, and all non-IQ1_S projections remain on the GPU. CUDA launch
interception remains the source of the live matrix-operation parameters. A
small llama.cpp sideband ABI provides unambiguous layer boundaries and routing
metadata.

The final result is a fail-closed correctness and throughput proof for the
fixed workload. A throughput value is reportable only after every request
finishes and all correctness, routing, residency, and hardware-evidence gates
pass.

## Motivation and measured baseline

The current implementation calls `iq1s_xrt::execute_captured` separately for
every captured component, waits for terminal `STALL`, copies its output, and
then advances to the next component. The latest partial live run produced:

- 4,801 logical component records;
- 153,632 physical submissions and completions, 38,408 on each CU;
- 22.671875 GiB of cumulative matrix DMA;
- a 39.6 percent resident-cache hit rate;
- 1.080 ms dispatch-to-stall p50 and 1.143 ms p95;
- no generated output token after 392.512 seconds.

That run has no reportable end-to-end TPS. The fresh forced-flash-attention
CUDA baseline completes the fixed workload at 64.1529631179 aggregate tok/s
mean. The target for each completed hybrid measurement pass remains at least
15 aggregate generated tok/s.

Removing host logging or adding asynchronous component threads cannot close
the measured gap. The new design therefore removes component submission as a
host-visible unit and consumes raw IQ1_S weights in RTL.

## Fixed model and workload contract

- Model: `Qwen3.5-397B-A17B`.
- GGUF: `/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`.
- GGUF byte count: exactly `94,155,830,880`.
- GGUF SHA-256:
  `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.
- Model context limit: 262,144 tokens.
- Benchmark context allocation: 512 tokens per active request.
- Workload: 64 requests, at most 16 active requests, exactly 32 generated
  tokens per completed request, and greedy deterministic decoding.
- Routed-expert tensors: 141 IQ1_S on U250; 24 IQ2_XXS, 4 IQ3_S, and 11 MXFP4
  on GPU.
- IQ1_S roles: 50 gate tensors, 50 up tensors, and 41 down tensors.
- IQ1_S storage: exactly 59,139,686,400 bytes, or 55.078125 GiB, before arena
  alignment and manifest metadata.
- U250 topology: three big CUs and one small CU connected to memory groups
  0/3/2/1, respectively.
- Correctness: identical greedy token IDs and sampled FFN comparison within
  `atol=1e-4`, `rtol=1e-3`.
- Compilation/build parallelism: no command may use more than 32 threads.

## Non-goals

This work does not move attention, DeltaNet, KV or recurrent state, router,
normalization, embeddings, sampling, or the 39 non-IQ1_S routed-expert tensors
to U250. It does not increase the fixed active-request limit to conceal poor
latency. It does not report modeled, extrapolated, partial-run, prompt, or
warm-up throughput as end-to-end TPS.

## End-to-end architecture

```text
GPU attention / router / non-IQ1_S projections
                   |
          layer_begin + route metadata
                   |
          CUDA launch interception
                   |
          LayerTransactionBuilder
             |               |
       Phase A: gate/up   Phase B: down
             |               |
       handwritten or AlgorithmTree compiler
                   |
     four persistent bank-local command rings
       |             |             |             |
      CU0           CU1           CU2           CU3
     DDR0           DDR3          DDR2          DDR1
       |             |             |             |
     batched completion and result-range DMA
                   |
GPU SiLU, multiply, mixed-projection merge, expert weighting, residual
                   |
             GPU attention continues
```

Each eligible layer is one logical transaction with at most two U250/GPU
phase fences. Phase A contains all IQ1_S gate and up work for that layer. The
GPU performs the activation and multiplication after Phase A and concurrently
executes any non-IQ1_S projection whose dependencies permit it. Phase B
contains IQ1_S down work. The GPU merges non-IQ1_S and IQ1_S results, applies
route weights and residuals, and continues the model.

The registered tensor manifest determines which roles are expected in each
layer. A layer is valid when every expected IQ1_S role appears exactly once in
the captured launch set; a role whose registered tensor is non-IQ1_S remains
GPU-native and is not expected by the U250 transaction.

## Versioned sideband ABI

The Qwen llama.cpp overlay adds a version-2 IQ1_S layer API while preserving
the existing tensor-registration API:

```c
int hetgpu_iq1s_layer_begin_v2(
    uint32_t abi_version,
    uint32_t layer_id,
    uint64_t transaction_id,
    uint32_t batch_count,
    void *cuda_stream);

int hetgpu_iq1s_layer_set_routes_v2(
    uint32_t abi_version,
    uint64_t transaction_id,
    const uint32_t *token_ids,
    const int32_t *expert_ids,
    const float *route_weights,
    uint32_t top_k);

int hetgpu_iq1s_layer_phase_commit_v2(
    uint32_t abi_version,
    uint64_t transaction_id,
    uint32_t phase);

int hetgpu_iq1s_layer_commit_v2(
    uint32_t abi_version,
    uint64_t transaction_id);

int hetgpu_iq1s_layer_abort_v2(
    uint32_t abi_version,
    uint64_t transaction_id,
    uint32_t reason);
```

The sideband ABI supplies identity, ordering, and layer boundaries. CUDA
launch interception remains authoritative for the live matrix, activation,
output, shape, stride, grid, and stream values. The layer coordinator rejects
a sideband record that does not match the intercepted stream and registered
tensor identity.

`transaction_id` is monotonically increasing within a model session and may
not be reused. `batch_count` is in `1..=16`, and `top_k` must match the audited
Qwen model. The three route pointers are CUDA device addresses on the stream
bound by `layer_begin`. `layer_set_routes` enqueues one coalesced device-to-host
copy into coordinator-owned staging, records a CUDA event, and returns after
enqueue. `layer_phase_commit(PHASE_A)` waits for that event before compiling
the first phase. The intercepted MoE launch must expose the same expert-ID
address and extent; otherwise strict execution aborts.

`layer_phase_commit(PHASE_A)` is called after the eligible gate/up launches.
It validates and submits Phase A, waits for its four-CU completion, and
publishes gate/up results onto the bound CUDA stream before returning. The GPU
can then execute SiLU and multiplication and launch the down projection.
`layer_commit` validates and submits Phase B when the layer has an IQ1_S down
role, waits for completion, and closes the transaction. For a layer without an
IQ1_S down role, `layer_commit` verifies the GPU-native down path and closes the
already completed Phase A transaction without a second U250 submission.

The transaction state machine is:

```text
EMPTY -> OPEN -> ROUTES_SET -> PHASE_A_CAPTURE
                                  |
                         PHASE_A_COMMITTED
                                  |
                           PHASE_A_DONE
                            /          \
                PHASE_B_CAPTURE     GPU_DOWN_VERIFIED
                       |                  |
                PHASE_B_COMMITTED         |
                       \__________________/
                                  |
                                CLOSED

Any nonterminal state -> ABORTED
```

Only `PHASE_A_COMMITTED` and `PHASE_B_COMMITTED` transactions can be submitted
to U250. A missing role, duplicate role, duplicate transaction ID, stream
mismatch, invalid expert, unexpected launch, or phase/layer commit with
incomplete capture moves the transaction to `ABORTED` and terminates strict
execution.

## Host software components

### LayerCoordinator

`LayerCoordinator` owns the transaction state machine. It joins sideband
metadata with intercepted launches and the immutable IQ1_S weight registry.
It produces one normalized `LayerTransaction` containing phase dependencies,
expert-major assignments, input/output spans, and the exact expected role set.

### LayerTraceCompiler

`LayerTraceCompiler` has two explicitly selected builders:

- `handwritten`, retained as the regression and semantic reference;
- `compiler`, which constructs and schedules the actual operations through
  `ptx::pass::tmatmul_algorithm_tree::AlgorithmTree`.

Both builders consume the same normalized transaction and emit the same
descriptor coverage, relocation schema, and semantic trace manifest. They
share the repository assembler and persistent executor. They may use different
safe instruction orderings, but must cover identical token, expert, role,
row-shard, and IQ1_S block coordinates.

### WeightArenaManager

`WeightArenaManager` verifies the fixed GGUF identity, reads the exact IQ1_S
tensor extents, row-shards every expert, constructs the four bank images, and
preloads them before measured inference. It owns all arena BOs for the model
session and exports an immutable address manifest to the compiler and RTL.

### PersistentXrtExecutor

`PersistentXrtExecutor` starts exactly one user-managed kernel on each of the
four CUs. It owns bank-local command/completion rings, program storage,
activation slabs, result slabs, doorbells, and fault registers. Normal layer
execution never creates a new XRT kernel run and never rewrites MM2S registers
for individual components.

### ProofLedger

`ProofLedger` writes append-only records for layer state transitions, routes,
resolved traces, DMA ranges, CU commands, completions, hardware faults,
comparisons, and timings. Evidence writes are buffered by layer and are not on
the per-component datapath.

## U250 weight layout and loading

Raw IQ1_S is the only resident representation. The implementation must not
materialize the expanded ternary component representation in host or U250 DDR
during measured execution.

Every tensor/expert matrix is divided by output rows across all four banks.
This row sharding makes every CU contribute to every selected expert and
avoids an expert-ID hotspot on a single CU. Shard boundaries obey IQ1_S row and
decoder alignment. The initial build uses equal-capacity shards; the live
microbenchmark may shift rows from the small CU to the three big CUs as long
as every bank remains within its verified capacity. The chosen ratio is fixed
before measured E2E passes.

Arena images are split into bounded superblocks rather than one BO per expert.
The superblock size is selected during implementation from sizes supported by
the installed XRT and U250 shell. Each manifest entry binds:

- model SHA-256;
- tensor name, layer, role, and expert ID;
- logical shape, strides, and IQ1_S block format;
- row start and row count;
- bank and superblock identity;
- byte offset and byte length;
- source and resident content hashes.

Each bank has 16 GiB. The unaligned equal split of raw weights is
13.76953125 GiB per bank, leaving more than 2 GiB per bank for alignment,
rings, programs, activations, and outputs. Preflight computes the actual
aligned layout and rejects any over-capacity bank before allocating or writing
a device BO.

Weight loading is a startup operation excluded from measured decode time. A
measurement is valid only when all 141 tensors are resident, every arena hash
matches, and weight DMA bytes remain exactly zero throughout the measured
window.

## Persistent kernel ABI

The new xclbin contains three parameterized big instances and one small
instance. Each instance is connected only to its assigned DDR bank. Its
AXI-Lite control plane exposes:

- command-ring base, capacity, producer head, and doorbell;
- completion-ring base, capacity, and consumer tail;
- program and arena-manifest bases;
- activation/result slab bases and lengths;
- session generation and model identity tag;
- sticky fault code and fault-detail registers;
- graceful shutdown and quiescent status.

A logical command descriptor contains at least:

- ABI version, descriptor length, and CRC;
- session generation, transaction ID, layer, phase, and role;
- program ID and resolved-trace ID;
- expert ID, lane mask, and token-to-lane mapping;
- arena entry and row-shard range;
- input/output slab offsets and validated lengths;
- dependency fence and completion slot.

The exact packed layout is generated from one shared schema consumed by Rust,
SystemVerilog, simulation, and proof validation. Descriptor and completion
layouts are little-endian, explicitly sized, naturally aligned, and covered
by compile-time/static assertions on both sides.

Completions return the descriptor identity, transaction and program IDs,
generation, CU ID, rows and lanes completed, cycle counters, DDR byte counts,
terminal status, and fault detail. The host accepts each completion exactly
once and only from the expected CU.

## RTL organization

The persistent kernel is divided into independently testable modules:

1. `iq1s_command_ring`: fetches descriptors and enforces ring ownership.
2. `iq1s_descriptor_check`: validates version, length, generation, bounds, and
   CRC before any memory request is issued.
3. `iq1s_program_fetch`: loads the cached program and relocation table.
4. `iq1s_trace_relocator`: binds descriptor/arena addresses and produces the
   resolved instruction stream.
5. `iq1s_block_decoder`: reads native 50-byte IQ1_S blocks and expands their
   grid/delta ternary symbols and scale metadata on the fly.
6. Existing buffered TernIP datapath: executes the ternary dot-product work.
7. `iq1s_scale_reconstruct`: combines exact integer partial sums and IQ1_S/Q8
   scales through a time-multiplexed FP32 pipeline.
8. `iq1s_result_writer`: coalesces row-shard output into result slab bursts.
9. `iq1s_completion_ring`: publishes completion records after all writes are
   globally visible.
10. `iq1s_fault_latch`: captures the first fault until session reset.

The core remains persistent until graceful shutdown or fault reset. A fault
stops descriptor consumption, drains or aborts the active AXI transaction
according to the module protocol, publishes fault detail when safe, and
requires host acknowledgement before reset.

## Native IQ1_S execution and trace truth

The existing host decomposition expands one IQ1_S launch into grid/delta
components and many physical tile submissions. The new block decoder performs
that expansion immediately before the TernIP datapath. Expanded symbols remain
in FIFOs or local buffers and are never written back to DDR.

The instruction set and assembler gain an explicit IQ1_S tmatmul operation.
AlgorithmTree models its imports, dependencies, execution, export, and
duration. The encoded operation names the descriptor/manifest relocation
slots rather than embedding unchecked runtime pointers. RTL expands the
operation into the exact grid/delta and block passes required by the matrix
shape.

Proof records contain both:

- the resolved hardware instruction sequence and its assembly/program hash;
- exact expanded counts for IQ1_S blocks, grid/delta passes, row shards,
  experts, lanes, and output rows.

This makes the trace an account of executed work rather than a synthetic
estimate. The compiler uses checked 64-bit or wider arithmetic for context,
token, expert, row, block, offset, trace-count, and duration calculations. It
is tested at the 262,144-token model limit, but emits work only for the current
active batch.

## Expert-major scheduling

Within a phase, assignments are grouped by expert ID. All active tokens that
select the same expert share a weight stream and occupy different TernIP lanes.
Groups larger than a CU's lane count are split deterministically. CU rings are
independent, so the three big CUs and one small CU may use different chunking
while processing their respective row shards.

The scheduler preserves token and role dependencies but may reorder independent
experts. It emits a phase fence only after all four row shards are complete.
No work may cross a layer dependency or consume a GPU-produced activation
before its CUDA event is complete.

## DMA and synchronization

Command, activation, and result memory use preallocated, bank-local,
double-buffered slabs. The host coalesces every layer/phase into the smallest
set of contiguous XRT BO range synchronizations supported by the runtime.
There is no DMA synchronization per expert, component, block, or row tile.

The steady-state sequence is:

1. wait for the reusable slab generation;
2. pack all phase activations and descriptors;
3. perform batched host-to-device range syncs;
4. publish producer heads and ring the four doorbells;
5. overlap eligible GPU work while U250 executes;
6. collect and validate all four completion ranges;
7. perform batched device-to-host result syncs;
8. publish results to the CUDA stream atomically;
9. release the slab generation after its CUDA consumer event.

Results from an incomplete or faulted phase are never exposed to downstream
CUDA work.

## Tuning policy

The following knobs may be tuned with matched, correctness-passing runs:

- four-CU row-shard ratios;
- command and completion ring depths;
- expert-major group order and lane packing;
- activation and result slab sizes;
- DMA coalescing threshold;
- descriptors per doorbell;
- gate/up issue order;
- overlap of GPU-native projections with U250 phases;
- completion polling and backoff behavior.

Attention placement is not a tuning knob. It remains forced to GPU flash
attention because the U250 design has no attention, softmax, KV, or recurrent
state datapath, and the measured GPU path is already the qualified baseline.

A configuration is eligible for performance comparison only after it produces
the same greedy tokens and passes sampled FFN tolerances. All tried
configurations, including rejected ones, are retained in the proof bundle.
The fastest correctness-passing configuration is frozen before the three E2E
measurement passes.

## Fail-closed behavior

Strict Qwen mode has no post-eligibility GPU fallback. It aborts on any of the
following:

- model, xclbin, arena, manifest, program, or tensor identity mismatch;
- missing, duplicated, unexpected, or cross-stream layer capture;
- invalid expert, shape, stride, pointer extent, relocation, or descriptor;
- ring overflow, stale generation, CRC failure, or duplicate completion;
- incomplete four-CU phase, timeout, AXI error, decoder error, or sticky fault;
- any measured-window weight DMA;
- output non-finiteness, tolerance failure, or token mismatch;
- attention or a non-IQ1_S tensor routed to U250;
- an eligible IQ1_S operation routed back to CUDA.

Abort records include the first failing transaction, layer, phase, descriptor,
CU, fault code, and last confirmed completion. A failed or incomplete run has
no reportable TPS.

## Verification strategy

### Software and compiler tests

- Sideband ABI lifecycle, copying, versioning, stream matching, and mutation
  tests.
- Per-layer expected-role tests from the audited 50/50/41 IQ1_S manifest.
- Handwritten/compiler semantic coverage and relocation mutation tests.
- AlgorithmTree checked-arithmetic tests at 262,144 tokens.
- Arena packing, alignment, hashing, capacity, row coverage, and bank-overflow
  tests.
- Ring wrap, stale generation, duplicate completion, and fail-closed tests.

All Cargo and build commands set `CARGO_BUILD_JOBS=32` or
`QWEN35_BUILD_JOBS=32` as appropriate.

### RTL simulation

Simulation consumes compiler-generated descriptors and program images rather
than handwritten test-only encodings. It compares the native IQ1_S decoder
and FP32 reconstruction against `iq1s_tmatmul.rs` fixtures and covers:

- gate/up/down shapes and all row-shard boundaries;
- batch counts 1, 6, 9, and 16;
- repeated and all-distinct expert selections;
- descriptor and completion ring wrap;
- backpressure on every AXI channel;
- malformed relocation, CRC, generation, bounds, and program records;
- injected AXI response failures and timeout recovery;
- exact completion ordering and output visibility.

### Hardware emulation and synthesis

Hardware emulation proves four independent CUs, local bank addressing,
persistent ring progress, batch DMA, and graceful shutdown. Synthesis records
resource utilization, requested and achieved clocks, timing slack, platform
identity, and bitstream/xclbin hashes. A generated xclbin is not accepted for
live inference if timing or platform compatibility checks fail.

### Live microbenchmarks

Live tests measure decoder throughput, TernIP utilization, DDR read/write
bandwidth, lane occupancy, command latency, phase latency, CU imbalance, and
DMA overlap. They also prove:

- all four CUs complete nonzero work;
- all 141 IQ1_S tensors are resident;
- measured-window weight DMA is zero;
- the four persistent kernels are not relaunched per layer;
- each offloaded layer uses no more than two FPGA/GPU phase fences.

### Numerical and end-to-end proof

CUDA, handwritten hybrid, and compiler hybrid use the same binary, model,
prompts, tokenization, scheduling limits, and greedy decoding. Sampled FFN
outputs must pass `atol=1e-4`, `rtol=1e-3`; every final token ID must be
identical to CUDA.

After one unmeasured warm-up, each hybrid mode runs three measured passes. TPS
is:

```text
2048 validated generated tokens
-----------------------------------------------
last completion timestamp - first enqueue timestamp
```

Model loading, arena preload, warm-up, prompt tokens, failed requests, and
incomplete requests are excluded from the numerator and measurement window.
The report includes each pass, minimum, median, and mean for CUDA,
handwritten, and compiler modes. Every hybrid pass must reach at least
15 aggregate generated tok/s. If it does not, the measured value may be
reported as a completed below-target result, but the implementation is not
declared to have passed the performance gate.

## Proof bundle

The final proof bundle contains:

- immutable model, software revision, RTL revision, toolchain, platform,
  bitstream, xclbin, and binary identities;
- model tensor audit and 50/50/41 layer-role manifest;
- per-bank arena maps, byte counts, and hashes;
- xclbin metadata and four-CU connectivity;
- handwritten/compiler traces, relocations, program hashes, and expanded
  operation counts;
- command/completion/DMA/telemetry records;
- U250 health, temperature, firewall, and fault state before and after runs;
- CUDA and hybrid numerical comparisons and token outputs;
- exact benchmark commands, environments, timestamps, and TPS calculations;
- the complete knob search and frozen winning configuration.

The proof validator recomputes counts and TPS from primary records and rejects
missing, duplicated, inconsistent, partial, or fallback evidence.

## Repository and integration boundary

Software changes belong in the isolated Qwen worktree. RTL and xclbin changes
belong in an isolated branch/worktree created from the authoritative
`ternary_matmul` revision selected during implementation planning. Existing
dirty files and `.proof/` artifacts in the software checkout are preserved and
never included in implementation commits unless explicitly named.

The current `MaxCores_370M.xclbin` remains the regression image. The new image
uses a distinct filename and kernel ABI so old Kimi and standalone tests cannot
silently load it. Runner preflight binds each mode to the exact expected
xclbin SHA and kernel metadata.

## Completion criteria

The work is complete only when:

1. the sideband ABI and CUDA interception form validated layer transactions;
2. exactly 141 IQ1_S tensors load into verified U250 bank arenas;
3. the new xclbin runs four persistent CUs with native IQ1_S decode;
4. handwritten and AlgorithmTree compiler programs are physically executed;
5. weight DMA is absent from the measured window;
6. attention and all 39 non-IQ1_S expert tensors remain GPU-native;
7. all numerical and exact-token gates pass;
8. all six hybrid measurement passes complete the 2,048-token workload;
9. every handwritten and compiler pass reaches at least 15 aggregate tok/s;
10. proof validation passes without overrides or inferred evidence.
