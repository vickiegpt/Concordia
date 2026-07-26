# Kimi IQ1_S FPGA Shadow Validation Design

**Date:** 2026-07-26
**Status:** Approved for spec review
**Target checkout:** `/home/victoryang00/hetGPU`

## Purpose

Add a correctness-first shadow path for one real Kimi K2.6 IQ1_S matrix-vector
launch. Native CUDA remains authoritative and continues to provide the output
used by inference. The FPGA observes one bounded 32-column IQ1_S subgroup,
computes its ternary integer partial product through CXL DAX, and compares the
result against a software reference.

This milestone proves that operands from a live Kimi CUDA launch can be decoded,
staged through DAX, executed by the ternary unit, and checked exactly without
changing Kimi's output.

## Decision Summary

The first implementation will:

1. Match only `mul_mat_vec_q<ggml_type19,...>`, where type 19 is IQ1_S.
2. Shadow at most one eligible launch per process by default.
3. Run native CUDA first and preserve its return status and output.
4. Capture operands in order on the actual intercepted CUDA stream.
5. Decode subgroup 0, covering input columns `[0, 32)`.
6. Pack its logical IQ1_S grid values into the FPGA's four-trits-per-byte
   matrix format.
7. Submit one replay-safe `tmatmul_go` program through `RUN_CSR_ONLY`.
8. Compare all valid FPGA rows with an exact signed-integer software dot.
9. Restore the first 2 MiB of DAX after proven terminal completion.
10. Disable further shadow attempts after any ambiguous hardware completion.

The implementation will not route FPGA output back into the CUDA destination.

## Confirmed Constraints

### Kimi operand ABI

The supported CUDA kernel has this parameter order:

```text
0 vx:        const void *   packed IQ1_S matrix
1 vy:        const void *   ordinary Q8_1 activation blocks
2 dst:       float *        native CUDA output
3 ncols_x:   int
4 nrows_x:   int
5 nrows_y:   int
6 nrows_dst: int
```

The pointers are borrowed CUDA allocations. The shadow path must not retain or
free them. It must copy the required bytes before returning from the intercepted
launch.

`mul_mat_q` is excluded from this milestone. Its Q8_1 layout is repacked for
MMQ, and stream-K variants have a separate fixup launch and temporary buffer.
Those semantics make it a larger and less isolated first target.

### IQ1_S is affine ternary, not a plain ternary matrix

One IQ1_S block covers 256 weights and contains:

```text
fp16 d
u8   qs[32]
u16  qh[8]
```

For a 32-element subgroup, the decoded logical grid is
`g[j] in {-1, 0, +1}`, but the full weight also includes a block scale, an odd
subgroup multiplier, and a signed `1/8` affine offset. One complete Kimi output
therefore cannot be represented by a single `tmatmul_go_nvint8` operation.

The first milestone compares only the exact integer primitive:

```text
partial[row] = sum(j=0..31, grid[row][j] * q8_1.qs[j])
```

This primitive cannot saturate signed int16:

```text
abs(partial[row]) <= 32 * 127 = 4064
```

The software reference also records the IQ1_S and Q8_1 scale/offset metadata
needed to reconstruct the corresponding floating-point subgroup contribution,
but that reconstructed value is telemetry and does not affect inference.

### Existing hardware proof

The current image already completed this packed subgroup primitive outside the
CUDA interception path:

- two sequential replay-safe `RUN_CSR_ONLY` submissions completed;
- each submission executed eight fetched instruction slots;
- instruction, load/store, tmatmul-read, and output-write counters advanced;
- all 2048 FPGA outputs matched the integer reference;
- the first 2 MiB DAX snapshot and restored image had the same SHA-256.

The implementation should reuse that proven packed format, program construction,
layout, and completion checks.

## Non-Goals

- No RTL changes.
- No driver ABI changes.
- No direct execution of packed IQ1_S bytes by the FPGA.
- No dense NVINT8 staging for this shadow milestone.
- No full 7168-column IQ1_S decomposition.
- No FPGA replacement of native CUDA output.
- No `mul_mat_q` or stream-K support.
- No attention, RoPE, normalization, KV-cache, or sampling offload.
- No throughput or TPS claim from the one-token validation.
- No modifications to the BitNet or llama.cpp checkout.

## Runtime Architecture

```text
llama.cpp cudaLaunchKernel(mul_mat_vec_q<IQ1_S>)
  |
  +-- existing prelaunch route/offload hook
  |     shadow-eligible launch => continue native
  |
  +-- real native cudaLaunchKernel
  |     failure => return native error, no shadow
  |
  +-- post-native shadow observer
        capture descriptor and operands on original stream
        decode one IQ1_S subgroup
        build packed ternary matrix and padded i16 input
        acquire process and device transaction locks
        snapshot first 2 MiB of DAX
        stage + verify + RUN_CSR_ONLY once
        compare integer output
        restore and verify DAX
        log evidence
        return original native result
```

Shadow mode is an observer, not another result from the existing prelaunch
handler. A successful shadow transaction must never cause cudart to suppress
the native launch.

## Components

### `cudart_shim.c`

The cudart shim retains the existing prelaunch behavior. When shadow mode is
enabled and a kernel is potentially eligible, the prelaunch diversion must
return "continue native" even when normal strict CXL routing would reject that
packed ABI.

After the real `cudaLaunchKernel` returns success, the shim calls a new
best-effort Rust observer with:

- kernel name;
- original `void **args`;
- grid and block dimensions;
- shared-memory size;
- the actual `cudaStream_t`.

The observer's result is diagnostic only. The shim returns the original native
CUDA status.

### `function.rs`

`function.rs` owns the CUDA ABI boundary:

- recognize exact IQ1_S MMVQ mangling (`ggml_type19`);
- copy scalar values and pointer values out of the argument slots;
- reject invalid dimensions and unsupported layouts before any CUDA or DAX
  access;
- query CUDA allocation base and remaining spans for all borrowed pointers;
- pass a value-owned launch descriptor to the shadow module;
- keep route-selection logging separate from hardware-execution logging.

The descriptor includes:

```text
launch_id
kernel name and ABI kind
cuda context/device
actual stream
matrix, activation, and native-output pointers
verified remaining allocation spans
ncols_x, nrows_x, nrows_y, nrows_dst
```

No `usize::MAX` allocation-size bypass is permitted.

### `kimi_iq1s_shadow.rs`

A new focused module owns:

- environment parsing;
- process-local cap and poison state;
- IQ1_S/Q8_1 host layouts;
- the exact 2048-entry IQ1_S grid table;
- subgroup decoding;
- packed-ternary matrix construction;
- padded signed-int16 input construction;
- software integer-dot reference;
- result comparison and numerical telemetry;
- orchestration of one synchronous shadow transaction.

The IQ1_S grid table is checked in with its upstream source revision and a
stable checksum. Tests compare representative entries and complete fixture
decodes against llama.cpp reference results.

### `cxl_tmatmul.rs`

Add a bounded packed-shadow submission API separate from the existing dense
NVINT8 CUDA-DAX API. The API accepts owned host payloads and never receives the
native CUDA destination pointer.

It owns:

- device and DAX preflight;
- process and cross-process locking;
- durable 2 MiB snapshot creation;
- packed payload staging and cache maintenance;
- CSR readback checks;
- one `RUN_CSR_ONLY`;
- terminal-state and counter validation;
- packed output readback;
- safe restore handling;
- structured evidence returned to the shadow module.

## Eligibility

A launch is eligible only when every condition is true:

- `HETGPU_KIMI_FPGA_SHADOW=1`;
- normalized name is `mul_mat_vec_q`;
- mangled template contains `ggml_type19`;
- `ncols_x` is positive and divisible by 256;
- `nrows_x`, `nrows_y`, and `nrows_dst` satisfy the MMVQ contract;
- valid output rows are in `[1, 2048]`;
- matrix allocation covers
  `nrows_x * (ncols_x / 256) * sizeof(block_iq1_s)`;
- activation allocation covers at least one ordinary Q8_1 block;
- output allocation covers `nrows_dst * sizeof(float)`;
- all dimensions and byte products pass checked arithmetic;
- the process has not exhausted its cap or entered poisoned state.

Unsupported launches continue on native CUDA and emit an ineligibility reason.

## Selection and Cap

Controls:

```text
HETGPU_KIMI_FPGA_SHADOW=1
HETGPU_KIMI_FPGA_SHADOW_MAX_LAUNCHES=1
HETGPU_KIMI_FPGA_SHADOW_TIMEOUT_MS=10000
HETGPU_KIMI_FPGA_SHADOW_LOG=/path/shadow.jsonl
HETGPU_KIMI_FPGA_SHADOW_SNAPSHOT_DIR=/var/tmp
HETGPU_KIMI_FPGA_SHADOW_DEV=/dev/cxl_tmatmul3b000
HETGPU_KIMI_FPGA_SHADOW_DAX=/dev/dax6.0
```

Rules:

- shadowing is disabled by default;
- the default cap is one;
- zero disables shadowing;
- invalid values disable shadowing and emit one configuration error;
- there is no unlimited setting;
- the cap applies when a launch claims the DAX transaction;
- preflight rejection and lock contention do not consume the cap;
- once durable snapshotting starts, the claim is consumed;
- a poisoned process performs no later hardware shadow transactions.

## CUDA Ordering and Operand Capture

The observer runs only after native `cudaLaunchKernel` reports success.

It queues operand copies on the original intercepted stream and synchronizes
that stream. This guarantees that input quantization and native MMVQ complete
before borrowed temporary data is read.

For the default subgroup:

- copy the packed matrix range required to address subgroup 0 for every valid
  output row;
- copy the first 36-byte ordinary Q8_1 activation block;
- optionally copy a bounded native-output telemetry sample;
- finish all CUDA copies before acquiring or mutating DAX.

The first implementation is synchronous. With a cap of one, this intentionally
trades latency for deterministic ownership and simple lifetime rules.

## Subgroup Conversion

For each valid output row:

1. Locate row block 0 using the checked packed row stride.
2. Read `qs[0..4]` and `qh[0]`.
3. Form each 11-bit IQ1_S grid index.
4. Decode 32 logical values in `{-1, 0, +1}`.
5. Place them in matrix columns `[0, 32)`.
6. Zero-pad columns `[32, 2048)` and rows after `nrows_dst`.

The matrix is packed four trits per byte using the existing tmatmul encoding.
The resulting 2048 x 2048 packed matrix is exactly 1 MiB.

The Q8_1 input uses raw signed `qs[0..32]` values widened to signed int16.
Input slots `[32, 2048)` are zero.

The program uses `tmatmul_go`, not `tmatmul_go_nvint8`. IQ1S's `1/8` affine
offset is not the NVINT8 threshold delta and is handled only in reconstruction
telemetry.

## Bounded DAX Layout

Only the first 2 MiB may be touched:

| Payload | DPA range | Size |
| --- | ---: | ---: |
| Packed ternary matrix | `0x000000-0x100000` | 1 MiB |
| Signed-int16 input | `0x100000-0x101000` | 4096 B |
| Output/sentinel | `0x110000-0x111000` | 4096 B |
| Replay-safe program | `0x120000-0x120080` | 128 B |
| Snapshot boundary | `0x000000-0x200000` | 2 MiB |

The program contains six semantic instructions:

```text
ldv v0, INPUT
tmatmul_import v0
tmatmul_go MATRIX
tmatmul_export v1
sv v1, OUTPUT
stall
```

It is encoded as a replay-safe 128-byte fetch image with the terminal stall in
the final 16-byte slot. A successful run reports eight fetched instruction
slots.

## Transaction State Machine

```text
disabled
  -> eligible
  -> claimed
  -> operands_captured
  -> locks_acquired
  -> snapshot_durable
  -> payload_staged
  -> submit_started
  -> terminal_completion_proven
  -> compared
  -> restored
  -> complete
```

Preflight requires:

- expected boot ID captured in evidence;
- matching BDF, control node, and DAX identity;
- UAPI version 2;
- `TMM1` device ID;
- `dim_d == 2048`;
- configured program DPA `0x120000`;
- sufficient DAX size and alignment;
- fresh idle DMA, no reset, no execution error, and no active stall;
- a process-local mutex;
- a nonblocking exclusive lock on the control device.

Before staging, write the 2 MiB snapshot to a unique file, `fsync` it, and
record its SHA-256 in a durable stage marker.

After staging:

- read back the complete input, output sentinel, and program through CSR;
- spot-check matrix words at the beginning, middle, and end;
- issue exactly one `RUN_CSR_ONLY`, with no retry and no BAR fallback.

Terminal success requires:

- ioctl success;
- DMA done;
- mirrored DMA done;
- `STALLED` result flag;
- no DMA-error result flag;
- stall status one;
- no reset or execution error;
- expected dimension and eight fetched instructions;
- positive expected instruction, load, tmatmul-read, and store counter deltas.

Only `submit_completed` may set `hardware_executed: true`.

## Failure and Restore Rules

Native CUDA errors are returned unchanged and skip shadow execution.

Shadow failures never replace a successful native result:

- before submit: restore DAX and verify the snapshot hash;
- after proven terminal completion: restore DAX and verify the snapshot hash,
  including after a numerical mismatch;
- after timeout, interrupted ioctl, missing stall, ambiguous DMA state, or any
  other unproven completion: do not write or restore DAX.

An ambiguous post-submit state:

1. writes `cleanup_blocked_launch_unproven` to the durable marker;
2. retains the snapshot file;
3. marks the process shadow path poisoned;
4. prevents every later hardware shadow attempt;
5. requires an externally proven quiescent boundary before recovery.

The observer still returns the successful native CUDA result.

## Logging

Shadow evidence is JSONL and uses a process-unique `launch_id`.

Event types:

```text
route_decision
shadow_claimed
operands_captured
snapshot_saved
dax_stage_verified
submit_started
submit_completed
numerical_comparison
restore_verified
shadow_complete
shadow_failed
```

`route_decision` always reports `hardware_executed: false`.

The numerical event records:

- subgroup and valid row range;
- matrix, input, expected-output, and FPGA-output SHA-256;
- mismatch count and first mismatch;
- maximum absolute integer difference;
- IQ1_S `d`, subgroup multiplier, and affine sign metadata;
- Q8_1 `d`, `s`, and sum of integer quants;
- reconstructed subgroup contribution error;
- saturation headroom;
- native output finite/NaN/Inf counts and bounded summary statistics;
- `authoritative_output: "native_cuda"`;
- `fpga_vs_native_directly_comparable: false`.

## Concurrency and Lifetime

- One process-local mutex serializes shadow selection and execution.
- An exclusive control-device lock prevents cross-process DAX overlap.
- The fixed DAX scratch region is never treated as a cache.
- The default one-launch mode does not cache converted matrices.
- Future host-side caches must use content hash, shape, subgroup, CUDA context,
  and device as the key; pointer identity is insufficient.
- CUDA mappings, host buffers, file descriptors, and locks live for one
  synchronous transaction.
- A PID change after `fork` disables shadowing in the child until explicit
  reinitialization.
- No background worker may retain CUDA pointers beyond the launch call.

## Testing

### Pure tests

- environment parsing and default cap;
- zero-disable and invalid-value behavior;
- concurrent claim behavior;
- exact type-19 MMVQ recognition;
- dimension, stride, and allocation-bound rejection;
- checked arithmetic failures occur before device access;
- IQ1_S grid decode against llama.cpp fixtures;
- four-trits-per-byte packing;
- Q8_1 signed widening;
- exact integer-dot reference and saturation bound;
- replay-safe 128-byte program construction;
- bounded layout overlap and end checks;
- event schema forbids route-only hardware claims;
- poison-state transitions.

### Mock integration tests

- native launch occurs exactly once and before shadow observation;
- native failure prevents shadow work;
- every injected shadow failure preserves the native return value;
- FPGA output is never copied to the native destination;
- the original CUDA stream is used and synchronized;
- cap one prevents a second hardware transaction;
- fake-DAX snapshot restores after every pre-submit failure;
- terminal submit restores after comparison failure;
- ambiguous submit performs no cleanup write and poisons the process.

### Model-free hardware gate

- use a synthetic IQ1_S/Q8_1 subgroup fixture;
- touch only the first 2 MiB;
- execute exactly one `RUN_CSR_ONLY`;
- prove terminal ioctl and BAR state;
- prove exact instruction and traffic counter deltas;
- compare all 2048 signed-int16 outputs bit-exactly;
- restore the full 2 MiB snapshot and verify SHA-256;
- exit zero with no dirty stage marker.

### Kimi one-token gate

Pin:

- `/home/eabban/BitNet/build-cuda128-gcc12/bin/llama-cli`;
- the six Kimi K2.6 IQ1_S shards;
- HetGPU library hashes;
- control device `/dev/cxl_tmatmul3b000`;
- DAX device `/dev/dax6.0`.

Run baseline and shadow with the same prompt, seed, deterministic sampling,
GPU-layer configuration, and one generated token.

Require:

- both runs exit zero;
- both runs emit the same token;
- exactly one IQ1_S MMVQ shadow claim;
- exactly one terminal hardware submit;
- exact FPGA/software integer subgroup comparison;
- one verified restore;
- no later hardware submits;
- unchanged native output ownership;
- no CXL, DAX, FPGA, or NVIDIA kernel error.

Load and timing values are smoke telemetry only. Generation TPS is not reported
from a one-token run.

## Acceptance Criteria

- Disabled mode is behaviorally unchanged.
- Shadow mode cannot suppress or replace native CUDA.
- Only exact type-19 MMVQ subgroup 0 is eligible.
- Default execution performs no more than one hardware shadow per process.
- Every memory span is validated before access.
- The actual CUDA stream orders operand capture.
- The FPGA integer subgroup output matches software for every valid row.
- Route logs alone never claim hardware execution.
- Successful and pre-submit-failed transactions restore the first 2 MiB
  exactly.
- Ambiguous post-submit state performs no DAX cleanup write and poisons shadow
  mode.
- Existing focused HetGPU tests pass.
- Model-free live proof passes before Kimi is attempted.
- Baseline and shadow Kimi one-token runs both exit zero with identical output.
- No RTL, driver, BitNet, or llama.cpp changes are required.
