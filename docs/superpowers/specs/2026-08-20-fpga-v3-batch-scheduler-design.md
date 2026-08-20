# FPGA CXL v3 Batch Scheduler Design

## Goal

Execute a logical IQ1_S matrix multiplication batch through the real CXL v3
UAPI, distribute capability-bounded work across all 16 FPGA instances, preserve
CUDA output ordering, and produce evidence that distinguishes software tests
from live FPGA execution.

## Corrected boundary

The existing `V3Session` already owns registration, commit, submission, wait,
request-ID demultiplexing, capability validation, and lease quarantine. The
batch scheduler therefore plans `TaskV3` descriptors; it does not introduce a
second memory allocator, userspace health model, or simulated execution
pipeline.

For a logical activation batch of `N`, the scheduler partitions `[0, N)` into
ordered slices no larger than live `CapsV3.max_batch`. Each physical matrix
component and slice becomes one `TaskV3` descriptor with:

- `batch = slice.count`;
- `lane = LANE_ANY`, leaving lane selection to the driver/FPGA;
- input and output offsets pointing at the slice's first row;
- capability-aligned input and output strides;
- a unique request ID used to restore logical order.

One-thread sequential decode cannot aggregate future CUDA calls without
violating synchronous launch semantics. This design accelerates real batched
MMQ launches and concurrent descriptor work; evaluation reports batch-one
latency separately from batched throughput.

## Components

`zluda/src/impl/batch_scheduler.rs` owns capability-bounded batch slicing and
completion-derived metrics. It is pure Rust and independently testable.

`zluda/src/impl/iq1s_tmatmul.rs` follows GGML's transposed Q8_1 MMQ record
layout. `stride11` is the pitch in 144-byte records between adjacent K groups,
not a byte stride between batch rows. For K-group `g` and logical batch item
`b`, the source record is `(g * stride11 + b) * 144`; only `b < ne11` is active.
The adapter stages those active records into component-by-slice input rows,
emits the v3 tasks, reconstructs every output row in original batch order, and
copies a contiguous `f32` output matrix back to CUDA.

`zluda/src/impl/cxl_tmatmul_v3.rs` remains the UAPI authority. It exposes the
live capability block needed by the planner and continues to validate every
descriptor and completion before results are accepted.

The NVIDIA named-kernel hook remains strict and synchronous. It logs scheduler
evidence after a completed v3 run but never reports success before output copy.

## Configuration and failure behavior

The scheduler uses live capabilities by default. `HETGPU_FPGA_BATCH_LIMIT` may
lower, but never raise, `max_batch` for evaluation. Zero, malformed, or
capability-exceeding values fail closed. Existing strict/fallback routing is
unchanged.

Any submit, wait, completion, output-range, or reconstruction failure aborts the
logical launch. Existing v3 lease quarantine rules remain authoritative.

## Evidence gates

Software acceptance requires planner edge-case tests, fake-v3 end-to-end
bit-exact batch reconstruction, the complete v3 unit suite, IQ1_S tests, and an
NVIDIA-feature build.

Live acceptance requires querying `/dev/cxl_tmatmul3b001`, observing 16
instances in its capability block, completing a real v3 batched fixture through
`/dev/dax6.0`, preserving bit-exact output, and recording descriptor count,
logical batch count, lane mask, per-lane completions, cycles, and elapsed time.
Kimi and MatMulFreeLM TPS are reported only when their processes exit normally
and emit timing blocks.
