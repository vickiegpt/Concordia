# Kimi IQ1_S GPU-Attention / AU250-BitLinear Hybrid Design

**Date:** 2026-08-25
**Status:** Approved
**Target checkout:** `/home/victoryang00/hetGPU`

## Goal

Run Kimi K2.6 IQ1_S with attention and all non-BitLinear CUDA kernels on the
NVIDIA GPU while qualified IQ1_S matrix products execute on the Alveo U250.
The AU250 path uses the existing Rust XRT backend, its four-BO host ABI, and
128-bit tmatmul instructions emitted by the repository assembler.

The route is correctness-first and fail-closed. A selected BitLinear launch
must either produce a fully reconstructed, validated FPGA result and copy it to
the CUDA destination or abort inference. It must never silently execute the
selected BitLinear operation on the GPU.

## Rebooted Live Contract

The live platform was revalidated after the 2026-08-25 reboot:

- NVIDIA RTX PRO 6000 Blackwell is available for attention and the remaining
  CUDA graph.
- AU250 functions `0000:64:00.0` and `0000:64:00.1` are bound to `xclmgmt`
  and `xocl` respectively.
- `/au250_xrt/example/MaxCores_370M.xclbin` is the only installed application
  image and targets `xilinx_u250_gen3x16_xdma_4_1_202210_1`.
- The image contains three `ternip_big` instances with D=1024 and batch size 9,
  plus one `ternip_small` instance with D=1024 and batch size 6.
- CU-to-bank placement is `ternip_big_1`/bank0, `ternip_big_2`/bank3,
  `ternip_big_3`/bank2, and `ternip_small_1`/bank1.
- Fixed-point storage is signed 16-bit with exponent -5; instructions are
  128-bit records and each core exposes four architectural vector registers.
- The known vector-add passes on this image through PYNQ.
- The canonical 1024x1024 ternary matmul passes exactly through the Rust
  four-BO XRT backend: nine lanes, 96 program bytes, and terminal STALL 1.

The host tools report XRT 2.21.75 while `au250-run` supplies the compatible
2.15 application runtime. Production and live tests therefore execute inside
`au250-run`; ordinary host processes must not bind against the incompatible
host XRT stack.

## Existing Interfaces Preserved

The existing four-BO submission contract remains authoritative for every
in-flight CU:

1. matrix BO;
2. input BO;
3. output BO;
4. program BO containing assembled little-endian 128-bit instructions.

The program continues to use BO device addresses bound by the existing
assembler and the user-managed MM2S/stall register contract. The CXL backend is
not changed and no CXL device, DAX mapping, CXL staging mode, or CXL environment
gate participates in this route.

Production selection adds a backend-neutral control:

```text
HETGPU_TMATMUL_BACKEND=xrt
```

The current BitNet route classifier remains repository-compatible. Existing
manifest spellings and CXL behavior stay accepted for old runs, but an XRT
backend selection dispatches a qualified tmatmul decision to the AU250 handler
before any CXL-only staging checks. Route evidence records both the logical
route and physical backend so an `xrt` execution cannot be confused with a CXL
candidate or a GPU fallback.

## Routing Boundary

Only qualified IQ1_S kernels are eligible:

- `mul_mat_q` specializations containing `ggml_type19`;
- `mul_mat_vec_q` specializations containing `ggml_type19`.

The existing launch-capture code owns CUDA ABI parsing, allocation-span checks,
packed IQ1_S matrix capture, Q8_1 activation capture, and output-pointer
validation. The XRT handler consumes its value-owned `CapturedLaunch`; it does
not retain borrowed CUDA pointers beyond the synchronized capture boundary.

Attention, flash attention, Q/K/V operations, RoPE, KV-cache operations,
normalization, softmax, sampling, and unknown kernels remain native CUDA.
Explicit GPU route markers continue to override generic matmul markers. In
strict mode, an IQ1_S launch selected for XRT but not supported by the exact
shape/layout contract is rejected rather than re-launched natively.

## Exact IQ1_S Decomposition on D=1024 TernIP

IQ1_S is affine ternary rather than one plain ternary matrix. Each 256-value
block contains eight 32-value groups with a grid ternary component, a delta
sign component, and scale metadata. The existing IQ1_S parser and
`reconstruct_from_raw` implementation remain the numerical source of truth.

### Geometry

For an arbitrary logical matrix:

- output rows are tiled in chunks of 1024;
- K columns are tiled in chunks of 1024;
- each K tile contains at most 32 logical 32-value groups;
- incomplete row/K tiles are zero-padded to the native 1024x1024 shape;
- batches and groups are scheduled together across the physical hardware
  lanes.

For each row/K tile, the host materializes two packed ternary matrices:

1. a grid matrix containing every IQ1_S grid group at its native columns;
2. a delta matrix containing the repeated delta sign for every group.

### Lane Multiplexing

A physical lane is assigned one `(logical_batch, global_group)` pair. Its input
vector is zero except for that group's 32 columns, where the Q8 quantized values
are stored directly as signed `i16` raw values. Because only 32 values are
active, each raw grid or delta dot is bounded by:

```text
-4096 <= dot <= 4064
```

This is exactly representable in signed `i16`. The fixed-point exponent does
not alter the integer primitive: the path reads the raw output bits and does
not convert the dot through floating-point fixed-point interpretation.

The three big CUs can evaluate nine batch/group pairs per submission and the
small CU can evaluate six. Unused lanes and padding remain zero. A scheduler
assigns deterministic work items to compatible CU capacity without changing
the reconstruction order.

### Reconstruction

Grid and delta raw dots are paired by row, logical batch, and global group.
The existing IQ1_S and Q8 scale metadata reconstructs the group contribution.
Contributions accumulate in deterministic global-group order into an f32
output buffer. Before CUDA copy-back, the handler requires:

- every planned raw component is present exactly once;
- every raw value is within the proven signed-16-bit bound;
- software reconstruction accepts each raw pair;
- every final f32 output is finite;
- output size exactly matches the captured launch contract.

Only after all checks pass is the complete output copied to the CUDA
destination. Partial output is never exposed.

## Persistent Four-CU XRT Executor

Opening the device and reloading the xclbin for every tile is infeasible for a
Kimi layer. A process-local executor therefore owns the application image for
the duration of the hybrid run:

1. open the XRT device once inside the `au250-run` process;
2. load and verify the configured xclbin once;
3. open exclusive native-IP contexts for all configured CUs;
4. allocate one reusable four-BO set in each CU's connected DDR bank;
5. bind stable BO addresses and assemble the tmatmul program once per CU;
6. run one worker per CU, synchronizing only that CU's BOs and registers;
7. return tagged results to a deterministic reconstruction coordinator;
8. quiesce all active CUs before releasing any BO or device address.

The default target table for `MaxCores_370M.xclbin` is:

| Native IP | Memory group | Lanes |
| --- | ---: | ---: |
| `ternip_big:ternip_big_1` | 0 | 9 |
| `ternip_big:ternip_big_2` | 3 | 9 |
| `ternip_big:ternip_big_3` | 2 | 9 |
| `ternip_small:ternip_small_1` | 1 | 6 |

The table is validated against the xclbin/runtime rather than trusted solely
from environment text. Each work item carries stable tile, component, lane,
batch, and group identifiers so out-of-order CU completion cannot corrupt
logical accumulation order.

Matrix materializations are cached by the existing matrix identity, content
hash, tile coordinates, and component kind. BO contents may be reused only
when all identity fields match. Activation/input BOs and output BOs are updated
for every scheduled submission.

## Configuration

The production wrapper sets at least:

```text
HETGPU_TMATMUL_BACKEND=xrt
HETGPU_BITNET_DISAGGREGATE=1
HETGPU_BITNET_DISAGG_STRICT=1
HETGPU_TMATMUL_HARDWARE_MATMUL=1
HETGPU_XRT_XCLBIN=/au250_xrt/example/MaxCores_370M.xclbin
HETGPU_XRT_TIMEOUT_MS=10000
HETGPU_BITNET_ROUTE_LOG=/work/.proof/.../routes.jsonl
HETGPU_XRT_EXECUTION_LOG=/work/.proof/.../xrt.jsonl
```

The CU table defaults to the validated image contract above and may be
overridden only through one versioned configuration value that is validated as
a whole. Per-variable mixtures that can associate a CU with the wrong DDR bank
are rejected.

CXL-specific variables are unset by the AU250 wrapper. The wrapper also pins
the known-good CUDA 13 llama runner and Kimi K2.6 IQ1_S model paths already used
by the GPU baseline.

## Error and Lifetime Rules

- Configuration, capture, packing, XRT, timeout, STALL, synchronization,
  reconstruction, CUDA copy-back, and evidence-log failures are fatal for a
  selected strict XRT launch.
- A failed CU is reset and its DMA is quiesced before BO reuse or teardown.
- If quiescence cannot be confirmed, that CU and its live BO/device handles are
  retained and poisoned for the rest of the process; their addresses are not
  released for reuse.
- A poisoned CU causes the strict run to abort rather than reducing silently to
  the GPU or remaining CUs.
- The application image is not reloaded while any worker owns an in-flight
  request.
- Route logs are part of the proof contract in strict mode and therefore fail
  closed when their configured destination cannot be written.

## Evidence and Verification Gates

Implementation is accepted only in this order:

1. **Pure tests:** tile planning, packed grid/delta matrices, lane assignment,
   raw-dot bounds, deterministic reconstruction, cache identity, multi-CU
   completion reordering, error poisoning, and strict no-fallback routing.
2. **Existing regression:** all current XRT four-BO, assembler, CXL tmatmul,
   IQ1_S decomposition, and BitNet routing tests remain passing.
3. **Live D=1024 regression:** vector-add and canonical exact tmatmul pass on
   `MaxCores_370M.xclbin` through `au250-run`.
4. **Live tiled proof:** a D=2048 synthetic IQ1_S fixture spans two K tiles,
   multiple row tiles, group/lane packing, and multiple CUs; every FPGA raw dot
   and reconstructed f32 output matches the software reference.
5. **Captured-layer proof:** at least one real Kimi IQ1_S launch executes on the
   AU250 and its complete reconstructed output matches the qualified software
   reference before replacement mode is enabled for the whole run.
6. **End-to-end proof:** a deterministic short Kimi prompt exits cleanly with
   valid token IDs; route logs prove attention remained native GPU and every
   selected IQ1_S BitLinear launch completed through XRT. FPGA temperature,
   firewall, and fatal-error reports remain healthy.
7. **Benchmark:** only after the correctness gates pass, record prompt and
   generation throughput, per-CU submissions/cycles, host staging time, FPGA
   time, reconstruction time, GPU utilization, FPGA power/temperature, and the
   exact model/xclbin/binary hashes.

Build success, route classification, loaded xclbin identity, or a nonzero
STALL alone does not constitute the hybrid proof. No TPS is reported unless
the end-to-end output and physical execution evidence are both valid.

## Non-Goals

- No CXL Type-2 path, DAX mapping, or CXL driver recovery.
- No RTL or xclbin rebuild in this phase.
- No approximation or direct ternarization of IQ1_S weights.
- No attention offload to the AU250.
- No silent GPU BitLinear fallback.
- No modification of the external BitNet/llama.cpp source tree unless a later
  measured ABI blocker proves the existing CUDA launch interception
  insufficient.
