# CXL tmatmul v2 Backend Design

## Purpose

Port the existing hetGPU ternary matmul path to the CXL Type-2 tmatmul v2
userspace ABI. The first implementation targets one real offload path:

```text
out[D] = input[D] * matrix[D x D]
```

where `D` is reported by the CXL tmatmul device. Unsupported kernels, missing
devices, incompatible shapes, or runtime errors must fall back to the existing
interpreter or CPU fallback paths.

## Current Context

hetGPU already has:

- a PTX-to-tmatmul assembly path in `ptx/src/pass/ptx_to_tmatmul.rs`
- a host scalar interpreter in `zluda/src/impl/tmatmul_interpreter.rs`
- virtual allocation tracking in `zluda/src/impl/memory.rs`
- launch/fallback logic in `zluda/src/impl/function.rs`

The CXL tree exposes the v2 ABI in `../cxl/include/uapi/linux/cxl_type2_accel.h`.
The current device on this machine is `/dev/cxl_tmatmul3b000`. The v2 ABI
uses the tmatmul misc device only for control. Program, matrix, input, and
output bytes are staged into the backing devdax CXL window by userspace.

## Approach

Add a focused Rust backend in `zluda/src/impl/cxl_tmatmul.rs`. The module owns
the CXL v2 userspace protocol:

1. Find the control device from `HETGPU_CXL_TMATMUL_DEV` or by scanning
   `/dev/cxl_tmatmul*`.
2. Find the devdax data window from `HETGPU_CXL_TMATMUL_DAX` or sysfs
   discovery equivalent to `../cxl/tools/testing/cxl/tmatmul_type2_run.c`.
3. Open the tmatmul misc device and read `CXL_TYPE2_TMATMUL_GET_INFO`.
4. Open and mmap the devdax window.
5. Stage the fixed v2 memory layout.
6. Flush staged cachelines with `clflush`/`sfence`.
7. Submit `CXL_TYPE2_TMATMUL_RUN_CSR_ONLY`.
8. Invalidate output cachelines with `clflush`/`lfence`.
9. Copy the output vector back into the hetGPU allocation.

The module returns structured Rust errors. The caller logs enough context for
debugging and then lets the existing fallback continue.

## Fixed v2 Layout

The implementation uses the layout from the current CXL smoke runner:

| Region | Device physical offset | Size |
| --- | ---: | ---: |
| Matrix | `0x000000` | `dim_d * dim_d / 4` |
| Input vector | `0x100000` | `dim_d * 2` |
| Output vector | `0x200000` | `dim_d * 2` |
| Program | `0x300000` | `6 * 16` |

The backend rejects devices whose devdax size is smaller than
`0x300000 + 96`.

## Program Encoding

The first backend emits the same six 128-bit instructions as the v2 smoke
runner:

1. `ldv v0, input`
2. `tmatmul_import v0`
3. `tmatmul_go matrix`
4. `tmatmul_export v1`
5. `sv v1, output`
6. `stall`

Instruction words are little-endian and match the smoke runner bit layout:

```text
word[0] = addr
word[1] = bits[2:0]=rms, [4:3]=tm, [6:5]=ls,
          [9:7]=va, [12:10]=vb, [15:13]=vy,
          [19:16]=op, [22:20]=fu
```

This intentionally avoids wiring arbitrary generated tmatmul assembly into the
CXL backend before the kernel ABI supports a variable program DPA and length.

## Data Contract

The supported operation uses the device-reported `dim_d`:

- input: `dim_d` elements, 16-bit Q8.8 fixed point
- output: `dim_d` elements, 16-bit Q8.8 fixed point
- matrix: packed 2-bit ternary values, `dim_d * dim_d / 4` bytes

The submit function receives host-backed hetGPU pointers and their remaining
allocation sizes. It validates:

- input has at least `dim_d * 2` bytes
- output has at least `dim_d * 2` bytes
- matrix has at least `dim_d * dim_d / 4` bytes
- devdax covers the fixed v2 layout
- the v2 UAPI version reported by `GET_INFO` is `2`

The backend does not convert f32 tensors in the first port. If a caller has f32
or another dtype, it is unsupported and must fall back.

## Environment

The CXL path is opt-in:

- `HETGPU_CXL_TMATMUL=1` enables submission attempts.
- `HETGPU_CXL_TMATMUL_DEV=/dev/cxl_tmatmul3b000` overrides the control device.
- `HETGPU_CXL_TMATMUL_DAX=/dev/dax0.0` overrides devdax discovery.
- `HETGPU_CXL_TMATMUL_TIMEOUT_MS=10000` overrides the ioctl timeout.
- `HETGPU_CXL_TMATMUL_DISABLE_AFTER_FAILURE=1` disables further CXL attempts
  in the process after the first runtime failure.

If `HETGPU_CXL_TMATMUL` is absent or false, behavior remains unchanged.

## hetGPU Integration

`zluda/src/impl/mod.rs` adds `cxl_tmatmul` behind the same feature gate used by
the current virtual tmatmul path.

`zluda/src/impl/function.rs` adds a narrow call from the existing tmatmul
fallback boundary. The call only happens when:

- `HETGPU_CXL_TMATMUL=1`
- the kernel name/parameters match the supported ternary matmul route
- three tracked allocations can be resolved as output, input, and matrix
- allocation sizes satisfy the data contract

The hook must not remove or weaken the existing interpreter and named CPU
fallbacks. CXL failure is reported as a fallback reason, not as a process-level
CUDA failure.

## Error Handling

All CXL errors are non-fatal to hetGPU unless the caller later opts into strict
behavior. The backend returns explicit errors for:

- no control device
- no devdax device
- open/mmap/ioctl failures
- UAPI version mismatch
- zero `dim_d`
- undersized allocations
- undersized devdax window
- missing `STALLED` result flag
- DMA error result flag
- timeout

The fallback hook logs the first few failures and then returns control to the
existing fallback path. With `HETGPU_CXL_TMATMUL_DISABLE_AFTER_FAILURE=1`, it
sets a process-local atomic flag to skip future CXL attempts after a runtime
failure.

## Testing

Pure tests cover:

- env parsing
- control device selection
- devdax sysfs discovery path handling
- ioctl number and UAPI struct sizes
- fixed DPA layout calculations
- six-instruction program encoding
- allocation-size validation

Hardware-facing checks are gated and not part of default test runs:

- `GET_INFO` succeeds against `/dev/cxl_tmatmul*`
- devdax mmap succeeds when `HETGPU_CXL_TMATMUL_DAX` is set or discoverable
- a submit reaches `STALLED`
- output bytes are copied back after cache invalidation

Default CI and local `cargo test` should not require root or device nodes.

## Non-Goals

- No kernel ABI changes.
- No arbitrary generated tmatmul assembly execution through CXL v2.
- No f32-to-Q8.8 or Q8.8-to-f32 conversion in the first port.
- No changes to the PTX compiler pipeline.
- No removal of interpreter or CPU fallbacks.
