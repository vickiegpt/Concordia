# XRT tmatmul Backend Design

## Goal

Add an isolated Rust backend that submits the existing hetGPU tmatmul assembly to the XRT-packaged `ternip_ip` accelerator in `./ternary_matmul`. The backend must preserve the existing CXL implementation and use the XRT interface that the current ternary-matmul bitstream actually exposes.

## Scope

The implementation adds `zluda/src/impl/xrt_tmatmul.rs` and the smallest module declaration needed to compile and test it. It does not modify `cxl_tmatmul.rs`, change the tmatmul instruction format, add a new RTL wrapper, or redefine the xclbin kernel metadata.

The backend is initially callable as a Rust module. Routing existing CUDA/ZLUDA launches to it is outside this change; the caller can be wired separately after the backend contract is validated.

## Source Contract

The authoritative XRT behavior comes from `./ternary_matmul`:

- `synth/pynqvivado_common/package_kernel.tcl` packages `ternip_ip` with `user_managed` control.
- `synth/pynqvivado_common/generate_kernel_xml.tcl` exposes the instruction DMA source at `MM2S_SA` offset `0x18`, its byte length at `MM2S_LENGTH` offset `0x28`, and the stall register at `0x1000`.
- `sw_utils/target/test_pynqvivado_basic.py` allocates data and instruction buffers in a connected DDR bank, assembles instructions after allocation using physical buffer addresses, synchronizes the buffers, starts the instruction DMA, waits for a stall, and then synchronizes results back to the host.
- `sw_utils/lib/asm.py` emits little-endian 128-bit instructions with 64-bit DDR addresses.

Consequently, the backend must use kernel register access rather than model `ternip_ip` as an ordinary four-argument `xrtRun` kernel.

## Four-BO Host Interface

Each submission owns four XRT buffer objects:

1. Matrix BO: read by tmatmul operations.
2. Input BO: read by vector load instructions.
3. Output BO: written by vector store instructions.
4. Program BO: contains the assembled little-endian 128-bit instruction stream.

These are a host-side submission contract, not four formal xclbin kernel arguments. All BOs are allocated in the DDR memory group connected to the selected `ternip_ip` instance. The backend obtains the group from the kernel metadata rather than assuming a bank number.

The submit API accepts:

- assembly text;
- the matrix, input, and output label names, such as `PARAM_0`, `PARAM_1`, and `PARAM_2`;
- matrix and input byte slices;
- a mutable output byte slice;
- an optional timeout through configuration.

The backend allocates the BOs before assembly, queries their XRT device addresses, binds the three supplied labels to those addresses, and calls `cxl_tmatmul::assemble_tmatmul_program`. This preserves the existing hetGPU instruction encoder while replacing the CXL logical DPA bindings with real XRT BO addresses.

## XRT Loading and Configuration

The Rust file runtime-loads XRT's C API from `libxrt_coreutil.so.2`, with `libxrt_coreutil.so` as a fallback. Runtime loading keeps ordinary hetGPU builds independent of the XRT linker configuration.

Configuration is fail-closed:

- `HETGPU_XRT_XCLBIN` is required and identifies the compatible `kernel.xclbin`.
- `HETGPU_XRT_DEVICE_INDEX` defaults to `0`.
- `HETGPU_XRT_KERNEL` defaults to `ternip_ip`.
- `HETGPU_XRT_INSTANCE` defaults to `0` and selects the per-instance register stride.
- `HETGPU_XRT_TIMEOUT_MS` defaults to `10000`.

The selected instance uses the repository's `0x4000` register stride. Its instruction-DMA and stall offsets are relative to that instance base.

## Submission Flow

The backend performs one submission in this order:

1. Load XRT functions and validate configuration.
2. Open the device, load the xclbin, obtain its UUID, and open `ternip_ip` exclusively.
3. Determine a compatible DDR memory group from the kernel metadata.
4. Allocate and populate matrix, input, and output BOs.
5. Query their device addresses and assemble the program with those address labels.
6. Validate the program and allocate/populate the program BO.
7. Synchronize matrix, input, and program BOs to the device.
8. Reset/start the instruction DMA, write the program BO address to `MM2S_SA`, and write the exact program byte length to `MM2S_LENGTH`.
9. Poll `STALL` until it becomes nonzero or the timeout expires.
10. Synchronize the output BO from the device, copy it into the caller's output slice, and acknowledge the stall.
11. Release BO, kernel, device, and dynamic-library handles in reverse ownership order.

The program length is not hard-coded by XRT. It must be nonzero and a multiple of 16 bytes. The existing assembler currently produces its replay-safe, terminal-stall image, and this backend transfers exactly the returned byte count.

## Errors and Cleanup

The backend returns a dedicated `XrtTmatmulError` with operation-specific context for configuration, dynamic loading, device/xclbin/kernel access, BO allocation or synchronization, assembly, register access, timeout, and invalid program errors.

Null handles, invalid memory-group results, zero BO addresses, address-width overflow, empty buffers, and malformed program lengths are rejected before launch. Output is copied to the caller only after a detected stall and a successful device-to-host synchronization.

Every XRT handle is wrapped in a small RAII owner. Partial initialization therefore releases all resources without requiring duplicated cleanup branches. A timeout returns an error and never reports the output as valid.

## Testing and Proof Boundary

Tests use an injected XRT function table so they can verify behavior without an FPGA. Focused tests cover:

- configuration parsing and defaults;
- label binding to BO device addresses;
- 128-bit program length validation;
- memory-group selection;
- instance-stride and register-offset calculation;
- transfer and launch ordering;
- timeout behavior;
- cleanup after failures at each owned-resource stage.

A real-ABI compile check verifies that the declared C signatures match the installed XRT headers closely enough to build. These tests prove host-side contract and error behavior only. Hardware completion and numeric correctness require a compatible `ternip_ip` xclbin and Alveo device and must be reported separately.
