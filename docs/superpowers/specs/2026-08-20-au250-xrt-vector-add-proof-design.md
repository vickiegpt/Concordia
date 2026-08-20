# AU250 XRT Four-BO Vector-Add Proof Design

## Goal

Prove the Rust `xrt_tmatmul` backend on the live AU250 by reproducing the known-good PYNQ vector-add result through `au250-run`. This is the hardware gate before replacing the add instruction with ternary-matmul instructions.

## Live Platform Contract

The authoritative artifact is `/au250_xrt/example/asym9_bs9_2641toks.xclbin`, already proven by `pynq_add_example.py`. Its relevant contract is:

- kernel `ternip_big`, with compute unit `ternip_big_1` connected to bank0;
- XRT kernel selector `ternip_big:{ternip_big_1}`;
- one kernel-local register aperture, so `HETGPU_XRT_INSTANCE=0`;
- the connected memory group is obtained from pointer argument 4 (`MM2S_SA`), so `HETGPU_XRT_MEMORY_ARG=4`;
- instruction DMA registers at `MM2S_DMACR=0x0000`, `MM2S_SA=0x0018`, and `MM2S_LENGTH=0x0028`;
- completion at `STALL=0x1000` and accelerator reset release at `RESET=0x2000`;
- vector shape `(9, 1024)` with signed little-endian 16-bit fixed point and exponent -5.

The xclbin has three `ternip_big` instances. This proof deliberately selects only `ternip_big_1`, matching the known-good example's bank0 placement. Multi-CU scheduling is outside this proof.

## Implementation

The existing four-BO ABI remains unchanged:

1. Matrix BO carries vector A.
2. Input BO carries vector B.
3. Output BO carries vector C.
4. Program BO carries the assembled 128-bit instruction image.

The proof assembly is:

```text
ldv v0, PARAM_MATRIX
ldv v1, PARAM_INPUT
add v2, v0, v1
sv v2, PARAM_OUTPUT
stall
```

The existing assembler binds the three labels to the real XRT BO addresses and emits little-endian 128-bit instructions. The backend transfers the complete replay-safe program image returned by that assembler.

Before starting instruction DMA, the backend writes zero to the selected instance's reset register at offset `0x2000`. This matches the known-good Python launch. The reset write is part of the common repository accelerator contract and remains instance-relative, like STALL and the DMA registers.

An ignored Rust hardware test is added in `xrt_tmatmul.rs`. It constructs the same values as the Python proof without depending on NumPy:

- A repeats floating-point values 0 through 7, encoded as raw fixed-point values 0, 32, ..., 224;
- B is 1.5 everywhere, encoded as raw value 48;
- expected C is the exact signed 16-bit sum.

The test calls `submit_xrt_tmatmul` directly and requires an explicit opt-in environment variable so ordinary test runs never program hardware.

## Execution Through `au250-run`

The `app215` container has XRT and PYNQ but no Rust toolchain. A binary linked on the gpu01 host is not portable into that container: the host uses glibc 2.43, the container uses glibc 2.35, and the current host test binary requires symbols through `GLIBC_2.39`. The proof therefore compiles and links the Rust test inside `au250-run` so the executable uses the same userspace ABI as the XRT runtime.

A host-side wrapper uses only ignored paths below `target/` for execution artifacts. It stages the host's backward-compatible Ninja executable there, bootstraps Rust 1.92.0 with the minimal profile into a persistent `target/au250-runtime` cache on the first run, and uses a separate `target/au250-app215` Cargo target directory. All compilation and linking occur inside `app215`; subsequent runs reuse those caches. No toolchain or generated binary is committed.

The run procedure is:

1. Check the AU250 temperature through the wrapper's existing guard.
2. Enter `app215` through `au250-run`, bootstrap or reuse the cached Rust toolchain, and build the ignored test with the Intel feature configuration already used for the backend tests.
3. Execute the generated test in the same `au250-run` container with:
   - `HETGPU_XRT_XCLBIN=/au250_xrt/example/asym9_bs9_2641toks.xclbin`
   - `HETGPU_XRT_KERNEL=ternip_big:{ternip_big_1}`
   - `HETGPU_XRT_INSTANCE=0`
   - `HETGPU_XRT_MEMORY_ARG=4`
   - the hardware-test opt-in variable enabled.
4. Run only the ignored AU250 vector-add test with exact output enabled.

The wrapper's existing 85 C temperature guard and device/container mapping remain authoritative. The implementation does not bypass either.

## Error Handling and Cleanup

Existing XRT errors remain fail-closed. A reset-register failure, BO failure, DMA launch failure, timeout, output synchronization failure, or incorrect output fails the test. Output is checked only after STALL and device-to-host synchronization.

RAII cleanup continues to release the four BOs, kernel, device, and dynamic-library handle in reverse ownership order. The test does not modify or remove the read-only xclbin.

## Acceptance Gates

The work is complete only when all of the following hold:

1. Unit tests verify that reset release occurs before DMA start and retain the existing four-BO, address-binding, timeout, and cleanup assertions.
2. Existing `cxl_tmatmul` tests still pass.
3. The app215-built test executable starts inside `au250-run` and resolves the container XRT library.
4. The live test selects `ternip_big_1`, completes with a nonzero STALL code, and reports exact equality for all 9 x 1024 output elements.
5. FPGA temperature remains below the wrapper guard and the post-run device state shows no fatal error.

This vector-add gate proves the real Rust four-BO/XRT path, BO address encoding, instruction DMA, accelerator execution, and output readback. It does not yet prove ternary-matmul numeric correctness; that is the next program-level gate.
