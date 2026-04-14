# AMD AIE Backend for Strix NPU — Design Spec

**Date:** 2026-04-13
**Status:** Draft — awaiting user review
**Target:** AMD Strix NPU (XDNA2 / AIE-ML, via Xilinx/mlir-aie toolchain)

## Goal

Compile PTX kernels to run natively on the AMD Strix NPU. End-to-end path:
PTX source → TOSA MLIR (AIE-shaped) → mlir-aie lowering → XCLBIN → XRT-loaded
kernel on NPU.

First-class support for:

- Native **INT4 matmul** (AIE-ML primitive)
- **Ternary-weight matmul** ({-1, 0, 1} packed weights) via IR-pattern recognition of
  BitNet-style unpack loops — no source annotation required from kernel authors

## Non-Goals

- No AIE Simulator integration. Validation is hardware-only on Strix.
- No fatbin/universal-binary packaging: AIE backend produces XCLBIN, consumed directly.
- No support (yet) for FP16/BF16 matmul, convolutions beyond
  `tosa.conv2d` forwarding, attention fused kernels, or multi-device
  partitioning. Tracked as follow-on work.
- No optimizations beyond what mlir-aie's default pipeline performs.

## Architecture Overview

```
PTX source
  ↓ (ptx_parser, existing)
PTX AST
  ↓ ptx/src/pass/emit_tosa_aie.rs (new)
TOSA MLIR (AIE-shaped: coarse tosa.matmul / tosa.conv2d / tosa.add, ternary ops)
  ↓ ext/aie_comgr-sys (new) — shell out to mlir-aie toolchain
    aie-opt (tosa→linalg→aievec→aie-ml)
    aie-translate (--aie-generate-xclbin)
  ↓
XCLBIN (artifact)
  ↓ ext/aie_runtime-sys (new) — XRT C API bindings
    xrtDeviceLoadXclbin, xrtKernelOpen, xrtRunStart, …
  ↓
Running on Strix NPU
```

Three new crates, one new compiler pass, one new comgr entry point.

## New Crates

### `ext/aie_comgr-sys`

Shell-out driver for the mlir-aie toolchain. Input: TOSA MLIR text. Output:
XCLBIN bytes. Mirrors the structure of `ext/pacc_comgr-sys`.

**Public API:**

```rust
pub struct AieCompileConfig {
    pub device: AieDevice,           // Strix1 (XDNA2); future: StrixHalo, Phoenix
    pub num_cols: u32,               // AIE tile columns (Strix NPU has 4)
    pub num_rows: u32,               // Rows (Strix NPU has 5, row 0 = shim)
    pub extra_aie_opt_flags: Vec<String>,
}

impl Default for AieCompileConfig { fn default() -> Self { Self::strix() } }
impl AieCompileConfig { pub fn strix() -> Self { /* 4 cols, 5 rows */ } }

pub fn compile_tosa_to_xclbin(
    tosa_mlir: &str,
    config: &AieCompileConfig,
) -> Result<Vec<u8>, AieComgrError>;
```

**Internal pipeline (each step = one subprocess invocation, temp files under
`tempfile::tempdir()`):**

1. Write input MLIR to `input.mlir`.
2. `aie-opt --pass-pipeline="builtin.module(tosa-to-linalg, ...)" input.mlir -o lowered.mlir`
3. `aie-opt --aie-objectFifo-stateful-transform --aie-lower-broadcast-packet lowered.mlir -o aie.mlir`
4. `aie-translate --aie-generate-cdo aie.mlir` → `aie_cdo.bin`
5. `aie-translate --aie-generate-ipu aie.mlir` → `ipu_insts.txt`
6. `aie-translate --aie-generate-xclbin --xclbin-name=kernel.xclbin aie.mlir`
7. Read `kernel.xclbin` into `Vec<u8>`, return.

Each step captures stdout+stderr; non-zero exit returns
`AieComgrError::ToolchainFailed { step, stderr, exit_code }` with the full
stderr preserved for diagnosis.

**Toolchain discovery:** `which::which("aie-opt")` at each call. Honor
`AIE_TOOLCHAIN_DIR` env var to override `$PATH` lookup. Fail with a clear
error message pointing to `https://github.com/Xilinx/mlir-aie` install docs
when not found.

**Errors (`AieComgrError`):**

- `ToolchainNotFound(String)` — binary not on PATH / AIE_TOOLCHAIN_DIR
- `ToolchainFailed { step: &'static str, stderr: String, exit_code: i32 }`
- `Io(std::io::Error)` — temp file / I/O issues
- `InvalidInput(String)` — MLIR obviously malformed before we invoke tools

**Dependencies:** `tempfile`, `which`, `thiserror`. No bindgen (pure shell-out).

### `ext/aie_runtime-sys`

Rust bindings to XRT C API for loading XCLBIN and running kernels on Strix.
Mirrors the pattern of `hip_runtime-sys` / `ze_runtime-sys` — a pure `-sys`
crate, no safe-Rust abstractions.

**Crate layout:**

```
ext/aie_runtime-sys/
  Cargo.toml          # deps: bindgen (build), libloading (runtime)
  build.rs            # runs bindgen against XRT headers
  wrapper.h           # #include <xrt/xrt_device.h>, xrt_bo.h, xrt_kernel.h, xrt_hw_context.h
  src/
    lib.rs            # re-exports bindgen output
```

**Binding surface:**

- Device: `xrtDeviceOpen`, `xrtDeviceClose`, `xrtDeviceLoadXclbinFile`,
  `xrtDeviceLoadXclbin`
- HW context: `xrtHwContextOpen`, `xrtHwContextClose`
- Kernel: `xrtKernelOpen`, `xrtKernelClose`, `xrtRunOpen`, `xrtRunSetArg`,
  `xrtRunStart`, `xrtRunWait`, `xrtRunClose`
- Buffer object: `xrtBOAlloc`, `xrtBOFree`, `xrtBOWrite`, `xrtBORead`,
  `xrtBOSync`, `xrtBOMap`

**Link configuration:** `build.rs` tries `pkg-config` for `xrt_coreutil` first
(XRT installs a `.pc` file at `/opt/xilinx/xrt/share/pkgconfig/`). Fallback:
`rustc-link-search=/opt/xilinx/xrt/lib`, `rustc-link-lib=xrt_coreutil`. Fails
at build time if XRT not installed — mirrors how `hip_runtime-sys` requires a
HIP install.

## New Compiler Pass: `ptx/src/pass/emit_tosa_aie.rs`

Sibling of existing `ptx/src/pass/emit_tosa_mlir.rs`. The Tenstorrent path
remains untouched — AIE gets its own emitter because its TOSA requirements
are fundamentally different (coarse-grained ops for mlir-aie to recognize vs.
fine-grained scalar-ish ops for TTIR).

**Signature:**

```rust
pub fn run<'input>(
    id_defs: GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<String, TranslateError>;
```

**Responsibilities:**

1. **Kernel shape analysis.** Build a lightweight CFG per `.entry`, identify
   perfectly-nested loops and their induction-variable ranges. Needed because
   the raising passes below operate on loop nests, not straight-line code.

2. **Matmul raising — two entry points:**
   - **Tensor-core intrinsics:** recognize `mma.sync.*`, `wmma.*`, `hmma.*` and
     map directly to `tosa.matmul` using the tile shapes encoded in the
     mnemonic. E.g., `mma.m16n8k16.f16` → `tosa.matmul` with
     `1x16x16 × 1x16x8`.
   - **Scalar loop-nest matmul (M5):** recognize the triple-nested
     `for i { for j { for k { C[i,j] += A[i,k]*B[k,j] } } }` shape plus common
     variants (accumulator in register, tiled blocks, K-reduction). Emit a
     single `tosa.matmul`.

3. **Ternary unpack pattern recognition (M4).** Match the canonical
   BitNet-style unpack shape:
   - Load of N packed bytes from a weight buffer
   - Shift+mask+subtract sequence producing values in {-1, 0, 1}
   - Multiply with activation tile
   When matched: replace unpack loop + matmul with a single
   `zluda.aie.ternary_matmul` op (or, if mlir-aie has no native ternary op,
   emit `tosa.matmul` with a pre-matmul dequantize prologue).
   Emit a diagnostic comment in the MLIR output when recognition fires, so
   mis-recognition is debuggable.

4. **INT4 matmul direct path.** Any PTX matmul where both inputs are `.s4`/`.u4`
   or packed 4-bit (via `dp4a`-style patterns) lowers to `tosa.matmul` with
   `tensor<MxKxi4>` × `tensor<KxNxi4>` → `tensor<MxNxi32>`.

5. **Elementwise fallback.** PTX ALU ops (add, mul, clamp, shift, …) map 1:1 to
   `tosa.add` / `tosa.mul` / `tosa.clamp` / etc. The type-system and value-map
   helpers from `emit_tosa_mlir.rs` are copied rather than cross-imported to
   keep the file boundaries clean (no shared mutable state between Tenstorrent
   and AIE paths).

6. **Unsupported-op handling.** When the pass can't raise a kernel (data-
   dependent control flow, scalar-only workload with no tensor shape, pattern
   that no raising rule matches) return
   `TranslateError::UnsupportedForAie { op: String }`. No fallback emission —
   aie-opt won't accept scalar code, so failing early is the honest answer.

## comgr Integration

**`comgr/Cargo.toml`:**

```toml
[features]
aie = ["dep:aie_comgr_sys", "dep:aie_runtime_sys"]

[dependencies]
aie_comgr_sys = { path = "../ext/aie_comgr-sys", optional = true }
aie_runtime_sys = { path = "../ext/aie_runtime-sys", optional = true }
```

**`comgr/src/lib.rs`:**

```rust
#[cfg(feature = "aie")]
pub fn compile_bitcode_aie(
    device: &CStr,              // e.g. "strix"
    main_buffer: &[u8],         // PTX text (not LLVM bitcode — see rationale)
    _ptx_impl: &[u8],           // unused, kept for signature symmetry
) -> Result<Vec<u8>, AieComgrError>;

#[cfg(feature = "aie")]
#[derive(Debug)]
pub enum AieComgrError { /* variants re-wrapped from aie_comgr_sys */ }
```

**Input format — PTX text, not LLVM bitcode.** Every other backend takes
bitcode because that's what `ptx` lowers to. AIE is intentionally different:
the raising pass (`emit_tosa_aie.rs`) pattern-matches on PTX shape, which is
more stable than LLVM IR for this purpose (especially for ternary unpack
recognition, where LLVM's own optimizations can obscure the unpack pattern).
The parameter name `main_buffer` is kept for signature symmetry with other
backends but documented as "PTX text" for the AIE entry point.

## ZLUDA Dispatch

ZLUDA's existing per-backend feature flags (`amd`, `intel`, `tenstorrent`,
`nvidia`) select which `compile_bitcode_*` function is called in the module-
load path. Add an `aie` feature to the set, wire into the existing dispatch
site (to be located during implementation — likely `zluda/src/impl/module.rs`
or a sibling). No new abstractions introduced.

## Testing Strategy — Hardware-Only

All validation happens on a physical Strix NPU. No AIE Simulator integration.
No golden-file tests.

- **`ext/aie_runtime-sys/tests/smoke.rs`** — opens device 0, loads a hand-built
  XCLBIN, launches with known inputs, reads back outputs, asserts numerics.
  Gated behind `#[cfg_attr(not(feature = "hw-test"), ignore)]`.
- **`comgr/tests/aie_int4_matmul.rs`** — end-to-end INT4 matmul: compile PTX →
  produce XCLBIN → load via runtime-sys → launch → verify numerics. Gated on
  `hw-test` feature.
- **`comgr/tests/aie_bitnet_gemm.rs`** — M4 gate: reference BitNet kernel
  source compiled to PTX, run through full pipeline, verify ternary
  recognition fires (via diagnostic-comment check on emitted MLIR) and output
  numerics match reference.

CI cannot run these. Developers running them locally set
`AIE_HW_TEST_DEVICE=/dev/accel/accel0` and pass `--features hw-test`.

## Risks

1. **Ternary pattern-recognition brittleness.** Recognition depends on IR
   shape, which can vary with PTX optimization level and source coding style.
   Mitigation: stage as M4, after M3 has proven the rest of the stack. If
   pattern matching proves unworkable, fall back to a `.pragma`-based escape
   hatch — non-trivial change but contained to `emit_tosa_aie.rs`.
2. **mlir-aie API drift.** Xilinx/mlir-aie's pass pipeline changes between
   releases. Mitigation: pin a specific mlir-aie commit in install
   documentation; keep the aie-opt pass-pipeline string as one constant in
   `aie_comgr-sys` so version bumps are a one-line edit.
3. **XRT + amdxdna kernel driver compatibility.** Strix NPU support in
   upstream XRT is recent and still evolving. Mitigation: M3 pins tested XRT
   and kernel driver versions in documentation; hw-test is the only
   regression signal (per the hardware-only testing choice).
4. **Hardware-only testing leaves no safety net.** No CI coverage means
   regressions are caught only during manual developer runs. Mitigation: keep
   each pipeline stage small and composable so individual components can be
   spot-checked during development without running the full pipeline.

## Milestones

- **M1 — TOSA emission:** `emit_tosa_aie.rs` with mma-intrinsic matmul raising,
  INT4 matmul, and elementwise passes. Scalar loop raising and ternary
  recognition deliberately out of scope.
- **M2 — Toolchain driver:** `ext/aie_comgr-sys` complete; drive a full
  `aie-opt`/`aie-translate` round-trip from hand-written TOSA to XCLBIN.
  Proves toolchain integration independent of the raising pass.
- **M3 — End-to-end INT4 matmul on Strix:** `ext/aie_runtime-sys` + comgr
  entry point wired. Go/no-go gate for the full stack.
- **M4 — Ternary pattern recognition:** unpack-pattern matcher + BitNet
  reference kernel end-to-end on Strix.
- **M5 — Scalar loop-nest matmul raising:** broaden input surface beyond
  mma-intrinsic CUDA code.

## Open Questions

- Exact Strix NPU configuration on the user's hardware (column/row count)
  — verify during M2.
- Pinned mlir-aie commit hash — pick during M2 once we exercise the toolchain.
- Ternary recognition: which exact IR shape to match first — decide during
  M4 with a reference BitNet kernel in hand.
