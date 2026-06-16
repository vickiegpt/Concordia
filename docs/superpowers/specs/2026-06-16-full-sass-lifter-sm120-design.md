# Full SASS Lifter for SM120 to PTX

**Date:** 2026-06-16
**Status:** Approved for planning
**Scope:** Implement a shared SASS-to-PTX lifter that handles both raw SM120 CUBIN input and cuobjdump/nvdisasm-style SASS text input.

## Goals

1. Add one shared lifter core for SASS instruction semantics to PTX text.
2. Support both existing input frontends:
   - raw CUBIN bytes through `CubinParser` and `SassDisassembler`
   - disassembly text through `TextDisassemblyParser`
3. Emit complete PTX modules, not isolated instruction snippets.
4. Make unsupported instructions visible through diagnostics and conservative comment fallbacks.
5. Keep existing `sass_inliner --recover-ptx` and `gpu_rr ptx` behavior compatible while moving them onto the shared lifter.

## Non-Goals

- Perfect decompilation back to original source PTX.
- Semantic equivalence for every Blackwell-only tensor or TMA instruction in the first pass.
- Dependence on NVIDIA tools in normal tests.
- Replacing the existing SASS-to-LLVM inliner.
- Rewriting the existing CUBIN parser or SM120 assembler crate.

## Existing Context

The repo already has the pieces needed for the front half of the pipeline:

- `ptx/src/sass/cubin_parser.rs` parses CUBIN ELF sections and extracts kernel code.
- `ptx/src/sass/disassembler.rs` decodes raw SASS bytes and parses cuobjdump-style text into `EnhancedSassInstruction`.
- `ptx/src/sass/instruction.rs` classifies opcodes, parses operands, and stores PTX template metadata.
- `ptx/src/sass/ptx_recovery.rs` has a per-instruction `PtxReconstructor`, but module emission and frontend integration are currently spread across CLI code.
- `ptx/src/bin/sass_inliner.rs` exposes `--recover-ptx`.
- `ptx/src/bin/gpu_rr.rs` exposes a separate PTX recovery path.

The new work should consolidate PTX lifting behavior instead of creating a parallel standalone tool.

## Architecture

Add a new module:

```text
ptx/src/sass/lifter.rs
```

This module owns the complete SASS-to-PTX pipeline after input has become `Vec<EnhancedSassInstruction>`.

Public API:

```rust
pub struct SassLiftOptions {
    pub sm_version: u32,
    pub kernel_name: String,
    pub include_sass_comments: bool,
    pub emit_unsupported_comments: bool,
}

pub struct SassLiftDiagnostic {
    pub address: Option<u64>,
    pub opcode: String,
    pub message: String,
}

pub struct SassLiftResult {
    pub ptx: String,
    pub diagnostics: Vec<SassLiftDiagnostic>,
}

pub fn lift_instructions_to_ptx(
    instructions: &[EnhancedSassInstruction],
    options: &SassLiftOptions,
) -> SassLiftResult;

pub fn lift_sass_text_to_ptx(
    text: &str,
    options: SassLiftOptions,
) -> Result<SassLiftResult, String>;

pub fn lift_cubin_to_ptx(
    cubin_data: &[u8],
    options: SassLiftOptions,
) -> Result<SassLiftResult, String>;
```

`lift_sass_text_to_ptx` uses `TextDisassemblyParser`. `lift_cubin_to_ptx` uses `CubinParser` and `SassDisassembler`, then calls `lift_instructions_to_ptx`.

## PTX Module Emission

The lifter emits a complete PTX module:

```ptx
.version 8.5
.target sm_120
.address_size 64

.visible .entry kernel()
{
    .reg .b32 %r<...>;
    .reg .b64 %rd<...>;
    .reg .pred %p<...>;

L_0000:
    // 0x0000: S2R
    mov.u32 %r0, %tid.x;
}
```

Register declarations are conservative and derived from observed SASS registers:

- `R*` registers map to `.b32 %r<N>`.
- address-like 64-bit memory operands can use `.b64 %rd<N>` when the source register syntax or instruction form implies 64-bit addressing.
- predicate registers map to `.pred %p<N>`.
- zero registers and `PT` are not declared.

Labels are generated for all instruction addresses and branch targets using `L_<hex address>`.

## Instruction Semantics

The first implementation covers common scalar, memory, control, and synchronization instructions:

- integer arithmetic and logic: `IADD`, `IADD3`, `IMAD`, `IMUL`, `SHL`, `SHR`, `LOP3`, `POPC`
- floating point: `FADD`, `FMUL`, `FFMA`, `FABS`, `FNEG`, `MUFU` fallbacks where the sub-op is unknown
- memory: `LDG`, `STG`, `LDS`, `STS`, `LDL`, `STL`, `LDC`
- data movement: `MOV`, `MOV32I`, `SEL`, `PRMT`
- special registers: `S2R`, `CS2R` for thread/block/lane/clock registers
- predicates and comparisons: `ISETP`, `FSETP`, `PSETP` in conservative `setp` form
- control flow: `BRA`, `BRX`, `JMP`, `RET`, `EXIT`
- synchronization: `BAR`, `DEPBAR`, `MEMBAR`
- atomics and tensor instructions: diagnostic plus PTX comment fallback unless a direct safe mapping is already obvious

The lifter must preserve predicates on lifted lines:

```ptx
@%p0 bra L_0050;
@!%p1 add.s32 %r2, %r3, %r4;
```

## Operand Normalization

The lifter normalizes operand roles before emitting PTX:

- Loads: destination is a register, source is memory or constant memory.
- Stores: destination is memory, source is value. Text parsing already puts the first operand in `dest_operands`; the lifter must not treat stores like loads.
- Branches: immediate, label, and address operands become `L_<target>`.
- Constant bank operands such as `c[0x0][0x160]` are preserved as comments or lifted to conservative symbolic operands when no parameter metadata is available.
- Special registers map to PTX names such as `%tid.x`, `%ctaid.x`, `%ntid.x`, `%nctaid.x`, `%laneid`, `%warpid`, `%clock`, and `%clock_hi`.

## Diagnostics

Unsupported or lossy instructions produce a `SassLiftDiagnostic`. The PTX output includes a comment fallback when enabled:

```ptx
// unsupported SASS HMMA at 0x0120: tensor instruction lifting is not implemented
// SASS: HMMA ...
```

This is preferable to emitting a misleading PTX instruction.

## CLI Integration

Update `ptx/src/bin/sass_inliner.rs`:

- In `--recover-ptx --stdin`, call `lift_sass_text_to_ptx`.
- In `--recover-ptx` for CUBIN input, call `lift_cubin_to_ptx`.
- Preserve `--ptx-output`, stdout behavior, `--sm`, and verbose diagnostic reporting.

Update `ptx/src/bin/gpu_rr.rs` only where it has a duplicate recovery path, so both CLIs use the same lifter.

## Public Exports

Export the lifter from:

- `ptx/src/sass/mod.rs`
- `ptx/src/lib.rs`

This gives downstream code a stable library path:

```rust
ptx::lift_sass_text_to_ptx(...)
ptx::lift_cubin_to_ptx(...)
```

## Testing

Tests should be text-first and not require CUDA tools:

1. Unit tests in `ptx/src/sass/lifter.rs` for individual instruction lifting.
2. Parser-plus-lifter tests using cuobjdump-style strings for:
   - arithmetic
   - memory load/store
   - special registers
   - predicates and branches
   - barriers
   - unsupported tensor fallback
3. CLI tests extending `ptx/tests/sass_inliner_test.rs` for `--recover-ptx --stdin --sm 120`.
4. A raw CUBIN integration test should use local synthetic CUBIN bytes only if the existing `nvidia_sass::cubin_builder` can produce bytes parseable by `CubinParser` without external tools. If that path is too brittle, keep CUBIN coverage at the library boundary with a malformed-input error test and rely on existing parser tests.

## Acceptance Criteria

- `sass_inliner --stdin --recover-ptx --sm 120 -` emits `.target sm_120` and a complete PTX module.
- The same lifter core is used for text input and CUBIN input.
- Common SM120 scalar and memory SASS snippets lift to readable PTX.
- Branch targets use generated labels.
- Unsupported instructions are reported in diagnostics and comments.
- Tests can run with `cargo test -p ptx sass` or narrower commands without requiring NVIDIA CUDA tools.
