# CXL tmatmul v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrow hetGPU backend that submits one D x D ternary matmul through the existing CXL Type-2 tmatmul v2 ABI.

**Architecture:** Add `zluda/src/impl/cxl_tmatmul.rs` for the CXL v2 userspace protocol and hook it from the current Intel virtual tmatmul fallback path. The backend is opt-in, validates the v2 data contract, and falls back to existing behavior on unsupported shapes or runtime errors.

**Tech Stack:** Rust 2021, `libc`, CUDA/ZLUDA virtual allocation tracking, Linux misc-device ioctl, devdax mmap.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `zluda/src/impl/cxl_tmatmul.rs` | CXL v2 constants, env parsing, device/devdax discovery, DPA layout validation, instruction encoding, ioctl/mmap submission, pure tests |
| `zluda/src/impl/mod.rs` | Declare the new module behind `#[cfg(feature = "intel")]` |
| `zluda/src/impl/function.rs` | Add helper definitions missing from the current fallback path and call the CXL submitter before CPU matmul fallback when enabled |

### Task 1: CXL v2 Pure Backend Surface

**Files:**
- Create: `zluda/src/impl/cxl_tmatmul.rs`
- Modify: `zluda/src/impl/mod.rs`
- Test: `zluda/src/impl/cxl_tmatmul.rs`

- [ ] **Step 1: Write failing pure tests**

Add tests that assert:

```rust
#[test]
fn encode_smoke_program_matches_v2_runner() {
    let prog = encode_smoke_program();
    assert_eq!(prog.len(), 96);
    assert_eq!(&prog[0..8], &TMATMUL_DPA_INPUT.to_le_bytes());
    assert_eq!(&prog[32..40], &TMATMUL_DPA_MATRIX.to_le_bytes());
    assert_eq!(&prog[64..72], &TMATMUL_DPA_OUTPUT.to_le_bytes());
}

#[test]
fn layout_requires_program_end() {
    assert_eq!(required_dax_len(), TMATMUL_DPA_PROGRAM as usize + TMATMUL_PROGRAM_BYTES);
}

#[test]
fn validates_supported_sizes() {
    let dim = 512;
    assert_eq!(matrix_bytes(dim).unwrap(), 512 * 512 / 4);
    assert_eq!(vector_bytes(dim).unwrap(), 512 * 2);
    assert!(validate_allocations(dim, 1024, 1024, 65536).is_ok());
    assert!(validate_allocations(dim, 1023, 1024, 65536).is_err());
}
```

- [ ] **Step 2: Run tests to verify red**

Run: `cargo test -p zluda --features intel --no-default-features cxl_tmatmul -- --nocapture`

Expected: FAIL because `cxl_tmatmul` and its functions are not implemented yet.

- [ ] **Step 3: Add minimal backend constants and pure helpers**

Implement constants for UAPI v2, DPA layout, instruction encoding, env parsing, and validation. Keep hardware I/O behind methods that are not exercised by pure tests.

- [ ] **Step 4: Expose the module**

Add to `zluda/src/impl/mod.rs`:

```rust
#[cfg(feature = "intel")]
pub(crate) mod cxl_tmatmul;
```

- [ ] **Step 5: Run tests to verify green**

Run: `cargo test -p zluda --features intel --no-default-features cxl_tmatmul -- --nocapture`

Expected: PASS for pure CXL tests.

### Task 2: CXL Runtime Submitter

**Files:**
- Modify: `zluda/src/impl/cxl_tmatmul.rs`

- [ ] **Step 1: Add runtime API**

Implement:

```rust
pub(crate) struct TernaryMatmulJob {
    pub input: *const u8,
    pub input_size: usize,
    pub matrix: *const u8,
    pub matrix_size: usize,
    pub output: *mut u8,
    pub output_size: usize,
}

pub(crate) fn enabled() -> bool;
pub(crate) unsafe fn submit(job: TernaryMatmulJob) -> Result<(), CxlTmatmulError>;
```

- [ ] **Step 2: Add Linux syscall protocol**

Implement `open`, `ioctl(GET_INFO)`, devdax discovery, `mmap`, `memcpy`, cache flush/invalidate, `ioctl(RUN_CSR_ONLY)`, and cleanup via `Drop`.

- [ ] **Step 3: Keep hardware tests gated**

Add an ignored test:

```rust
#[test]
#[ignore]
fn cxl_get_info_if_device_present() {
    let cfg = Config::from_env().unwrap();
    let info = get_info_for_test(&cfg).unwrap();
    assert_eq!(info.version, CXL_TYPE2_TMATMUL_UAPI_VERSION);
    assert!(info.dim_d > 0);
}
```

- [ ] **Step 4: Run pure tests**

Run: `cargo test -p zluda --features intel --no-default-features cxl_tmatmul -- --nocapture`

Expected: PASS with the hardware test ignored.

### Task 3: Function Fallback Hook

**Files:**
- Modify: `zluda/src/impl/function.rs`

- [ ] **Step 1: Make existing tmatmul fallback helpers coherent**

Define the currently referenced helper functions if they are still absent:

```rust
#[cfg(feature = "intel")]
fn tmatmul_named_fallback_enabled() -> bool { ... }

#[cfg(feature = "intel")]
fn tmatmul_hardware_matmul_enabled() -> bool { ... }

#[cfg(feature = "intel")]
fn tmatmul_is_matmul_kernel_name(name_lower: &str) -> bool { ... }
```

- [ ] **Step 2: Add CXL attempt before CPU matmul fallback**

Inside `execute_tmatmul_hardware_matmul_fallback`, scan tracked allocation pointers, choose output/input/matrix using allocation sizes, and call:

```rust
let job = super::cxl_tmatmul::TernaryMatmulJob {
    input: input_ptr as *const u8,
    input_size,
    matrix: matrix_ptr as *const u8,
    matrix_size,
    output: output_ptr as *mut u8,
    output_size,
};
match super::cxl_tmatmul::submit(job) { ... }
```

- [ ] **Step 3: Preserve fallback behavior**

If the CXL path is disabled, unsupported, or fails, log the reason and return to the existing CPU/interpreter fallback without returning a CUDA error.

- [ ] **Step 4: Run targeted compile/test**

Run: `cargo test -p zluda --features intel --no-default-features cxl_tmatmul -- --nocapture`

Expected: PASS.

### Task 4: Final Verification

**Files:**
- Verify: `zluda/src/impl/cxl_tmatmul.rs`
- Verify: `zluda/src/impl/function.rs`
- Verify: `zluda/src/impl/mod.rs`

- [ ] **Step 1: Format changed Rust files**

Run: `cargo fmt --package zluda`

Expected: no formatting errors.

- [ ] **Step 2: Run focused tests**

Run: `cargo test -p zluda --features intel --no-default-features cxl_tmatmul -- --nocapture`

Expected: PASS.

- [ ] **Step 3: Check status**

Run: `git status --short`

Expected: only intended implementation files are changed in addition to pre-existing user changes.
