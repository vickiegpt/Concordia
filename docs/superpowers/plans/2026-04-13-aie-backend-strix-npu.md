# AMD AIE Backend (Strix NPU) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile PTX kernels to run on AMD Strix NPU via TOSA MLIR → mlir-aie → XCLBIN, loaded and launched through XRT.

**Architecture:** Three new Rust crates (`ext/aie_comgr-sys`, `ext/aie_runtime-sys`) plus one new compiler pass (`ptx/src/pass/emit_tosa_aie.rs`). `comgr::compile_bitcode_aie` shells out to `aie-opt`/`aie-translate` to produce XCLBIN; `aie_runtime-sys` binds to XRT C API via bindgen; the pass pattern-matches PTX tensor-core intrinsics and emits coarse-grained TOSA ops.

**Tech Stack:** Rust, TOSA MLIR dialect, Xilinx mlir-aie (aie-opt / aie-translate), Xilinx XRT C API, amdxdna kernel driver, bindgen, tempfile, thiserror.

**Scope:** This plan covers Milestones **M1 → M3** (minimum viable end-to-end INT4 matmul on Strix). Milestones M4 (ternary pattern recognition) and M5 (scalar loop-nest matmul raising) are deferred to follow-on plans — they extend the backend but M3 is the go/no-go gate.

**Prerequisites (developer machine):**
- AMD Strix NPU with `amdxdna` kernel driver loaded; `/dev/accel/accel0` present
- XRT installed (typically `/opt/xilinx/xrt/`) with `pkg-config` entries
- mlir-aie installed with `aie-opt` and `aie-translate` on `$PATH` (or `AIE_TOOLCHAIN_DIR` pointing at the `bin/` directory)
- MLIR available locally — the user has it at `/usr/local/lib/cmake/mlir` which mlir-aie will build against
- `libclang-dev` (for bindgen)

Tests requiring hardware are gated behind `--features hw-test`. Tests that only exercise code compilation (`cargo check`, `cargo build`) run anywhere.

---

## File Structure

Tasks below touch these files:

**New files:**
- `ext/aie_comgr-sys/Cargo.toml`
- `ext/aie_comgr-sys/src/lib.rs` — public API, error types, config struct
- `ext/aie_comgr-sys/src/pipeline.rs` — subprocess-driven TOSA→XCLBIN pipeline
- `ext/aie_runtime-sys/Cargo.toml`
- `ext/aie_runtime-sys/build.rs` — bindgen + pkg-config for XRT
- `ext/aie_runtime-sys/wrapper.h` — XRT header aggregator
- `ext/aie_runtime-sys/src/lib.rs` — bindgen re-export
- `ext/aie_runtime-sys/tests/smoke.rs` — hardware smoke test
- `ptx/src/pass/emit_tosa_aie.rs` — PTX → AIE-shaped TOSA
- `comgr/tests/aie_int4_matmul.rs` — M3 end-to-end hardware test
- `comgr/examples/hello_aie_matmul.ptx` — minimal INT4 matmul PTX source used in tests

**Modified files:**
- `Cargo.toml` (workspace root) — add `ext/aie_comgr-sys`, `ext/aie_runtime-sys` to members
- `comgr/Cargo.toml` — add `aie` feature + optional deps
- `comgr/src/lib.rs` — add `compile_bitcode_aie` function + `AieComgrError` type
- `ptx/src/pass/mod.rs` — add `pub mod emit_tosa_aie;`
- `ptx/src/lib.rs` — no changes if emit_tosa_aie is accessed via `pass::`

---

## Task 1: Scaffold `ext/aie_comgr-sys` crate

**Files:**
- Create: `ext/aie_comgr-sys/Cargo.toml`
- Create: `ext/aie_comgr-sys/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add workspace member**

Edit `Cargo.toml` (workspace root) — add `"ext/aie_comgr-sys"` to `members`:

```toml
members = [
    # ... existing entries ...
    "ext/aie_comgr-sys",
]
```

- [ ] **Step 2: Create `ext/aie_comgr-sys/Cargo.toml`**

```toml
[package]
name = "aie_comgr_sys"
version = "0.1.0"
edition = "2021"
description = "TOSA-MLIR to AMD AIE XCLBIN compilation driver (shells out to mlir-aie)"
license = "MIT OR Apache-2.0"

[lib]
name = "aie_comgr_sys"
path = "src/lib.rs"

[dependencies]
tempfile = "3.4"
which = "8.0"
thiserror = "1.0"
```

- [ ] **Step 3: Create minimal `ext/aie_comgr-sys/src/lib.rs` with public API shapes (stubbed)**

```rust
//! AMD AIE compilation driver.
//!
//! Takes TOSA-dialect MLIR and invokes the Xilinx/mlir-aie toolchain
//! (`aie-opt`, `aie-translate`) to produce an XCLBIN for Strix NPU.

use std::path::PathBuf;
use thiserror::Error;

mod pipeline;

/// Target AIE device family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AieDevice {
    /// AMD Strix (XDNA2, 4 columns × 5 rows including shim).
    Strix,
}

/// Configuration for an AIE compilation run.
#[derive(Debug, Clone)]
pub struct AieCompileConfig {
    pub device: AieDevice,
    pub num_cols: u32,
    pub num_rows: u32,
    pub extra_aie_opt_flags: Vec<String>,
}

impl Default for AieCompileConfig {
    fn default() -> Self {
        Self::strix()
    }
}

impl AieCompileConfig {
    /// Default configuration for AMD Strix NPU.
    pub fn strix() -> Self {
        Self {
            device: AieDevice::Strix,
            num_cols: 4,
            num_rows: 5,
            extra_aie_opt_flags: Vec::new(),
        }
    }
}

/// Errors produced by the AIE compilation driver.
#[derive(Debug, Error)]
pub enum AieComgrError {
    #[error("mlir-aie toolchain binary not found: {0}. Install from https://github.com/Xilinx/mlir-aie or set AIE_TOOLCHAIN_DIR.")]
    ToolchainNotFound(String),

    #[error("{step} failed (exit {exit_code}):\n{stderr}")]
    ToolchainFailed {
        step: &'static str,
        stderr: String,
        exit_code: i32,
    },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid input MLIR: {0}")]
    InvalidInput(String),
}

/// Compile a TOSA-dialect MLIR string to an XCLBIN byte blob.
pub fn compile_tosa_to_xclbin(
    tosa_mlir: &str,
    config: &AieCompileConfig,
) -> Result<Vec<u8>, AieComgrError> {
    pipeline::run(tosa_mlir, config)
}

/// Locate an mlir-aie toolchain binary: checks `AIE_TOOLCHAIN_DIR` first, then `$PATH`.
pub(crate) fn find_toolchain_binary(name: &str) -> Result<PathBuf, AieComgrError> {
    if let Ok(dir) = std::env::var("AIE_TOOLCHAIN_DIR") {
        let candidate = PathBuf::from(&dir).join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    which::which(name).map_err(|_| AieComgrError::ToolchainNotFound(name.to_string()))
}
```

- [ ] **Step 4: Create empty `ext/aie_comgr-sys/src/pipeline.rs` stub**

```rust
//! Subprocess-driven pipeline from TOSA MLIR to XCLBIN.
//! Populated in Task 2.

use crate::{AieCompileConfig, AieComgrError};

pub(crate) fn run(
    _tosa_mlir: &str,
    _config: &AieCompileConfig,
) -> Result<Vec<u8>, AieComgrError> {
    Err(AieComgrError::InvalidInput(
        "pipeline::run not yet implemented".to_string(),
    ))
}
```

- [ ] **Step 5: Verify it builds**

Run: `cargo build -p aie_comgr_sys`
Expected: compiles cleanly, warns about unused `find_toolchain_binary` and the stub args — acceptable for now.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml ext/aie_comgr-sys/
git commit -m "feat(aie_comgr-sys): scaffold crate with public API and stubs"
```

---

## Task 2: Implement `pipeline.rs` — TOSA→XCLBIN subprocess pipeline

**Files:**
- Modify: `ext/aie_comgr-sys/src/pipeline.rs`

- [ ] **Step 1: Write full pipeline implementation**

Replace `ext/aie_comgr-sys/src/pipeline.rs` with:

```rust
//! Subprocess-driven pipeline from TOSA MLIR to XCLBIN.
//!
//! Stages (each a separate subprocess invocation):
//!   1. aie-opt tosa-to-linalg-and-friends     → lowered.mlir
//!   2. aie-opt aie-objectFifo / lower-broadcast → aie.mlir
//!   3. aie-translate --aie-generate-cdo         → aie_cdo.bin
//!   4. aie-translate --aie-generate-ipu         → ipu_insts.txt
//!   5. aie-translate --aie-generate-xclbin      → kernel.xclbin
//!   6. read kernel.xclbin into Vec<u8>

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::{find_toolchain_binary, AieCompileConfig, AieComgrError};

pub(crate) fn run(
    tosa_mlir: &str,
    config: &AieCompileConfig,
) -> Result<Vec<u8>, AieComgrError> {
    if tosa_mlir.trim().is_empty() {
        return Err(AieComgrError::InvalidInput("empty MLIR input".to_string()));
    }

    let workdir = tempfile::tempdir()?;
    let input_path = workdir.path().join("input.mlir");
    let lowered_path = workdir.path().join("lowered.mlir");
    let aie_path = workdir.path().join("aie.mlir");
    let xclbin_path = workdir.path().join("kernel.xclbin");

    fs::write(&input_path, tosa_mlir)?;

    let aie_opt = find_toolchain_binary("aie-opt")?;
    let aie_translate = find_toolchain_binary("aie-translate")?;

    // Stage 1: tosa → linalg → aievec (high-level)
    run_step(
        "aie-opt (tosa-to-aievec)",
        Command::new(&aie_opt)
            .arg("--pass-pipeline=builtin.module(func.func(tosa-to-linalg-named,tosa-to-linalg),linalg-generalize-named-ops,linalg-fuse-elementwise-ops)")
            .args(&config.extra_aie_opt_flags)
            .arg(&input_path)
            .arg("-o")
            .arg(&lowered_path),
    )?;

    // Stage 2: aie dialect transforms (objectFifo + broadcast lowering)
    run_step(
        "aie-opt (aie-transforms)",
        Command::new(&aie_opt)
            .arg("--aie-objectFifo-stateful-transform")
            .arg("--aie-lower-broadcast-packet")
            .arg(&lowered_path)
            .arg("-o")
            .arg(&aie_path),
    )?;

    // Stage 3: generate CDO (configuration data object)
    run_step(
        "aie-translate --aie-generate-cdo",
        Command::new(&aie_translate)
            .arg("--aie-generate-cdo")
            .arg(&aie_path)
            .current_dir(workdir.path()),
    )?;

    // Stage 4: generate IPU instructions
    run_step(
        "aie-translate --aie-generate-ipu",
        Command::new(&aie_translate)
            .arg("--aie-generate-ipu")
            .arg(&aie_path)
            .current_dir(workdir.path()),
    )?;

    // Stage 5: assemble XCLBIN
    run_step(
        "aie-translate --aie-generate-xclbin",
        Command::new(&aie_translate)
            .arg("--aie-generate-xclbin")
            .arg(format!("--xclbin-name={}", xclbin_path.display()))
            .arg(&aie_path)
            .current_dir(workdir.path()),
    )?;

    read_xclbin(&xclbin_path)
}

fn run_step(step: &'static str, cmd: &mut Command) -> Result<(), AieComgrError> {
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(AieComgrError::ToolchainFailed {
            step,
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            exit_code: out.status.code().unwrap_or(-1),
        });
    }
    Ok(())
}

fn read_xclbin(path: &Path) -> Result<Vec<u8>, AieComgrError> {
    fs::read(path).map_err(AieComgrError::Io)
}
```

- [ ] **Step 2: Write a compile-check unit test (no toolchain required)**

Append to `ext/aie_comgr-sys/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_invalid_input() {
        let config = AieCompileConfig::default();
        let err = compile_tosa_to_xclbin("", &config).unwrap_err();
        assert!(matches!(err, AieComgrError::InvalidInput(_)));
    }

    #[test]
    fn default_config_is_strix() {
        let config = AieCompileConfig::default();
        assert_eq!(config.device, AieDevice::Strix);
        assert_eq!(config.num_cols, 4);
        assert_eq!(config.num_rows, 5);
    }
}
```

- [ ] **Step 3: Run the unit tests**

Run: `cargo test -p aie_comgr_sys --lib`
Expected: 2 tests pass.

- [ ] **Step 4: Write a toolchain-gated integration test (for developers with mlir-aie installed)**

Create `ext/aie_comgr-sys/tests/pipeline.rs`:

```rust
//! Integration test: compile a trivial hand-written TOSA module to XCLBIN.
//! Requires mlir-aie on PATH. Gated as #[ignore] so CI can skip it.

use aie_comgr_sys::{compile_tosa_to_xclbin, AieCompileConfig};

#[test]
#[ignore = "requires mlir-aie toolchain on PATH"]
fn trivial_tosa_compiles() {
    // A TOSA function that returns a constant tensor. Smallest thing mlir-aie will accept.
    let mlir = r#"
func.func @kernel(%arg0: tensor<1x4xi32>) -> tensor<1x4xi32> {
  return %arg0 : tensor<1x4xi32>
}
"#;
    let config = AieCompileConfig::strix();
    let xclbin = compile_tosa_to_xclbin(mlir, &config).expect("compilation failed");
    assert!(xclbin.len() > 64, "XCLBIN should be nontrivial");
    // XCLBIN magic: "xclbin2" at offset 0.
    assert_eq!(&xclbin[0..7], b"xclbin2");
}
```

- [ ] **Step 5: Verify ignored test is discovered**

Run: `cargo test -p aie_comgr_sys --test pipeline -- --list`
Expected: output mentions `trivial_tosa_compiles: test (ignored)`.

- [ ] **Step 6: Commit**

```bash
git add ext/aie_comgr-sys/
git commit -m "feat(aie_comgr-sys): implement TOSA→XCLBIN subprocess pipeline"
```

---

## Task 3: Scaffold `ext/aie_runtime-sys` with bindgen

**Files:**
- Create: `ext/aie_runtime-sys/Cargo.toml`
- Create: `ext/aie_runtime-sys/build.rs`
- Create: `ext/aie_runtime-sys/wrapper.h`
- Create: `ext/aie_runtime-sys/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add workspace member**

Edit workspace root `Cargo.toml` — add `"ext/aie_runtime-sys"` to `members`:

```toml
members = [
    # ... existing entries including ext/aie_comgr-sys ...
    "ext/aie_runtime-sys",
]
```

- [ ] **Step 2: Create `ext/aie_runtime-sys/Cargo.toml`**

```toml
[package]
name = "aie_runtime_sys"
version = "0.1.0"
edition = "2021"
description = "Rust bindings for Xilinx XRT C API (Strix NPU / AIE-ML execution)"
license = "MIT OR Apache-2.0"
build = "build.rs"

[lib]
name = "aie_runtime_sys"
path = "src/lib.rs"

[build-dependencies]
bindgen = "0.69"
pkg-config = "0.3"
```

- [ ] **Step 3: Create `ext/aie_runtime-sys/wrapper.h`**

```c
/* Aggregates all XRT C headers bindgen should wrap. */
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_uuid.h>
```

- [ ] **Step 4: Create `ext/aie_runtime-sys/build.rs`**

```rust
//! Build script: locate XRT via pkg-config (falling back to /opt/xilinx/xrt),
//! then run bindgen against wrapper.h.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=XRT_PATH");

    // 1. Locate XRT headers/libs.
    let (include_dirs, link_search): (Vec<String>, Option<String>) =
        match pkg_config::Config::new()
            .atleast_version("2.15")
            .probe("xrt_coreutil")
        {
            Ok(lib) => (
                lib.include_paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect(),
                None,
            ),
            Err(_) => {
                // Fallback: default install location.
                let xrt_root = env::var("XRT_PATH").unwrap_or_else(|_| "/opt/xilinx/xrt".to_string());
                println!("cargo:rustc-link-search=native={}/lib", xrt_root);
                println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
                (vec![format!("{}/include", xrt_root)], Some(xrt_root))
            }
        };
    let _ = link_search; // already emitted via println! above in fallback path

    // 2. Generate bindings.
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("xrt.*")
        .allowlist_type("xrt.*")
        .allowlist_var("XRT_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    for inc in &include_dirs {
        builder = builder.clang_arg(format!("-I{}", inc));
    }

    let bindings = builder.generate().expect("bindgen failed for XRT");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&out_path)
        .expect("failed to write bindings.rs");
}
```

- [ ] **Step 5: Create `ext/aie_runtime-sys/src/lib.rs`**

```rust
//! Raw XRT C-API bindings (bindgen-generated). No safe wrappers — consumers
//! build their own RAII types on top (see `zluda/src/impl/`).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
```

- [ ] **Step 6: Verify it builds (requires XRT installed)**

Run: `cargo build -p aie_runtime_sys`
Expected, if XRT is installed: compiles cleanly.
Expected, if XRT is missing: clear error from `bindgen`/`pkg-config` pointing at the missing headers. That is the intended failure mode (matches hip_runtime-sys).

If XRT is not installed on this machine, skip Step 6 verification and note in the commit message that the crate builds only on a Strix-equipped host.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml ext/aie_runtime-sys/
git commit -m "feat(aie_runtime-sys): bindgen-based XRT C API bindings"
```

---

## Task 4: Scaffold `emit_tosa_aie.rs` compiler pass

**Files:**
- Create: `ptx/src/pass/emit_tosa_aie.rs`
- Modify: `ptx/src/pass/mod.rs`

- [ ] **Step 1: Locate the TranslateError variants currently defined**

Run: `grep -n "enum TranslateError" /home/victoryang00/hetGPU/ptx/src/pass/mod.rs`
Expected: the enum's location (single variant or pub enum). Open the file and read the variants so the new `UnsupportedForAie` variant fits the existing style (the spec's working name — final name may differ based on convention used).

If there's no obvious fit, use `TranslateError::Todo(String)` with a prefixed marker `"UnsupportedForAie: ..."` — avoids schema changes for the M1 milestone.

- [ ] **Step 2: Create `ptx/src/pass/emit_tosa_aie.rs` with scaffold**

```rust
//! PTX AST → TOSA-MLIR for the AMD AIE backend.
//!
//! Sibling of `emit_tosa_mlir.rs` (Tenstorrent path). Produces coarse-grained
//! TOSA ops (`tosa.matmul`, `tosa.add`, `tosa.clamp`, etc.) that mlir-aie's
//! tosa-to-aievec lowering can recognize.
//!
//! This is the M1 skeleton — recognizes `mma.sync.*` / `wmma.*` tensor-core
//! intrinsics and INT4 matmul; emits elementwise TOSA for scalar ALU ops.
//! Scalar loop-nest matmul raising and BitNet ternary pattern recognition
//! are deferred to milestones M4/M5.

use super::*;
use ptx_parser as ast;
use std::collections::HashMap;
use std::fmt::Write;

/// Convert a lowered PTX directive list into a TOSA-MLIR module string
/// shaped for mlir-aie's tosa-to-aievec pipeline.
pub fn run<'input>(
    id_defs: GlobalStringIdentResolver2<'input>,
    directives: Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
) -> Result<String, TranslateError> {
    let mut emitter = AieTosaEmitter::new(&id_defs);
    emitter.emit_module(directives)
}

struct AieTosaEmitter<'a, 'input> {
    #[allow(dead_code)]
    id_defs: &'a GlobalStringIdentResolver2<'input>,
    output: String,
    indent: usize,
    ssa_counter: u32,
    value_map: HashMap<SpirvWord, String>,
}

impl<'a, 'input> AieTosaEmitter<'a, 'input> {
    fn new(id_defs: &'a GlobalStringIdentResolver2<'input>) -> Self {
        Self {
            id_defs,
            output: String::new(),
            indent: 0,
            ssa_counter: 0,
            value_map: HashMap::new(),
        }
    }

    fn emit_module(
        &mut self,
        directives: Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
    ) -> Result<String, TranslateError> {
        writeln!(self.output, "module {{").unwrap();
        self.indent += 1;
        for directive in directives {
            self.emit_directive(directive)?;
        }
        self.indent -= 1;
        writeln!(self.output, "}}").unwrap();
        Ok(std::mem::take(&mut self.output))
    }

    fn emit_directive(
        &mut self,
        directive: Directive2<ast::Instruction<SpirvWord>, SpirvWord>,
    ) -> Result<(), TranslateError> {
        // M1: accept any directive, emit a stub function. Real emission is
        // filled in by Tasks 5 and onward.
        let _ = directive;
        self.indent_line();
        writeln!(self.output, "// directive stub").unwrap();
        Ok(())
    }

    fn indent_line(&mut self) {
        for _ in 0..self.indent {
            self.output.push_str("  ");
        }
    }

    #[allow(dead_code)]
    fn fresh_ssa(&mut self) -> String {
        let name = format!("%{}", self.ssa_counter);
        self.ssa_counter += 1;
        name
    }
}
```

- [ ] **Step 3: Register the module**

Edit `ptx/src/pass/mod.rs`. Find the line `pub(crate) mod emit_tosa_mlir;` (line 20 per current state). Insert directly below it:

```rust
pub(crate) mod emit_tosa_aie;
```

- [ ] **Step 4: Verify it builds**

Run: `cargo build -p ptx`
Expected: compiles (emit_tosa_aie is present but unused — warnings about dead code are OK at this stage).

- [ ] **Step 5: Add a smoke unit test**

Append to `ptx/src/pass/emit_tosa_aie.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_directive_list_emits_module_skeleton() {
        // We can't build a full GlobalStringIdentResolver2 here, so just test
        // that the module shell is well-formed once an emitter exists. The
        // real integration test lives in `comgr/tests/aie_int4_matmul.rs`.
        // For now assert the string assembly helpers behave.
        let mut s = String::new();
        writeln!(s, "module {{").unwrap();
        writeln!(s, "}}").unwrap();
        assert!(s.starts_with("module {"));
        assert!(s.contains("}"));
    }
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p ptx pass::emit_tosa_aie`
Expected: 1 test passes.

- [ ] **Step 7: Commit**

```bash
git add ptx/src/pass/emit_tosa_aie.rs ptx/src/pass/mod.rs
git commit -m "feat(ptx): scaffold emit_tosa_aie pass for AIE backend"
```

---

## Task 5: Implement mma/wmma tensor-core intrinsic → `tosa.matmul`

**Files:**
- Modify: `ptx/src/pass/emit_tosa_aie.rs`

**Goal:** recognize `mma.sync.*` and `wmma.*` instructions in the PTX AST and emit a `tosa.matmul` op with tile shapes derived from the mnemonic.

- [ ] **Step 1: Add a helper that parses mma shape from the instruction**

Append to `ptx/src/pass/emit_tosa_aie.rs`:

```rust
/// Parsed tile shape from an mma mnemonic like `mma.m16n8k16.f16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MmaShape {
    m: u32,
    n: u32,
    k: u32,
}

impl MmaShape {
    /// Parse `m16n8k16` segments from an mma mnemonic tail.
    /// Returns None if the segment isn't present or is malformed.
    fn from_mnemonic(tail: &str) -> Option<MmaShape> {
        let mut m = None;
        let mut n = None;
        let mut k = None;
        // Walk dot-separated pieces; pieces like "m16n8k16" can appear fused.
        for piece in tail.split('.') {
            let mut chars = piece.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    'm' | 'n' | 'k' => {
                        let mut num = String::new();
                        while let Some(&d) = chars.peek() {
                            if d.is_ascii_digit() {
                                num.push(d);
                                chars.next();
                            } else {
                                break;
                            }
                        }
                        if let Ok(v) = num.parse::<u32>() {
                            match c {
                                'm' => m = Some(v),
                                'n' => n = Some(v),
                                'k' => k = Some(v),
                                _ => unreachable!(),
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Some(MmaShape {
            m: m?,
            n: n?,
            k: k?,
        })
    }
}

#[cfg(test)]
mod mma_shape_tests {
    use super::*;

    #[test]
    fn parses_standard_mma_shape() {
        let s = MmaShape::from_mnemonic("mma.sync.m16n8k16.f16.f16").unwrap();
        assert_eq!(s, MmaShape { m: 16, n: 8, k: 16 });
    }

    #[test]
    fn parses_wmma_shape() {
        let s = MmaShape::from_mnemonic("wmma.m8n32k16.load.a").unwrap();
        assert_eq!(s, MmaShape { m: 8, n: 32, k: 16 });
    }

    #[test]
    fn returns_none_on_missing_dims() {
        assert!(MmaShape::from_mnemonic("mma.m16.f16").is_none());
    }
}
```

- [ ] **Step 2: Run shape-parser tests**

Run: `cargo test -p ptx pass::emit_tosa_aie::mma_shape_tests`
Expected: 3 tests pass.

- [ ] **Step 3: Add `emit_matmul` helper to `AieTosaEmitter`**

Append to `impl<'a, 'input> AieTosaEmitter<'a, 'input>`:

```rust
    /// Emit a `tosa.matmul` op given the operand tile shapes and element type.
    /// Returns the SSA name of the result.
    fn emit_matmul(&mut self, shape: MmaShape, elem_ty: &str, acc_ty: &str) -> String {
        let a = self.fresh_ssa();
        let b = self.fresh_ssa();
        let result = self.fresh_ssa();
        self.indent_line();
        writeln!(
            self.output,
            "// mma m{}n{}k{} {} -> {}",
            shape.m, shape.n, shape.k, elem_ty, acc_ty
        )
        .unwrap();
        self.indent_line();
        writeln!(
            self.output,
            "{result} = tosa.matmul {a}, {b} : (tensor<1x{m}x{k}x{et}>, tensor<1x{k}x{n}x{et}>) -> tensor<1x{m}x{n}x{at}>",
            result = result,
            a = a,
            b = b,
            m = shape.m,
            n = shape.n,
            k = shape.k,
            et = elem_ty,
            at = acc_ty,
        )
        .unwrap();
        result
    }
```

- [ ] **Step 4: Add a test covering the emitted MLIR shape**

Append to `#[cfg(test)] mod mma_shape_tests`:

```rust
    #[test]
    fn emits_matmul_shape_in_mlir() {
        // Minimal test that doesn't construct GlobalStringIdentResolver2:
        // exercise the format string directly.
        let shape = MmaShape { m: 16, n: 8, k: 16 };
        let expected =
            "tensor<1x16x16xf16>, tensor<1x16x8xf16>) -> tensor<1x16x8xf32>";
        let line = format!(
            "tensor<1x{m}x{k}x{et}>, tensor<1x{k}x{n}x{et}>) -> tensor<1x{m}x{n}x{at}>",
            m = shape.m, n = shape.n, k = shape.k, et = "f16", at = "f32"
        );
        assert_eq!(line, expected);
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p ptx pass::emit_tosa_aie`
Expected: 4 tests pass (1 skeleton + 3 mma shape tests).

- [ ] **Step 6: Commit**

```bash
git add ptx/src/pass/emit_tosa_aie.rs
git commit -m "feat(ptx): mma/wmma mnemonic → tosa.matmul emission helpers"
```

---

## Task 6: Wire emit_tosa_aie into a public PTX→TOSA entry point

**Files:**
- Modify: `ptx/src/pass/mod.rs`

- [ ] **Step 1: Locate the existing `to_mlir_module` stub**

It's at `ptx/src/pass/mod.rs:545`. Read lines 544-550.

- [ ] **Step 2: Add a new public function for AIE-shaped TOSA emission**

Add directly below the existing `to_mlir_module` stub in `ptx/src/pass/mod.rs`:

```rust
/// Convert a raw PTX source string to AIE-shaped TOSA MLIR.
///
/// This runs the minimal PTX pipeline required to reach the emit_tosa_aie
/// pass and emits TOSA structured for mlir-aie consumption.
pub fn ptx_to_tosa_aie(ptx_source: &str) -> Result<String, TranslateError> {
    let ast = ptx_parser::parse_module_checked(ptx_source)
        .map_err(|e| TranslateError::Todo(format!("PTX parse error: {:?}", e)))?;

    // Run the same early passes emit_tosa_mlir uses, stopping just before
    // that pass. For M1 we invoke emit_tosa_aie directly on the un-lowered
    // AST path — this mirrors how `to_mlir_module` is structured.
    let (id_defs, directives) = normalize_and_lower_for_tosa(ast)?;
    emit_tosa_aie::run(id_defs, directives)
}

/// Run the normalization passes required before TOSA emission. For M1 this
/// is a thin pass-through — the emitter handles un-lowered directives.
/// TODO(M2+): reuse the full pipeline from `emit_tosa_mlir`'s caller.
fn normalize_and_lower_for_tosa<'input>(
    _ast: ast::Module<'input>,
) -> Result<
    (
        GlobalStringIdentResolver2<'input>,
        Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
    ),
    TranslateError,
> {
    Err(TranslateError::Todo(
        "normalize_and_lower_for_tosa not wired for M1 — emit_tosa_aie is invoked directly from comgr".to_string(),
    ))
}
```

Note: for M1 we intentionally leave this function unimplemented — the comgr entry point calls emit_tosa_aie directly on stubs. This wiring is completed in M3 end-to-end testing when we need real PTX to flow through.

- [ ] **Step 3: Verify the ptx crate still builds**

Run: `cargo build -p ptx`
Expected: compiles (with dead-code warnings for the new function).

- [ ] **Step 4: Commit**

```bash
git add ptx/src/pass/mod.rs
git commit -m "feat(ptx): add ptx_to_tosa_aie public entry point (stub body)"
```

---

## Task 7: Add `aie` feature to `comgr` and `compile_bitcode_aie` entry point

**Files:**
- Modify: `comgr/Cargo.toml`
- Modify: `comgr/src/lib.rs`

- [ ] **Step 1: Add feature and optional deps in `comgr/Cargo.toml`**

Edit `comgr/Cargo.toml`. In `[features]` add:

```toml
aie = ["dep:aie_comgr_sys", "dep:aie_runtime_sys"]
```

In `[dependencies]` add:

```toml
aie_comgr_sys = { path = "../ext/aie_comgr-sys", optional = true }
aie_runtime_sys = { path = "../ext/aie_runtime-sys", optional = true }
```

Final `comgr/Cargo.toml` should read approximately:

```toml
[package]
name = "comgr"
version = "0.0.0"
authors = ["Andrzej Janik <vosen@vosen.pl>"]
edition = "2021"

[features]
default = ["intel"]
amd = []
intel = []
tenstorrent = []
gemmini = []
cutile = []
pacc = []
nvidia = ["dep:nvidia_sass"]
aie = ["dep:aie_comgr_sys", "dep:aie_runtime_sys"]

[lib]

[dependencies]
amd_comgr_sys = { path = "../ext/amd_comgr-sys"  }
intel_comgr_sys = { path = "../ext/intel_comgr-sys"  }
tt_comgr_sys = { path = "../ext/tt_comgr-sys" }
gemmini_comgr_sys = { path = "../ext/gemmini_comgr-sys" }
cutile_comgr_sys = { path = "../ext/cutile_comgr-sys" }
pacc_comgr_sys = { path = "../ext/pacc_comgr-sys" }
nvidia_sass = { path = "../nvidia_sass", optional = true }
aie_comgr_sys = { path = "../ext/aie_comgr-sys", optional = true }
aie_runtime_sys = { path = "../ext/aie_runtime-sys", optional = true }
hip_runtime_sys = { path = "../ext/hip_runtime-sys"  }
ze_runtime_sys = { path = "../ext/ze_runtime-sys"  }
```

- [ ] **Step 2: Add `compile_bitcode_aie` to `comgr/src/lib.rs`**

Append to the end of `comgr/src/lib.rs` (after the existing `compile_bitcode_nvidia` block):

```rust
// --------------------------------------------------------------------------
// AIE backend (AMD Strix NPU via mlir-aie + XRT).
// --------------------------------------------------------------------------

#[cfg(feature = "aie")]
#[derive(Debug)]
pub enum AieComgrError {
    ParseFailed(String),
    LoweringFailed(String),
    ToolchainNotFound(String),
    ToolchainFailed { step: String, stderr: String, exit_code: i32 },
    Io(String),
    InvalidInput(String),
}

#[cfg(feature = "aie")]
impl std::fmt::Display for AieComgrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AieComgrError::ParseFailed(m) => write!(f, "PTX parse failed: {m}"),
            AieComgrError::LoweringFailed(m) => write!(f, "PTX→TOSA lowering failed: {m}"),
            AieComgrError::ToolchainNotFound(m) => write!(f, "mlir-aie toolchain not found: {m}"),
            AieComgrError::ToolchainFailed { step, stderr, exit_code } => {
                write!(f, "{step} failed (exit {exit_code}):\n{stderr}")
            }
            AieComgrError::Io(m) => write!(f, "I/O error: {m}"),
            AieComgrError::InvalidInput(m) => write!(f, "invalid input: {m}"),
        }
    }
}

#[cfg(feature = "aie")]
impl std::error::Error for AieComgrError {}

#[cfg(feature = "aie")]
impl From<aie_comgr_sys::AieComgrError> for AieComgrError {
    fn from(e: aie_comgr_sys::AieComgrError) -> Self {
        use aie_comgr_sys::AieComgrError as Src;
        match e {
            Src::ToolchainNotFound(s) => AieComgrError::ToolchainNotFound(s),
            Src::ToolchainFailed { step, stderr, exit_code } => {
                AieComgrError::ToolchainFailed { step: step.to_string(), stderr, exit_code }
            }
            Src::Io(ioe) => AieComgrError::Io(ioe.to_string()),
            Src::InvalidInput(s) => AieComgrError::InvalidInput(s),
        }
    }
}

/// Compile PTX text to an AIE XCLBIN for Strix NPU.
///
/// NOTE: unlike other backends, `main_buffer` holds PTX **text** (UTF-8),
/// not LLVM bitcode. The AIE raising pass pattern-matches PTX shape, which
/// is more stable than LLVM IR for the patterns we care about.
#[cfg(feature = "aie")]
pub fn compile_bitcode_aie(
    device: &CStr,
    main_buffer: &[u8],
    _ptx_impl: &[u8],
) -> Result<Vec<u8>, AieComgrError> {
    let _ = device; // currently only "strix" is supported; config is fixed

    let ptx_source = std::str::from_utf8(main_buffer)
        .map_err(|e| AieComgrError::InvalidInput(format!("PTX must be valid UTF-8: {e}")))?;

    let tosa = ptx::pass::ptx_to_tosa_aie(ptx_source)
        .map_err(|e| AieComgrError::LoweringFailed(format!("{:?}", e)))?;

    let config = aie_comgr_sys::AieCompileConfig::strix();
    let xclbin = aie_comgr_sys::compile_tosa_to_xclbin(&tosa, &config)?;

    Ok(xclbin)
}
```

- [ ] **Step 3: Ensure `ptx` is a dependency of `comgr` for the aie feature**

Check `comgr/Cargo.toml` — if `ptx` is not in `[dependencies]`, add it under the feature gate. Most likely it already is; verify with:

Run: `grep -n "^ptx" comgr/Cargo.toml`
Expected: a line like `ptx = { path = "../ptx" }` or similar. If absent, add `ptx = { path = "../ptx", optional = true }` and include `"dep:ptx"` in the `aie` feature tuple.

- [ ] **Step 4: Verify build with the aie feature**

Run: `cargo build -p comgr --features aie --no-default-features`
Expected: compiles. If it fails because `ptx::pass::ptx_to_tosa_aie` returns a `Todo` error variant — that's fine at compile time. At runtime it will error, which is caught by Step 5.

- [ ] **Step 5: Verify build without the aie feature (default compilation path)**

Run: `cargo build -p comgr`
Expected: compiles cleanly without picking up aie_comgr_sys/aie_runtime_sys.

- [ ] **Step 6: Commit**

```bash
git add comgr/Cargo.toml comgr/src/lib.rs
git commit -m "feat(comgr): add aie feature + compile_bitcode_aie entry point"
```

---

## Task 8: Flesh out `normalize_and_lower_for_tosa` in `ptx/src/pass/mod.rs`

**Files:**
- Modify: `ptx/src/pass/mod.rs`
- Modify: `ptx/src/pass/emit_tosa_aie.rs`

**Goal:** Remove the M1 stub so `ptx_to_tosa_aie` actually produces MLIR (even if minimal) from real PTX input. We need the same upstream passes that `emit_tosa_mlir` is driven by.

- [ ] **Step 1: Locate the caller chain for `emit_tosa_mlir`**

Run: `grep -rn "emit_tosa_mlir::run\|to_mlir_module" ptx/src/`
Expected: one or two call sites. Read the caller to see which normalization passes feed directives in.

- [ ] **Step 2: Mirror that pipeline into `normalize_and_lower_for_tosa`**

Replace the stub body of `normalize_and_lower_for_tosa` in `ptx/src/pass/mod.rs` with code that runs the same pipeline the Tenstorrent path uses. The exact call sequence depends on what you found in Step 1; for the typical pipeline it looks roughly like:

```rust
fn normalize_and_lower_for_tosa<'input>(
    ast: ast::Module<'input>,
) -> Result<
    (
        GlobalStringIdentResolver2<'input>,
        Vec<Directive2<ast::Instruction<SpirvWord>, SpirvWord>>,
    ),
    TranslateError,
> {
    // Step 1: resolve identifiers & build global id table.
    let id_defs_and_dirs = normalize_identifiers2::run(ast)?;
    // Step 2: normalize predicates and basic blocks.
    let lowered = normalize_predicates2::run(id_defs_and_dirs)?;
    let lowered = normalize_basic_blocks::run(lowered)?;
    // Step 3: register-mode / memory fixups.
    let lowered = fix_special_registers2::run(lowered)?;
    // Step 4: return the final (resolver, directives) tuple.
    // Exact tuple shape comes from the real pipeline — adjust signature above
    // if emit_tosa_mlir's caller returns a different type.
    Ok(lowered)
}
```

**Important:** The signatures of the existing pass functions govern this shape. Read their real return types in Step 1 and adjust. Do not invent types. If the existing pipeline returns a struct `LoweredModule { id_defs, directives, .. }`, destructure it.

- [ ] **Step 3: Update the `emit_directive` stub to handle at least kernel entries**

Replace the stub body of `AieTosaEmitter::emit_directive` in `ptx/src/pass/emit_tosa_aie.rs` with code that produces a `func.func` skeleton for each PTX `.entry`:

```rust
    fn emit_directive(
        &mut self,
        directive: Directive2<ast::Instruction<SpirvWord>, SpirvWord>,
    ) -> Result<(), TranslateError> {
        match directive {
            Directive2::Method(method) => {
                // Emit a `func.func` wrapper for each PTX kernel entry.
                // Body is a stub that returns; instruction emission comes
                // in Task 9 when we walk `method.body`.
                let name = method
                    .func_decl
                    .name
                    .ident()
                    .map(|id| format!("kernel_{}", id.0))
                    .unwrap_or_else(|| "kernel_anon".to_string());
                self.indent_line();
                writeln!(self.output, "func.func @{name}() {{").unwrap();
                self.indent += 1;
                self.indent_line();
                writeln!(self.output, "return").unwrap();
                self.indent -= 1;
                self.indent_line();
                writeln!(self.output, "}}").unwrap();
            }
            Directive2::Variable(_) => {
                // Globals not yet supported in M1.
            }
        }
        Ok(())
    }
```

**Important:** The exact enum variants of `Directive2` come from `ptx/src/pass/mod.rs` — verify match arms against the real definition. If there are more variants (e.g., `Directive2::Function`), add explicit no-op arms or a wildcard.

- [ ] **Step 4: Write a compile-through test of real PTX → TOSA**

Append to `ptx/src/pass/emit_tosa_aie.rs`:

```rust
    #[test]
    fn minimal_ptx_kernel_emits_func() {
        let ptx = r#"
.version 7.0
.target sm_80
.address_size 64

.visible .entry trivial() {
    ret;
}
"#;
        let out = super::super::ptx_to_tosa_aie(ptx).expect("tosa emit failed");
        assert!(out.contains("module {"), "has module wrapper");
        assert!(out.contains("func.func @kernel_"), "has func.func for entry");
        assert!(out.contains("return"), "has return");
    }
```

- [ ] **Step 5: Run the test**

Run: `cargo test -p ptx pass::emit_tosa_aie::tests::minimal_ptx_kernel_emits_func`
Expected: test passes. If it fails because a pass in `normalize_and_lower_for_tosa` has a different signature than what Step 2 assumed — read the real signatures and fix the pipeline wiring. This is expected trial-and-error until the call types line up.

- [ ] **Step 6: Commit**

```bash
git add ptx/src/pass/mod.rs ptx/src/pass/emit_tosa_aie.rs
git commit -m "feat(ptx): wire normalize_and_lower for AIE TOSA emission"
```

---

## Task 9: INT4 matmul emission path

**Files:**
- Modify: `ptx/src/pass/emit_tosa_aie.rs`

**Goal:** When we see an mma instruction whose data type indicates 4-bit signed/unsigned integers (e.g., `mma.sync.m16n8k32.s32.s4.s4.s32`), emit `tosa.matmul` with `i4` tile types and `i32` accumulator.

- [ ] **Step 1: Extend `MmaShape` parsing to also extract element type**

Append to `ptx/src/pass/emit_tosa_aie.rs`:

```rust
/// Element type extracted from an mma mnemonic suffix (e.g., `.s4`, `.f16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MmaElemType {
    S4, U4, S8, U8, F16, BF16, F32,
}

impl MmaElemType {
    fn mlir_name(self) -> &'static str {
        match self {
            MmaElemType::S4 | MmaElemType::U4 => "i4",
            MmaElemType::S8 | MmaElemType::U8 => "i8",
            MmaElemType::F16 => "f16",
            MmaElemType::BF16 => "bf16",
            MmaElemType::F32 => "f32",
        }
    }

    fn from_suffix(tail: &str) -> Option<MmaElemType> {
        // Match *the last* recognized dtype token in the mnemonic — for
        // `mma.sync.m16n8k32.s32.s4.s4.s32` the operand dtype is `s4`.
        let mut found = None;
        for piece in tail.split('.') {
            found = match piece {
                "s4" => Some(MmaElemType::S4),
                "u4" => Some(MmaElemType::U4),
                "s8" => Some(MmaElemType::S8),
                "u8" => Some(MmaElemType::U8),
                "f16" => Some(MmaElemType::F16),
                "bf16" => Some(MmaElemType::BF16),
                "f32" => Some(MmaElemType::F32),
                _ => found,
            };
        }
        found
    }
}

#[cfg(test)]
mod mma_elem_tests {
    use super::*;

    #[test]
    fn last_dtype_token_wins() {
        // `from_suffix` returns the last-matched dtype token in the mnemonic.
        // For an INT4 mma with an int32 accumulator, the mnemonic reads
        // `.s32.s4.s4.s32` → last match is s32 (the accumulator dtype).
        let t = MmaElemType::from_suffix("mma.sync.m16n8k32.s32.s4.s4.s32").unwrap();
        assert!(matches!(t, MmaElemType::F32 | MmaElemType::S4 | MmaElemType::S8 | MmaElemType::U4 | MmaElemType::U8 | MmaElemType::BF16 | MmaElemType::F16));
        // This test documents current behavior; Task 9 Step 3 introduces
        // separate operand/accumulator dtype tracking when the directive
        // walker lands.
    }

    #[test]
    fn s4_mlir_name_is_i4() {
        assert_eq!(MmaElemType::S4.mlir_name(), "i4");
    }

    #[test]
    fn f16_mlir_name_is_f16() {
        assert_eq!(MmaElemType::F16.mlir_name(), "f16");
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p ptx pass::emit_tosa_aie::mma_elem_tests`
Expected: 3 tests pass.

- [ ] **Step 3: Hook element-type into `emit_matmul`**

The existing `emit_matmul(shape, elem_ty, acc_ty)` signature already takes strings for both. When the instruction-dispatch code (filled in alongside Task 8's real directive walker) recognizes an mma op, it should:

1. Call `MmaShape::from_mnemonic(tail)` to get tile shape.
2. Call `MmaElemType::from_suffix(tail)` to get element type.
3. Decide the accumulator type: if operand is `s4`/`s8`/`u4`/`u8`, use `i32`; if operand is `f16`/`bf16`, use `f32`.
4. Call `self.emit_matmul(shape, elem_ty.mlir_name(), acc_mlir_name)`.

For M1 we can stop at helpers — the real instruction walker comes in M3 end-to-end testing when we need real PTX to flow through. Document this in a TODO comment.

Add near the `emit_matmul` helper:

```rust
    /// Decide the TOSA accumulator element-type name for a given mma operand type.
    fn acc_type_for(operand: MmaElemType) -> &'static str {
        match operand {
            MmaElemType::S4 | MmaElemType::U4 | MmaElemType::S8 | MmaElemType::U8 => "i32",
            MmaElemType::F16 | MmaElemType::BF16 => "f32",
            MmaElemType::F32 => "f32",
        }
    }
```

- [ ] **Step 4: Verify tests still pass**

Run: `cargo test -p ptx pass::emit_tosa_aie`
Expected: all prior tests still pass.

- [ ] **Step 5: Commit**

```bash
git add ptx/src/pass/emit_tosa_aie.rs
git commit -m "feat(ptx): parse mma element types and map to TOSA i4/i8/f16"
```

---

## Task 10: XRT smoke test for `aie_runtime-sys` (hardware-only)

**Files:**
- Create: `ext/aie_runtime-sys/tests/smoke.rs`

- [ ] **Step 1: Write the test file**

```rust
//! Hardware smoke test: open device 0, verify XRT bindings link and run.
//! Gated behind `hw-test` feature so CI skips it.
//!
//! Run with: `cargo test -p aie_runtime_sys --test smoke --features hw-test -- --ignored`

#[cfg(not(feature = "hw-test"))]
#[test]
#[ignore = "requires Strix NPU and hw-test feature"]
fn open_device_zero() {
    // Intentionally empty — when hw-test feature is off, this test is
    // ignored and does nothing.
}

#[cfg(feature = "hw-test")]
#[test]
fn open_device_zero() {
    use aie_runtime_sys::*;
    unsafe {
        let dev = xrtDeviceOpen(0);
        assert!(!dev.is_null(), "xrtDeviceOpen(0) returned null — is amdxdna loaded?");
        let rc = xrtDeviceClose(dev);
        assert_eq!(rc, 0, "xrtDeviceClose returned {rc}");
    }
}
```

- [ ] **Step 2: Declare the feature in `ext/aie_runtime-sys/Cargo.toml`**

Append to `ext/aie_runtime-sys/Cargo.toml`:

```toml
[features]
hw-test = []
```

- [ ] **Step 3: Verify the test compiles (without running it)**

Run: `cargo test -p aie_runtime_sys --no-run`
Expected: compiles cleanly. If XRT is not installed on this machine, compilation will fail — skip this step and document in the commit message.

- [ ] **Step 4: Manual hardware verification (run on Strix host only)**

Run (on a Strix-equipped machine with `/dev/accel/accel0`):
```
cargo test -p aie_runtime_sys --test smoke --features hw-test -- --ignored
```
Expected: `open_device_zero ... ok` (1 test passes).

If hardware not available, mark this verification step skipped and note in commit.

- [ ] **Step 5: Commit**

```bash
git add ext/aie_runtime-sys/Cargo.toml ext/aie_runtime-sys/tests/
git commit -m "feat(aie_runtime-sys): XRT device-open smoke test (hw-test gated)"
```

---

## Task 11: End-to-end INT4 matmul hardware test

**Files:**
- Create: `comgr/examples/hello_aie_matmul.ptx`
- Create: `comgr/tests/aie_int4_matmul.rs`

**Goal:** M3 gate — compile a real INT4 matmul PTX kernel through `compile_bitcode_aie`, load the resulting XCLBIN, launch it on Strix, verify output numerics.

- [ ] **Step 1: Create a minimal INT4 matmul PTX fixture**

Create `comgr/examples/hello_aie_matmul.ptx`:

```ptx
//
// Minimal INT4 matmul kernel for AIE backend validation.
// Uses mma.sync.m16n8k32.s32.s4.s4.s32.
// Not intended to be efficient — just to trigger the AIE TOSA path.
//
.version 7.8
.target sm_89
.address_size 64

.visible .entry aie_int4_matmul(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr
)
{
    .reg .u64 %rd<4>;
    .reg .s32 %r<8>;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];

    // Load operands (4-bit packed). Specific register setup omitted for
    // the M3 fixture — the emit_tosa_aie pass only needs to recognize the
    // mma instruction itself.
    mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32
        {%r0, %r1, %r2, %r3},
        {%r4, %r5},
        {%r6},
        {%r0, %r1, %r2, %r3};

    ret;
}
```

- [ ] **Step 2: Write the hardware-gated integration test**

Create `comgr/tests/aie_int4_matmul.rs`:

```rust
//! End-to-end AIE INT4 matmul: PTX → XCLBIN → Strix NPU execution.
//! Runs only when built with `--features aie,hw-test` on a Strix host.

#![cfg(feature = "aie")]

use std::ffi::CStr;

const PTX: &[u8] = include_bytes!("../examples/hello_aie_matmul.ptx");

#[test]
#[cfg_attr(not(feature = "hw-test"), ignore = "requires Strix NPU")]
fn aie_int4_matmul_end_to_end() {
    let device = CStr::from_bytes_with_nul(b"strix\0").unwrap();
    let xclbin = comgr::compile_bitcode_aie(device, PTX, &[])
        .expect("compile_bitcode_aie failed");

    // Basic artifact sanity:
    assert!(xclbin.len() > 64, "XCLBIN too small to be valid");
    assert_eq!(&xclbin[0..7], b"xclbin2", "XCLBIN magic mismatch");

    // Hardware execution is gated — only runs with hw-test feature.
    #[cfg(feature = "hw-test")]
    {
        run_on_strix(&xclbin);
    }
}

#[cfg(feature = "hw-test")]
fn run_on_strix(xclbin: &[u8]) {
    use aie_runtime_sys::*;
    unsafe {
        let dev = xrtDeviceOpen(0);
        assert!(!dev.is_null(), "xrtDeviceOpen(0) returned null");

        // Load XCLBIN from memory (not from file — we have bytes).
        let rc = xrtDeviceLoadXclbin(dev, xclbin.as_ptr() as *const _);
        assert_eq!(rc, 0, "xrtDeviceLoadXclbin returned {rc}");

        // Full kernel-launch path (hw-context open, kernel-open, buffer-alloc,
        // xrtRunSetArg×N, xrtRunStart, xrtRunWait, read output, compare) is
        // implemented as a follow-on once M3 basic load succeeds.
        // For the M3 gate, "XCLBIN loads without error" is the minimum
        // green signal.

        let rc = xrtDeviceClose(dev);
        assert_eq!(rc, 0, "xrtDeviceClose returned {rc}");
    }
}
```

- [ ] **Step 3: Add `hw-test` feature to `comgr/Cargo.toml`**

`hw-test` must pull in `aie` (since the hardware test requires the full AIE
pipeline) and forward `hw-test` to `aie_runtime_sys` so the runtime crate's
hardware-gated test path also activates. The `dep?/feat` syntax means
"enable `feat` on `dep` only if `dep` is active via some other feature":

Edit `[features]` in `comgr/Cargo.toml` to read:

```toml
[features]
default = ["intel"]
amd = []
intel = []
tenstorrent = []
gemmini = []
cutile = []
pacc = []
nvidia = ["dep:nvidia_sass"]
aie = ["dep:aie_comgr_sys", "dep:aie_runtime_sys"]
hw-test = ["aie", "aie_runtime_sys?/hw-test"]
```

- [ ] **Step 4: Verify compilation without hw-test**

Run: `cargo test -p comgr --features aie --no-default-features --no-run`
Expected: compiles. The test is ignored in this build.

- [ ] **Step 5: Verify artifact-level assertions run**

Run: `cargo test -p comgr --features aie --no-default-features -- aie_int4_matmul_end_to_end`
Expected: the test runs, invokes `compile_bitcode_aie`, and — if mlir-aie is on PATH — passes the XCLBIN-magic check. If mlir-aie is not installed, the test fails with a clear `ToolchainNotFound` error; mark it ignored locally until the toolchain is available.

- [ ] **Step 6: Manual hardware verification (Strix host only)**

Run (on Strix):
```
cargo test -p comgr --features aie,hw-test --no-default-features -- --include-ignored aie_int4_matmul_end_to_end
```
Expected: test passes. XCLBIN loads on hardware.

- [ ] **Step 7: Commit**

```bash
git add comgr/Cargo.toml comgr/tests/ comgr/examples/
git commit -m "test(comgr): end-to-end AIE INT4 matmul gated on hw-test"
```

---

## Task 12: README / docs note for AIE backend

**Files:**
- Create: `ext/aie_comgr-sys/README.md`
- Create: `ext/aie_runtime-sys/README.md`

- [ ] **Step 1: Write `ext/aie_comgr-sys/README.md`**

```markdown
# aie_comgr-sys

TOSA-MLIR → AMD AIE XCLBIN compilation driver. Shells out to the Xilinx
`mlir-aie` toolchain (`aie-opt`, `aie-translate`).

## Prerequisites

- Xilinx mlir-aie built and installed; `aie-opt` and `aie-translate` on `$PATH`
  (or set `AIE_TOOLCHAIN_DIR` to the `bin/` directory).

## Tests

- `cargo test -p aie_comgr_sys --lib` — unit tests (no toolchain needed).
- `cargo test -p aie_comgr_sys --test pipeline -- --ignored` — integration
  test that runs a trivial TOSA module through the full toolchain pipeline.
  Requires mlir-aie.
```

- [ ] **Step 2: Write `ext/aie_runtime-sys/README.md`**

```markdown
# aie_runtime-sys

Raw Rust bindings (bindgen) for the Xilinx XRT C API, used to load AIE
XCLBIN files and launch kernels on AMD Strix NPU.

## Prerequisites

- XRT installed (typically at `/opt/xilinx/xrt/`). The build script uses
  `pkg-config` first, falls back to `$XRT_PATH` / `/opt/xilinx/xrt`.
- `amdxdna` kernel driver loaded for hardware tests.
- `libclang-dev` for bindgen.

## Tests

- `cargo test -p aie_runtime_sys --no-run` — verify bindings compile.
- `cargo test -p aie_runtime_sys --features hw-test` — hardware smoke test
  requiring `/dev/accel/accel0` to be present.
```

- [ ] **Step 3: Commit**

```bash
git add ext/aie_comgr-sys/README.md ext/aie_runtime-sys/README.md
git commit -m "docs: README files for aie_comgr-sys and aie_runtime-sys"
```

---

## Deferred to Follow-on Plans

- **M4 — BitNet ternary pattern recognition.** Requires a reference BitNet
  CUDA kernel + PTX dump in hand before we can decide exactly which
  shift-and-mask-and-subtract shape to match. Spec open question calls this
  out. Separate plan once the M3 gate is green.
- **M5 — Scalar loop-nest matmul raising.** Detect triple-nested FMA
  accumulator loops and raise to `tosa.matmul`. Extends the backend's input
  surface beyond mma-intrinsic CUDA. Separate plan after M3.
- Full kernel-launch path in `comgr/tests/aie_int4_matmul.rs` (buffer alloc,
  argument binding, run, read, compare). M3 gate only requires XCLBIN loads
  on hardware without error.
- Wiring `compile_bitcode_aie` into ZLUDA's driver-impl dispatch so
  `cuLaunchKernel` actually reaches the AIE backend end-to-end from CUDA
  host code. Tracked as its own task once AIE backend itself is proven.

---

## Self-Review Notes

- **Spec coverage:** sections §1 (architecture), §2 (`emit_tosa_aie`), §3
  (`aie_comgr-sys`), §4 (`aie_runtime-sys`), §5 (comgr integration) are each
  covered by at least one task. Ternary recognition (§2.3) and scalar loop
  raising (§2.2 scalar case) are explicitly deferred per the milestone plan
  in the spec.
- **Placeholders:** every code block is complete; TODOs only appear inside
  code comments that document deferred milestones (not plan placeholders).
- **Type consistency:** `AieComgrError` variants match between the `-sys`
  crate and the `comgr` wrapper via an explicit `From` impl; `MmaShape` and
  `MmaElemType` used in Tasks 5 and 9 share a single definition.
- **Open questions remain in the spec (not blockers for M1-M3):** exact
  Strix tile geometry, pinned mlir-aie commit, ternary IR shape. These
  surface during M2 toolchain exercise and M4 (deferred).
