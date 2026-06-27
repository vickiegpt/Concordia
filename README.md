# hetGPU

hetGPU is a CUDA-compatible runtime and compiler workspace derived from ZLUDA, extended for heterogeneous GPU targets and Concordia-style fault-tolerant execution experiments.

The repository is organized around three linked paths:

- A `libcuda`/`libnvcuda` shim that lets CUDA applications load through the hetGPU runtime.
- A PTX compiler and recovery pipeline that can translate, instrument, lower, and debug CUDA kernels across several backend targets.
- A staged Concordia runtime substrate for delta checkpointing, persistent-kernel dispatch, NCCL boundary hooks, and MPI-aware recovery logs.

This is research infrastructure. Some pieces are production-shaped, but backend coverage and hardware paths vary by target.

## What Is In This Repository

| Area | Main paths | Purpose |
| --- | --- | --- |
| CUDA API shim | `zluda/`, `cuda_base/`, `cuda_types/` | CUDA Driver API compatibility layer, module loading, kernel launch interposition, and backend dispatch. |
| PTX compiler passes | `ptx/`, `ptx_parser/` | PTX parsing, normalization, LLVM emission, MLIR/TOSA/Tenstorrent/SIFIVE/tmatmul lowering, debug mapping, and state recovery helpers. |
| SASS lifter | `ptx/src/sass/`, `ptx/src/bin/sass_inliner.rs` | CUBIN parsing, NVIDIA SASS disassembly, SASS-to-PTX lifting, LLVM inlining, DWARF-assisted recovery, diagnostics, and fuzzing. |
| Open ptxas path | `nvidia_sass/`, `ptxas/` | Experimental PTX-to-SASS-to-CUBIN assembler pipeline for NVIDIA SM120-style targets. |
| Concordia runtime | `zluda/src/impl/concordia_*.rs`, `zluda/src/impl/persistent_router.rs` | Delta checkpoint records, append-only log replay, PTX safe-point annotation, NVIDIA persistent worker, and opt-in persistent routing. |
| NCCL/MPI hooks | `zluda/src/nccl_shim.c` | Minimal NCCL-compatible shim with all-reduce rendezvous, MPI env detection, and Concordia checkpoint boundary hooks. |
| CXL/tmatmul backend | `zluda/src/impl/cxl_tmatmul.rs`, `zluda/src/impl/tmatmul_interpreter.rs`, `ptx/src/pass/ptx_to_tmatmul.rs` | PTX-to-tmatmul lowering, host/CXL staging, simulator and hardware-oriented execution paths. |
| Backend compiler glue | `comgr/`, `ext/*_comgr-sys`, `ext/*_runtime-sys` | AMD, Intel Level Zero, Tenstorrent, SIFIVE, Cuttlefish, AIE, and NVIDIA support crates. |

## Architecture

```text
CUDA application / PyTorch / Triton
        |
        v
hetGPU libcuda shim (`zluda`)
        |
        +-- PTX text --------------------+
        |                                |
        +-- CUBIN/fatbin -> SASS lifter -+-> PTX recovery / annotation
                                         |
                                         v
                               PTX pass pipeline
                                         |
          +------------------------------+------------------------------+
          |                              |                              |
          v                              v                              v
   LLVM/COMGR backends             tmatmul/CXL path              NVIDIA pass-through
   AMD / Intel / SIFIVE / TT       emulator or hardware          optional persistent worker

Concordia hooks sit on the module-load and launch path:

PTX/SASS recovery -> Concordia safe-point labels -> checkpoint registration
kernel/NCCL boundary -> delta checkpoint -> rank-scoped AOF log
simple elementwise launch -> optional persistent-kernel route
```

## Concordia Integration

The Concordia code in this tree is a compile-safe staged port of the paper design, not a full production fault-tolerant LLM serving stack.

Implemented pieces:

- PTX safe-point discovery and annotation at `.entry`, `bar.sync`, and `ret` sites.
- Binary-module recovery through the existing Rust SASS-to-PTX lifter path. There is no JEB dependency.
- Host-testable delta checkpoint state for opaque shadow-diff regions and allocator-bitmap regions.
- Append-only AOF records with committed-record replay and truncated-suffix tolerance.
- C-callable region registration and checkpoint APIs:
  - `hetgpu_concordia_register_host_region`
  - `hetgpu_concordia_register_bitmap_region`
  - `hetgpu_concordia_checkpoint_host_region`
  - `hetgpu_concordia_checkpoint_bitmap_region`
  - `hetgpu_concordia_checkpoint_boundary`
- MPI-aware rank, world-size, and local-rank detection from OpenMPI, PMIx, PMI, MVAPICH, Slurm, PyTorch-style, and Concordia-specific environment variables.
- Rank-scoped AOF path expansion for parallel jobs.
- NVIDIA-only persistent worker with pinned host-mapped ring buffer and simple ops: add, mul, sub, SiLU, ReLU, scale, fused add+ReLU, and dirty-page scan.
- NCCL all-reduce boundary hooks that can call Concordia checkpoint boundaries before and after collectives.

Not yet complete:

- Full SASS binary patching with live register capture.
- Full GPU-resident AOF append from persistent worker tasks.
- Production NCCL communicator replacement and multi-node failure orchestration.
- Complete LLM KV-cache allocator integration.
- Cross-architecture CTX migration from the Concordia paper.

## MPI And Parallel Runs

The Concordia runtime does not require linking against MPI. It reads common MPI environment variables so independent ranks can run the same shim safely.

Rank detection keys include:

- Rank: `CONCORDIA_MPI_RANK`, `HETGPU_CONCORDIA_MPI_RANK`, `OMPI_COMM_WORLD_RANK`, `PMIX_RANK`, `PMI_RANK`, `MV2_COMM_WORLD_RANK`, `SLURM_PROCID`, `RANK`
- World size: `CONCORDIA_MPI_WORLD_SIZE`, `HETGPU_CONCORDIA_MPI_WORLD_SIZE`, `OMPI_COMM_WORLD_SIZE`, `PMIX_SIZE`, `PMI_SIZE`, `MV2_COMM_WORLD_SIZE`, `SLURM_NTASKS`, `WORLD_SIZE`
- Local rank: `CONCORDIA_MPI_LOCAL_RANK`, `HETGPU_CONCORDIA_MPI_LOCAL_RANK`, `OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `MV2_COMM_WORLD_LOCAL_RANK`, `SLURM_LOCALID`, `PMI_LOCAL_RANK`, `LOCAL_RANK`

If `CONCORDIA_AOF_PATH=/tmp/concordia/session.aof` and the job has multiple ranks, rank 3 of 16 writes:

```text
/tmp/concordia/session.rank0003-of-0016.aof
```

Template tokens are also supported:

```bash
export CONCORDIA_AOF_PATH=/tmp/concordia/r{rank}-w{world}-l{local_rank}.aof
```

The persistent worker device defaults to MPI local rank unless overridden:

```bash
export CONCORDIA_PERSISTENT_DEVICE=0
```

## Build

Clone with submodules:

```bash
git clone --recursive <repo-url>
cd hetGPU
```

General Rust build:

```bash
cargo build --release
```

Focused checks used for the current staged Concordia port:

```bash
cargo test -p zluda --features intel --no-default-features concordia_ -- --nocapture --test-threads=1
cargo test -p zluda --features intel --no-default-features persistent_router -- --nocapture --test-threads=1
cargo check -p zluda --features nvidia --no-default-features
cargo check -p zluda --features nvidia,tmatmul --no-default-features
```

Build the developer tools:

```bash
cargo build -p ptx --bin sass_inliner
cargo build -p ptx --bin gpu_rr
cargo build -p ptx --bin ptx_to_tmatmul_hw
cargo build -p ptxas --bin ptxas
cargo build -p nvidia_sass
```

### Caveats: LLVMTarget not built by default

```
error: linking with `cc` failed: exit status: 1
  |
  = note:  "cc" "-Wl,--version-script=/tmp/rustctsJxAq/list" "-Wl,--no-undefined-version" "-m64" "/tmp/rustctsJxAq/symbols.o" "<89 object files omitted>" "-Wl,--as-needed" "-Wl,-Bstatic" "/xx/hetGPU/target/debug/deps/{libptx-7bdaff80a2317f98.rlib,libwhich-f30bb1abebdc81fc.rlib,libhome-7e8a5bcbbae9393a.rlib,libeither-64643123df8b89ff.rlib,librustix-af06f22e44d470fe.rlib,liblinux_raw_sys-3594dde066d7bd9d.rlib,libtempfile-95c2d4d89752ec6d.rlib,libgetrandom-b0234ab360848404.rlib,libfastrand-e6b7bf4f5223db08.rlib,libonce_cell-3762ade5b54492ce.rlib,librustix-7214c2022c62c874.rlib,libbitflags-df5e13556f379dca.rlib,liblinux_raw_sys-93ad30317744a679.rlib,libserde_json-96cad77e399e48a3.rlib,libitoa-e93891d541c4924d.rlib,libryu-89b8058e107f9d11.rlib,libregex-cc941c181af0db7a.rlib,libregex_automata-bda824cb2424702e.rlib,libaho_corasick-084dd9c39f7ac0ae.rlib,libmemchr-098566f0fddadd5e.rlib,libregex_syntax-c89bdbd6e66e0c64.rlib,libstrum-b548b1a93b339f7b.rlib,libquick_error-20ff717f463892e5.rlib,libptx_parser-1e5967146d4cc314.rlib,libthiserror-182054611c23cf3e.rlib,libbitflags-fdc24f4688603348.rlib,libwinnow-42c14bebbe4544ba.rlib,librustc_hash-50d45f4cee64a708.rlib,liblogos-b3726b5a3afef72d.rlib,libderive_more-9910237a94e5634b.rlib,libllvm_hetGPU-be4f1e966c7c00b7.rlib,libllvm_sys-dec2a59b25406277.rlib,libserde-50cedc28066d9a15.rlib,librand-7f4d8b7fe7531938.rlib,librand_chacha-376b75692958a1cc.rlib,libppv_lite86-a6f6214d46654bf2.rlib,libzerocopy-e3604abd9a947746.rlib,librand_core-82ab10e7dd7b6353.rlib,libgetrandom-cc67d104b46bae27.rlib,liblibc-83ba1c66a8baec75.rlib,libcfg_if-e3a2a86c3f7f7605.rlib,libbase64-abde09cee03ffba5.rlib,librustc_hash-bb091058ca505f29.rlib,libcuda_types-a88f0ad52e7d6047.rlib,libze_runtime_sys-56ce8c843283f7fa.rlib}.rlib" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib/{libstd-*,libpanic_unwind-*,libobject-*,libmemchr-*,libaddr2line-*,libgimli-*,librustc_demangle-*,libstd_detect-*,libhashbrown-*,librustc_std_workspace_alloc-*,libminiz_oxide-*,libadler2-*,libunwind-*,libcfg_if-*,liblibc-*,liballoc-*,librustc_std_workspace_core-*,libcore-*,libcompiler_builtins-*}.rlib" "-Wl,-Bdynamic" "-lLLVMBitWriter" "-lLLVMAnalysis" "-lLLVMProfileData" "-lLLVMSymbolize" "-lLLVMDebugInfoBTF" "-lLLVMDebugInfoPDB" "-lLLVMDebugInfoMSF" "-lLLVMDebugInfoCodeView" "-lLLVMDebugInfoDWARF" "-lLLVMObject" "-lLLVMTextAPI" "-lLLVMMCParser" "-lLLVMIRReader" "-lLLVMAsmParser" "-lLLVMMC" "-lLLVMBitReader" "-lLLVMCore" "-lLLVMRemarks" "-lLLVMBitstreamReader" "-lLLVMBinaryFormat" "-lLLVMTargetParser" "-lLLVMSupport" "-lLLVMDemangle" "-lLLVMTarget" "-lstdc++" "-lze_loader" "-lgcc_s" "-lutil" "-lrt" "-lpthread" "-lm" "-ldl" "-lc" "-L" "/tmp/rustctsJxAq/raw-dylibs" "-Wl,--eh-frame-hdr" "-Wl,-z,noexecstack" "-L" "/xx/hetGPU/target/debug/build/ze_runtime_sys-1222225a9a9570c5/out" "-L" "/xx/hetGPU/target/debug/build/lz4-sys-d3a5e3b6c2386e54/out" "-L" "/xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out/build/lib" "-L" "/xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out/build/lib/../../../../../../../ext/llvm-project/build/lib" "-L" "/xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out" "-L" "/xx/hetGPU/target/debug/build/tt_runtime_sys-2582b52c9d1e3544/out" "-L" "/opt/rocm/lib/" "-L" "/opt/rocm/lib/" "-L" "/usr/lib/x86_64-linux-gnu/" "-L" "lib" "-L" "<sysroot>/lib/rustlib/x86_64-unknown-linux-gnu/lib" "-o" "/xx/hetGPU/target/debug/deps/libnvcuda.so" "-Wl,--gc-sections" "-shared" "-Wl,-z,relro,-z,now" "-nodefaultlibs"
  = note: some arguments are omitted. use `--verbose` to show all linker arguments
  = note: /usr/bin/ld: cannot find -lLLVMTarget
```
fix:
```
pushd /xx/hetGPU/target/debug/build/llvm_hetGPU-f16113f7bf4ec5f8/out/build/
ninja LLVMTarget
popd
```
### Linux

If you are building on Linux you must also symlink (or rename) the hetGPU output binaries after hetGPU build finishes:
```
ln -s libnvcuda.so target/release/libcuda.so
ln -s libnvcuda.so target/release/libcuda.so.1
ln -s libnvml.so target/release/libnvidia-ml.so
```

## Developer Tools

hetGPU includes several developer tools for GPU debugging and analysis:

### SASS Inliner (`sass_inliner`)

Convert SASS (NVIDIA GPU assembly) to LLVM IR for analysis and cross-platform compilation:

```bash
# Build the tool
cargo build -p ptx --bin sass_inliner

# Inline SASS from CUBIN file
sass_inliner kernel.cubin -o output.ll

# Use cuobjdump output
cuobjdump -sass -lineinfo kernel.cubin | sass_inliner --stdin -o output.ll

# Dump SASS instructions
sass_inliner kernel.cubin --dump-sass
```

#### PTX Recovery

Recover PTX source code from SASS using DWARF debug information or semantic reconstruction:

```bash
# Recover PTX from CUBIN with debug info
sass_inliner kernel.cubin --recover-ptx --ptx-output recovered.ptx

# Recover PTX from cuobjdump output (semantic reconstruction)
cuobjdump -sass kernel.cubin | sass_inliner --stdin --recover-ptx
```

### GPU Record/Replay (`gpu_rr`)

rr-style debugging for GPU kernels with record, replay, and analysis capabilities:

```bash
# Build the tool
cargo build -p ptx --bin gpu_rr

# Record kernel execution
gpu_rr record kernel.cubin -o trace.gpur

# Replay with breakpoint
gpu_rr replay --break 0x100 trace.gpur

# Analyze execution hotspots
gpu_rr analyze trace.gpur -o report.json

# Recover PTX from SASS
gpu_rr ptx kernel.cubin -o recovered.ptx
```

### Inlining Strategies

The SASS inliner supports multiple strategies:
- `asm` / `inline-asm` - Preserve exact SASS encoding as inline assembly
- `ptx` / `reconstruct` - Convert to PTX-equivalent LLVM IR
- `meta` / `metadata` - Metadata only, no IR changes
- `hybrid` (default) - Balance precision and compatibility

## Contributing

If you want to develop hetGPU itself, read [CONTRIBUTING.md](CONTRIBUTING.md), it contains instructions how to set up dependencies and run tests


## License

This software is dual-licensed under either the Apache 2.0 license or the MIT license. See [LICENSE-APACHE](LICENSE-APACHE) or [LICENSE-MIT](LICENSE-MIT) for details
