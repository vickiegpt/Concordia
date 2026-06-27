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

Linux `libcuda` symlinks after a release build:

```bash
ln -s libnvcuda.so target/release/libcuda.so
ln -s libnvcuda.so target/release/libcuda.so.1
ln -s libnvml.so target/release/libnvidia-ml.so
```

## Runtime Usage

Run a CUDA application through the shim:

```bash
LD_LIBRARY_PATH=/path/to/hetGPU/target/release <application> <arguments>
```

Enable Concordia checkpoint boundary logging:

```bash
export HETGPU_CONCORDIA_BOUNDARY=1
export HETGPU_CONCORDIA_LOGS=1
export CONCORDIA_AOF_PATH=/tmp/concordia/session.aof
LD_LIBRARY_PATH=/path/to/hetGPU/target/release <application> <arguments>
```

Enable NCCL boundary hooks:

```bash
export HETGPU_CONCORDIA_NCCL_BOUNDARY=1
export HETGPU_NCCL_LOGS=1
```

Enable the NVIDIA persistent worker path for simple elementwise kernels:

```bash
export CONCORDIA_PERSISTENT=1
export CONCORDIA_PERSISTENT_CAPACITY=1024
export CONCORDIA_PERSISTENT_BLOCKS=1
export CONCORDIA_ARCH=sm_80
export CONCORDIA_NVCC=/usr/local/cuda-12.8/bin/nvcc
```

The persistent worker path is currently compiled only for the pure NVIDIA feature configuration.

## SASS Lifter And PTX Recovery

The shared lifter can recover PTX from NVIDIA CUBINs or `cuobjdump -sass` text:

```bash
cargo run -p ptx --bin sass_inliner -- kernel.cubin --recover-ptx --ptx-output recovered.ptx
cuobjdump -sass kernel.cubin | cargo run -p ptx --bin sass_inliner -- --stdin --recover-ptx
```

Useful SASS lifter runtime flags:

```bash
export HETGPU_SASS_LIFTER_LOG=1
export HETGPU_SASS_LIFTER_DUMP=/tmp/lifted.ptx
export HETGPU_SASS_LIFTER_CUBIN_DUMP_DIR=/tmp/hetgpu-cubins
export HETGPU_SASS_LIFTER_DIAGNOSTIC_LIMIT=32
```

The module loader first uses embedded PTX when available. When PTX is unavailable and the lifter is requested, it selects a CUBIN payload from ELF/fatbin input, supports zstd/LZ4-compressed fatbin entries, lifts through `ptx::lift_cubin_to_ptx`, and then lets Concordia annotate the recovered PTX with safe points.

## PTX Pass Pipeline

The main PTX-to-LLVM path normalizes and lowers PTX through passes such as:

- `normalize_identifiers`
- `replace_known_functions`
- `normalize_predicates2`
- `resolve_function_pointers`
- `fix_special_registers`
- `expand_operands`
- `insert_post_saturation`
- `deparamize_functions`
- `replace_instructions_with_functions_fp_required`
- `normalize_basic_blocks`
- `remove_unreachable_basic_blocks`
- `instruction_mode_to_global_mode`
- `insert_explicit_load_store`
- `insert_implicit_conversions2`
- `replace_instructions_with_functions`
- `hoist_globals`
- `emit_llvm`

Target-specific passes and emitters include tmatmul assembly, SIFIVE VCIX, TOSA MLIR, AIE/TOSA, and debug/MLIR integration helpers.

## Open ptxas / NVIDIA SASS Assembler

`nvidia_sass` implements an experimental PTX-to-SASS pipeline:

```text
PTX subset -> parse -> instruction selection -> register allocation
           -> control-code scheduling -> encoding validation -> CUBIN builder
```

The `ptxas` crate exposes that path as a command-line replacement for simple kernels:

```bash
cargo run -p ptxas -- -o kernel.cubin -arch=sm_120 kernel.ptx
```

This is not a complete NVIDIA ptxas replacement. It is useful for controlled SM120 assembler experiments and round-trip validation.

## CXL/tmatmul Path

The tmatmul path lowers PTX matrix operations to a ternary-matmul assembly form and can run through emulator, simulator, or CXL Type-2 hardware-oriented paths depending on the environment.

Common flags:

```bash
export HETGPU_TMATMUL_COCOTB=1
export HETGPU_TMATMUL_ASM_DIR=/tmp/tmatmul-asm
export HETGPU_CXL_TMATMUL_STAGING=mmap
export HETGPU_TMATMUL_MATRIX_STAGE=host
export HETGPU_TMATMUL_IO_STAGE=host
export HETGPU_TMATMUL_OUTPUT_DTYPE=f32
```

The default hardware paths in code are `/dev/cxl_tmatmul3b000` and `/dev/dax0.0`; real hardware runs depend on the host CXL topology and driver state.

## Current Limitations

- This tree intentionally keeps multiple research paths side by side; not every feature combination is meaningful.
- The Concordia path is staged and compile-safe. It implements checkpoint metadata, AOF logging, safe-point annotation, MPI scoping, NCCL boundary hooks, and an NVIDIA persistent worker, but not the full paper system.
- SASS lifting is best-effort. Unsupported opcodes are surfaced as diagnostics or comments rather than silently translated.
- CXL/tmatmul hardware execution requires matching kernel drivers, device nodes, DAX setup, and platform topology.
- The persistent worker compiles embedded CUDA with `nvcc`; set `CONCORDIA_NVCC` and `CONCORDIA_ARCH` when the default toolchain is not correct.

## License

The code in this repository is dual-licensed under Apache 2.0 or MIT. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).

The Concordia paper text that guided the staged runtime port is CC BY 4.0; code in this repository remains governed by the repository license unless a file says otherwise.
