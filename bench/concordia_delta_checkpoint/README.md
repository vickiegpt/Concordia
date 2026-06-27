# Concordia Delta Checkpoint Benchmark

This is a Rust/cuda-oxide reproduction harness for the Concordia paper's transparent delta-checkpoint microbenchmarks. The GPU dirty-page scanner is written as a Rust `#[kernel]` inside a `#[cuda_module]` and compiled to PTX by NVlabs `cuda-oxide`.

This bench intentionally does not embed hand-written PTX and does not use this repo's local CUDA bindings for kernel launch. It uses:

- `cuda-device` for `#[kernel]`, `#[cuda_module]`, thread indexing, and `DisjointSlice`
- `cuda-core` for CUDA context, typed device buffers, stream synchronization, and subrange DtoH payload copies
- `cargo oxide` from `https://github.com/NVlabs/cuda-oxide`

## Setup

Install `cargo-oxide` with the upstream pinned nightly:

```bash
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

The local smoke script can also use a checked-out cuda-oxide repo:

```bash
git clone https://github.com/NVlabs/cuda-oxide /tmp/cuda-oxide
```

## Run

```bash
cd bench/concordia_delta_checkpoint

# Correctness smoke, defaulting to Blackwell sm_120.
CONCORDIA_BENCH_SELF_TEST=1 cargo oxide run --arch sm_120

# Paper-style tables.
CONCORDIA_BENCH_WARMUP=5 CONCORDIA_BENCH_ITERS=20 cargo oxide run --arch sm_120

# Launch-overhead-only mode.
CONCORDIA_BENCH_LAUNCH_ONLY=1 CONCORDIA_BENCH_ITERS=100 cargo oxide run --arch sm_120
```

If `cargo-oxide` is not installed but `/tmp/cuda-oxide` exists:

```bash
CONCORDIA_BENCH_SELF_TEST=1 \
  cargo run --manifest-path /tmp/cuda-oxide/crates/cargo-oxide/Cargo.toml -- run --arch sm_120
```

The benchmark also accepts CLI flags when running the produced binary directly:

```text
--self-test
--launch-only
--warmup N
--iters N
--device N
```

For `cargo oxide run`, prefer the environment variables because the cargo subcommand does not forward arbitrary program args:

```text
CONCORDIA_BENCH_SELF_TEST=1
CONCORDIA_BENCH_LAUNCH_ONLY=1
CONCORDIA_BENCH_WARMUP=N
CONCORDIA_BENCH_ITERS=N
CONCORDIA_BENCH_DEVICE=N
```

## Output

The benchmark prints CSV sections for:

- Table 5-style sparse delta checkpointing: 16, 50, 128, and 256 MiB regions with one dirty 4 KiB page
- Table 6-style dirty-page scaling: 256 MiB region with 1, 4, 10, and 32 dirty pages
- CUDA launch overhead through the generated cuda-oxide typed launcher

The CPU baseline copies the full device region back to host and scans pages in Rust. The GPU path runs the cuda-oxide Rust kernel to mark dirty pages, then appends only detected dirty 4 KiB payloads to a host-side log buffer.

## MPI

MPI launchers are supported through common local-rank environment variables. By default the benchmark chooses:

```text
device = local_rank % cuda_device_count
```

Recognized local-rank variables are `OMPI_COMM_WORLD_LOCAL_RANK`, `MPI_LOCALRANKID`, `PMI_LOCAL_RANK`, and `SLURM_LOCALID`. Override with `CONCORDIA_BENCH_DEVICE=N`.

Example:

```bash
mpirun -np 2 \
  -x CONCORDIA_BENCH_SELF_TEST=1 \
  -x CUDA_OXIDE_ARCH=sm_120 \
  bash -lc 'cd /home/victoryang00/hetGPU/bench/concordia_delta_checkpoint && ./test_smoke.sh'
```

## Caveats

This harness is intended to validate the Concordia delta-checkpoint shape with cuda-oxide. The one-thread-per-page Rust scanner is compile-safe and transparent, but it is not the optimized block-cooperative PTX/SASS scanner described in the paper. Use the printed timings to compare scaling and correctness in this repo, not as a claim that the implementation matches the paper's fastest GPU kernel.
