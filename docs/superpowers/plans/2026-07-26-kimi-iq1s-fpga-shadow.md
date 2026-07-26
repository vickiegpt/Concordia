# Kimi IQ1_S FPGA Shadow Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that one real Kimi K2.6 type-19 IQ1_S `mul_mat_vec_q` launch can run natively on NVIDIA, feed a bounded IQ1_S subgroup through CXL DAX to the ternary FPGA, and produce a bit-exact integer result without changing Kimi output.

**Architecture:** Keep CUDA authoritative. The existing cudart prelaunch hook must continue native execution for a shadow-eligible IQ1_S MMVQ launch. Immediately after the real Rust-side `cuLaunchKernel` succeeds, capture operands in order on its actual stream, decode subgroup 0, submit one replay-safe packed `tmatmul_go` transaction through the first 2 MiB of DAX, compare all valid rows, restore only after a proven terminal boundary, and emit JSONL evidence. The hardware layer accepts owned host payloads and never receives the CUDA destination pointer.

**Tech Stack:** Rust 2021 (`zluda`), C cudart shim, CUDA Driver API, Linux CXL tmatmul UAPI v2, device DAX, `serde_json`, Rust unit tests with fake CUDA/DAX adapters, Python 3 evidence validators, Bash launch wrappers.

---

## Fixed Contract

Implement against
`docs/superpowers/specs/2026-07-26-kimi-iq1s-fpga-shadow-design.md`.
These values are not configurable in this milestone:

```text
kernel                 mul_mat_vec_q with template type ggml_type19
subgroup               0, columns [0, 32)
FPGA dimension         2048
matrix DPA             0x000000, 0x100000 bytes
input DPA              0x100000, 0x001000 bytes
output DPA             0x110000, 0x001000 bytes
program DPA            0x120000, 0x000080 bytes
snapshot boundary      [0x000000, 0x200000)
submission             exactly one RUN_CSR_ONLY
instruction            tmatmul_go, never tmatmul_go_nvint8
authoritative output   native CUDA only
```

The implementation refines one source-location detail from the design without
changing behavior: the post-native observer belongs in
`zluda/src/impl/function.rs::launch_kernel`, directly after the real
`nvidia_runtime_sys::cuLaunchKernel` succeeds. That is the actual Driver API
launch seam reached by `cudart_shim.c`, and it already owns the real driver
stream and argument slots. Do not add a second C-to-Rust postlaunch FFI that
could observe or submit twice.

## Execution Safety

- Do not edit RTL, the CXL driver, BitNet, llama.cpp, or model shards.
- Do not push any branch or commit.
- Do not use `/dev/dax*` until all pure and fake-device tests pass.
- Do not attempt Kimi until the model-free hardware gate passes.
- Do not restore DAX after an ambiguous post-submit state.
- Do not stage all existing dirty files from the original checkout.

### Task 0: Preserve the Existing Dirty Baseline in an Isolated Worktree

**Files:**
- Source: `/home/victoryang00/hetGPU`
- Create: `/home/victoryang00/hetGPU-kimi-iq1s-shadow`
- Preserve:
  - `zluda/src/cublas_shim.c`
  - `zluda/src/impl/cxl_tmatmul.rs`
  - `zluda/src/impl/function.rs`
  - `zluda/src/impl/kimi_concordia.rs`

- [ ] **Step 1: Record the source state**

Run:

```bash
cd /home/victoryang00/hetGPU
git status --short
git rev-parse HEAD
git diff --check
git diff --binary -- \
  zluda/src/cublas_shim.c \
  zluda/src/impl/cxl_tmatmul.rs \
  zluda/src/impl/function.rs \
  zluda/src/impl/kimi_concordia.rs \
  > /tmp/hetgpu-kimi-iq1s-baseline.patch
sha256sum /tmp/hetgpu-kimi-iq1s-baseline.patch
```

Expected: HEAD is `d808e1a`; the four listed source files are modified; the
patch passes `git diff --check`. The untracked PDF and directory are not copied.

- [ ] **Step 2: Create the isolated branch and worktree**

Run:

```bash
cd /home/victoryang00/hetGPU
git worktree add -b kimi-iq1s-shadow \
  /home/victoryang00/hetGPU-kimi-iq1s-shadow d808e1a
cd /home/victoryang00/hetGPU-kimi-iq1s-shadow
git apply --check /tmp/hetgpu-kimi-iq1s-baseline.patch
git apply /tmp/hetgpu-kimi-iq1s-baseline.patch
git diff --check
```

Expected: the isolated checkout contains only the four preserved source diffs.
The original checkout remains unchanged.

- [ ] **Step 3: Commit the preserved baseline only in the isolated branch**

Run:

```bash
cd /home/victoryang00/hetGPU-kimi-iq1s-shadow
git add \
  zluda/src/cublas_shim.c \
  zluda/src/impl/cxl_tmatmul.rs \
  zluda/src/impl/function.rs \
  zluda/src/impl/kimi_concordia.rs
git diff --cached --check
git commit -m "chore: preserve current tmatmul integration baseline"
```

Expected: a local baseline commit. Do not push it.

### Task 1: Add Shadow Configuration, Cap, and Poison State

**Files:**
- Create: `zluda/src/impl/kimi_iq1s_shadow.rs`
- Modify: `zluda/src/impl/mod.rs:50-75`

- [ ] **Step 1: Write failing configuration tests**

Add tests named:

```rust
config_disabled_by_default
config_defaults_to_one_launch_and_ten_second_timeout
config_zero_cap_disables_shadow
config_invalid_value_disables_shadow
claim_is_single_winner_under_concurrency
poison_prevents_later_claims
forked_child_is_disabled
```

Use a pure `ShadowConfig::from_lookup(|name| -> Option<String>)` parser so tests
never mutate process-global environment. Assert these defaults:

```rust
enabled: false
max_launches: 1
timeout_ms: 10_000
control_dev: "/dev/cxl_tmatmul3b000"
dax_dev: "/dev/dax6.0"
snapshot_dir: "/var/tmp"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run:

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests::config_ -- --nocapture --test-threads=1
```

Expected: compile failure because `kimi_iq1s_shadow` and its types do not exist.

- [ ] **Step 3: Implement the smallest state module**

Implement:

```rust
pub(crate) struct ShadowConfig { /* typed fields above */ }
pub(crate) enum ClaimResult { Claimed(LaunchClaim), Disabled, Exhausted, Poisoned }
pub(crate) struct ShadowProcessState {
    owner_pid: u32,
    consumed: AtomicUsize,
    poisoned: AtomicBool,
    transaction: Mutex<()>,
}
```

Parsing rules must exactly match the design. `LaunchClaim` is consumed only when
durable snapshotting starts. A PID mismatch returns `Disabled`.

Register the module only for the NVIDIA backend:

```rust
#[cfg(all(
    feature = "nvidia",
    not(feature = "amd"),
    not(feature = "intel"),
    not(feature = "tenstorrent")
))]
pub(crate) mod kimi_iq1s_shadow;
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
```

Expected: all configuration/state tests pass without opening CUDA, CXL, or DAX.

- [ ] **Step 5: Commit**

```bash
git add zluda/src/impl/mod.rs zluda/src/impl/kimi_iq1s_shadow.rs
git commit -m "feat: add Kimi FPGA shadow process state"
```

### Task 2: Check In and Verify the Exact IQ1_S Grid

**Files:**
- Create: `zluda/src/impl/kimi_iq1s_grid.rs`
- Modify: `zluda/src/impl/mod.rs`
- Test fixture source: the pinned llama.cpp/BitNet IQ1_S implementation used by
  `/home/eabban/BitNet/build-cuda128-gcc12/bin/llama-cli`

- [ ] **Step 1: Pin provenance before copying data**

Record these already verified values in the module header:

```text
BitNet repository commit     01eb415772c342d9f20dc42772f1583ae1e5b102
llama.cpp submodule commit   1f86f058de0c3f4098dedae2ae8653c335c868a1
source file                  ggml/src/ggml-common.h
source symbol                iq1s_grid
source file SHA-256          40046ac43a8089fe03fb05784dd23f66c7c851e00dcaee0afb7601b5945a1fc1
flattened grid SHA-256       07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a
```

Obtain the revision and source symbol with:

```bash
cd /home/eabban/BitNet
git rev-parse HEAD
rg -n "iq1s_grid|iq1_s|kgrid" .
```

Do not derive the grid from a remembered formula.

- [ ] **Step 2: Write failing grid tests**

Add tests:

```rust
grid_has_2048_entries_of_32_trits
grid_values_are_strictly_ternary
grid_checksum_matches_pinned_upstream
grid_representative_indices_match_upstream
```

The table has 2048 entries of eight signed trits. Checksum the 16,384 flattened
signed bytes, encoding `-1` as `0xff`, `0` as `0x00`, and `+1` as `0x01`.

- [ ] **Step 3: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_grid::tests:: -- --nocapture
```

Expected: compile failure because the table is absent.

- [ ] **Step 4: Add the checked-in table**

Expose only:

```rust
pub(crate) const IQ1S_GRID: [[i8; 8]; 2048] = [
    // Exact values decoded from the pinned upstream uint64_t table.
];
pub(crate) const IQ1S_GRID_SHA256: &str =
    "07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a";
```

Do not add a runtime dependency on the BitNet checkout.

- [ ] **Step 5: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_grid::tests:: -- --nocapture
git add zluda/src/impl/mod.rs zluda/src/impl/kimi_iq1s_grid.rs
git commit -m "feat: pin the IQ1_S ternary grid"
```

Expected: all four tests pass and the test prints the pinned revision/checksum.

### Task 3: Decode IQ1_S and Build the FPGA Payload

**Files:**
- Modify: `zluda/src/impl/kimi_iq1s_shadow.rs`
- Create: `zluda/test_data/kimi_iq1s_subgroup0_fixture.json`

- [ ] **Step 1: Generate one independent llama.cpp fixture**

Create a small one-off extractor outside the repo or use the pinned upstream
decode routine to produce a fixture with fields `upstream_commit`,
`block_iq1_s_hex`, `q8_1_hex`, `grid`, `q8_qs`, and `integer_dot`.
`upstream_commit` must equal
`1f86f058de0c3f4098dedae2ae8653c335c868a1`; the two hex strings decode to
exactly 50 and 36 bytes; both arrays contain exactly 32 signed integers; and
`integer_dot` is computed by the independent upstream extractor. The Rust test
must reject malformed lengths.

- [ ] **Step 2: Write failing decode and packing tests**

Add:

```rust
block_iq1_s_is_exactly_50_bytes
q8_1_block_is_exactly_36_bytes
fixture_decodes_to_upstream_grid
four_trits_pack_in_existing_hardware_order
packed_matrix_is_exactly_one_mib
q8_qs_are_sign_extended_to_i16
padded_input_is_exactly_4096_bytes
integer_reference_matches_fixture
integer_reference_cannot_exceed_4064
```

The packing test must include all values and byte boundaries, not only zeros.
Use the existing `cxl_tmatmul` trit encoder as the format authority.

- [ ] **Step 3: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
```

Expected: tests fail because decode/build functions are absent.

- [ ] **Step 4: Implement pure conversion**

Implement checked functions:

```rust
fn decode_subgroup0(block: &[u8; 50]) -> Result<DecodedSubgroup, ShadowError>;
fn build_packed_matrix(
    rows: &[DecodedSubgroup],
    valid_rows: usize,
) -> Result<Vec<u8>, ShadowError>;
fn build_i16_input(q8_1: &[u8; 36]) -> Result<Vec<u8>, ShadowError>;
fn integer_reference(
    rows: &[DecodedSubgroup],
    q8_qs: &[i8; 32],
) -> Vec<i16>;
```

Use checked arithmetic. Zero-fill columns `[32,2048)` and rows after
`valid_rows`. Preserve IQ1_S `d`, subgroup multiplier, affine sign, Q8_1 `d`,
and `s` as telemetry only.

- [ ] **Step 5: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
git add \
  zluda/src/impl/kimi_iq1s_shadow.rs \
  zluda/test_data/kimi_iq1s_subgroup0_fixture.json
git commit -m "feat: convert IQ1_S subgroup to packed ternary"
```

### Task 4: Recognize Only the Supported CUDA ABI

**Files:**
- Modify: `zluda/src/impl/function.rs:8496-8581`
- Modify: `zluda/src/impl/kimi_iq1s_shadow.rs`

- [ ] **Step 1: Write failing ABI tests**

Create pure descriptor parsing tests for:

```rust
accepts_exact_type19_mul_mat_vec_q
rejects_type19_mul_mat_q
rejects_non_type19_mul_mat_vec_q
rejects_missing_argument_slot
rejects_nonpositive_dimensions
rejects_ncols_not_divisible_by_256
rejects_more_than_2048_output_rows
rejects_checked_stride_overflow_before_pointer_query
```

Use explicit fake argument slots in test-owned storage. Never dereference a
synthetic device pointer.

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
```

Expected: tests fail because the exact `MmvqIq1SLaunch` parser is absent.

- [ ] **Step 3: Implement a value-owned descriptor**

Add:

```rust
pub(crate) struct MmvqIq1SLaunch {
    launch_id: String,
    kernel_name: String,
    stream: CUstream,
    matrix_ptr: CUdeviceptr,
    activation_ptr: CUdeviceptr,
    native_output_ptr: CUdeviceptr,
    matrix_remaining: usize,
    activation_remaining: usize,
    output_remaining: usize,
    ncols_x: usize,
    nrows_x: usize,
    nrows_y: usize,
    nrows_dst: usize,
}
```

Split recognition, scalar extraction, checked byte requirements, and allocation
span checks into pure/testable stages. Keep `mul_mat_q` excluded.

- [ ] **Step 4: Prevent the prelaunch hook from consuming a shadow candidate**

At the beginning of `nvidia_try_launch_named_cxl_tmatmul`, return `None` when:

```text
HETGPU_KIMI_FPGA_SHADOW=1
and the normalized name is mul_mat_vec_q
and the mangling contains ggml_type19
```

Do not return success. `None` maps to C shim result `1`, which means continue to
the registered native launch. Keep all existing prelaunch behavior unchanged
when shadow mode is disabled.

- [ ] **Step 5: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
cargo check -p zluda --features nvidia --no-default-features
git add zluda/src/impl/function.rs zluda/src/impl/kimi_iq1s_shadow.rs
git commit -m "feat: identify Kimi IQ1_S MMVQ shadow launches"
```

### Task 5: Validate CUDA Allocation Spans and Capture on the Actual Stream

**Files:**
- Modify: `zluda/src/impl/function.rs:8683-8731`
- Modify: `zluda/src/impl/kimi_iq1s_shadow.rs`

- [ ] **Step 1: Write failing adapter tests**

Introduce a private trait implemented by the real CUDA backend and a fake:

```rust
trait ShadowCuda {
    fn allocation_span(&self, ptr: CUdeviceptr) -> Result<(CUdeviceptr, usize), ShadowError>;
    fn copy_dtoh_async(&self, dst: &mut [u8], src: CUdeviceptr, stream: CUstream)
        -> Result<(), ShadowError>;
    fn stream_synchronize(&self, stream: CUstream) -> Result<(), ShadowError>;
}
```

Tests:

```rust
remaining_span_accounts_for_interior_pointer
undersized_matrix_is_rejected_before_copy
undersized_activation_is_rejected_before_copy
undersized_output_is_rejected_before_copy
capture_uses_original_stream_for_every_copy
capture_synchronizes_original_stream_once
capture_copies_only_required_matrix_blocks_and_one_q8_1_block
```

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
```

Expected: compile failure until the adapter and capture layer exist.

- [ ] **Step 3: Implement the real CUDA adapter**

Use CUDA pointer-allocation queries to obtain allocation base and size, then
calculate `remaining = size - (ptr - base)` with checked arithmetic. Delete or
bypass the current `usize::MAX` acceptance for this shadow path.

Queue:

```text
nrows_dst IQ1_S block-0 copies at checked row strides
one 36-byte ordinary Q8_1 block
optional bounded native output telemetry sample
```

on `h_stream`, then synchronize that same stream before any DAX lock or access.
No background task may retain device pointers.

- [ ] **Step 4: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
cargo check -p zluda --features nvidia --no-default-features
git add zluda/src/impl/function.rs zluda/src/impl/kimi_iq1s_shadow.rs
git commit -m "feat: capture Kimi operands on the native CUDA stream"
```

### Task 6: Add the Bounded Packed-DAX Transaction Contract

**Files:**
- Modify: `zluda/src/impl/cxl_tmatmul.rs:457-700`

- [ ] **Step 1: Write failing layout and program tests**

Add constants and tests:

```rust
shadow_layout_is_non_overlapping_and_inside_two_mib
shadow_matrix_is_exactly_one_mib
shadow_input_and_output_are_4096_bytes
shadow_program_is_128_bytes
shadow_program_uses_tmatmul_go_not_nvint8
shadow_program_has_terminal_stall_in_slot_seven
shadow_program_decodes_to_six_semantic_operations
```

Assert exact DPA values from the Fixed Contract section.

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  cxl_tmatmul::tests::shadow_ -- --nocapture
```

Expected: tests fail because the shadow layout and builder are absent.

- [ ] **Step 3: Implement owned request/evidence types**

Add:

```rust
pub(crate) struct PackedShadowRequest {
    pub matrix: Vec<u8>,
    pub input: Vec<u8>,
    pub output_sentinel: Vec<u8>,
    pub timeout: Duration,
    pub launch_id: String,
}

pub(crate) struct PackedShadowEvidence {
    pub terminal_completion_proven: bool,
    pub hardware_executed: bool,
    pub output: Vec<i16>,
    pub counters_before: CounterSnapshot,
    pub counters_after: CounterSnapshot,
    pub snapshot_sha256: String,
    pub restored_sha256: Option<String>,
}
```

Expose one new API:

```rust
pub(crate) fn submit_packed_shadow(
    config: &PackedShadowDeviceConfig,
    request: PackedShadowRequest,
) -> Result<PackedShadowEvidence, PackedShadowError>;
```

It must not accept any CUDA pointer or native output destination.

- [ ] **Step 4: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  cxl_tmatmul::tests::shadow_ -- --nocapture
git add zluda/src/impl/cxl_tmatmul.rs
git commit -m "feat: define bounded packed DAX shadow transaction"
```

### Task 7: Implement Snapshot, Locks, Restore, and Poison-Safe Outcomes

**Files:**
- Modify: `zluda/src/impl/cxl_tmatmul.rs`
- Modify: `zluda/src/impl/kimi_iq1s_shadow.rs`

- [ ] **Step 1: Add a fake device boundary**

Define a private hardware trait that covers only:

```text
open/identity/preflight
exclusive nonblocking lock
read and write DAX ranges
fsync snapshot and marker
CSR readback
RUN_CSR_ONLY
status and counters
```

The production implementation uses the existing ioctl/CSR helpers. Unit tests
use an in-memory 2 MiB fake and an ordered operation log.

- [ ] **Step 2: Write failing state-machine tests**

Add:

```rust
preflight_rejection_does_not_consume_claim
lock_contention_does_not_consume_claim
snapshot_is_fsynced_before_first_dax_write
every_presubmit_failure_restores_and_verifies_sha256
terminal_numerical_mismatch_still_restores
ambiguous_postsubmit_state_performs_no_cleanup_write
ambiguous_postsubmit_state_retains_snapshot_and_marker
ambiguous_postsubmit_state_poisons_process
only_submit_completed_sets_hardware_executed
```

Inject a failure after every operation. Assert exact final DAX bytes and ordered
calls for each case.

- [ ] **Step 3: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  shadow_transaction_ -- --nocapture --test-threads=1
cargo test -p zluda --features nvidia --no-default-features \
  ambiguous_ -- --nocapture --test-threads=1
```

Expected: tests fail until the explicit state machine exists.

- [ ] **Step 4: Implement the transaction state machine**

Use an enum with the design states. Snapshot exactly `0x200000` bytes to a
unique file under the configured directory, `fsync` it, hash it, and atomically
write a durable stage marker before staging.

Preflight must prove:

```text
boot ID, BDF/device identity, UAPI v2, TMM1, dim_d=2048,
program_dpa=0x120000, DAX >= 2 MiB, idle/no-reset/no-error/no-stall
```

Acquire both the process mutex and nonblocking exclusive control-device lock.
Do not introduce retries.

- [ ] **Step 5: Implement restore policy exactly**

Before submit or after proven terminal completion:

```text
restore 2 MiB -> flush -> reread 2 MiB -> compare SHA-256
```

After submit begins without proven terminal completion:

```text
write cleanup_blocked_launch_unproven marker
retain snapshot
perform no DAX write
return AmbiguousCompletion
poison process shadow state
```

- [ ] **Step 6: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  shadow_transaction_ -- --nocapture --test-threads=1
cargo test -p zluda --features nvidia --no-default-features \
  ambiguous_ -- --nocapture --test-threads=1
git add zluda/src/impl/cxl_tmatmul.rs zluda/src/impl/kimi_iq1s_shadow.rs
git commit -m "feat: make packed DAX shadow transactions recoverable"
```

### Task 8: Execute Exactly One RUN_CSR_ONLY and Validate Completion

**Files:**
- Modify: `zluda/src/impl/cxl_tmatmul.rs`

- [ ] **Step 1: Write failing submission tests**

Add:

```rust
staging_reads_back_input_output_and_program
staging_spot_checks_matrix_start_middle_end
submission_calls_run_csr_only_exactly_once
submission_never_uses_bar_fallback
submission_never_retries
terminal_requires_ioctl_success_and_both_dma_done_bits
terminal_requires_stalled_flag_and_stall_status
terminal_rejects_dma_reset_and_execution_errors
terminal_requires_dimension_2048_and_eight_fetches
terminal_requires_positive_expected_counter_deltas
```

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  cxl_tmatmul::tests::shadow_submit_ -- --nocapture
```

Expected: failures identify each missing terminal predicate.

- [ ] **Step 3: Wire the existing proven UAPI helpers**

Build and stage the replay-safe program:

```text
slot 0 ldv v0, 0x100000
slot 1 tmatmul_import v0
slot 2 tmatmul_go 0x000000
slot 3 tmatmul_export v1
slot 4 sv v1, 0x110000
slot 5 stall
slot 6 replay-safe padding
slot 7 terminal stall encoding
```

Use the established assembler encoding rather than duplicating bit fields.
Perform exactly one `RUN_CSR_ONLY` at program DPA `0x120000`.

- [ ] **Step 4: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  cxl_tmatmul::tests::shadow_ -- --nocapture --test-threads=1
cargo check -p zluda --features nvidia --no-default-features
git add zluda/src/impl/cxl_tmatmul.rs
git commit -m "feat: submit one replay-safe packed tmatmul program"
```

### Task 9: Wire the Post-Native Observer Without Changing CUDA Results

**Files:**
- Modify: `zluda/src/impl/function.rs:8683-8731`
- Modify: `zluda/src/impl/kimi_iq1s_shadow.rs`

- [ ] **Step 1: Write failing launch-order tests**

Extract the native launch/observer sequence behind a testable helper. Add:

```rust
native_launch_occurs_exactly_once
native_launch_precedes_shadow_observer
native_failure_skips_shadow
shadow_success_preserves_native_success
every_shadow_failure_preserves_native_success
observer_receives_exact_native_stream
observer_never_writes_native_output
cap_one_prevents_second_hardware_transaction
disabled_mode_does_not_call_shadow_code
```

The fake native launcher must count calls and return chosen CUDA status values.

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  function::tests:: -- --nocapture --test-threads=1
```

Expected: tests fail until the observer is integrated.

- [ ] **Step 3: Add the observer after native success**

In `launch_kernel`:

```rust
let result = nvidia_runtime_sys::cuLaunchKernel(/* existing exact arguments */);
if result != 0 {
    return Err(CUerror::UNKNOWN);
}

kimi_iq1s_shadow::observe_after_native_launch(
    &f.function_name,
    (grid_dim_x, grid_dim_y, grid_dim_z),
    (block_dim_x, block_dim_y, block_dim_z),
    shared_mem_bytes,
    h_stream,
    kernel_params,
);
```

The observer returns `()` and logs its own diagnostic outcome. It cannot change
the native result. Preserve the existing `kimi_concordia` observer and define a
stable order; the IQ1_S observer must receive untouched args and the same stream.
Do the equivalent only for `launch_kernel_ex` if an exact type-19 MMVQ route is
proven to use it; otherwise emit a tested ineligibility event and stay native.

- [ ] **Step 4: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  function::tests:: -- --nocapture --test-threads=1
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests:: -- --nocapture --test-threads=1
cargo check -p zluda --features nvidia --no-default-features
git add zluda/src/impl/function.rs zluda/src/impl/kimi_iq1s_shadow.rs
git commit -m "feat: observe Kimi IQ1_S after native CUDA launch"
```

### Task 10: Add Structured Evidence and a Strict Validator

**Files:**
- Modify: `zluda/src/impl/kimi_iq1s_shadow.rs`
- Modify: `zluda/src/impl/cxl_tmatmul.rs`
- Create: `tools/validate_kimi_iq1s_shadow.py`
- Create: `tools/tests/test_validate_kimi_iq1s_shadow.py`

- [ ] **Step 1: Write failing event-schema tests**

Add Rust tests:

```rust
route_event_can_never_claim_hardware_execution
submit_completed_is_only_hardware_execution_event
comparison_event_records_all_hashes_and_mismatch_fields
complete_event_requires_restore_verified
ambiguous_failure_records_cleanup_blocked_marker
```

Add Python fixture tests with one valid JSONL and mutations for duplicate
submits, missing restore, mismatch, wrong devices, and route-only hardware
claims.

- [ ] **Step 2: Run and confirm failure**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests::event_ -- --nocapture
python3 -m unittest discover -s tools/tests \
  -p 'test_validate_kimi_iq1s_shadow.py'
```

Expected: failure because the schema/validator is incomplete.

- [ ] **Step 3: Implement JSONL events**

Serialize the exact events from the design. Every event includes:

```text
schema_version, timestamp, boot_id, pid, launch_id, event,
kernel, control_dev, dax_dev, hardware_executed
```

`numerical_comparison` additionally includes all hashes, mismatch count, first
mismatch, maximum integer difference, scale/offset telemetry, saturation
headroom, bounded native-output statistics, and:

```json
{
  "authoritative_output": "native_cuda",
  "fpga_vs_native_directly_comparable": false
}
```

- [ ] **Step 4: Implement the validator**

The validator accepts:

```bash
python3 tools/validate_kimi_iq1s_shadow.py \
  --log /path/shadow.jsonl \
  --expected-control /dev/cxl_tmatmul3b000 \
  --expected-dax /dev/dax6.0 \
  --require-launches 1
```

Exit nonzero unless there is exactly one claim, one submit, terminal completion,
zero mismatches, one verified restore, and no later submit.

- [ ] **Step 5: Run and commit**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow::tests::event_ -- --nocapture
python3 -m unittest discover -s tools/tests \
  -p 'test_validate_kimi_iq1s_shadow.py'
git add \
  zluda/src/impl/kimi_iq1s_shadow.rs \
  zluda/src/impl/cxl_tmatmul.rs \
  tools/validate_kimi_iq1s_shadow.py \
  tools/tests/test_validate_kimi_iq1s_shadow.py
git commit -m "test: validate Kimi FPGA shadow evidence"
```

### Task 11: Add Reproducible Model-Free and Kimi Launch Wrappers

**Files:**
- Create: `tools/run_kimi_iq1s_fpga_shadow.sh`
- Create: `tools/run_kimi_iq1s_model_free_gate.sh`
- Modify: `tools/validate_kimi_iq1s_shadow.py`

- [ ] **Step 1: Write shell contract tests**

Add `--dry-run` to both scripts and test that output contains:

```text
/home/eabban/BitNet/build-cuda128-gcc12/bin/llama-cli
/dev/cxl_tmatmul3b000
/dev/dax6.0
HETGPU_KIMI_FPGA_SHADOW_MAX_LAUNCHES=1
HETGPU_KIMI_FPGA_SHADOW_TIMEOUT_MS=10000
one generated token
deterministic seed and sampling
```

The Kimi script must discover exactly the six files matching
`/root/models/kimi-k2.6-iq1_s/moonshotai_Kimi-K2.6-IQ1_S/moonshotai_Kimi-K2.6-IQ1_S-0000[1-6]-of-00006.gguf`
and abort otherwise. It must record executable, model, and HetGPU library
SHA-256 values.

- [ ] **Step 2: Run and confirm failure**

```bash
bash tools/run_kimi_iq1s_model_free_gate.sh --dry-run
bash tools/run_kimi_iq1s_fpga_shadow.sh --dry-run
```

Expected: commands fail until scripts exist.

- [ ] **Step 3: Implement wrappers**

The model-free script:

```text
preflights devices without writes
runs one synthetic packed subgroup transaction
validates 2048 outputs
validates 2 MiB restore hash
writes a machine-readable result JSON
```

The Kimi script runs two separate processes:

```text
baseline: HETGPU_KIMI_FPGA_SHADOW unset
shadow:   HETGPU_KIMI_FPGA_SHADOW=1
```

Both use identical prompt, seed, sampling, GPU layers, context, batch settings,
and `-n 1`. Capture stdout, stderr, route JSONL, shadow JSONL, environment, hashes,
exit status, and extracted token under a unique results directory.

- [ ] **Step 4: Run dry-run validation and commit**

```bash
bash tools/run_kimi_iq1s_model_free_gate.sh --dry-run
bash tools/run_kimi_iq1s_fpga_shadow.sh --dry-run
bash -n tools/run_kimi_iq1s_model_free_gate.sh
bash -n tools/run_kimi_iq1s_fpga_shadow.sh
git add \
  tools/run_kimi_iq1s_model_free_gate.sh \
  tools/run_kimi_iq1s_fpga_shadow.sh \
  tools/validate_kimi_iq1s_shadow.py
git commit -m "test: add Kimi IQ1_S FPGA shadow proof runners"
```

Expected: dry runs identify exact commands without opening devices.

### Task 12: Run the Offline Regression Gate

**Files:**
- Verify only

- [ ] **Step 1: Format and inspect**

```bash
cargo fmt --all -- --check
git diff --check
git status --short
```

Expected: no formatting or whitespace failures; only planned files differ from
the committed task sequence.

- [ ] **Step 2: Run all focused tests**

```bash
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow -- --nocapture --test-threads=1
cargo test -p zluda --features nvidia --no-default-features \
  cxl_tmatmul::tests::shadow_ -- --nocapture --test-threads=1
python3 -m unittest discover -s tools/tests \
  -p 'test_validate_kimi_iq1s_shadow.py'
```

Expected: all pass without root and without device nodes.

- [ ] **Step 3: Build both relevant NVIDIA configurations**

```bash
cargo check -p zluda --features nvidia --no-default-features
cargo check -p zluda --features nvidia,tmatmul --no-default-features
cargo build -p zluda --release --features nvidia --no-default-features
```

Expected: all exit zero.

- [ ] **Step 4: Prove disabled-mode compatibility**

Run the existing focused NVIDIA/Kimi tests with shadow variables unset:

```bash
env -u HETGPU_KIMI_FPGA_SHADOW \
  cargo test -p zluda --features nvidia --no-default-features \
  kimi_concordia -- --nocapture --test-threads=1
```

Expected: existing tests pass and no shadow log is created.

- [ ] **Step 5: Commit any test-only corrections**

```bash
git add -u
git diff --cached --check
git commit -m "test: complete offline Kimi FPGA shadow regression"
```

Skip this commit if no correction was needed.

### Task 13: Run the Live Model-Free Hardware Gate

**Files:**
- Produce: the result directory printed by
  `tools/run_kimi_iq1s_model_free_gate.sh`

- [ ] **Step 1: Capture immutable live topology before device access**

```bash
date -u
cat /proc/sys/kernel/random/boot_id
lspci -Dnn
ls -l /dev/cxl_tmatmul3b000 /dev/dax6.0
cat /sys/class/dax/dax6.0/size
sudo dmesg --ctime | tail -n 200
```

Abort without writes if either node, `TMM1`, UAPI v2, `dim_d=2048`, program DPA
`0x120000`, or at least 2 MiB of DAX cannot be proven.

- [ ] **Step 2: Ensure exclusive ownership**

Check no benchmark, Kimi process, previous proof, or DAX writer is running. Take
the same exclusive control-device lock the runtime will use. Abort on contention.

- [ ] **Step 3: Run exactly one model-free transaction**

```bash
sudo tools/run_kimi_iq1s_model_free_gate.sh \
  --control /dev/cxl_tmatmul3b000 \
  --dax /dev/dax6.0 \
  --snapshot-bytes 0x200000 \
  --program-dpa 0x120000
```

Expected:

```text
RUN_CSR_ONLY calls: 1
fetched slots: 8
valid output rows: 2048
mismatches: 0
snapshot SHA-256 == restored SHA-256
dirty marker: none
```

- [ ] **Step 4: Validate post-state**

```bash
sudo dmesg --ctime | tail -n 200
RESULT_DIR="$(find .live -maxdepth 1 -type d \
  -name 'kimi_iq1s_shadow_model_free_*' -printf '%T@ %p\n' |
  sort -nr | head -n1 | cut -d' ' -f2-)"
python3 tools/validate_kimi_iq1s_shadow.py \
  --log "${RESULT_DIR}/shadow.jsonl" \
  --expected-control /dev/cxl_tmatmul3b000 \
  --expected-dax /dev/dax6.0 \
  --require-launches 1
```

Do not proceed to Kimi on an empty `RESULT_DIR`, warning, mismatch, retained
marker, or ambiguous completion.

### Task 14: Run the Kimi Baseline and One-Token Shadow Gate

**Files:**
- Produce: the result directory printed by
  `tools/run_kimi_iq1s_fpga_shadow.sh`

- [ ] **Step 1: Recheck live state**

Repeat the boot ID, nodes, idle status, exclusive-lock, NVIDIA health, and kernel
log checks from Task 13. Confirm the model-free proof completed in the same boot.

- [ ] **Step 2: Run baseline and shadow**

```bash
tools/run_kimi_iq1s_fpga_shadow.sh \
  --llama-cli /home/eabban/BitNet/build-cuda128-gcc12/bin/llama-cli \
  --control /dev/cxl_tmatmul3b000 \
  --dax /dev/dax6.0 \
  --max-launches 1 \
  --timeout-ms 10000 \
  --tokens 1
```

Expected: baseline exits zero, shadow exits zero, and the script prints the
unique result directory.

- [ ] **Step 3: Validate evidence**

```bash
RESULT_DIR="$(find .live -maxdepth 1 -type d \
  -name 'kimi_iq1s_shadow_kimi_*' -printf '%T@ %p\n' |
  sort -nr | head -n1 | cut -d' ' -f2-)"
python3 tools/validate_kimi_iq1s_shadow.py \
  --log "${RESULT_DIR}/shadow.jsonl" \
  --expected-control /dev/cxl_tmatmul3b000 \
  --expected-dax /dev/dax6.0 \
  --require-launches 1
```

Also assert:

```text
baseline token == shadow token
exactly one IQ1_S MMVQ claim
exactly one RUN_CSR_ONLY
all valid FPGA integer rows == CPU reference
exactly one restore_verified
no later hardware submit
native CUDA remains authoritative
no CXL, DAX, FPGA, or NVIDIA kernel error
```

Do not report TPS from this one-token correctness run.

- [ ] **Step 4: Capture final system evidence**

```bash
nvidia-smi
sudo dmesg --ctime | tail -n 300
git status --short
git log --oneline --decorate -15
```

Store command outputs in the result directory. Do not commit generated logs,
snapshots, model data, or `.live` results.

### Task 15: Final Review and Local Handoff

**Files:**
- Review all planned source and tool files
- Update: `docs/superpowers/specs/2026-07-26-kimi-iq1s-fpga-shadow-design.md`
  only if implementation evidence requires a factual clarification

- [ ] **Step 1: Verify specification coverage**

Build a local checklist mapping every acceptance criterion at design lines
`539-556` to a test name and evidence path. Any unproven criterion remains
explicitly blocked.

- [ ] **Step 2: Scan for incomplete implementation**

```bash
rg -n 'TODO|TBD|FIXME|placeholder|usize::MAX|tmatmul_go_nvint8' \
  zluda/src/impl/kimi_iq1s_shadow.rs \
  zluda/src/impl/kimi_iq1s_grid.rs \
  zluda/src/impl/cxl_tmatmul.rs \
  zluda/src/impl/function.rs \
  tools/run_kimi_iq1s_fpga_shadow.sh \
  tools/run_kimi_iq1s_model_free_gate.sh \
  tools/validate_kimi_iq1s_shadow.py
```

Expected:

- no placeholder markers in new code;
- no `usize::MAX` bypass in the IQ1_S shadow path;
- `tmatmul_go_nvint8` appears only in a negative assertion/test or unrelated
  pre-existing paths.

- [ ] **Step 3: Review type and boundary consistency**

Confirm:

```text
CUDA pointers remain CUdeviceptr at the ABI boundary
DPA offsets are u64
byte lengths are checked usize conversions
dimensions are validated before conversion
FPGA outputs are signed i16
reference accumulation is at least i32 before checked i16 conversion
launch IDs and hashes are stable strings
```

- [ ] **Step 4: Run the full final command set**

```bash
cargo fmt --all -- --check
cargo test -p zluda --features nvidia --no-default-features \
  kimi_iq1s_shadow -- --nocapture --test-threads=1
cargo test -p zluda --features nvidia --no-default-features \
  cxl_tmatmul::tests::shadow_ -- --nocapture --test-threads=1
cargo check -p zluda --features nvidia --no-default-features
cargo check -p zluda --features nvidia,tmatmul --no-default-features
python3 -m unittest discover -s tools/tests \
  -p 'test_validate_kimi_iq1s_shadow.py'
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 5: Create the final local commit**

```bash
git status --short
git add \
  docs/superpowers/specs/2026-07-26-kimi-iq1s-fpga-shadow-design.md \
  zluda/src/impl/mod.rs \
  zluda/src/impl/function.rs \
  zluda/src/impl/cxl_tmatmul.rs \
  zluda/src/impl/kimi_iq1s_grid.rs \
  zluda/src/impl/kimi_iq1s_shadow.rs \
  zluda/test_data/kimi_iq1s_subgroup0_fixture.json \
  tools/run_kimi_iq1s_model_free_gate.sh \
  tools/run_kimi_iq1s_fpga_shadow.sh \
  tools/validate_kimi_iq1s_shadow.py \
  tools/tests/test_validate_kimi_iq1s_shadow.py
git diff --cached --check
git commit -m "feat: validate Kimi IQ1_S on the ternary FPGA shadow path"
```

Skip unchanged paths rather than using `git add -A`. Do not push.

## Completion Evidence

The work is complete only when all of these artifacts exist and validate:

```text
offline unit and mock test logs
NVIDIA and NVIDIA+tmatmul cargo check logs
model-free hardware result JSON
model-free JSONL with one terminal submit and verified restore
Kimi baseline stdout/stderr/token record
Kimi shadow stdout/stderr/token record
Kimi JSONL with exact integer comparison and verified restore
same-boot topology and dmesg evidence
local git commit history
```

Build success alone is not live hardware proof. A route event is not hardware
execution. A one-token correctness run is not a TPS benchmark.
