# Kimi IQ1_S AU250 Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Kimi K2.6 with attention on the NVIDIA GPU and strict, numerically reconstructed IQ1_S BitLinear launches on all four AU250 TernIP compute units.

**Architecture:** Add a backend-neutral route selector, refactor the existing four-BO XRT code into a persistent multi-CU wave executor, and add an IQ1_S-to-D=1024 planner that multiplexes `(batch, group32)` work over TernIP lanes. Capture and reconstruction reuse `iq1s_tmatmul`; a selected XRT launch copies back a complete finite f32 result or aborts without native-GPU BitLinear fallback.

**Tech Stack:** Rust 2021, CUDA Driver API interception, XRT C APIs loaded with `dlopen`, AU250 `MaxCores_370M.xclbin`, Bash/Docker `au250-run` environment, Python 3 proof validation, Cargo tests.

---

## File Map

- Modify `zluda/src/impl/bitnet_disagg.rs`: parse `HETGPU_TMATMUL_BACKEND`, preserve existing logical route names, and add physical-backend evidence.
- Modify `zluda/src/impl/xrt_tmatmul.rs`: retain the one-shot API and add persistent device/CU/four-BO ownership plus concurrent wave execution.
- Create `zluda/src/impl/iq1s_xrt.rs`: own AU250-specific D=1024 geometry, packed grid/delta matrices, lane inputs, result demultiplexing, reconstruction, and execution evidence.
- Modify `zluda/src/impl/iq1s_tmatmul.rs`: expose only the captured Q8 group and checked output helpers needed by `iq1s_xrt`.
- Modify `zluda/src/impl/function.rs`: select XRT before CXL-only staging checks and route exact IQ1_S MMQ/MMVQ launches fail-closed.
- Modify `zluda/src/impl/mod.rs`: register `iq1s_xrt` only for the supported Unix/NVIDIA build.
- Modify `zluda/tests/run_au250_xrt_vector_add.sh`: use the installed `MaxCores_370M.xclbin`.
- Modify `zluda/tests/run_au250_xrt_tmatmul.sh`: use the installed `MaxCores_370M.xclbin`.
- Create `tools/au250_hybrid_run.sh`: launch one app215 container with both NVIDIA and AU250 devices and required mounts.
- Create `tools/build_au250_kimi_runtime.sh`: build a glibc-2.35-compatible CUDA 13 BitNet runner and Rust shim inside app215.
- Create `zluda/tests/test_au250_hybrid_runtime_static.sh`: fail-closed static validation of the hybrid container/build wrappers.
- Create `zluda/tests/run_au250_xrt_iq1s.sh`: execute the ignored live tiled IQ1_S hardware test and health gates.
- Create `tools/run_kimi_k26_iq1s_au250_hybrid.sh`: run captured-layer qualification, strict E2E inference, evidence validation, and benchmark collection.
- Create `zluda/tests/validate_au250_hybrid_proof.py`: validate route, physical XRT execution, tokens, health, hashes, and timing evidence.
- Create `zluda/tests/test_validate_au250_hybrid_proof.py`: unit tests for proof acceptance and fail-closed rejection.
- Create `docs/kimi_k26_au250_hybrid_proof.txt`: exact commands and proof boundary after live qualification.

### Task 1: Establish a GPU-capable app215 execution environment

**Files:**
- Create: `tools/au250_hybrid_run.sh`
- Create: `tools/build_au250_kimi_runtime.sh`
- Create: `zluda/tests/test_au250_hybrid_runtime_static.sh`

- [ ] **Step 1: Write the failing static wrapper test**

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
runner="${repo_root}/tools/au250_hybrid_run.sh"
builder="${repo_root}/tools/build_au250_kimi_runtime.sh"

bash -n "${runner}"
bash -n "${builder}"
rendered="$(${runner} --print-docker true)"
grep -Fq -- '--gpus all' <<<"${rendered}"
grep -Fq -- '-v /au250_xrt:/au250_xrt:ro' <<<"${rendered}"
grep -Fq -- '/usr/local/cuda-13.0:/usr/local/cuda-13.0:ro' <<<"${rendered}"
grep -Fq -- '/home/eabban/BitNet:/bitnet:ro' <<<"${rendered}"
grep -Fq -- '/root/models/kimi-k2.6-iq1_s/moonshotai_Kimi-K2.6-IQ1_S:/models/kimi:ro' <<<"${rendered}"
grep -Fq -- 'app215' <<<"${rendered}"
grep -Fq -- 'CMAKE_CUDA_ARCHITECTURES=120' "${builder}"
grep -Fq -- 'GGML_CUDA=ON' "${builder}"
```

- [ ] **Step 2: Run the test and verify the wrappers are absent**

Run: `bash zluda/tests/test_au250_hybrid_runtime_static.sh`

Expected: FAIL because `tools/au250_hybrid_run.sh` does not exist.

- [ ] **Step 3: Implement the hybrid container wrapper**

`tools/au250_hybrid_run.sh` must render and execute the same command path:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
bitnet_root="${AU250_BITNET_ROOT:-/home/eabban/BitNet}"
model_root="${AU250_KIMI_MODEL_ROOT:-/root/models/kimi-k2.6-iq1_s/moonshotai_Kimi-K2.6-IQ1_S}"
cuda_root="${AU250_CUDA_ROOT:-/usr/local/cuda-13.0}"
source /au250_xrt/env.sh >/dev/null

if [[ "${1:-}" == "--print-docker" ]]; then
    shift
    printf 'docker run --rm --gpus all --privileged %s -v /sys:/sys -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro -v /au250_xrt:/au250_xrt:ro -v %q:/work -v %q:/bitnet:ro -v %q:/models/kimi:ro -v %q:/usr/local/cuda-13.0:ro app215 %q\n' \
        "$(_au250_devflags)" "${repo_root}" "${bitnet_root}" "${model_root}" "${cuda_root}" "$*"
    exit 0
fi

[[ $# -gt 0 ]] || { echo "usage: $0 <command...>" >&2; exit 2; }
temperature="$(_au250_fpga_temp)"
[[ -z "${temperature}" || "${temperature}" -lt "${AU250_TEMP_LIMIT:-85}" ]] || {
    echo "AU250 temperature ${temperature}C exceeds guard" >&2
    exit 1
}

docker run --rm --gpus all --privileged $(_au250_devflags) \
    -v /sys:/sys \
    -v /lib/firmware/xilinx:/lib/firmware/xilinx:ro \
    -v /au250_xrt:/au250_xrt:ro \
    -v "${repo_root}":/work -w /work \
    -v "${bitnet_root}":/bitnet:ro \
    -v "${model_root}":/models/kimi:ro \
    -v "${cuda_root}":/usr/local/cuda-13.0:ro \
    app215 bash -lc 'source /XRT/build/Release/opt/xilinx/xrt/setup.sh >/dev/null 2>&1; export PATH=/usr/local/cuda-13.0/bin:$PATH; export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}; exec "$@"' _ "$@"
```

- [ ] **Step 4: Implement the app215-compatible build wrapper**

`tools/build_au250_kimi_runtime.sh` invokes the hybrid wrapper with these exact build gates:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

"${repo_root}/tools/au250_hybrid_run.sh" bash -lc '
set -euo pipefail
cmake -S /bitnet -B /work/target/au250-bitnet-cuda130 \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build /work/target/au250-bitnet-cuda130 --target llama-cli llama-tokenize -j"$(nproc)"

export RUSTUP_HOME=/work/target/au250-runtime/rustup
export CARGO_HOME=/work/target/au250-runtime/cargo
export CARGO_TARGET_DIR=/work/target/au250-app215
export PATH=/work/target/au250-runtime/bin:${CARGO_HOME}/bin:${PATH}
install -d /work/target/au250-runtime/bin
test -x /work/target/au250-runtime/bin/ninja || install -m 0755 /opt/miniconda3/bin/ninja /work/target/au250-runtime/bin/ninja
test -x "${CARGO_HOME}/bin/cargo" || {
  curl --proto "=https" --tlsv1.2 -fsS https://sh.rustup.rs -o /work/target/au250-runtime/rustup-init.sh
  sh /work/target/au250-runtime/rustup-init.sh -y --profile minimal --default-toolchain 1.92.0
}
cargo build -p zluda --features nvidia --no-default-features

/work/target/au250-bitnet-cuda130/bin/llama-cli --version
ldd /work/target/au250-app215/debug/libnvcuda.so | grep -F libxrt_coreutil || true
nvidia-smi -L
xbutil examine -d 0000:64:00.1 -r platform
'
```

- [ ] **Step 5: Run static and live environment probes**

Run:

```bash
bash zluda/tests/test_au250_hybrid_runtime_static.sh
tools/au250_hybrid_run.sh bash -lc 'nvidia-smi -L; xbutil examine -d 0000:64:00.1 -r platform'
```

Expected: static test PASS; one NVIDIA GPU and the U250 platform are visible in the same app215 process namespace.

- [ ] **Step 6: Commit the environment boundary**

```bash
git add tools/au250_hybrid_run.sh tools/build_au250_kimi_runtime.sh zluda/tests/test_au250_hybrid_runtime_static.sh
git commit -m "build: add combined GPU AU250 runtime"
```

### Task 2: Add backend-neutral route selection and evidence

**Files:**
- Modify: `zluda/src/impl/bitnet_disagg.rs:55-116,423-472,517-1010`
- Modify: `zluda/src/impl/function.rs:8107-8138,8885-8920`

- [ ] **Step 1: Add failing backend-selection and log tests**

Add tests that assert the exact contract:

```rust
#[test]
fn backend_env_selects_xrt_without_enabling_cxl() {
    let _guard = EnvGuard::set(&[("HETGPU_TMATMUL_BACKEND", Some("xrt"))]);
    assert_eq!(tmatmul_backend_from_env().unwrap(), Some(TmatmulBackend::Xrt));
}

#[test]
fn backend_env_rejects_unknown_values() {
    let _guard = EnvGuard::set(&[("HETGPU_TMATMUL_BACKEND", Some("cuda"))]);
    assert!(tmatmul_backend_from_env().unwrap_err().contains("cuda"));
}

#[test]
fn xrt_route_log_preserves_logical_route_and_records_physical_backend() {
    let path = tempfile::NamedTempFile::new().unwrap();
    let decision = classify_kernel_name("ffn_mul_mat_q", &config(true));
    append_route_log(path.path(), &decision, RouteHardware::xrt(true)).unwrap();
    let value: serde_json::Value = serde_json::from_str(
        std::fs::read_to_string(path.path()).unwrap().lines().next().unwrap()
    ).unwrap();
    assert_eq!(value["route"], "cxl_tmatmul");
    assert_eq!(value["backend"], "xrt");
    assert_eq!(value["xrt_enabled"], true);
    assert_eq!(value["cxl_enabled"], false);
}
```

Include `HETGPU_TMATMUL_BACKEND` in `BITNET_DISAGG_ENV_VARS` so tests restore the environment.

- [ ] **Step 2: Verify the new tests fail**

Run: `cargo test -p zluda --features nvidia --no-default-features bitnet_disagg -- --nocapture`

Expected: FAIL because `TmatmulBackend` and `RouteHardware` are undefined.

- [ ] **Step 3: Implement the selector and evidence context**

Add these types and parsing behavior:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TmatmulBackend { Cxl, Xrt }

impl TmatmulBackend {
    pub(crate) fn as_str(self) -> &'static str {
        match self { Self::Cxl => "cxl", Self::Xrt => "xrt" }
    }
}

pub(crate) fn tmatmul_backend_from_env() -> Result<Option<TmatmulBackend>, String> {
    match std::env::var("HETGPU_TMATMUL_BACKEND").ok().as_deref().map(str::trim) {
        None | Some("") => Ok(None),
        Some("cxl") => Ok(Some(TmatmulBackend::Cxl)),
        Some("xrt") => Ok(Some(TmatmulBackend::Xrt)),
        Some(value) => Err(format!("unsupported HETGPU_TMATMUL_BACKEND={value:?}; expected cxl or xrt")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouteHardware {
    pub(crate) backend: Option<TmatmulBackend>,
    pub(crate) enabled: bool,
    pub(crate) hardware_matmul_enabled: bool,
}

impl RouteHardware {
    pub(crate) fn xrt(hardware_matmul_enabled: bool) -> Self {
        Self { backend: Some(TmatmulBackend::Xrt), enabled: true, hardware_matmul_enabled }
    }
    pub(crate) fn cxl(enabled: bool, hardware_matmul_enabled: bool) -> Self {
        Self { backend: Some(TmatmulBackend::Cxl), enabled, hardware_matmul_enabled }
    }
}
```

Change `append_route_log` and `append_route_log_from_env` to accept `RouteHardware`; retain `route`, `source`, and existing CXL fields while adding `backend` and `xrt_enabled`.

- [ ] **Step 4: Update all existing log call sites without changing CXL behavior**

Pass `RouteHardware::cxl(cxl_enabled, hardware_enabled)` at existing CXL/native call sites. In the named pre-native handler, parse the backend once; return a strict configuration error for an invalid value and dispatch XRT before checking `cxl_tmatmul_enabled()` or CUDA-DAX staging.

- [ ] **Step 5: Run route tests**

Run:

```bash
cargo test -p zluda --features nvidia --no-default-features bitnet_disagg -- --nocapture
cargo test -p zluda --features nvidia --no-default-features nvidia_named_cxl -- --nocapture
```

Expected: all selected tests PASS and legacy records still contain `"route":"cxl_tmatmul"`.

- [ ] **Step 6: Commit route selection**

```bash
git add zluda/src/impl/bitnet_disagg.rs zluda/src/impl/function.rs
git commit -m "feat: select physical tmatmul backend"
```

### Task 3: Refactor XRT into a persistent four-CU wave executor

**Files:**
- Modify: `zluda/src/impl/xrt_tmatmul.rs:49-1023`

- [ ] **Step 1: Add failing configuration and lifecycle tests**

Add exact public request/result types and tests around the existing `FakeXrt` event stream:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XrtCuTarget {
    pub(crate) ip_name: String,
    pub(crate) memory_group: u32,
    pub(crate) lanes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XrtWaveJob {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) matrix: Arc<[u8]>,
    pub(crate) input: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct XrtWaveCompletion {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) stall_code: u32,
    pub(crate) output: Vec<u8>,
}

#[test]
fn maxcores_targets_match_xclbin_layout() {
    assert_eq!(XrtPoolConfig::maxcores_targets(), vec![
        XrtCuTarget { ip_name: "ternip_big:ternip_big_1".into(), memory_group: 0, lanes: 9 },
        XrtCuTarget { ip_name: "ternip_big:ternip_big_2".into(), memory_group: 3, lanes: 9 },
        XrtCuTarget { ip_name: "ternip_big:ternip_big_3".into(), memory_group: 2, lanes: 9 },
        XrtCuTarget { ip_name: "ternip_small:ternip_small_1".into(), memory_group: 1, lanes: 6 },
    ]);
}

#[test]
fn cu_table_override_is_one_versioned_atomic_value() {
    let json = r#"{"version":1,"cus":[{"ip_name":"ternip_big:ternip_big_1","memory_group":0,"lanes":9}]}"#;
    let parsed = XrtPoolConfig::parse_cu_table(json).unwrap();
    assert_eq!(parsed.len(), 1);
    assert!(XrtPoolConfig::parse_cu_table(r#"{"version":2,"cus":[]}"#).is_err());
    assert!(XrtPoolConfig::parse_cu_table(r#"{"version":1,"cus":[]}"#).is_err());
}

#[test]
fn pool_loads_xclbin_once_and_allocates_four_bos_per_cu() {
    let xrt = FakeXrt::new([1, 1, 1, 1]);
    let pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
    assert_eq!(pool.ops.events().iter().filter(|e| matches!(e, Event::LoadXclbin(_))).count(), 1);
    assert_eq!(pool.ops.events().iter().filter(|e| matches!(e, Event::BoAlloc { .. })).count(), 16);
}

#[test]
fn wave_returns_request_order_after_out_of_order_stalls() {
    let xrt = FakeXrt::with_per_ip_stalls(vec![vec![0, 0, 1], vec![1], vec![0, 0, 0, 1], vec![0, 1]]);
    let mut pool = Pool::open_with_ops(xrt, pool_test_config()).unwrap();
    let completions = pool.run_wave(test_wave_jobs()).unwrap();
    assert_eq!(completions.iter().map(|c| c.request_id).collect::<Vec<_>>(), vec![10, 11, 12, 13]);
}

fn test_wave_jobs() -> Vec<XrtWaveJob> {
    XrtPoolConfig::maxcores_targets().into_iter().enumerate().map(|(cu_index, target)| {
        XrtWaveJob {
            request_id: 10 + cu_index as u64,
            cu_index,
            matrix: Arc::from(vec![0_u8; AU250_MATRIX_BYTES]),
            input: vec![0_u8; expected_vector_bytes(target.lanes)],
        }
    }).collect()
}

fn pool_test_config() -> XrtPoolConfig {
    XrtPoolConfig {
        xclbin: PathBuf::from("/tmp/MaxCores_370M.xclbin"),
        device_index: 0,
        targets: XrtPoolConfig::maxcores_targets(),
        num_vector_registers: 4,
        timeout_ms: 20,
    }
}
```

Extend `FakeState` with
`stall_reads_by_ip: HashMap<u32, VecDeque<u32>>`; implement
`FakeXrt::with_per_ip_stalls` by assigning each vector to IP indices 0 through
N-1, and make `xcl_reg_read` consume only the queue for its `index`. This makes
the out-of-order completion test deterministic without sleeping.

- [ ] **Step 2: Verify lifecycle tests fail**

Run: `cargo test -p zluda --features nvidia --no-default-features xrt_tmatmul -- --nocapture`

Expected: FAIL because the persistent pool types are undefined.

- [ ] **Step 3: Split shared device and reusable CU ownership**

Implement these ownership boundaries in `xrt_tmatmul.rs`:

```rust
const AU250_DIM: usize = 1024;
const AU250_MATRIX_BYTES: usize = AU250_DIM * AU250_DIM / 4;
const AU250_TMATMUL_ASSEMBLY: &str = "ldv v0, PARAM_INPUT\ntmatmul_import v0\ntmatmul_go PARAM_MATRIX\ntmatmul_export v1\nsv v1, PARAM_OUTPUT\nstall\n";

struct ReusableCu {
    target: XrtCuTarget,
    ip_device: Handle,
    ip_index: u32,
    matrix_bo: Handle,
    input_bo: Handle,
    output_bo: Handle,
    program_bo: Handle,
    program_address: u64,
    program_bytes: usize,
    release_handles: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct XrtPoolConfig {
    xclbin: PathBuf,
    device_index: u32,
    targets: Vec<XrtCuTarget>,
    num_vector_registers: u8,
    timeout_ms: u32,
}

struct Pool<O: XrtOps> {
    ops: O,
    device: Handle,
    uuid: Xuid,
    cus: Vec<ReusableCu>,
    timeout_ms: u32,
    poisoned: bool,
}

pub(crate) struct XrtTmatmulPool {
    inner: Pool<RealXrt>,
}
```

`Pool::open_with_ops` owns the injected operations table and implements all
lifecycle behavior, which lets the existing `FakeXrt` observe it directly.
`XrtTmatmulPool::open_from_env()` constructs `Pool<RealXrt>`, loads the xclbin
once, obtains one UUID, opens each native-IP context, allocates
matrix/input/output/program BOs in matrix-input-output-program order, binds
stable addresses, assembles the six-instruction program once per CU, and syncs
each program BO once. Preserve `submit_xrt_tmatmul` as the one-shot
compatibility API.

`HETGPU_XRT_CU_CONFIG` is the only override and contains the complete version-1
JSON table; reject unknown versions, zero/duplicate CUs, duplicate IP names,
duplicate memory groups, lane counts other than 6 or 9, and more than four CUs.
When it is absent, use the validated `MaxCores_370M` table.

- [ ] **Step 4: Implement concurrent wave start/poll/finish**

Use one host coordinator: stage at most one job per CU, start every staged CU, then poll all pending STALL registers until each completes or the common deadline expires. Validate exact payload sizes:

```rust
fn expected_vector_bytes(lanes: usize) -> usize { AU250_DIM * lanes * 2 }

fn validate_wave_job(target: &XrtCuTarget, job: &XrtWaveJob) -> Result<(), XrtTmatmulError> {
    if job.matrix.len() != AU250_MATRIX_BYTES {
        return Err(XrtTmatmulError::InvalidBuffer(format!("matrix has {} bytes, expected {AU250_MATRIX_BYTES}", job.matrix.len())));
    }
    let expected = expected_vector_bytes(target.lanes);
    if job.input.len() != expected {
        return Err(XrtTmatmulError::InvalidBuffer(format!("input has {} bytes, expected {expected}", job.input.len())));
    }
    Ok(())
}
```

Read a full CU-sized output, acknowledge STALL, and sort completions by the input request order. On any start/poll/read failure, quiesce every CU launched in that wave; if any quiesce fails, set `poisoned=true`, retain its handles, and reject every subsequent call.

The pool is globally serialized by the IQ1_S executor. Add `unsafe impl Send for
XrtTmatmulPool` with a safety comment that every raw XRT handle is owned by the
pool, never accessed without its process-global mutex, and destroyed only by
the owning pool. Do not add `Sync` to any raw-handle owner.

- [ ] **Step 5: Run low-level regressions**

Run:

```bash
cargo test -p zluda --features nvidia --no-default-features xrt_tmatmul -- --nocapture
cargo test -p zluda --features nvidia --no-default-features cxl_tmatmul -- --nocapture
```

Expected: all one-shot and pool tests PASS, including four-BO ordering and quiescence retention.

- [ ] **Step 6: Commit persistent XRT ownership**

```bash
git add zluda/src/impl/xrt_tmatmul.rs
git commit -m "feat: keep AU250 XRT sessions persistent"
```

### Task 4: Implement D=1024 IQ1_S packing and lane planning

**Files:**
- Create: `zluda/src/impl/iq1s_xrt.rs`
- Modify: `zluda/src/impl/iq1s_tmatmul.rs:858-922`
- Modify: `zluda/src/impl/mod.rs:89-96`

- [ ] **Step 1: Expose bounded captured-launch helpers**

Change only these methods to `pub(crate)`:

```rust
pub(crate) fn checked_output_element_count(signature: &GgmlType19Signature) -> Result<usize, String>;
pub(crate) fn q8_group(&self, batch_index: usize, global_group: usize) -> Result<Q8_1Block, String>;
```

Do not expose packed activation storage or mutable matrix internals.

- [ ] **Step 2: Write failing AU250 geometry, packing, and lane tests**

Create `iq1s_xrt.rs` with constants and tests that assert:

```rust
pub(crate) const AU250_DIM: usize = 1024;
pub(crate) const AU250_GROUP_VALUES: usize = 32;
pub(crate) const AU250_GROUPS_PER_K_TILE: usize = 32;
pub(crate) const AU250_MATRIX_BYTES: usize = AU250_DIM * AU250_DIM / 4;

fn kimi_signature() -> GgmlType19Signature {
    GgmlType19Signature {
        kernel: "mul_mat_q".into(),
        ne00: 7168,
        ne01: 2048,
        stride01: 28,
        ne10: 7168,
        ne11: 1,
        stride11: 8064,
        ne0: 2048,
    }
}

#[test]
fn kimi_7168_by_2048_uses_seven_k_tiles_and_two_row_tiles() {
    let geometry = plan_au250_tiles(&kimi_signature()).unwrap();
    assert_eq!(geometry.iter().map(|t| t.k_tile).max(), Some(6));
    assert_eq!(geometry.iter().map(|t| t.row_tile).max(), Some(1));
    assert_eq!(geometry.last().unwrap().valid_in, 1024);
}

#[test]
fn lane_input_is_dimension_major_and_group_sparse() {
    let q8 = Q8_1Block { d: 1.0, s: 0.0, qs: [7; 32] };
    let bytes = pack_lane_input(9, &[(3, 5, q8)]).unwrap();
    let raw = bytes.chunks_exact(2).map(|b| i16::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    assert_eq!(raw[5 * 32 * 9 + 3], 7);
    assert_eq!(raw.iter().filter(|v| **v == 7).count(), 32);
}

#[test]
fn direct_q8_raw_values_keep_every_group_dot_in_i16() {
    assert_eq!(raw_dot_bounds(), (-4096, 4096));
}
```

- [ ] **Step 3: Verify packing tests fail**

Run: `cargo test -p zluda --features nvidia --no-default-features iq1s_xrt -- --nocapture`

Expected: FAIL because planning and packing functions are undefined.

- [ ] **Step 4: Implement tile and matrix materialization**

Define stable geometry and component keys:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Au250Tile {
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) valid_out: usize,
    pub(crate) valid_in: usize,
    pub(crate) group_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Au250MatrixKey {
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) kind: ComponentKind,
}
```

`plan_au250_tiles` steps by 1024 over `ne0` and `ne00`. `pack_component_matrix` iterates valid rows and groups, calls `MatrixSource::group(global_row, global_group)`, selects `grid_values` or repeated `delta_sign`, and writes four two-bit trits per byte with `-1 -> 3`, `0 -> 0`, `1 -> 1`. All padding remains zero and the returned length is exactly 262144 bytes.

- [ ] **Step 5: Implement deterministic lane work partitioning**

Use these identifiers:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaneAssignment {
    pub(crate) lane: usize,
    pub(crate) batch_index: usize,
    pub(crate) global_group: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedAu250Job {
    pub(crate) request_id: u64,
    pub(crate) cu_index: usize,
    pub(crate) matrix_key: Au250MatrixKey,
    pub(crate) assignments: Vec<LaneAssignment>,
}
```

For each tile/component, enumerate `(batch_index, global_group)` in batch-major then increasing-group order. Partition the stream across CU capacities `[9,9,9,6]`, with at most one job per CU in a wave. Reject zero capacities, duplicate lane assignments, out-of-range groups, and any request-ID overflow.

- [ ] **Step 6: Run decomposition tests and commit**

Run:

```bash
cargo test -p zluda --features nvidia --no-default-features iq1s_xrt -- --nocapture
cargo test -p zluda --features nvidia --no-default-features iq1s_tmatmul -- --nocapture
```

Expected: all selected tests PASS.

```bash
git add zluda/src/impl/iq1s_xrt.rs zluda/src/impl/iq1s_tmatmul.rs zluda/src/impl/mod.rs
git commit -m "feat: plan IQ1_S work for D1024 TernIP"
```

### Task 5: Execute and reconstruct strict IQ1_S results

**Files:**
- Modify: `zluda/src/impl/iq1s_xrt.rs`

- [ ] **Step 1: Write a failing mock-backend reconstruction test**

Define an injectable boundary and test full output replacement:

```rust
pub(crate) trait Au250WaveExecutor {
    fn lane_capacities(&self) -> Vec<usize>;
    fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String>;
}

#[test]
fn executor_demultiplexes_grid_delta_and_matches_reference_bits() {
    let captured = two_k_tile_two_row_tile_fixture();
    let expected = software_reference(&captured).unwrap();
    let mut backend = CpuDotWaveExecutor::new(vec![9, 9, 9, 6]);
    let result = execute_captured_with(&captured, &mut backend).unwrap();
    assert_eq!(result.outputs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
               expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert!(result.evidence.submission_count > 4);
    assert_eq!(result.evidence.backend, "xrt");
}

#[test]
fn missing_or_duplicate_completion_fails_before_output_copy() {
    let captured = small_fixture();
    for mode in [CompletionFault::Missing, CompletionFault::Duplicate] {
        let mut backend = FaultingWaveExecutor::new(mode);
        assert!(execute_captured_with(&captured, &mut backend).unwrap_err().contains("completion"));
    }
}
```

Add one shared test constructor in this module rather than hand-building each
case:

```rust
#[cfg(test)]
fn captured_fixture(ne00: u64, ne01: u64, batch: u64) -> CapturedLaunch {
    assert!(ne00.is_multiple_of(256));
    let blocks_per_row = usize::try_from(ne00 / 256).unwrap();
    let signature = GgmlType19Signature {
        kernel: "mul_mat_q".into(),
        ne00,
        ne01,
        stride01: blocks_per_row as u64,
        ne10: ne00,
        ne11: batch,
        stride11: batch,
        ne0: ne01,
    };
    let mut grid = [[0_i8; 8]; GRID_ENTRIES];
    for (index, values) in grid.iter_mut().enumerate() {
        for (column, value) in values.iter_mut().enumerate() {
            *value = [-1, 0, 1][(index + column) % 3];
        }
    }
    let mut one_block = [0_u8; IQ1S_BLOCK_BYTES];
    one_block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let mut matrix = Vec::with_capacity(ne01 as usize * blocks_per_row * IQ1S_BLOCK_BYTES);
    for row in 0..ne01 as usize {
        for block in 0..blocks_per_row {
            one_block[2] = ((row + block) & 0xff) as u8;
            matrix.extend_from_slice(&one_block);
        }
    }
    let records = usize::try_from((ne00 / 128 - 1) * batch + batch).unwrap();
    let mut activations = vec![0_u8; records * Q8_1_MMQ_BYTES];
    for record in activations.chunks_exact_mut(Q8_1_MMQ_BYTES) {
        for pair in 0..4 {
            record[pair * 4..pair * 4 + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            record[pair * 4 + 2..pair * 4 + 4].copy_from_slice(&0x3c00_u16.to_le_bytes());
            for (index, value) in record[16 + pair * 32..16 + (pair + 1) * 32].iter_mut().enumerate() {
                *value = (index as i16 - 16) as i8 as u8;
            }
        }
    }
    capture_from_host(
        LogicalLaunch {
            matrix_ptr: 0x1000,
            activation_ptr: 0x2000,
            output_ptr: 0x3000,
            allocation_generation: 1,
            content_hash: [0x5a; 32],
            signature,
        },
        &matrix,
        &activations,
        &grid,
    ).unwrap()
}

#[cfg(test)]
fn small_fixture() -> CapturedLaunch { captured_fixture(256, 1, 1) }

#[cfg(test)]
fn two_k_tile_two_row_tile_fixture() -> CapturedLaunch {
    captured_fixture(2048, 1030, 1)
}

#[cfg(test)]
fn software_reference(captured: &CapturedLaunch) -> Result<Vec<f32>, String> {
    let batch = usize::try_from(captured.launch.signature.ne11).map_err(|_| "batch overflow")?;
    let rows = usize::try_from(captured.launch.signature.ne0).map_err(|_| "row overflow")?;
    let groups = usize::try_from(captured.launch.signature.ne00 / 32).map_err(|_| "group overflow")?;
    let mut outputs = vec![0_f32; batch.checked_mul(rows).ok_or("output overflow")?];
    for batch_index in 0..batch {
        for row in 0..rows {
            for global_group in 0..groups {
                let q8 = captured.q8_group(batch_index, global_group)?;
                let (d, group) = captured.matrix.group(row, global_group)?;
                let (grid, delta) = raw_component_dots(&group, &q8);
                let contribution = reconstruct_from_raw(&group, d, &q8, grid << 8, delta << 8)?;
                let output = &mut outputs[batch_index * rows + row];
                *output = (*output + contribution) as f32;
            }
        }
    }
    Ok(outputs)
}
```

- [ ] **Step 2: Verify reconstruction tests fail**

Run: `cargo test -p zluda --features nvidia --no-default-features iq1s_xrt -- --nocapture`

Expected: FAIL because `execute_captured_with` and evidence types are undefined.

- [ ] **Step 3: Implement raw-component collection and reconstruction**

Use dense checked slots indexed by `(component, batch, row, global_group)`:

```rust
#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct XrtIq1sEvidence {
    pub(crate) backend: &'static str,
    pub(crate) logical_batch: usize,
    pub(crate) row_tiles: usize,
    pub(crate) k_tiles: usize,
    pub(crate) submission_count: u64,
    pub(crate) per_cu_submissions: Vec<u64>,
    pub(crate) raw_min: i16,
    pub(crate) raw_max: i16,
    pub(crate) reference_checked_components: u64,
    pub(crate) comparison_status: &'static str,
}

#[derive(Debug)]
pub(crate) struct XrtIq1sResult {
    pub(crate) outputs: Vec<f32>,
    pub(crate) evidence: XrtIq1sEvidence,
}
```

For every completion, validate request/CU identity, exact output length, and nonzero STALL. Decode dimension-major output at `(row * lanes + lane) * 2`; require each slot empty before insertion. After all waves, require every planned grid and delta slot filled exactly once. Iterate `batch`, `row`, then increasing `global_group`; fetch `captured.q8_group` and `captured.matrix.group`, call existing `reconstruct_from_raw` with `i64::from(raw) << 8` for compatibility with its Q8.8 validator, and accumulate with the same f32 rounding expression as the existing CXL executor. `reconstruct_from_raw` independently computes `raw_component_dots`, so every grid/delta hardware value is reference-checked before contributing; increment `reference_checked_components` only after both checks pass and set `comparison_status="pass"` only when every planned component was checked.

Cache packed `Arc<[u8]>` component matrices in a byte-accounted LRU keyed by
matrix identity, row tile, K tile, and component kind. Parse
`HETGPU_XRT_MATRIX_CACHE_BYTES` with a 512 MiB default, reject zero/overflow,
evict least-recently-used entries before insertion, and never evict a value
still held by an in-flight wave.

- [ ] **Step 4: Bind the real persistent pool and strict log writer**

Implement:

```rust
pub(crate) fn execute_captured(captured: &CapturedLaunch) -> Result<XrtIq1sResult, String> {
    static POOL: OnceLock<Mutex<Result<XrtTmatmulPool, String>>> = OnceLock::new();
    let pool = POOL.get_or_init(|| Mutex::new(XrtTmatmulPool::open_from_env().map_err(|e| e.to_string())));
    let mut guard = pool.lock().map_err(|_| "AU250 XRT pool mutex poisoned".to_string())?;
    let pool = match &mut *guard {
        Ok(pool) => pool,
        Err(error) => return Err(error.clone()),
    };
    execute_captured_with(captured, pool)
}
```

Implement `Au250WaveExecutor for XrtTmatmulPool` by forwarding capacities and
`run_wave`. Store initialization failures permanently for the process, append
one JSONL completion record to `HETGPU_XRT_EXECUTION_LOG`, and fail if the
configured log cannot be written.

- [ ] **Step 5: Run strict executor tests**

Run: `cargo test -p zluda --features nvidia --no-default-features iq1s_xrt -- --nocapture`

Expected: exact CPU/mock reconstruction PASS; missing, duplicate, nonzero-padding, timeout, and poisoned-backend cases FAIL closed as asserted.

- [ ] **Step 6: Commit the strict executor**

```bash
git add zluda/src/impl/iq1s_xrt.rs
git commit -m "feat: reconstruct IQ1_S from AU250 results"
```

### Task 6: Wire exact IQ1_S launches into the NVIDIA pre-native hook

**Files:**
- Modify: `zluda/src/impl/function.rs:8527-9114,9280-9620`

- [ ] **Step 1: Add failing XRT route tests**

Add tests using an injected captured-execution closure so no hardware is required:

```rust
#[test]
fn xrt_backend_bypasses_all_cxl_staging_gates() {
    let _guard = EnvGuard::set(&[
        ("HETGPU_TMATMUL_BACKEND", Some("xrt")),
        ("HETGPU_BITNET_DISAGGREGATE", Some("1")),
        ("HETGPU_BITNET_DISAGG_STRICT", Some("1")),
        ("HETGPU_TMATMUL_HARDWARE_MATMUL", Some("1")),
        ("HETGPU_CXL_TMATMUL", None),
        ("HETGPU_TMATMUL_MATRIX_STAGE", None),
        ("HETGPU_TMATMUL_IO_STAGE", None),
    ]);
    assert_eq!(selected_named_backend().unwrap(), Some(TmatmulBackend::Xrt));
}

#[test]
fn strict_xrt_failure_is_consumed_and_never_falls_through_native() {
    let result = xrt_post_execution_route("mul_mat_q<ggml_type19>", true, Err("stall timeout".into()), |_| {});
    assert!(matches!(result, Some(Err(message)) if message.contains("stall timeout")));
}

#[test]
fn non_iq1s_kernel_is_not_claimed_by_xrt_strict_mode() {
    assert!(unsafe {
        nvidia_try_launch_named_xrt_tmatmul("flash_attn", std::ptr::null_mut())
    }.is_none());
}
```

- [ ] **Step 2: Verify hook tests fail**

Run: `cargo test -p zluda --features nvidia --no-default-features nvidia_ -- --nocapture`

Expected: FAIL because the XRT handler is undefined.

- [ ] **Step 3: Implement MMQ and MMVQ XRT handlers**

Add `nvidia_try_launch_iq1s_xrt_tmatmul` and `nvidia_try_launch_iq1s_vec_xrt_tmatmul`. Reuse the existing shape readers, signature constructors, CUDA pointer readers, `capture_launch`, and `capture_vec_launch`. Replace only the physical executor call:

```rust
let captured = super::iq1s_tmatmul::capture_launch(launch)
    .map_err(|error| format!("capture IQ1_S CUDA launch: {error}"))?;
let execution = super::iq1s_xrt::execute_captured(&captured)?;
unsafe { super::iq1s_tmatmul::copy_outputs_to_cuda(&captured, &execution.outputs)?; }
```

`iq1s_xrt::execute_captured` must not copy CUDA output itself, so `function.rs` retains the ABI boundary and performs exactly one copy after complete validation.

Parse `HETGPU_XRT_COMPARE_MAX_LAUNCHES` as a nonnegative u64. For the first N
successful XRT launches, emit a `captured_layer_comparison` record containing
the output hash, `reference_checked_components`, and
`comparison_status="pass"`. The comparison record is evidence-only: every
launch still uses the same always-reference-checked reconstruction, and every
selected launch still routes to XRT.

- [ ] **Step 4: Dispatch XRT before the CXL path**

Rename the outer handler to `nvidia_try_launch_named_tmatmul`. Its order is:

```rust
match super::bitnet_disagg::tmatmul_backend_from_env() {
    Err(error) => return Some(Err(error)),
    Ok(Some(TmatmulBackend::Xrt)) => nvidia_try_launch_named_xrt_tmatmul(kernel_name, kernel_params),
    Ok(Some(TmatmulBackend::Cxl)) | Ok(None) => nvidia_try_launch_named_cxl_tmatmul(kernel_name, kernel_params),
}
```

The XRT handler first checks for exact IQ1_S MMQ/MMVQ names. Non-IQ1_S kernels return `None` even in strict mode. Qualified capture, XRT, evidence, reconstruction, or CUDA-copy errors return `Some(Err)` unconditionally when strict is set; no path calls native CUDA for that launch.

- [ ] **Step 5: Run hook and full library tests**

Run:

```bash
cargo test -p zluda --features nvidia --no-default-features nvidia_ -- --nocapture
cargo test -p zluda --features nvidia --no-default-features iq1s_ -- --nocapture
cargo test -p zluda --features nvidia --no-default-features
```

Expected: all tests PASS; attention markers remain native GPU, legacy CXL tests remain unchanged, and strict XRT candidates cannot fall through.

- [ ] **Step 6: Commit production routing**

```bash
git add zluda/src/impl/function.rs
git commit -m "feat: route Kimi IQ1_S launches to AU250"
```

### Task 7: Qualify the installed xclbin and live D=2048 tiled execution

**Files:**
- Modify: `zluda/tests/run_au250_xrt_vector_add.sh`
- Modify: `zluda/tests/run_au250_xrt_tmatmul.sh`
- Create: `zluda/tests/run_au250_xrt_iq1s.sh`
- Modify: `zluda/src/impl/iq1s_xrt.rs`

- [ ] **Step 1: Replace the removed xclbin path in existing wrappers**

Set both wrappers to:

```bash
export HETGPU_XRT_XCLBIN=/au250_xrt/example/MaxCores_370M.xclbin
```

Keep the native-IP selector, memory group, temperature guard, and firewall checks unchanged.

- [ ] **Step 2: Add an ignored live tiled IQ1_S test**

The test constructs a valid host-captured fixture with `ne00=2048`, `ne01=1030`, `ne10=2048`, `ne11=1`, `ne0=1030`; patterned grid indices/signs; finite half scales; and Q8 values spanning `-128..127`. It runs `iq1s_xrt::execute_captured`, computes the existing software reference, and asserts f32 bit equality plus physical coverage:

```rust
#[test]
#[ignore = "requires HETGPU_XRT_AU250_IQ1S_TEST=1 and live MaxCores AU250"]
fn au250_iq1s_two_by_two_tiles_match_reference() {
    assert_eq!(std::env::var("HETGPU_XRT_AU250_IQ1S_TEST").as_deref(), Ok("1"));
    let captured = two_k_tile_two_row_tile_fixture();
    let expected = software_reference(&captured).unwrap();
    let actual = execute_captured(&captured).unwrap();
    assert_eq!(actual.outputs.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
               expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    assert!(actual.evidence.per_cu_submissions.iter().filter(|n| **n > 0).count() >= 2);
}
```

- [ ] **Step 3: Create the live wrapper**

`zluda/tests/run_au250_xrt_iq1s.sh` uses `tools/au250_hybrid_run.sh`, the cached app215 Rust toolchain, and:

```bash
export HETGPU_XRT_AU250_IQ1S_TEST=1
export HETGPU_XRT_XCLBIN=/au250_xrt/example/MaxCores_370M.xclbin
export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
export HETGPU_XRT_TIMEOUT_MS=10000
export HETGPU_XRT_EXECUTION_LOG=/work/target/au250-iq1s-live.jsonl
cargo test -p zluda --features nvidia --no-default-features \
  au250_iq1s_two_by_two_tiles_match_reference -- --ignored --nocapture
```

Before the test, run `xclbinutil --info` and require the exact four instance/bank
pairs from the approved design. After the test, require `xbutil examine`
firewall `GOOD`, all used CUs `DONE`, no fatal errors, and a temperature below
85 C.

- [ ] **Step 4: Run the three live gates**

Run:

```bash
bash zluda/tests/run_au250_xrt_vector_add.sh
bash zluda/tests/run_au250_xrt_tmatmul.sh
bash zluda/tests/run_au250_xrt_iq1s.sh
```

Expected: vector-add exact PASS, canonical tmatmul exact PASS, and tiled IQ1_S f32-bit PASS with at least two physical CUs used.

- [ ] **Step 5: Commit live qualification**

```bash
git add zluda/tests/run_au250_xrt_vector_add.sh zluda/tests/run_au250_xrt_tmatmul.sh zluda/tests/run_au250_xrt_iq1s.sh zluda/src/impl/iq1s_xrt.rs
git commit -m "test: qualify tiled IQ1_S on AU250"
```

### Task 8: Build strict Kimi E2E proof and benchmark validation

**Files:**
- Create: `tools/run_kimi_k26_iq1s_au250_hybrid.sh`
- Create: `zluda/tests/validate_au250_hybrid_proof.py`
- Create: `zluda/tests/test_validate_au250_hybrid_proof.py`

- [ ] **Step 1: Write failing proof-validator tests**

Use temporary JSONL/log fixtures and assert rejection of each missing boundary:

```python
def test_accepts_gpu_attention_and_physical_xrt_bitlinear(tmp_path):
    proof = make_valid_proof(tmp_path)
    result = validate(proof)
    assert result["status"] == "pass"
    assert result["xrt_completions"] > 0
    assert result["gpu_attention_routes"] > 0

@pytest.mark.parametrize("mutation", [
    "missing_xrt_completion", "bitlinear_gpu_fallback", "missing_attention_gpu_route",
    "invalid_token", "nonzero_exit", "firewall_bad", "fatal_error", "hash_missing",
])
def test_rejects_incomplete_or_ambiguous_proof(tmp_path, mutation):
    proof = make_valid_proof(tmp_path)
    apply_mutation(proof, mutation)
    with pytest.raises(ProofError):
        validate(proof)
```

- [ ] **Step 2: Verify validator tests fail**

Run: `python3 -m pytest -q zluda/tests/test_validate_au250_hybrid_proof.py`

Expected: FAIL because the validator module does not exist.

- [ ] **Step 3: Implement the proof validator**

`validate_au250_hybrid_proof.py` accepts a proof directory and requires:

```python
REQUIRED_SUMMARY_KEYS = {
    "model_sha256", "xclbin_sha256", "libnvcuda_sha256", "runner_sha256",
    "exit_code", "generated_token_ids", "prompt_tokens_per_second",
    "generation_tokens_per_second", "gpu_name", "fpga_bdf",
}
```

Parse `routes.jsonl` and require at least one `backend=xrt` IQ1_S route, at least one GPU attention route, and no selected IQ1_S record with backend `gpu`, `fallback`, or `cxl`. Parse `xrt.jsonl` and require every selected request completed once, every STALL nonzero, every used CU named in the four-CU table, and raw bounds within `[-4096,4096]`. Require nonempty integer token IDs, exit code zero, firewall GOOD, no fatal text, temperature below 85 C, and all four hashes present.

- [ ] **Step 4: Implement the strict hybrid runner**

The shell wrapper creates `.proof/kimi-au250-<UTC timestamp>/`, writes that
exact relative path to ignored `target/au250-last-proof-path`, calls
`tools/build_au250_kimi_runtime.sh`, then passes
`HETGPU_AU250_PROOF_DIR=/work/<that relative path>` into one deterministic
prompt inside `tools/au250_hybrid_run.sh` with:

```bash
export LD_PRELOAD=/work/target/au250-app215/debug/libnvcuda.so
export LD_LIBRARY_PATH=/work/target/au250-bitnet-cuda130/3rdparty/llama.cpp/src:/work/target/au250-bitnet-cuda130/3rdparty/llama.cpp/ggml/src:/work/target/au250-bitnet-cuda130/3rdparty/llama.cpp/ggml/src/ggml-cuda:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}
export HETGPU_TMATMUL_BACKEND=xrt
export HETGPU_BITNET_DISAGGREGATE=1
export HETGPU_BITNET_DISAGG_STRICT=1
export HETGPU_TMATMUL_HARDWARE_MATMUL=1
export HETGPU_XRT_XCLBIN=/au250_xrt/example/MaxCores_370M.xclbin
export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
export HETGPU_XRT_TIMEOUT_MS=10000
export HETGPU_BITNET_ROUTE_LOG=${HETGPU_AU250_PROOF_DIR}/routes.jsonl
export HETGPU_XRT_EXECUTION_LOG=${HETGPU_AU250_PROOF_DIR}/xrt.jsonl
unset HETGPU_CXL_TMATMUL HETGPU_TMATMUL_CXL HETGPU_TMATMUL_MATRIX_STAGE HETGPU_TMATMUL_IO_STAGE
```

Run `/work/target/au250-bitnet-cuda130/bin/llama-cli` on `/models/kimi/moonshotai_Kimi-K2.6-IQ1_S-00001-of-00006.gguf` with `--seed 42 --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --no-display-prompt --simple-io --no-warmup -c 512 -n 1 -ngl 99`. Tokenize the generated stdout with `/work/target/au250-bitnet-cuda130/bin/llama-tokenize` and the same model, reject an empty/noninteger token list, then capture stdout, token IDs, stderr, route/XRT JSONL, `nvidia-smi`, `xbutil` platform/firewall/error/thermal reports, hashes, and llama timing into `summary.json`.

- [ ] **Step 5: Run validator unit tests and shell syntax checks**

Run:

```bash
python3 -m pytest -q zluda/tests/test_validate_au250_hybrid_proof.py
bash -n tools/run_kimi_k26_iq1s_au250_hybrid.sh
```

Expected: all validator tests PASS and shell syntax is valid.

- [ ] **Step 6: Commit the proof harness**

```bash
git add tools/run_kimi_k26_iq1s_au250_hybrid.sh zluda/tests/validate_au250_hybrid_proof.py zluda/tests/test_validate_au250_hybrid_proof.py
git commit -m "test: add strict Kimi AU250 hybrid proof"
```

### Task 9: Run captured-layer qualification, E2E inference, and benchmark

**Files:**
- Create: `docs/kimi_k26_au250_hybrid_proof.txt`
- Generated only: `.proof/kimi-au250-*/`

- [ ] **Step 1: Run the app215-compatible build**

Run: `tools/build_au250_kimi_runtime.sh`

Expected: `llama-cli --version`, `nvidia-smi -L`, and `xbutil examine` all succeed inside the same container; the build outputs are under ignored `target/` paths.

- [ ] **Step 2: Run one captured IQ1_S launch in compare mode**

Set `HETGPU_XRT_COMPARE_MAX_LAUNCHES=1` so the first complete real Kimi XRT
launch emits its output hash and the count of raw grid/delta components checked
by `reconstruct_from_raw`. Run the deterministic one-token command and require
that captured-layer record before accepting the full-run proof.

Run:

```bash
HETGPU_XRT_COMPARE_MAX_LAUNCHES=1 N_PREDICT=1 \
  tools/run_kimi_k26_iq1s_au250_hybrid.sh
```

Expected: one complete captured launch comparison PASS, no GPU BitLinear fallback record, and healthy FPGA reports.

- [ ] **Step 3: Run strict one-token E2E replacement**

Run:

```bash
HETGPU_XRT_COMPARE_MAX_LAUNCHES=0 N_PREDICT=1 \
  tools/run_kimi_k26_iq1s_au250_hybrid.sh
```

Expected: process exit 0, a valid generated token ID, GPU attention route evidence, physical XRT completion evidence for every selected IQ1_S launch, and proof-validator status `pass`.

- [ ] **Step 4: Run the benchmark only after E2E passes**

Run:

```bash
N_PREDICT=8 HETGPU_AU250_BENCHMARK=1 \
  tools/run_kimi_k26_iq1s_au250_hybrid.sh
```

Expected: a validated summary containing prompt/generation tok/s, per-CU submission counts, XRT staging/execution/reconstruction time, NVIDIA utilization snapshot, AU250 power/temperature, and artifact hashes. A timeout or invalid token makes the benchmark fail rather than report partial TPS.

- [ ] **Step 5: Write the proof record**

Create `docs/kimi_k26_au250_hybrid_proof.txt` containing the exact commit, branch, UTC timestamp, model/xclbin/runner/lib hashes, commands, generated proof-directory path, token IDs/text, route counts, per-CU completions, timing, GPU identity, AU250 health, and explicit statement that attention was GPU-native while selected IQ1_S BitLinear executed on XRT.

- [ ] **Step 6: Run final regression and proof validation**

Run:

```bash
cargo test -p zluda --features nvidia --no-default-features
python3 -m pytest -q zluda/tests/test_validate_au250_hybrid_proof.py
bash zluda/tests/test_au250_hybrid_runtime_static.sh
python3 zluda/tests/validate_au250_hybrid_proof.py "$(cat target/au250-last-proof-path)"
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 7: Commit the qualified proof record**

```bash
git add docs/kimi_k26_au250_hybrid_proof.txt
git commit -m "docs: record Kimi AU250 hybrid qualification"
```

## Self-Review Checklist

- Spec coverage: backend selection, GPU attention, exact IQ1_S decomposition, D=1024 tiling, lane multiplexing, all four CUs, four-BO persistence, strict error handling, same-process GPU/AU250 runtime, live tiled proof, captured launch, E2E evidence, and benchmark all map to Tasks 1-9.
- Compatibility: existing CXL route strings and one-shot `submit_xrt_tmatmul` remain available; XRT dispatch occurs before CXL-only environment checks.
- Numerical contract: raw AU250 dots use direct signed Q8 values, remain within signed i16, are shifted only when passed to the existing Q8.8 validator, and accumulate in deterministic group order.
- Lifetime contract: xclbin is loaded once, BO addresses stay stable, program BOs are assembled once, failures quiesce launched CUs, and uncertain addresses are retained rather than reused.
- Proof contract: build, route logs, and STALL alone are insufficient; tokens, physical completions, health, and hashes are mandatory before TPS.
