# Qwen3.5 TQ1_0 AU250 Hybrid Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify a fail-closed Qwen3.5-397B-A17B runtime in which CUDA executes attention and all non-routed-expert work while the existing four-BO AU250 backend executes eligible TQ1_0 routed-expert `mul_mat_id` operations, then compare it fairly with CUDA-only execution.

**Architecture:** Apply a repository-owned bridge overlay to pinned current llama.cpp revision `925e1179947ea0c0ebfb0032df18af3a729822be`. The overlay registers GGUF-backed TQ1_0 expert tensors and offers eligible CUDA `GGML_OP_MUL_MAT_ID` nodes to two versioned Rust C-ABI functions; Rust reads verified weight ranges, reproduces upstream Q8_K quantization, submits assembler-produced 128-bit programs through the existing persistent four-CU/four-BO XRT pool, reconstructs f32 results, and copies complete outputs back to CUDA. A dedicated runner holds one model process per mode, performs correctness gates before timing, captures machine-readable evidence, and rejects results unless token equality, 100% eligible-route coverage, all-CU activity, device health, and complete timings pass.

**Tech Stack:** Rust 2021, C ABI, current llama.cpp/CMake/CUDA 13, GGUF TQ1_0 and Q8_K, Xilinx XRT on Alveo U250, repository tmatmul assembler, Bash, Python 3 standard library, pytest, Docker `app215`.

---

## Repository boundaries and fixed interfaces

The implementation must preserve the existing Kimi/IQ1_S route and all unrelated dirty-tree changes. New Qwen code lives beside the existing modules and scripts. No existing model file is deleted or rewritten.

The versioned bridge types are defined once in `tools/qwen35-tq1-bridge.h` and mirrored with `#[repr(C)]` in Rust:

```c
#define HETGPU_TQ1_ABI_VERSION 1u
#define HETGPU_TQ1_NOT_HANDLED 0
#define HETGPU_TQ1_HANDLED 1
#define HETGPU_TQ1_ERROR (-1)

enum hetgpu_tq1_role_v1 {
    HETGPU_TQ1_ROLE_GATE_EXPS = 1,
    HETGPU_TQ1_ROLE_UP_EXPS = 2,
    HETGPU_TQ1_ROLE_DOWN_EXPS = 3,
    HETGPU_TQ1_ROLE_GATE_UP_EXPS = 4,
};

struct hetgpu_tq1_tensor_v1 {
    uint32_t abi_version;
    uint32_t ggml_type;
    uint32_t role;
    uint32_t file_index;
    const char * name;
    const char * path;
    uint64_t file_offset;
    uint64_t nbytes;
    int64_t ne[4];
    uint64_t nb[4];
};

struct hetgpu_tq1_mul_mat_id_v1 {
    uint32_t abi_version;
    uint32_t src0_type;
    uint32_t src1_type;
    uint32_t ids_type;
    uint32_t dst_type;
    uint32_t reserved;
    const char * src0_name;
    const void * src1_device;
    const void * ids_device;
    void * dst_device;
    void * cuda_stream;
    int64_t src0_ne[4];
    uint64_t src0_nb[4];
    int64_t src1_ne[4];
    uint64_t src1_nb[4];
    int64_t ids_ne[4];
    uint64_t ids_nb[4];
    int64_t dst_ne[4];
    uint64_t dst_nb[4];
};

typedef int (*hetgpu_tq1_register_tensor_v1_fn)(const struct hetgpu_tq1_tensor_v1 *);
typedef int (*hetgpu_tq1_try_mul_mat_id_v1_fn)(const struct hetgpu_tq1_mul_mat_id_v1 *);
```

The Rust route uses these fixed return meanings: `0` means the node is outside the enabled qualified boundary, `1` means the complete destination was installed, and `-1` means the process must fail. Once strict mode classifies a node as eligible, returning `0` is itself an error.

The physical executor remains unchanged at the kernel boundary:

```text
kernel argument 0 = matrix BO
kernel argument 1 = input BO
kernel argument 2 = output BO
kernel argument 3 = program BO
```

The program is assembled through the existing assembler from:

```text
ldv v0, PARAM_INPUT
tmatmul_import v0
tmatmul_go PARAM_MATRIX
tmatmul_export v1
sv v1, PARAM_OUTPUT
stall
```

## File map

- Create `tools/fetch_qwen35_tq1_model.sh`: fixed-model resumable download, capacity check, size/hash verification, atomic publication.
- Create `zluda/tests/test_fetch_qwen35_tq1_model.sh`: isolated model-fetch contract tests with command shims.
- Modify `zluda/src/impl/xrt_tmatmul.rs`: own the one process-global persistent pool used by every XRT quantization route.
- Modify `zluda/src/impl/iq1s_xrt.rs`: consume the shared pool instead of declaring an IQ1_S-only static.
- Create `zluda/src/impl/tq1_tmatmul.rs`: tensor registry, GGUF range reader, exact TQ1_0 decode, Q8_K quantization, cache identity, and CPU reference reconstruction.
- Create `zluda/src/impl/tq1_xrt.rs`: D=1024 tile/lane planner, two-bit packing, XRT job creation, completion validation, deterministic reconstruction, evidence.
- Create `zluda/src/impl/tq1_bridge.rs`: versioned C ABI, CUDA copies, strict route classification, shared executor invocation, atomic output publication.
- Modify `zluda/src/impl/mod.rs`: enable the three Qwen modules only for the supported Unix NVIDIA build.
- Modify `zluda/src/lib.rs`: export the two bridge entry points.
- Create `tools/qwen35-tq1-bridge.h`: the single C/C++ ABI declaration.
- Create `tools/llama-qwen35-tq1-hetgpu.patch`: pinned llama.cpp loader registration and CUDA dispatch hook.
- Create `tools/prepare_au250_qwen35_source.sh`: revision-checked, idempotent pristine-source overlay preparation.
- Create `zluda/tests/test_prepare_au250_qwen35_source.sh`: synthetic-source and real-pinned-source patch qualification.
- Create `tools/au250_qwen35_run.sh`: Qwen-specific AU250/GPU container wrapper.
- Create `tools/build_au250_qwen35_runtime.sh`: pinned source, overlay, CUDA llama.cpp, and Rust shim build.
- Create `zluda/tests/test_au250_qwen35_runtime_static.sh`: static contract and shell syntax test.
- Create `zluda/tests/tq1_upstream_reference.cpp`: executable oracle linked to the pinned llama.cpp quant routines.
- Create `zluda/tests/run_au250_xrt_tq1.sh`: live single-tile and tiled numerical qualification.
- Create `tools/qwen35_au250_eval.py`: server lifecycle, prompt construction, semantic and timed A/B requests, telemetry capture.
- Create `zluda/tests/test_qwen35_au250_eval.py`: fake-server orchestration and report-rendering tests.
- Create `tools/run_qwen35_tq1_au250_hybrid.sh`: end-to-end preflight and proof-directory orchestration.
- Create `zluda/tests/validate_qwen35_tq1_au250_proof.py`: fail-closed proof validator.
- Create `zluda/tests/test_validate_qwen35_tq1_au250_proof.py`: validator acceptance and rejection cases.
- Create `zluda/evaluation/fixtures/qwen35_prompt_seed.txt`: deterministic source text from which the exact 256-token timed prompt is derived.
- Create `zluda/evaluation/2026-08-26-qwen35-tq1-au250-evaluation.md`: final evidence-backed A/B report, written only after live validation.

### Task 1: Add fixed-model acquisition and verification

**Files:**
- Create: `tools/fetch_qwen35_tq1_model.sh`
- Test: `zluda/tests/test_fetch_qwen35_tq1_model.sh`

- [ ] **Step 1: Write the failing shell contract test**

The test creates a fake `curl` and `sha256sum` in a temporary `PATH`, verifies that insufficient space fails before download, verifies that a successful download is first written as `.partial`, and verifies that the production constants occur literally in the script:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
script="${repo_root}/tools/fetch_qwen35_tq1_model.sh"
grep -Fq 'Qwen3.5-397B-A17B-UD-TQ1_0.gguf' "${script}"
grep -Fq '94155830880' "${script}"
grep -Fq '0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568' "${script}"
bash -n "${script}"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
mkdir -p "${work}/bin" "${work}/model"
cat >"${work}/bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
out=""
while (($#)); do
    if [[ "$1" == "--output" ]]; then out="$2"; shift 2; else shift; fi
done
printf 'fixture' >"${out}"
SH
chmod +x "${work}/bin/curl"
cat >"${work}/bin/sha256sum" <<'SH'
#!/usr/bin/env bash
printf '%s  %s\n' "${HETGPU_TEST_SHA}" "$1"
SH
chmod +x "${work}/bin/sha256sum"
HETGPU_MODEL_FETCH_TESTING=1 HETGPU_TEST_MODEL_SIZE=7 \
HETGPU_TEST_SHA=0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568 \
PATH="${work}/bin:${PATH}" "${script}" "${work}/model"
test -f "${work}/model/Qwen3.5-397B-A17B-UD-TQ1_0.gguf"
test ! -e "${work}/model/Qwen3.5-397B-A17B-UD-TQ1_0.gguf.partial"
```

- [ ] **Step 2: Run the test and observe the missing-script failure**

Run: `bash zluda/tests/test_fetch_qwen35_tq1_model.sh`

Expected: exit nonzero because `tools/fetch_qwen35_tq1_model.sh` does not exist.

- [ ] **Step 3: Implement the fetch script**

Use fixed production constants and allow the reduced byte count only when the explicit test guard is set:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo='nohurry/Qwen3.5-397B-A17B-TQ1_0-GGUF'
name='Qwen3.5-397B-A17B-UD-TQ1_0.gguf'
url="https://huggingface.co/${repo}/resolve/main/${name}?download=true"
expected_size=94155830880
expected_sha='0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568'
destination="${1:-/root/models/qwen35-tq1}"
if [[ "${HETGPU_MODEL_FETCH_TESTING:-0}" == 1 ]]; then
    expected_size="${HETGPU_TEST_MODEL_SIZE:?}"
fi
mkdir -p "${destination}"
final="${destination}/${name}"
partial="${final}.partial"
available="$(df -PB1 "${destination}" | awk 'NR==2 {print $4}')"
current=0
[[ ! -e "${partial}" ]] || current="$(stat -c %s "${partial}")"
remaining=$((expected_size - current))
(( remaining >= 0 && available >= remaining + 1073741824 )) || {
    echo "insufficient free bytes for verified Qwen model" >&2
    exit 1
}
if [[ -f "${final}" ]] && [[ "$(stat -c %s "${final}")" == "${expected_size}" ]] \
   && [[ "$(sha256sum "${final}" | awk '{print $1}')" == "${expected_sha}" ]]; then
    exit 0
fi
curl --fail --location --retry 8 --retry-all-errors --continue-at - \
    --output "${partial}" "${url}"
[[ "$(stat -c %s "${partial}")" == "${expected_size}" ]] || {
    echo "Qwen model size mismatch" >&2; exit 1;
}
[[ "$(sha256sum "${partial}" | awk '{print $1}')" == "${expected_sha}" ]] || {
    echo "Qwen model SHA-256 mismatch" >&2; exit 1;
}
mv -f -- "${partial}" "${final}"
```

Before `mkdir -p`, reject a destination that already exists as a non-directory:

```bash
[[ ! -e "${destination}" || -d "${destination}" ]] || {
    echo "model destination exists and is not a directory: ${destination}" >&2
    exit 1
}
```

After the atomic rename, print only verified, non-secret artifact fields:

```bash
printf 'verified_model=%s\nverified_size=%s\nverified_sha256=%s\n' \
    "${final}" "${expected_size}" "${expected_sha}"
```

- [ ] **Step 4: Run the contract test**

Run: `bash zluda/tests/test_fetch_qwen35_tq1_model.sh`

Expected: exit 0 and no output containing `mismatch`.

- [ ] **Step 5: Commit the acquisition boundary**

```bash
git add tools/fetch_qwen35_tq1_model.sh zluda/tests/test_fetch_qwen35_tq1_model.sh
git commit -m "feat: add verified Qwen TQ1 model fetch"
```

### Task 2: Centralize the persistent four-CU XRT pool

**Files:**
- Modify: `zluda/src/impl/xrt_tmatmul.rs:724-744`
- Modify: `zluda/src/impl/iq1s_xrt.rs:742-757`

- [ ] **Step 1: Add failing shared-pool unit tests**

Under `xrt_tmatmul.rs` tests, add a test-only initialization counter and assert that two calls use one initialized pool and that a poisoned initialization error is returned unchanged:

```rust
#[test]
fn persistent_pool_initializes_once_and_serializes_callers() {
    let state = OnceLock::new();
    let calls = AtomicUsize::new(0);
    let first = with_pool_state(
        &state,
        || { calls.fetch_add(1, Ordering::SeqCst); Ok(FakePool::new()) },
        |pool| { pool.calls += 1; Ok(pool.calls) },
    ).unwrap();
    let second = with_pool_state(
        &state,
        || { calls.fetch_add(1, Ordering::SeqCst); Ok(FakePool::new()) },
        |pool| { pool.calls += 1; Ok(pool.calls) },
    ).unwrap();
    assert_eq!((first, second, calls.load(Ordering::SeqCst)), (1, 2, 1));
}
```

The production-facing test calls `with_persistent_pool` with an unset `HETGPU_XRT_XCLBIN` twice and asserts both errors contain the same required-variable text without a second open attempt.

- [ ] **Step 2: Verify the tests fail because the shared accessor is absent**

Run: `cargo test -p zluda --no-default-features --features nvidia xrt_tmatmul::tests::persistent_pool -- --nocapture`

Expected: compilation fails with `cannot find function with_persistent_pool`.

- [ ] **Step 3: Add the shared accessor and move ownership into `xrt_tmatmul.rs`**

Add the process-global state next to `XrtTmatmulPool`:

```rust
static PERSISTENT_POOL: OnceLock<Mutex<Result<XrtTmatmulPool, String>>> = OnceLock::new();

fn with_pool_state<P, T>(
    state: &OnceLock<Mutex<Result<P, String>>>,
    initialize: impl FnOnce() -> Result<P, String>,
    operation: impl FnOnce(&mut P) -> Result<T, String>,
) -> Result<T, String> {
    let state = state.get_or_init(|| Mutex::new(initialize()));
    let mut guard = state
        .lock()
        .map_err(|_| "AU250 XRT pool mutex poisoned".to_string())?;
    match &mut *guard {
        Ok(pool) => operation(pool),
        Err(error) => Err(error.clone()),
    }
}

pub(crate) fn with_persistent_pool<T>(
    operation: impl FnOnce(&mut XrtTmatmulPool) -> Result<T, String>,
) -> Result<T, String> {
    with_pool_state(
        &PERSISTENT_POOL,
        || XrtTmatmulPool::open_from_env().map_err(|error| error.to_string()),
        operation,
    )
}
```

Keep `XrtTmatmulPool::open_from_env`, `lane_capacities`, and `run_wave` private to the module family. Do not change `XrtWaveJob`, `XrtWaveCompletion`, assembler invocation, BO allocation, BO order, or CU configuration parsing.

- [ ] **Step 4: Make IQ1_S consume the shared accessor**

Replace the IQ1_S-local `OnceLock<Mutex<...>>` in `execute_captured` with:

```rust
pub(crate) fn execute_captured(captured: &CapturedLaunch) -> Result<XrtIq1sResult, String> {
    let result = super::xrt_tmatmul::with_persistent_pool(|pool| {
        execute_captured_with(captured, pool)
    })?;
    append_execution_log_from_env(&result.evidence)?;
    Ok(result)
}
```

Remove only imports made unused by this replacement.

- [ ] **Step 5: Run XRT and IQ1_S regression tests**

Run: `cargo test -p zluda --no-default-features --features nvidia xrt_tmatmul iq1s_xrt -- --nocapture`

Expected: all selected tests pass, including four-BO/order and poisoned-pool cases.

- [ ] **Step 6: Commit the shared executor ownership**

```bash
git add zluda/src/impl/xrt_tmatmul.rs zluda/src/impl/iq1s_xrt.rs
git commit -m "refactor: share persistent AU250 XRT pool"
```

### Task 3: Implement exact TQ1_0 and Q8_K numerical primitives

**Files:**
- Create: `zluda/src/impl/tq1_tmatmul.rs`
- Modify: `zluda/src/impl/iq1s_tmatmul.rs:1913`
- Modify: `zluda/src/impl/mod.rs`

- [ ] **Step 1: Add decoder, quantizer, and reference-dot tests**

The new module tests must cover all 256 positions, not only payload prefixes:

```rust
#[test]
fn tq1_decode_covers_payload_tail_and_scale() {
    let mut bytes = [0u8; TQ1_BLOCK_BYTES];
    for (index, byte) in bytes[..48].iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
    }
    bytes[48..52].copy_from_slice(&[3, 81, 127, 242]);
    bytes[52..54].copy_from_slice(&0x3800u16.to_le_bytes());
    let block = Tq1Block::decode(&bytes).unwrap();
    assert_eq!(block.scale, 0.5);
    assert_eq!(block.trits.len(), 256);
    assert!(block.trits.iter().all(|value| (-1..=1).contains(value)));
    assert_eq!(block.trits, upstream_decode_fixture(&bytes));
}

#[test]
fn q8_k_matches_upstream_rounding_and_zero_block() {
    let values: Vec<f32> = (0..256)
        .map(|index| ((index as i32 - 127) as f32) / 31.0)
        .collect();
    let quantized = Q8KBlock::quantize(&values).unwrap();
    assert_eq!(quantized.qs, upstream_q8_fixture(&values));
    assert!(quantized.scale.is_finite());
    assert_eq!(Q8KBlock::quantize(&[0.0; 256]).unwrap().scale, 0.0);
}

#[test]
fn tq1_q8_dot_uses_per_block_scales() {
    let weights = fixture_tq1_blocks(4);
    let activations = fixture_f32_values(1024);
    let got = reference_dot(&weights, &activations).unwrap();
    let expected = upstream_tq1_q8_dot_fixture(&weights, &activations);
    assert!((got - expected).abs() <= 1.0e-6);
}
```

Add rejection cases for 53/55-byte blocks, NaN/Infinity activations, non-multiple-of-256 K, overflowing file ranges, invalid UTF-8 names, and non-finite FP16-derived scales.

- [ ] **Step 2: Verify the numerical tests fail**

Run: `cargo test -p zluda --no-default-features --features nvidia tq1_tmatmul -- --nocapture`

Expected: compilation fails because `tq1_tmatmul` and its types do not exist.

- [ ] **Step 3: Expose the existing FP16 decoder for reuse**

Change only the visibility of the existing helper:

```rust
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let f32_bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = fraction.leading_zeros() - 22;
            let normalized = (fraction << (leading + 1)) & 0x03ff;
            sign | ((127 - 15 - leading) << 23) | (normalized << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent + (127 - 15)) << 23) | (fraction << 13),
    };
    f32::from_bits(f32_bits)
}
```

- [ ] **Step 4: Add exact TQ1_0 decoding**

Define the fixed layout and use the upstream ordering exactly:

```rust
pub(crate) const TQ1_VALUES: usize = 256;
pub(crate) const TQ1_BLOCK_BYTES: usize = 54;
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tq1Block {
    pub(crate) trits: [i8; TQ1_VALUES],
    pub(crate) scale: f32,
}

fn decode_digit(byte: u8, power: u8) -> i8 {
    let q = byte.wrapping_mul(power);
    ((((q as u16) * 3) >> 8) as i16 - 1) as i8
}

impl Tq1Block {
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != TQ1_BLOCK_BYTES {
            return Err(format!("TQ1_0 block has {} bytes, expected 54", bytes.len()));
        }
        let mut trits = [0i8; TQ1_VALUES];
        let mut out = 0;
        for j in (0..32).step_by(32) {
            for n in 0..5 {
                for m in 0..32 {
                    trits[out] = decode_digit(bytes[j + m], POW3[n]);
                    out += 1;
                }
            }
        }
        for j in (32..48).step_by(16) {
            for n in 0..5 {
                for m in 0..16 {
                    trits[out] = decode_digit(bytes[j + m], POW3[n]);
                    out += 1;
                }
            }
        }
        for n in 0..4 {
            for j in 0..4 {
                trits[out] = decode_digit(bytes[48 + j], POW3[n]);
                out += 1;
            }
        }
        if out != TQ1_VALUES {
            return Err(format!("TQ1_0 decoded {out} values, expected 256"));
        }
        let scale = super::iq1s_tmatmul::half_to_f32(u16::from_le_bytes([bytes[52], bytes[53]]));
        if !scale.is_finite() {
            return Err("TQ1_0 scale is not finite".to_string());
        }
        Ok(Self { trits, scale })
    }
}
```

- [ ] **Step 5: Add upstream-compatible Q8_K quantization**

Implement the exact sign choice and float bit rounding, with 256 signed quants and one f32 scale:

```rust
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Q8KBlock {
    pub(crate) qs: [i8; 256],
    pub(crate) scale: f32,
}

fn nearest_int(value: f32) -> Result<i32, String> {
    if !value.is_finite() || value.abs() > 4_194_303.0 {
        return Err("Q8_K rounding input is outside the upstream bound".to_string());
    }
    let bits = (value + 12_582_912.0).to_bits() as i32;
    Ok((bits & 0x007f_ffff) - 0x0040_0000)
}

impl Q8KBlock {
    pub(crate) fn quantize(values: &[f32]) -> Result<Self, String> {
        if values.len() != 256 || values.iter().any(|value| !value.is_finite()) {
            return Err("Q8_K requires 256 finite f32 values".to_string());
        }
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &value in values {
            if value.abs() > amax { amax = value.abs(); max = value; }
        }
        if amax == 0.0 {
            return Ok(Self { qs: [0; 256], scale: 0.0 });
        }
        let iscale = -127.0 / max;
        let mut qs = [0i8; 256];
        for (dst, &value) in qs.iter_mut().zip(values) {
            *dst = nearest_int(iscale * value)?.min(127) as i8;
        }
        Ok(Self { qs, scale: 1.0 / iscale })
    }
}
```

The reference dot iterates TQ1 blocks in increasing K order and accumulates `(integer_dot as f32) * weight.scale * activation.scale` into f32.

- [ ] **Step 6: Run numerical tests and all existing IQ1_S numerical tests**

Run: `cargo test -p zluda --no-default-features --features nvidia tq1_tmatmul iq1s_tmatmul -- --nocapture`

Expected: all selected tests pass.

- [ ] **Step 7: Commit numerical primitives**

```bash
git add zluda/src/impl/tq1_tmatmul.rs zluda/src/impl/iq1s_tmatmul.rs zluda/src/impl/mod.rs
git commit -m "feat: add exact TQ1 Q8K numerical primitives"
```

### Task 4: Add the validated GGUF expert-tensor registry and bounded reader

**Files:**
- Modify: `zluda/src/impl/tq1_tmatmul.rs`

- [ ] **Step 1: Write registry and file-range tests**

Use a temporary file containing two expert slices and assert identity, strides, bounds, and duplicate behavior:

```rust
#[test]
fn registry_reads_only_the_registered_expert_range() {
    let file = NamedTempFile::new().unwrap();
    write_fixture_gguf_payload(file.path(), 4096, 2);
    let registration = registration_for(file.path(), "blk.0.ffn_down_exps.weight", [1024, 4096, 2, 1]);
    let source = Tq1TensorSource::register(registration).unwrap();
    let first = source.read_row_blocks(0, 0, 0, 4).unwrap();
    let second = source.read_row_blocks(1, 0, 0, 4).unwrap();
    assert_ne!(first, second);
    assert_eq!(first.len(), 4 * TQ1_BLOCK_BYTES);
}

#[test]
fn registry_rejects_changed_metadata_for_an_existing_name() {
    let registry = TensorRegistry::default();
    registry.register(registration("blk.1.ffn_up_exps.weight", 100, 540)).unwrap();
    let error = registry.register(registration("blk.1.ffn_up_exps.weight", 101, 540)).unwrap_err();
    assert!(error.contains("conflicting TQ1_0 registration"));
}
```

Also assert that an identical duplicate is idempotent; a symlink path is canonicalized; the opened file is regular and read-only; `offset + nbytes` cannot overflow or exceed file size; `ne[3] == 1`; K is a multiple of 256; `nb[0] == 54`; expert stride contains all rows; names match exactly `blk.<decimal>.ffn_{gate,up,down,gate_up}_exps.weight`; and roles agree with names.

- [ ] **Step 2: Verify registry tests fail**

Run: `cargo test -p zluda --no-default-features --features nvidia tq1_tmatmul::tests::registry -- --nocapture`

Expected: compilation fails because `TensorRegistry` and `Tq1TensorSource` are absent.

- [ ] **Step 3: Implement immutable registrations and positional reads**

Use these central types and `FileExt::read_exact_at` so reads never mutate a shared seek cursor:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExpertRole { Gate, Up, Down, GateUp }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TensorIdentity {
    pub(crate) canonical_path: PathBuf,
    pub(crate) file_offset: u64,
    pub(crate) nbytes: u64,
    pub(crate) name: String,
    pub(crate) ne: [u64; 4],
    pub(crate) nb: [u64; 4],
    pub(crate) role: ExpertRole,
    pub(crate) device: u64,
    pub(crate) inode: u64,
    pub(crate) modified_ns: u128,
}

pub(crate) struct Tq1TensorSource {
    pub(crate) identity: TensorIdentity,
    file: File,
}

impl Tq1TensorSource {
    pub(crate) fn read_exact(&self, relative: u64, output: &mut [u8]) -> Result<(), String> {
        let end = relative.checked_add(output.len() as u64)
            .ok_or_else(|| "TQ1_0 relative file range overflow".to_string())?;
        if end > self.identity.nbytes {
            return Err("TQ1_0 read exceeds registered tensor span".to_string());
        }
        self.file.read_exact_at(output, self.identity.file_offset + relative)
            .map_err(|error| format!("read TQ1_0 tensor {}: {error}", self.identity.name))
    }
}
```

Store sources in `OnceLock<RwLock<HashMap<String, Arc<Tq1TensorSource>>>>`. Never replace an existing name with different metadata. Derive each expert/row/block byte address with checked arithmetic from the registered `nb[]`; do not assume expert-major order independently of the metadata.

- [ ] **Step 4: Run the complete TQ1 module test set**

Run: `cargo test -p zluda --no-default-features --features nvidia tq1_tmatmul -- --nocapture`

Expected: all TQ1 tests pass, including malformed range and duplicate rejection.

- [ ] **Step 5: Commit the registry**

```bash
git add zluda/src/impl/tq1_tmatmul.rs
git commit -m "feat: register bounded GGUF TQ1 expert tensors"
```

### Task 5: Implement D=1024 TQ1 tile planning and deterministic XRT reconstruction

**Files:**
- Create: `zluda/src/impl/tq1_xrt.rs`
- Modify: `zluda/src/impl/mod.rs`
- Modify: `zluda/src/impl/xrt_tmatmul.rs:523-538,907-1026`

- [ ] **Step 1: Write planner, packer, and reconstruction tests with a fake executor**

Define a fake executor that returns physical completions in reverse order. Required assertions are:

```rust
#[test]
fn nine_lane_cu_gets_one_job_and_six_lane_cu_gets_six_plus_two() {
    let tile = logical_tile_fixture(1024, 1024, 1);
    let nine = plan_tile_jobs(&tile, &[9]).unwrap();
    assert_eq!(nine.iter().map(|job| job.assignments.len()).collect::<Vec<_>>(), vec![8]);
    let six = plan_tile_jobs(&tile, &[6]).unwrap();
    assert_eq!(six.iter().map(|job| job.assignments.len()).collect::<Vec<_>>(), vec![6, 2]);
}

#[test]
fn reversed_four_cu_completions_reconstruct_in_logical_order() {
    let mut fake = FakeTq1WaveExecutor::new(vec![9, 9, 9, 6]).reversed();
    let result = execute_mul_mat_id_with(&fixture_operation(), &mut fake).unwrap();
    assert_eq!(result.outputs, fixture_reference_outputs());
    assert_eq!(result.evidence.per_cu_submissions.len(), 4);
    assert!(result.evidence.per_cu_submissions.iter().all(|count| *count > 0));
}
```

Add tests for two-bit codes (`-1 -> 3`, `0 -> 0`, `1 -> 1`), four trits per byte, zero padding, 262,144-byte matrix tiles, each lane nonzero in only one 128-value half-group, unused lane zeros, request ID uniqueness, wrong-CU completion, duplicated/missing completion, bad output length, invalid STALL, nonzero padding, raw values outside `[-16384, 16384]`, four block scales per K tile, multi-row/multi-K tiling, matrix-cache identity/LRU eviction, nonzero dispatch-to-STALL nanoseconds, and checked clock-normalized cycle calculation.

- [ ] **Step 2: Verify the new module tests fail**

Run: `cargo test -p zluda --no-default-features --features nvidia tq1_xrt -- --nocapture`

Expected: compilation fails because `tq1_xrt` does not exist.

- [ ] **Step 3: Define the planner and evidence interfaces**

```rust
pub(crate) const AU250_DIM: usize = 1024;
pub(crate) const HALF_GROUP: usize = 128;
pub(crate) const TQ1_BLOCKS_PER_K_TILE: usize = 4;
pub(crate) const HALF_GROUPS_PER_K_TILE: usize = 8;
pub(crate) const AU250_MATRIX_BYTES: usize = 262_144;
pub(crate) const RAW_DOT_MIN: i32 = -16_384;
pub(crate) const RAW_DOT_MAX: i32 = 16_384;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Tq1LogicalGroup {
    pub(crate) token: usize,
    pub(crate) expert_slot: usize,
    pub(crate) expert: usize,
    pub(crate) row_tile: usize,
    pub(crate) k_tile: usize,
    pub(crate) block: usize,
    pub(crate) half: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tq1LaneAssignment {
    pub(crate) lane: usize,
    pub(crate) group: Tq1LogicalGroup,
}

pub(crate) trait Tq1WaveExecutor {
    fn lane_capacities(&self) -> Vec<usize>;
    fn run_wave(&mut self, jobs: Vec<XrtWaveJob>) -> Result<Vec<XrtWaveCompletion>, String>;
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct Tq1XrtEvidence {
    pub(crate) backend: &'static str,
    pub(crate) eligible_operations: u64,
    pub(crate) handled_operations: u64,
    pub(crate) submission_count: u64,
    pub(crate) completion_count: u64,
    pub(crate) per_cu_submissions: Vec<u64>,
    pub(crate) per_cu_completions: Vec<u64>,
    pub(crate) stall_codes: Vec<u32>,
    pub(crate) raw_min: i32,
    pub(crate) raw_max: i32,
    pub(crate) matrix_bytes: u64,
    pub(crate) input_bytes: u64,
    pub(crate) output_bytes: u64,
    pub(crate) program_bytes: u64,
    pub(crate) dispatch_to_stall_ns: u64,
    pub(crate) clock_hz: u64,
    pub(crate) derived_accelerator_cycles: u64,
    pub(crate) decode_ns: u64,
    pub(crate) pack_ns: u64,
    pub(crate) xrt_ns: u64,
    pub(crate) reconstruct_ns: u64,
}
```

Implement `Tq1WaveExecutor for XrtTmatmulPool` using the existing `lane_capacities` and `run_wave` methods.

Extend `XrtWaveCompletion` with `dispatch_to_stall_ns` and `program_bytes`. In `Pool::run_wave`, capture `Instant::now()` immediately before the per-CU program DMA/start-register sequence and capture elapsed time on the first nonzero STALL read. Tests inject a monotonic-clock trait so fake completions have deterministic elapsed values. This is a host-observed interval, not a hardware cycle counter.

For strict TQ1 execution, require `HETGPU_XRT_CLOCK_HZ`. The build/runner obtains it from `xclbinutil --info`, stores the raw tool output, and passes the parsed positive integer. Calculate a clearly labelled clock-normalized value with checked u128 arithmetic:

```rust
fn derived_cycles(elapsed_ns: u64, clock_hz: u64) -> Result<u64, String> {
    if elapsed_ns == 0 || clock_hz == 0 {
        return Err("dispatch-to-STALL time and XRT clock must be positive".to_string());
    }
    let numerator = u128::from(elapsed_ns)
        .checked_mul(u128::from(clock_hz))
        .ok_or_else(|| "derived accelerator cycle multiplication overflow".to_string())?;
    u64::try_from((numerator + 999_999_999) / 1_000_000_000)
        .map_err(|_| "derived accelerator cycle count does not fit u64".to_string())
}
```

Evidence and the final report must call this field `derived_accelerator_cycles` and separately retain `dispatch_to_stall_ns`, `clock_hz`, and the xclbin metadata. Do not describe it as a hardware counter.

- [ ] **Step 4: Implement matrix and lane packing**

Pack a matrix byte from four trits with:

```rust
fn trit_code(trit: i8) -> Result<u8, String> {
    match trit { -1 => Ok(0b11), 0 => Ok(0b00), 1 => Ok(0b01), value => Err(format!("invalid trit {value}")) }
}

fn pack_four(trits: [i8; 4]) -> Result<u8, String> {
    Ok(trit_code(trits[0])?
        | (trit_code(trits[1])? << 2)
        | (trit_code(trits[2])? << 4)
        | (trit_code(trits[3])? << 6))
}
```

Each physical matrix contains the full ternary 1024x1024 tile. Each assigned lane input contains Q8_K signed quants only at `[block * 256 + half * 128, +128)` and zeros elsewhere. Convert each signed quant to little-endian i16 for the existing vector ABI. A nine-lane target receives eight assignments in one job. A six-lane target receives deterministic assignment slices `[0..6]` and `[6..8]` with distinct request IDs.

- [ ] **Step 5: Implement scheduling, validation, cache, and reconstruction**

Schedule logical work round-robin over all configured CUs, then submit one wave per set of free CUs. Validate returned request ownership with a `HashMap<u64, PlannedJob>` and reject any duplicate or absent ID before mutating accumulation storage.

Reconstruct only after all eight half-groups exist:

```rust
let block_integer_dot = i32::from(raw_half_0) + i32::from(raw_half_1);
let contribution = block_integer_dot as f32 * weight_scale * activation_scale;
if !contribution.is_finite() {
    return Err("TQ1_0 reconstruction produced a non-finite value".to_string());
}
output[logical_output] += contribution;
```

Use deterministic loop order: token, expert slot, row tile, output row, K tile, TQ1 block, half. Cache packed matrix tiles with a byte-bounded LRU keyed by canonical model identity, tensor identity, file generation, expert, row tile, K tile, and content hash. Read `HETGPU_TQ1_MATRIX_CACHE_BYTES`, defaulting to 1 GiB, and reject zero or non-numeric capacities.

- [ ] **Step 6: Run planner and existing executor tests**

Run: `cargo test -p zluda --no-default-features --features nvidia tq1_xrt xrt_tmatmul iq1s_xrt -- --nocapture`

Expected: all selected tests pass and the fake reordered-completion test reports nonzero use of all four configured CUs.

- [ ] **Step 7: Commit the TQ1 XRT adapter**

```bash
git add zluda/src/impl/tq1_xrt.rs zluda/src/impl/mod.rs
git commit -m "feat: execute TQ1 expert tiles on AU250"
```

### Task 6: Export the strict Rust C-ABI bridge and CUDA copy boundary

**Files:**
- Create: `zluda/src/impl/tq1_bridge.rs`
- Modify: `zluda/src/impl/mod.rs`
- Modify: `zluda/src/lib.rs`

- [ ] **Step 1: Write ABI layout, classification, and atomic-copy tests**

Tests must construct raw C records and assert:

```rust
#[test]
fn abi_v1_layout_and_return_codes_are_stable() {
    assert_eq!(HETGPU_TQ1_ABI_VERSION, 1);
    assert_eq!(HETGPU_TQ1_NOT_HANDLED, 0);
    assert_eq!(HETGPU_TQ1_HANDLED, 1);
    assert_eq!(HETGPU_TQ1_ERROR, -1);
    assert_eq!(align_of::<HetgpuTq1MulMatIdV1>(), align_of::<u64>());
}

#[test]
fn strict_eligible_failure_never_becomes_not_handled() {
    let _env = crate::r#impl::test_env::lock();
    std::env::set_var("HETGPU_QWEN_TQ1_XRT", "1");
    std::env::set_var("HETGPU_QWEN_TQ1_STRICT", "1");
    let operation = eligible_operation_fixture();
    let result = try_mul_mat_id_with(&operation, &FailingCudaCopies, &FailingExecutor);
    assert_eq!(result, HETGPU_TQ1_ERROR);
}
```

Fake CUDA copies must prove the destination remains unchanged when activation copy, ID copy, XRT execution, reconstruction, or final copy fails. Add non-handled tests for disabled env and unqualified names, and error tests for wrong ABI/type/shape/stride, invalid or duplicate expert IDs, null pointers, unknown registration, and output-size overflow.

- [ ] **Step 2: Verify bridge tests fail**

Run: `cargo test -p zluda --no-default-features --features nvidia tq1_bridge -- --nocapture`

Expected: compilation fails because the bridge module and symbols are absent.

- [ ] **Step 3: Define mirrored ABI records and the CUDA copier**

Mirror every field from the header with `#[repr(C)]`, fixed-width integers, and four-element arrays. Never expose Rust enums across the ABI. Define:

```rust
trait CudaCopies {
    unsafe fn synchronize(&self, stream: *mut c_void) -> Result<(), String>;
    unsafe fn device_to_host(&self, dst: &mut [u8], src: *const c_void) -> Result<(), String>;
    unsafe fn host_to_device(&self, dst: *mut c_void, src: &[u8]) -> Result<(), String>;
}
```

The production implementation calls `cuStreamSynchronize_ckpt`, `cuMemcpyDtoH_v2`, and `cuMemcpyHtoD_v2`, checks every return code, and copies the full f32 destination only after validation completes. Synchronize before D2H so the input/ID producers are complete and after H2D before returning handled.

- [ ] **Step 4: Implement exact `mul_mat_id` semantics and strict classification**

Validate the GGML shape contract:

```text
src0 = [K, N, n_experts, 1], TQ1_0
src1 = [K, activation_channels, n_tokens, 1], F32
ids  = [n_expert_used, n_tokens, 1, 1], I32
dst  = [N, n_expert_used, n_tokens, 1], F32
src0.K == src1.K
ids.n_tokens == src1.n_tokens
ids.n_expert_used % src1.activation_channels == 0
```

For each `(token, expert_slot)`, use `ids[token, expert_slot]` as the source expert and `src1[token, expert_slot % activation_channels]` as the activation vector, matching upstream `ggml_mul_mat_id`. Reject duplicate expert IDs within one token. Qualify only registered Qwen names with type numbers `TQ1_0=34`, `F32=0`, and `I32=26`; also verify those numeric constants in the pinned overlay at build time rather than trusting the hard-coded values alone.

- [ ] **Step 5: Export no-unwind entry points**

In `lib.rs`, add C entry points that catch panics and map them to `-1`:

```rust
#[no_mangle]
pub unsafe extern "C" fn hetgpu_tq1_register_tensor_v1(
    tensor: *const crate::r#impl::tq1_bridge::HetgpuTq1TensorV1,
) -> i32 {
    std::panic::catch_unwind(|| crate::r#impl::tq1_bridge::register_tensor(tensor))
        .unwrap_or(HETGPU_TQ1_ERROR)
}

#[no_mangle]
pub unsafe extern "C" fn hetgpu_tq1_try_mul_mat_id_v1(
    operation: *const crate::r#impl::tq1_bridge::HetgpuTq1MulMatIdV1,
) -> i32 {
    std::panic::catch_unwind(|| crate::r#impl::tq1_bridge::try_mul_mat_id(operation))
        .unwrap_or(HETGPU_TQ1_ERROR)
}
```

Use a process-global counters/evidence writer. `HETGPU_TQ1_EVIDENCE_LOG` is required and must be writable in strict mode. Append one JSON object per operation with route, dimensions, IDs hash, timing breakdown, per-CU work, byte counts, STALL codes, and error text. Serialize writes behind a mutex and call `sync_data` before returning handled.

- [ ] **Step 6: Run bridge tests and a release build**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia tq1_bridge -- --nocapture
cargo build -p zluda --release --no-default-features --features nvidia,embed_cudart,evaluation
nm -D target/release/libnvcuda.so | grep -E 'hetgpu_tq1_(register_tensor|try_mul_mat_id)_v1'
```

Expected: bridge tests pass and `nm` prints exactly both exported symbols.

- [ ] **Step 7: Commit the bridge**

```bash
git add zluda/src/impl/tq1_bridge.rs zluda/src/impl/mod.rs zluda/src/lib.rs
git commit -m "feat: export strict Qwen TQ1 AU250 bridge"
```

### Task 7: Create the pinned llama.cpp bridge overlay

**Files:**
- Create: `tools/qwen35-tq1-bridge.h`
- Create: `tools/llama-qwen35-tq1-hetgpu.patch`
- Create: `tools/prepare_au250_qwen35_source.sh`
- Test: `zluda/tests/test_prepare_au250_qwen35_source.sh`

- [ ] **Step 1: Write the overlay preparation test**

The test creates a temporary git repository from the exact patch-target files in a caller-supplied source fixture and invokes the script with `HETGPU_OVERLAY_TESTING=1` plus `HETGPU_TEST_LLAMA_REVISION=$(git rev-parse HEAD)`. Production execution rejects both variables unless the testing guard is exactly `1`, and the script still contains and defaults to the literal pinned revision. Check these applied markers:

```bash
grep -Fq 'hetgpu_tq1_register_tensor_v1' "${overlay}/src/llama-model-loader.cpp"
grep -Fq 'hetgpu_tq1_try_mul_mat_id_v1' "${overlay}/ggml/src/ggml-cuda/ggml-cuda.cu"
grep -Fq 'const std::string & path() const' "${overlay}/src/llama-mmap.h"
grep -Fq 'HETGPU_TQ1_ABI_VERSION' "${overlay}/tools/qwen35-tq1-bridge.h"
grep -Fq "925e1179947ea0c0ebfb0032df18af3a729822be" "${repo_root}/tools/prepare_au250_qwen35_source.sh"
test "$(git -C "${source}" rev-parse HEAD)" = "${HETGPU_TEST_LLAMA_REVISION}"
```

Run the preparation twice and assert the second run changes no file or mtime. Assert that a source at a different revision and a nonempty unqualified destination both fail.

- [ ] **Step 2: Verify the overlay test fails**

Run: `bash zluda/tests/test_prepare_au250_qwen35_source.sh`

Expected: exit nonzero because the preparation script and patch do not exist.

- [ ] **Step 3: Add the shared ABI header**

Write exactly the ABI declarations from the repository-boundaries section, with `#pragma once`, `<stdint.h>`, and `extern "C"` guards. The header must contain no llama.cpp or CUDA headers and must compile as both C11 and C++17.

- [ ] **Step 4: Patch loader registration**

The patch adds `llama_file::path() const` returning its stored canonical filename, resolves `hetgpu_tq1_register_tensor_v1` with `dlsym(RTLD_DEFAULT, ...)`, and registers only matching TQ1_0 expert weights after GGUF offsets are validated. It passes `weights_map` metadata, `files[w.idx]->path()`, `w.offs`, `ggml_nbytes`, `ne`, and `nb`. Registration `-1` throws a model-load error; absence of the symbol is allowed only when hybrid mode is not requested.

Name parsing accepts exactly:

```cpp
static uint32_t hetgpu_tq1_role(const char * name) {
    if (std::regex_match(name, std::regex("blk\\.[0-9]+\\.ffn_gate_exps\\.weight"))) return HETGPU_TQ1_ROLE_GATE_EXPS;
    if (std::regex_match(name, std::regex("blk\\.[0-9]+\\.ffn_up_exps\\.weight"))) return HETGPU_TQ1_ROLE_UP_EXPS;
    if (std::regex_match(name, std::regex("blk\\.[0-9]+\\.ffn_down_exps\\.weight"))) return HETGPU_TQ1_ROLE_DOWN_EXPS;
    if (std::regex_match(name, std::regex("blk\\.[0-9]+\\.ffn_gate_up_exps\\.weight"))) return HETGPU_TQ1_ROLE_GATE_UP_EXPS;
    return 0;
}
```

- [ ] **Step 5: Patch CUDA `mul_mat_id` dispatch**

At the beginning of `ggml_cuda_mul_mat_id`, resolve the try function once. When `HETGPU_QWEN_TQ1_XRT=1` and the node has a qualified TQ1_0 expert name, populate the ABI struct from `src0`, `src1`, `ids`, `dst`, and `ctx.stream()`. Handle return values as:

```cpp
const int route = hook(&operation);
if (route == HETGPU_TQ1_HANDLED) return;
if (route == HETGPU_TQ1_ERROR) GGML_ABORT("strict HetGPU TQ1 route failed");
if (getenv("HETGPU_QWEN_TQ1_STRICT") && strcmp(getenv("HETGPU_QWEN_TQ1_STRICT"), "1") == 0) {
    GGML_ABORT("eligible TQ1 mul_mat_id returned not-handled in strict mode");
}
```

When routing is disabled, execute the untouched native CUDA body. Add `static_assert(GGML_TYPE_TQ1_0 == 34)`, `static_assert(GGML_TYPE_F32 == 0)`, and `static_assert(GGML_TYPE_I32 == 26)` adjacent to bridge record creation. Disable CUDA graph capture when strict routing is enabled because the host/XRT bridge synchronizes the CUDA stream.

- [ ] **Step 6: Implement safe, pinned overlay preparation**

`prepare_au250_qwen35_source.sh` takes `<pristine-source> <overlay-destination>`, verifies the source is a clean git checkout at the pinned revision, rejects `/`, identical source/destination, and nonempty unqualified destinations, copies with `tar` excluding `.git` and build directories, installs the ABI header as `tools/qwen35-tq1-bridge.h`, applies the patch with `patch --batch --forward -p1`, and verifies all markers. A marker file records source revision and patch SHA-256; an identical marker makes subsequent calls exit successfully without rewriting.

- [ ] **Step 7: Run synthetic and real-source overlay tests**

Run:

```bash
bash zluda/tests/test_prepare_au250_qwen35_source.sh
work="$(mktemp -d)"
tools/prepare_au250_qwen35_source.sh /tmp/llama.cpp-qwen-context "${work}/overlay"
cmake -S "${work}/overlay" -B "${work}/overlay/build-smoke" -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "${work}/overlay/build-smoke" --target llama-cli -j2
```

Expected: preparation is idempotent and the CPU smoke build finishes with target `llama-cli` built.

- [ ] **Step 8: Commit the overlay**

```bash
git add tools/qwen35-tq1-bridge.h tools/llama-qwen35-tq1-hetgpu.patch tools/prepare_au250_qwen35_source.sh zluda/tests/test_prepare_au250_qwen35_source.sh
git commit -m "feat: bridge pinned llama.cpp TQ1 experts to HetGPU"
```

### Task 8: Add the Qwen-specific container and build workflow

**Files:**
- Create: `tools/au250_qwen35_run.sh`
- Create: `tools/build_au250_qwen35_runtime.sh`
- Create: `zluda/tests/test_au250_qwen35_runtime_static.sh`

- [ ] **Step 1: Write static workflow tests**

The test runs `bash -n` and asserts the wrapper mounts the Qwen model read-only at `/models/qwen`, mounts pristine llama.cpp read-only at `/llama-pristine`, uses a separate writable overlay/build root, exposes all AU250 devices through the known environment helper, and includes CUDA 13. It asserts the build script checks the pinned revision, builds `llama-server` and `llama-cli`, builds the Rust `nvidia,embed_cudart,evaluation` feature set, and records SHA-256 hashes.

- [ ] **Step 2: Verify static tests fail**

Run: `bash zluda/tests/test_au250_qwen35_runtime_static.sh`

Expected: exit nonzero because the Qwen workflow scripts are absent.

- [ ] **Step 3: Implement the Qwen container wrapper**

Use the existing temperature guard and `_au250_devflags` pattern from `tools/au250_hybrid_run.sh`, but define independent defaults:

```bash
qwen_model_root="${AU250_QWEN_MODEL_ROOT:-/root/models/qwen35-tq1}"
llama_source_root="${AU250_QWEN_LLAMA_ROOT:-/tmp/llama.cpp-qwen-context}"
cuda_root="${AU250_CUDA_ROOT:-/usr/local/cuda-13.0}"
```

Mount the repository at `/work`, model at `/models/qwen:ro`, source at `/llama-pristine:ro`, CUDA at `/usr/local/cuda-13.0:ro`, XRT at `/au250_xrt:ro`, and the caller-selected `AU250_QWEN_BUILD_ROOT` at `/qwen-build`. Preserve the `--print-docker` mode and quote every host path.

- [ ] **Step 4: Implement the reproducible build**

The build script verifies the source revision, prepares `/qwen-build/llama-overlay`, configures `/qwen-build/llama-build` with:

```bash
cmake -S /qwen-build/llama-overlay -B /qwen-build/llama-build \
  -DGGML_CUDA=ON -DGGML_CUDA_F16=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /qwen-build/llama-build --target llama-server llama-cli -j"$(nproc)"
cargo build -p zluda --release --no-default-features \
  --features nvidia,embed_cudart,evaluation
```

Write `/qwen-build/manifest.json` with pinned llama revision, overlay patch hash, HetGPU commit and dirty manifest hash, compiler versions, binary hashes, library hash, and build command. Fail if either bridge symbol is missing from `libnvcuda.so`.

- [ ] **Step 5: Run static tests and build in the AU250 container**

Run:

```bash
bash zluda/tests/test_au250_qwen35_runtime_static.sh
AU250_QWEN_BUILD_ROOT=/root/qwen35-au250-build tools/au250_qwen35_run.sh \
  /work/tools/build_au250_qwen35_runtime.sh
```

Expected: static tests pass; build exits 0; manifest names the pinned revision; both `llama-server` and `libnvcuda.so` hashes are nonempty.

- [ ] **Step 6: Commit the build workflow**

```bash
git add tools/au250_qwen35_run.sh tools/build_au250_qwen35_runtime.sh zluda/tests/test_au250_qwen35_runtime_static.sh
git commit -m "build: add Qwen TQ1 AU250 runtime workflow"
```

### Task 9: Add an upstream-linked numerical oracle and live XRT gates

**Files:**
- Create: `zluda/tests/tq1_upstream_reference.cpp`
- Create: `zluda/tests/run_au250_xrt_tq1.sh`
- Modify: `zluda/src/lib.rs`

- [ ] **Step 1: Add an evaluation-only Rust fixture entry point and failing tests**

Export an evaluation-only function accepting raw 54-byte blocks, f32 activations, dimensions, and f32 output. Tests reject null pointers, nonmultiples of 256, and undersized buffers. The function calls the same TQ1 adapter and shared XRT pool used by inference; it does not contain a second XRT implementation.

- [ ] **Step 2: Add the upstream oracle executable**

`tq1_upstream_reference.cpp` reads a binary fixture header followed by TQ1_0 blocks and f32 activations, calls `quantize_row_q8_K_ref` and `ggml_vec_dot_tq1_0_q8_K`, and writes f32 outputs plus Q8_K scales/quants. Compile it against the pinned overlay sources:

```bash
c++ -O2 -std=c++17 \
  -I/qwen-build/llama-overlay/ggml/include \
  -I/qwen-build/llama-overlay/ggml/src \
  -I/qwen-build/llama-overlay/ggml/src/ggml-cpu \
  /work/zluda/tests/tq1_upstream_reference.cpp \
  /qwen-build/llama-build/ggml/src/libggml.a \
  /qwen-build/llama-build/ggml/src/libggml-base.a \
  /qwen-build/llama-build/ggml/src/ggml-cpu/libggml-cpu.a \
  -lpthread -ldl -lm -o /qwen-build/tq1_upstream_reference
```

- [ ] **Step 3: Implement deterministic live fixtures and fail-closed comparisons**

The live script generates two fixtures with a fixed xorshift64 seed:

```text
single tile: N=1024, K=1024, tokens=1, experts used=1
tiled: N=2048, K=2048, tokens=4, experts used=10
```

It runs the upstream oracle and the Rust evaluation entry point, then requires every element to satisfy:

```text
absolute_error <= 1e-4 + 1e-5 * abs(reference)
```

It also requires exact Q8_K quants/scales, all four CUs with nonzero submissions and completions in the tiled case, only terminal STALL codes, raw bounds within `[-16384, 16384]`, zero padding, and exact matrix/input/output/program byte counts. Store fixtures, references, outputs, and JSON evidence beneath the supplied proof directory.

- [ ] **Step 4: Run Rust tests, single-tile hardware gate, and tiled hardware gate**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation tq1 -- --nocapture
tools/au250_qwen35_run.sh /work/zluda/tests/run_au250_xrt_tq1.sh \
  /qwen-build/tq1_upstream_reference /qwen-build/tq1-live-proof
```

Expected: exit 0; evidence reports `single_tile.status="pass"`, `tiled.status="pass"`, maximum error within tolerance, and four positive per-CU completion counts.

- [ ] **Step 5: Commit numerical qualification**

```bash
git add zluda/tests/tq1_upstream_reference.cpp zluda/tests/run_au250_xrt_tq1.sh zluda/src/lib.rs
git commit -m "test: qualify TQ1 tiles against upstream on AU250"
```

### Task 10: Build the fail-closed proof validator

**Files:**
- Create: `zluda/tests/validate_qwen35_tq1_au250_proof.py`
- Create: `zluda/tests/test_validate_qwen35_tq1_au250_proof.py`

- [ ] **Step 1: Write acceptance and one-fault-at-a-time rejection fixtures**

The test creates a minimal valid proof directory and parametrically mutates one field. Required rejection cases are: model size/hash mismatch, llama revision mismatch, binary mismatch between modes, any CPU layer placement, prompt token count other than 256, generated token mismatch, semantic result other than exact `OK`, eligible count zero, handled count different from eligible count, fallback/error count nonzero, any CU with zero work, missing/duplicate request ID, invalid STALL, raw bound violation, numerical gate failure, process exit nonzero, device-health regression, fewer than five measurements, missing timing field, and non-finite/negative timing.

Core assertions should read:

```python
assert cuda["binary_sha256"] == hybrid["binary_sha256"]
assert cuda["model_sha256"] == EXPECTED_MODEL_SHA256
assert cuda["generated_token_ids"] == hybrid["generated_token_ids"]
assert hybrid["routes"]["eligible"] == hybrid["routes"]["handled"]
assert hybrid["routes"]["fallback"] == 0
assert hybrid["routes"]["error"] == 0
assert all(count > 0 for count in hybrid["xrt"]["per_cu_completions"])
assert len(cuda["measurements"]) == len(hybrid["measurements"]) == 5
```

- [ ] **Step 2: Verify validator tests fail**

Run: `python3 -m pytest -q zluda/tests/test_validate_qwen35_tq1_au250_proof.py`

Expected: import failure because the validator is absent.

- [ ] **Step 3: Implement strict schema validation and metric summaries**

Use only Python standard library in the validator. Reject unknown schema versions and missing keys. Calculate min, max, median, population standard deviation, and coefficient of variation for model load, prompt tokens/s, TTFT, generation tokens/s, and end-to-end latency. Emit one normalized JSON object only after all gates pass:

```json
{
  "schema_version": 1,
  "status": "pass",
  "modes": {
    "cuda": {"measurements": 5},
    "hybrid": {"measurements": 5}
  },
  "token_ids_match": true,
  "eligible_route_coverage": 1.0,
  "all_cus_active": true
}
```

On failure, print `QWEN_TQ1_PROOF_INVALID: <specific reason>` to stderr and return nonzero. Never print throughput summaries for an invalid proof.

- [ ] **Step 4: Run the validator tests**

Run: `python3 -m pytest -q zluda/tests/test_validate_qwen35_tq1_au250_proof.py`

Expected: all acceptance and rejection tests pass.

- [ ] **Step 5: Commit the proof validator**

```bash
git add zluda/tests/validate_qwen35_tq1_au250_proof.py zluda/tests/test_validate_qwen35_tq1_au250_proof.py
git commit -m "test: validate Qwen TQ1 hybrid proof fail closed"
```

### Task 11: Implement semantic and timed A/B orchestration

**Files:**
- Create: `zluda/evaluation/fixtures/qwen35_prompt_seed.txt`
- Create: `tools/qwen35_au250_eval.py`
- Create: `tools/run_qwen35_tq1_au250_hybrid.sh`
- Create: `zluda/tests/test_qwen35_au250_eval.py`
- Modify: `zluda/tests/test_au250_qwen35_runtime_static.sh`

- [ ] **Step 1: Add runner contract tests**

Extend the static test to assert the runner uses one identical `llama-server` path and hash for both modes, `--ctx-size 512`, `--n-gpu-layers 999`, greedy sampling, semantic generation before timing, one warm-up, five measurements, 32 generated tokens, and validation as the final command. Add a Python fake-server test that returns deterministic `/tokenize`, `/detokenize`, `/health`, and `/completion` responses and verifies request equality across modes.

- [ ] **Step 2: Verify runner tests fail**

Run:

```bash
bash zluda/tests/test_au250_qwen35_runtime_static.sh
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
```

Expected: failures identify the absent runner and evaluator.

- [ ] **Step 3: Add the fixed prompt seed and exact 256-token construction**

The seed file contains this text repeated until it is comfortably above 256 tokens:

```text
Explain how a heterogeneous inference runtime preserves numerical correctness while assigning attention to a GPU and ternary expert matrix multiplication to an FPGA. Discuss tensor shapes, quantization scales, deterministic scheduling, error handling, and evidence required for a fair performance comparison. Keep the explanation technical, concrete, and internally consistent.
```

Store the paragraph above exactly eight times, separated by one newline. At evaluation start, call `/tokenize` with `add_special=false`, take the first 256 token IDs, call `/detokenize`, and retokenize the returned text with `add_special=false`; require exact equality with those 256 IDs. Store both the resulting text and IDs in the proof directory. Submit the token-ID array, not a string, to `/completion` in both modes so an implicit BOS cannot change the counted workload.

- [ ] **Step 4: Implement the Python server/evaluation controller**

For each mode, launch a fresh server while retaining it across one warm-up and five measured requests:

```python
command = [
    server, "--model", model, "--ctx-size", "512", "--n-gpu-layers", "999",
    "--threads", str(threads), "--host", "127.0.0.1", "--port", str(port),
    "--seed", "42", "--no-webui",
]
request = {
    "prompt": prompt_token_ids,
    "n_predict": 32,
    "temperature": 0.0,
    "seed": 42,
    "cache_prompt": False,
    "return_tokens": True,
    "stream": True,
    "timings_per_token": True,
    "return_progress": True,
}
```

Before timed requests, send the semantic prompt `Reply with exactly OK and no other text.` with greedy sampling. Decode and trim only surrounding whitespace, require exact `OK`, and require CUDA/hybrid token IDs equal. Record server-reported prompt-eval and eval timings, client monotonic latency, TTFT from streamed response timestamps, model load time parsed from logs, process status, and complete stdout/stderr.

Use the same allowlisted environment in both modes except:

```text
CUDA:   HETGPU_QWEN_TQ1_XRT=0, HETGPU_QWEN_TQ1_STRICT=0
Hybrid: HETGPU_QWEN_TQ1_XRT=1, HETGPU_QWEN_TQ1_STRICT=1
```

Both modes preload the same built `libnvcuda.so`. Hybrid additionally sets `HETGPU_TQ1_EVIDENCE_LOG`, `HETGPU_XRT_EXECUTION_LOG`, `HETGPU_XRT_XCLBIN`, and the verified four-CU JSON configuration. Do not change placement, threads, context, prompt, sampling, or binary between modes.

- [ ] **Step 5: Implement preflight and proof capture in the shell runner**

The shell runner requires model, build manifest, xclbin, and output directory arguments. It verifies model byte count/SHA, pinned llama revision, binary/library/xclbin hashes, at least 100 GiB free disk before acquisition, CUDA visibility, enough aggregate free GPU memory for full placement, all four expected XRT CUs, temperature below 85 C, and clean `xbutil examine` health before starting.

Run CUDA-only first, then the live numerical gate, then strict hybrid. Parse both server logs and require that all model layers were offloaded to CUDA; if full placement fails, stop without producing A/B TPS. Capture `nvidia-smi --query-compute-apps` and `xbutil examine --report platform,thermal,electrical` before and after each mode. Record the exact command, allowlisted environment, repository status and diff hash without including secret-valued variables.

- [ ] **Step 6: Run static and fake-server tests**

Run:

```bash
bash zluda/tests/test_au250_qwen35_runtime_static.sh
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
```

Expected: all tests pass; fake proof contains one warm-up plus five measured requests per mode and exact token equality.

- [ ] **Step 7: Commit the A/B runner**

```bash
git add zluda/evaluation/fixtures/qwen35_prompt_seed.txt tools/qwen35_au250_eval.py tools/run_qwen35_tq1_au250_hybrid.sh zluda/tests/test_au250_qwen35_runtime_static.sh zluda/tests/test_qwen35_au250_eval.py
git commit -m "feat: orchestrate strict Qwen CUDA AU250 evaluation"
```

### Task 12: Run the full regression and live Qwen qualification

**Files:**
- Create at runtime: `.proof/qwen35-tq1-au250-20260826T<UTC>/`

- [ ] **Step 1: Run formatting, static analysis, and unit tests**

Run:

```bash
cargo fmt --all -- --check
cargo clippy -p zluda --no-default-features --features nvidia,embed_cudart,evaluation -- -D warnings
cargo test -p zluda --no-default-features --features nvidia,evaluation -- --nocapture
bash zluda/tests/test_fetch_qwen35_tq1_model.sh
bash zluda/tests/test_prepare_au250_bitnet_source.sh
bash zluda/tests/test_prepare_au250_qwen35_source.sh
bash zluda/tests/test_au250_hybrid_runtime_static.sh
bash zluda/tests/test_au250_qwen35_runtime_static.sh
python3 -m pytest -q zluda/tests/test_validate_au250_hybrid_proof.py zluda/tests/test_validate_qwen35_tq1_au250_proof.py zluda/tests/test_qwen35_au250_eval.py
git diff --check
```

Expected: every command exits 0. Existing Kimi/IQ1_S tests remain green.

- [ ] **Step 2: Fetch and independently verify the selected checkpoint**

Run:

```bash
tools/fetch_qwen35_tq1_model.sh /root/models/qwen35-tq1
stat -c '%n %s' /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf
sha256sum /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf
```

Expected: size `94155830880` and SHA-256 `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.

- [ ] **Step 3: Build the exact runtime and run the full proof**

Run:

```bash
export AU250_QWEN_MODEL_ROOT=/root/models/qwen35-tq1
export AU250_QWEN_LLAMA_ROOT=/tmp/llama.cpp-qwen-context
export AU250_QWEN_BUILD_ROOT=/root/qwen35-au250-build
tools/au250_qwen35_run.sh /work/tools/build_au250_qwen35_runtime.sh
proof="/work/.proof/qwen35-tq1-au250-$(date -u +%Y%m%dT%H%M%SZ)"
tools/au250_qwen35_run.sh /work/tools/run_qwen35_tq1_au250_hybrid.sh \
  /models/qwen/Qwen3.5-397B-A17B-UD-TQ1_0.gguf \
  /qwen-build/llama-build/bin/llama-server \
  /au250_xrt/example/asym9_bs9_2641toks.xclbin \
  "${proof}"
python3 zluda/tests/validate_qwen35_tq1_au250_proof.py "${proof}"
```

Expected: the validator prints one JSON object with `"status":"pass"`, `"token_ids_match":true`, `"eligible_route_coverage":1.0`, and `"all_cus_active":true`. If any gate fails, retain the proof directory and report the exact failure without throughput claims.

- [ ] **Step 4: Inspect the physical and semantic evidence manually**

Run:

```bash
jq '{routes:.routes,xrt:.xrt,semantic:.semantic}' "${proof}/hybrid/result.json"
jq '[.measurements[] | {prompt_tps,generation_tps,ttft_ms,e2e_ms}]' "${proof}/cuda/result.json"
jq '[.measurements[] | {prompt_tps,generation_tps,ttft_ms,e2e_ms}]' "${proof}/hybrid/result.json"
rg -n 'fallback|ERROR|timeout|poison|non-finite|mismatch' "${proof}"
```

Expected: eligible equals handled, fallback/error are zero, all four CUs have positive validated work, both modes have identical token IDs, and the final search has no unaccounted failure line.

- [ ] **Step 5: Commit implementation only after all non-hardware and hardware gates pass**

```bash
git status --short
git log --oneline --max-count=12
```

Do not add `.proof/`, downloaded models, build directories, or generated binaries. Commit any final source-only corrections in narrowly scoped commits with their corresponding tests.

### Task 13: Write the evidence-backed evaluation report

**Files:**
- Create: `zluda/evaluation/2026-08-26-qwen35-tq1-au250-evaluation.md`
- Modify: `tools/qwen35_au250_eval.py`
- Modify: `zluda/tests/test_qwen35_au250_eval.py`

- [ ] **Step 1: Generate the report only from a passing normalized proof**

Add `render_report(normalized, proof_path)` to the Python evaluator and build all rows directly from validated numeric fields:

```python
def render_report(normalized: dict, proof_path: pathlib.Path) -> str:
    cuda = normalized["modes"]["cuda"]["summary"]
    hybrid = normalized["modes"]["hybrid"]["summary"]
    metrics = [
        ("Prompt tokens/s", "prompt_tps"),
        ("Generation tokens/s", "generation_tps"),
        ("Time to first token (ms)", "ttft_ms"),
        ("End-to-end latency (ms)", "e2e_ms"),
    ]
    rows = []
    for label, key in metrics:
        c = cuda[key]
        h = hybrid[key]
        ratio = h["median"] / c["median"]
        rows.append(
            f"| {label} | {c['median']:.6g} | {h['median']:.6g} | "
            f"{ratio:.6g} | {c['min']:.6g}-{c['max']:.6g} / "
            f"{h['min']:.6g}-{h['max']:.6g} | {c['stddev']:.6g} / {h['stddev']:.6g} |"
        )
    xrt = normalized["modes"]["hybrid"]["xrt"]
    return "\n".join([
        "# Qwen3.5-397B-A17B TQ1_0 CUDA vs AU250 Hybrid Evaluation",
        "", "## Qualification result", "",
        "- Proof status: PASS",
        "- Model: Qwen3.5-397B-A17B-UD-TQ1_0.gguf",
        "- Model SHA-256: 0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568",
        "- llama.cpp revision: 925e1179947ea0c0ebfb0032df18af3a729822be",
        "- Token IDs identical: yes",
        "- Eligible expert operations handled by AU250: 100%",
        f"- Active CUs: {sum(count > 0 for count in xrt['per_cu_completions'])}/4",
        "", "## Method", "",
        "Both modes used the same binary, fully CUDA-resident model, 512-token context, "
        "greedy sampling, exact 256-token timed prompt, and 32 generated tokens. Each "
        "mode used one warm-up and five measured requests in a fresh process while "
        "retaining the model across requests. Hybrid changed only strict TQ1_0 routed-"
        "expert `mul_mat_id` dispatch; attention, linear attention, routing, shared "
        "experts, normalization, KV/state work, and sampling remained on CUDA.",
        "", "## Performance", "",
        "| Metric | CUDA-only median | Hybrid median | Hybrid/CUDA | CUDA / hybrid min-max | CUDA / hybrid stddev |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "", "## Offload evidence", "",
        f"Per-CU completions: `{xrt['per_cu_completions']}`. "
        f"The validated proof, including transfer bytes, accelerator cycles, STALL codes, "
        f"stage timings, and pre/post device telemetry, is `{proof_path}`.",
        "", "## Limitations", "",
        "This is batch-size-one text generation on one host and one selected checkpoint. "
        "It does not measure vision, multi-user serving, speculative decoding, attention "
        "offload, or a CPU-expert baseline.", "",
    ])
```

Add a unit test that passes four known metric summaries, checks exact ratios and CU count in the rendered Markdown, and confirms that every numeric table cell parses as finite.

Add a `render-report` subcommand that reads the normalized JSON, requires `status == "pass"`, refuses an output path outside the repository `zluda/evaluation` directory, and writes through a sibling temporary file followed by `os.replace`. Generate the report with:

```bash
python3 tools/qwen35_au250_eval.py render-report \
  --normalized "${proof}/normalized.json" \
  --proof-path "${proof}" \
  --output zluda/evaluation/2026-08-26-qwen35-tq1-au250-evaluation.md
```

- [ ] **Step 2: Verify every report statement against artifacts**

Run:

```bash
python3 zluda/tests/validate_qwen35_tq1_au250_proof.py "${proof}" >"${proof}/normalized.json"
python3 tools/qwen35_au250_eval.py render-report --normalized "${proof}/normalized.json" --proof-path "${proof}" --output zluda/evaluation/2026-08-26-qwen35-tq1-au250-evaluation.md
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
git diff --check -- zluda/evaluation/2026-08-26-qwen35-tq1-au250-evaluation.md
```

Expected: proof validation exits 0, report-rendering tests pass, and `git diff --check` exits 0.

- [ ] **Step 3: Commit the report without proof binaries or model data**

Because repository Markdown is ignored, force-add only the exact report and normally add the tested renderer changes:

```bash
git add -f zluda/evaluation/2026-08-26-qwen35-tq1-au250-evaluation.md
git add tools/qwen35_au250_eval.py zluda/tests/test_qwen35_au250_eval.py
git commit -m "docs: report Qwen TQ1 AU250 hybrid evaluation"
```

## Final proof boundary

Completion requires all of the following at once:

- pinned llama.cpp and verified 94,155,830,880-byte GGUF;
- same hashed binary and full CUDA model placement in both modes;
- exact upstream-compatible TQ1_0 decode and Q8_K activation quantization;
- assembler-produced 128-bit program and unchanged matrix/input/output/program BO order;
- successful single-tile and multi-tile AU250 numerical comparisons;
- eligible equals handled, with no eligible native fallback;
- valid unique completions and nonzero work on all four CUs;
- exact CUDA/hybrid token equality and semantic `OK` gate;
- one warm-up plus five valid measured requests per mode;
- clean pre/post GPU and FPGA health evidence;
- fail-closed proof validation before any throughput result is reported.

If any item is absent, the deliverable is a retained failed-proof directory and its first exact failure, not a performance number.
