# Kimi K2.6 Lifter Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Kimi K2.6-style SASS lifter correctness benchmarks plus an optional GGUF end-to-end LD_PRELOAD capture script.

**Architecture:** Keep the existing `zluda/tests/sass_roundtrip_bench/run.sh` synthetic CUBIN-to-lifted-PTX comparator as the strict correctness gate. Add five Kimi-style PTX templates using the existing `(uint32_t *out, const uint32_t *in, uint32_t n)` ABI, and add a separate slow e2e script that validates local GGUF runner inputs before building or running the hook.

**Tech Stack:** Bash, PTX 8.7 templates, CUDA Driver API C harness, `ptxas`, Rust `zluda` NVIDIA passthrough build, `LD_PRELOAD=target/debug/libnvcuda.so`.

---

## File Structure

- Modify: `zluda/tests/sass_roundtrip_bench/run.sh`
  - Owns synthetic case registration, dry-run rows, ptxas assembly, LD_PRELOAD lifter invocation, and CSV output.
- Modify: `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh`
  - Owns shell-level regression tests for case registration, dry-run case selection, PTX template presence, e2e skip behavior, and static hook markers.
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_iq1m_unpack.ptx`
  - Exercises packed low-bit extraction, sign extension, scale-like integer arithmetic, and predicate selection.
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_rmsnorm_bits.ptx`
  - Exercises integer-to-float conversion, FMA, multiply, rsqrt approximation, and conversion back to integer output.
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_swiglu_mix.ptx`
  - Exercises gate/value mixing with f32 arithmetic, predicates, and integer reinterpretation.
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_rope_mix.ptx`
  - Exercises adjacent-index address arithmetic, paired global loads, shifts, and rotate-like mixing.
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_attention_mask.ptx`
  - Exercises attention-style predicate masking, comparisons, and selected zeroing.
- Create: `zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh`
  - Owns optional actual Kimi K2.6 IQ1_M GGUF LD_PRELOAD capture, skip status rows, runtime status rows, and e2e CSV output.
- Read-only context: `tools/run_kimi_k26_iq1m_bitnet.sh`
  - Existing wrapper invoked by the e2e script after paths are validated.
- Read-only context: `tools/download_kimi_k26_iq1m.sh`
  - Source of the six expected Kimi K2.6 IQ1_M GGUF shard names.

## Task 1: Register Synthetic Kimi Cases Test-First

**Files:**
- Modify: `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh`
- Modify: `zluda/tests/sass_roundtrip_bench/run.sh`

- [ ] **Step 1: Write the failing case-registration test**

Edit `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh` after the existing four `grep -Fxq` case checks:

```bash
grep -Fxq "kimi_iq1m_unpack" <<<"${cases}"
grep -Fxq "kimi_rmsnorm_bits" <<<"${cases}"
grep -Fxq "kimi_swiglu_mix" <<<"${cases}"
grep -Fxq "kimi_rope_mix" <<<"${cases}"
grep -Fxq "kimi_attention_mask" <<<"${cases}"
```

Edit the same file after the existing custom dry-run selection block:

```bash
kimi_work_dir="$(mktemp -d /tmp/hetgpu-roundtrip-kimi-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}"' EXIT
HETGPU_ROUNDTRIP_WORKDIR="${kimi_work_dir}" \
HETGPU_ROUNDTRIP_SM=120 \
HETGPU_ROUNDTRIP_CASES=kimi_iq1m_unpack,kimi_attention_mask \
    "${SCRIPT_DIR}/run.sh" --dry-run >/dev/null
kimi_csv="${kimi_work_dir}/bench.csv"
grep -Fq "kimi_iq1m_unpack,sm_120,dry_run" "${kimi_csv}"
grep -Fq "kimi_attention_mask,sm_120,dry_run" "${kimi_csv}"
if grep -Fq "kimi_rope_mix,sm_120,dry_run" "${kimi_csv}"; then
    echo "round-trip dry-run ignored Kimi HETGPU_ROUNDTRIP_CASES selection" >&2
    exit 1
fi
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
bash zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
```

Expected: nonzero exit before dry-run selection reaches the new Kimi rows, because `run.sh --list-cases` does not print `kimi_iq1m_unpack`.

- [ ] **Step 3: Register the Kimi cases**

Edit `zluda/tests/sass_roundtrip_bench/run.sh` and replace the current `CASES=(...)` line with:

```bash
CASES=(
    int_add
    pred_select
    fma_bits
    shared_reverse
    kimi_iq1m_unpack
    kimi_rmsnorm_bits
    kimi_swiglu_mix
    kimi_rope_mix
    kimi_attention_mask
)
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
bash zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
```

Expected: exit 0. This task only checks list and dry-run behavior, so PTX files are not needed yet.

- [ ] **Step 5: Commit**

```bash
git add zluda/tests/sass_roundtrip_bench/run.sh zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
git commit -m "test: register Kimi roundtrip cases"
```

## Task 2: Add Kimi PTX Templates Test-First

**Files:**
- Modify: `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh`
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_iq1m_unpack.ptx`
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_rmsnorm_bits.ptx`
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_swiglu_mix.ptx`
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_rope_mix.ptx`
- Create: `zluda/tests/sass_roundtrip_bench/ptx/kimi_attention_mask.ptx`

- [ ] **Step 1: Write the failing PTX-template test**

Edit `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh` after the dry-run selection tests:

```bash
for kimi_case in \
    kimi_iq1m_unpack \
    kimi_rmsnorm_bits \
    kimi_swiglu_mix \
    kimi_rope_mix \
    kimi_attention_mask
do
    ptx_file="${SCRIPT_DIR}/ptx/${kimi_case}.ptx"
    test -s "${ptx_file}"
    grep -Fq ".visible .entry ${kimi_case}(" "${ptx_file}"
    grep -Fq ".param .u64 out" "${ptx_file}"
    grep -Fq ".param .u64 in" "${ptx_file}"
    grep -Fq ".param .u32 n" "${ptx_file}"
done
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
bash zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
```

Expected: nonzero exit at `test -s .../kimi_iq1m_unpack.ptx`, because the new PTX files do not exist.

- [ ] **Step 3: Create `kimi_iq1m_unpack.ptx`**

Create `zluda/tests/sass_roundtrip_bench/ptx/kimi_iq1m_unpack.ptx`:

```ptx
.version 8.7
.target sm_120
.address_size 64

.visible .entry kimi_iq1m_unpack(
    .param .u64 out,
    .param .u64 in,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [in];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];

    and.b32 %r7, %r6, 3;
    shr.u32 %r8, %r6, 2;
    and.b32 %r8, %r8, 3;
    shr.u32 %r9, %r6, 4;
    and.b32 %r9, %r9, 3;
    shr.u32 %r10, %r6, 6;
    and.b32 %r10, %r10, 3;

    shl.b32 %r11, %r7, 30;
    shr.s32 %r11, %r11, 30;
    shl.b32 %r12, %r8, 30;
    shr.s32 %r12, %r12, 30;
    shl.b32 %r13, %r9, 30;
    shr.s32 %r13, %r13, 30;
    shl.b32 %r14, %r10, 30;
    shr.s32 %r14, %r14, 30;

    mad.lo.u32 %r15, %r11, 7, %r12;
    mad.lo.u32 %r15, %r13, 13, %r15;
    mad.lo.u32 %r15, %r14, 17, %r15;
    and.b32 %r16, %r5, 1;
    setp.eq.u32 %p2, %r16, 0;
    xor.b32 %r17, %r15, 0x9e3779b9;
    selp.u32 %r17, %r17, %r15, %p2;

    add.u64 %rd5, %rd1, %rd3;
    st.global.u32 [%rd5], %r17;

DONE:
    ret;
}
```

- [ ] **Step 4: Create `kimi_rmsnorm_bits.ptx`**

Create `zluda/tests/sass_roundtrip_bench/ptx/kimi_rmsnorm_bits.ptx`:

```ptx
.version 8.7
.target sm_120
.address_size 64

.visible .entry kimi_rmsnorm_bits(
    .param .u64 out,
    .param .u64 in,
    .param .u32 n
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<13>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [in];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];

    and.b32 %r7, %r6, 2047;
    cvt.rn.f32.u32 %f1, %r7;
    mov.f32 %f2, 0f3A83126F;
    mov.f32 %f3, 0f3F000000;
    fma.rn.f32 %f4, %f1, %f2, %f3;
    mul.rn.f32 %f5, %f4, %f4;
    mov.f32 %f6, 0f3C23D70A;
    add.rn.f32 %f7, %f5, %f6;
    rsqrt.approx.ftz.f32 %f8, %f7;
    mul.rn.f32 %f9, %f4, %f8;
    mov.f32 %f10, 0f42C80000;
    mov.f32 %f11, 0f43000000;
    fma.rn.f32 %f12, %f9, %f10, %f11;
    cvt.rzi.u32.f32 %r8, %f12;
    xor.b32 %r9, %r8, %r6;

    add.u64 %rd5, %rd1, %rd3;
    st.global.u32 [%rd5], %r9;

DONE:
    ret;
}
```

- [ ] **Step 5: Create `kimi_swiglu_mix.ptx`**

Create `zluda/tests/sass_roundtrip_bench/ptx/kimi_swiglu_mix.ptx`:

```ptx
.version 8.7
.target sm_120
.address_size 64

.visible .entry kimi_swiglu_mix(
    .param .u64 out,
    .param .u64 in,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<14>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [in];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];

    and.b32 %r7, %r6, 1023;
    shr.u32 %r8, %r6, 10;
    and.b32 %r8, %r8, 1023;
    cvt.rn.f32.u32 %f1, %r7;
    cvt.rn.f32.u32 %f2, %r8;
    mov.f32 %f3, 0f3C23D70A;
    mov.f32 %f4, 0f3F800000;
    fma.rn.f32 %f5, %f1, %f3, %f4;
    mul.rn.f32 %f6, %f2, %f5;
    mov.f32 %f7, 0f43800000;
    setp.gt.f32 %p2, %f6, %f7;
    selp.f32 %f8, %f7, %f6, %p2;
    cvt.rzi.u32.f32 %r9, %f8;
    shl.b32 %r10, %r9, 3;
    shr.u32 %r11, %r6, 21;
    xor.b32 %r12, %r10, %r11;
    add.u32 %r13, %r12, %r5;

    add.u64 %rd5, %rd1, %rd3;
    st.global.u32 [%rd5], %r13;

DONE:
    ret;
}
```

- [ ] **Step 6: Create `kimi_rope_mix.ptx`**

Create `zluda/tests/sass_roundtrip_bench/ptx/kimi_rope_mix.ptx`:

```ptx
.version 8.7
.target sm_120
.address_size 64

.visible .entry kimi_rope_mix(
    .param .u64 out,
    .param .u64 in,
    .param .u32 n
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<10>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [in];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    xor.b32 %r6, %r5, 1;
    setp.ge.u32 %p2, %r6, %r1;
    selp.u32 %r6, %r5, %r6, %p2;

    mul.wide.u32 %rd3, %r5, 4;
    mul.wide.u32 %rd4, %r6, 4;
    add.u64 %rd5, %rd2, %rd3;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.u32 %r7, [%rd5];
    ld.global.u32 %r8, [%rd6];

    and.b32 %r9, %r5, 1;
    setp.eq.u32 %p2, %r9, 0;
    shl.b32 %r10, %r7, 5;
    shr.u32 %r11, %r7, 27;
    or.b32 %r12, %r10, %r11;
    shl.b32 %r13, %r8, 7;
    shr.u32 %r14, %r8, 25;
    or.b32 %r15, %r13, %r14;
    sub.u32 %r16, %r12, %r15;
    add.u32 %r17, %r12, %r15;
    selp.u32 %r17, %r16, %r17, %p2;

    add.u64 %rd7, %rd1, %rd3;
    st.global.u32 [%rd7], %r17;

DONE:
    ret;
}
```

- [ ] **Step 7: Create `kimi_attention_mask.ptx`**

Create `zluda/tests/sass_roundtrip_bench/ptx/kimi_attention_mask.ptx`:

```ptx
.version 8.7
.target sm_120
.address_size 64

.visible .entry kimi_attention_mask(
    .param .u64 out,
    .param .u64 in,
    .param .u32 n
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [in];
    ld.param.u32 %r1, [n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r5, %r3, %r4, %r2;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];

    and.b32 %r7, %r5, 31;
    shr.u32 %r8, %r6, 11;
    and.b32 %r8, %r8, 31;
    setp.gt.u32 %p2, %r8, %r7;
    and.b32 %r9, %r6, 0x0000ffff;
    xor.b32 %r10, %r9, 0x00005a5a;
    selp.u32 %r11, 0, %r10, %p2;
    setp.eq.u32 %p3, %r7, %r8;
    add.u32 %r12, %r11, 127;
    selp.u32 %r13, %r12, %r11, %p3;
    shl.b32 %r14, %r7, 16;
    or.b32 %r15, %r13, %r14;

    add.u64 %rd5, %rd1, %rd3;
    st.global.u32 [%rd5], %r15;

DONE:
    ret;
}
```

- [ ] **Step 8: Run static tests**

Run:

```bash
bash zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
```

Expected: exit 0.

- [ ] **Step 9: Validate PTX assembly for Blackwell when ptxas is available**

Run:

```bash
tmp_dir="$(mktemp -d /tmp/hetgpu-kimi-ptxas.XXXXXX)"
for case_name in kimi_iq1m_unpack kimi_rmsnorm_bits kimi_swiglu_mix kimi_rope_mix kimi_attention_mask; do
    /usr/local/cuda-12.8/bin/ptxas -arch=sm_120 \
        "zluda/tests/sass_roundtrip_bench/ptx/${case_name}.ptx" \
        -o "${tmp_dir}/${case_name}.cubin"
done
```

Expected: each command exits 0 and writes a nonempty CUBIN under `${tmp_dir}`.

- [ ] **Step 10: Commit**

```bash
git add \
    zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh \
    zluda/tests/sass_roundtrip_bench/ptx/kimi_iq1m_unpack.ptx \
    zluda/tests/sass_roundtrip_bench/ptx/kimi_rmsnorm_bits.ptx \
    zluda/tests/sass_roundtrip_bench/ptx/kimi_swiglu_mix.ptx \
    zluda/tests/sass_roundtrip_bench/ptx/kimi_rope_mix.ptx \
    zluda/tests/sass_roundtrip_bench/ptx/kimi_attention_mask.ptx
git commit -m "test: add Kimi SASS roundtrip PTX cases"
```

## Task 3: Add Optional Kimi GGUF E2E Capture Test-First

**Files:**
- Modify: `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh`
- Create: `zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh`

- [ ] **Step 1: Write the failing e2e skip tests**

Edit `zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh` near the end, before the existing `shared_reverse` barrier check:

```bash
e2e_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${e2e_work_dir}"' EXIT
HETGPU_KIMI_E2E_WORKDIR="${e2e_work_dir}" \
BITNET_LLAMA_CLI="${e2e_work_dir}/missing-llama-cli" \
MODEL_DIR="${e2e_work_dir}/missing-model" \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null
e2e_csv="${e2e_work_dir}/bench_kimi_k26_e2e.csv"
head -n 1 "${e2e_csv}" | grep -Fxq "case,status,total_ms,exit_code,stdout_bytes,stderr_bytes,lifter_markers,lifted_ptx_files,lifted_ptx_bytes,message"
grep -Fq "kimi_k26_iq1m,skipped_missing_runner" "${e2e_csv}"

fake_runner="${e2e_work_dir}/fake-llama-cli"
printf '#!/usr/bin/env bash\nprintf "fake kimi output\\n"\n' >"${fake_runner}"
chmod +x "${fake_runner}"
e2e_model_work_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-model-test.XXXXXX)"
trap 'rm -rf "${work_dir}" "${custom_work_dir}" "${kimi_work_dir}" "${e2e_work_dir}" "${e2e_model_work_dir}"' EXIT
HETGPU_KIMI_E2E_WORKDIR="${e2e_model_work_dir}" \
BITNET_LLAMA_CLI="${fake_runner}" \
MODEL_DIR="${e2e_model_work_dir}/missing-model" \
    "${SCRIPT_DIR}/run_kimi_k26_e2e.sh" >/dev/null
e2e_model_csv="${e2e_model_work_dir}/bench_kimi_k26_e2e.csv"
grep -Fq "kimi_k26_iq1m,skipped_missing_model" "${e2e_model_csv}"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
bash zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
```

Expected: nonzero exit at `run_kimi_k26_e2e.sh`, because the e2e script does not exist.

- [ ] **Step 3: Create the e2e script**

Create `zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

CSV_HEADER="case,status,total_ms,exit_code,stdout_bytes,stderr_bytes,lifter_markers,lifted_ptx_files,lifted_ptx_bytes,message"
CASE_NAME="kimi_k26_iq1m"
CARGO="${CARGO:-cargo}"

WORK_DIR="${HETGPU_KIMI_E2E_WORKDIR:-$(mktemp -d /tmp/hetgpu-kimi-k26-e2e.XXXXXX)}"
if [[ "${HETGPU_KIMI_E2E_KEEP:-0}" != "1" && -z "${HETGPU_KIMI_E2E_WORKDIR:-}" ]]; then
    trap 'rm -rf "${WORK_DIR}"' EXIT
else
    echo "[kimi-k26-e2e] keeping work dir: ${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}/logs"

csv="${HETGPU_KIMI_E2E_CSV:-${WORK_DIR}/bench_kimi_k26_e2e.csv}"
printf '%s\n' "${CSV_HEADER}" >"${csv}"

append_row() {
    local status="$1"
    local total_ms="$2"
    local exit_code="$3"
    local stdout_bytes="$4"
    local stderr_bytes="$5"
    local lifter_markers="$6"
    local lifted_ptx_files="$7"
    local lifted_ptx_bytes="$8"
    local message="$9"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${CASE_NAME}" "${status}" "${total_ms}" "${exit_code}" \
        "${stdout_bytes}" "${stderr_bytes}" "${lifter_markers}" \
        "${lifted_ptx_files}" "${lifted_ptx_bytes}" "${message}" >>"${csv}"
}

runner="${BITNET_LLAMA_CLI:-/root/hetGPU/BitNet-work/build/bin/llama-cli}"
model_dir="${MODEL_DIR:-/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M}"

required_shards=(
    "moonshotai_Kimi-K2.6-IQ1_M-00001-of-00006.gguf"
    "moonshotai_Kimi-K2.6-IQ1_M-00002-of-00006.gguf"
    "moonshotai_Kimi-K2.6-IQ1_M-00003-of-00006.gguf"
    "moonshotai_Kimi-K2.6-IQ1_M-00004-of-00006.gguf"
    "moonshotai_Kimi-K2.6-IQ1_M-00005-of-00006.gguf"
    "moonshotai_Kimi-K2.6-IQ1_M-00006-of-00006.gguf"
)

if [[ ! -x "${runner}" ]]; then
    append_row "skipped_missing_runner" 0 0 0 0 0 0 0 "missing_runner:${runner}"
    echo "[kimi-k26-e2e] CSV: ${csv}"
    exit 0
fi

if [[ ! -d "${model_dir}" ]]; then
    append_row "skipped_missing_model" 0 0 0 0 0 0 0 "missing_model_dir:${model_dir}"
    echo "[kimi-k26-e2e] CSV: ${csv}"
    exit 0
fi

for shard in "${required_shards[@]}"; do
    if [[ ! -f "${model_dir}/${shard}" ]]; then
        append_row "skipped_missing_model" 0 0 0 0 0 0 0 "missing_shard:${shard}"
        echo "[kimi-k26-e2e] CSV: ${csv}"
        exit 0
    fi
done

echo "[kimi-k26-e2e] building libnvcuda.so with NVIDIA passthrough"
if ! "${CARGO}" build -p zluda --no-default-features --features nvidia >"${WORK_DIR}/logs/cargo-build.log" 2>&1; then
    append_row "run_failed" 0 1 0 "$(stat -c%s "${WORK_DIR}/logs/cargo-build.log")" 0 0 0 "cargo_build_failed"
    tail -n 120 "${WORK_DIR}/logs/cargo-build.log" >&2
    echo "[kimi-k26-e2e] CSV: ${csv}" >&2
    exit 1
fi

stdout_log="${WORK_DIR}/logs/kimi.stdout"
stderr_log="${WORK_DIR}/logs/kimi.stderr"
ptx_dump="${WORK_DIR}/lifted_kimi_k26.ptx"
prompt="${KIMI_PROMPT:-Say that you have started in one short sentence.}"

start_ms="$(date +%s%3N)"
set +e
env \
    LD_PRELOAD="${REPO_ROOT}/target/debug/libnvcuda.so" \
    HETGPU_SASS_LIFTER_LOG=1 \
    HETGPU_SASS_LIFTER_DUMP="${ptx_dump}" \
    BITNET_LLAMA_CLI="${runner}" \
    MODEL_DIR="${model_dir}" \
    "${REPO_ROOT}/tools/run_kimi_k26_iq1m_bitnet.sh" "${prompt}" \
    >"${stdout_log}" 2>"${stderr_log}"
exit_code="$?"
set -e
end_ms="$(date +%s%3N)"
total_ms="$((end_ms - start_ms))"

stdout_bytes="$(stat -c%s "${stdout_log}")"
stderr_bytes="$(stat -c%s "${stderr_log}")"
lifter_markers="$(grep -c "\\[hetGPU SASS\\] lifted" "${stderr_log}" || true)"
if [[ -s "${ptx_dump}" ]]; then
    lifted_ptx_files=1
    lifted_ptx_bytes="$(stat -c%s "${ptx_dump}")"
else
    lifted_ptx_files=0
    lifted_ptx_bytes=0
fi

status="pass"
message="ok"
if [[ "${exit_code}" != "0" ]]; then
    status="run_failed"
    message="runner_exit_${exit_code}"
elif [[ "${stdout_bytes}" == "0" ]]; then
    status="empty_output"
    message="empty_stdout"
elif [[ "${lifter_markers}" == "0" ]]; then
    status="missing_lifter_marker"
    message="no_lifter_marker"
fi

append_row "${status}" "${total_ms}" "${exit_code}" "${stdout_bytes}" "${stderr_bytes}" \
    "${lifter_markers}" "${lifted_ptx_files}" "${lifted_ptx_bytes}" "${message}"

echo "[kimi-k26-e2e] ${CASE_NAME}: ${status} lifter_markers=${lifter_markers} lifted_ptx_bytes=${lifted_ptx_bytes}"
echo "[kimi-k26-e2e] CSV: ${csv}"

if [[ "${status}" != "pass" && "${HETGPU_KIMI_E2E_ALLOW_FAILURES:-0}" != "1" ]]; then
    tail -n 200 "${stderr_log}" >&2
    exit 1
fi
```

Set the executable bit:

```bash
chmod +x zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```bash
bash zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
```

Expected: exit 0. The e2e checks return skip rows without building because the runner or model path is absent.

- [ ] **Step 5: Commit**

```bash
git add zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh
git commit -m "test: add Kimi GGUF e2e lifter capture"
```

## Task 4: Runtime Benchmark Verification

**Files:**
- No source edits.
- Read generated artifacts under the `WORK_DIR` printed by the scripts.

- [ ] **Step 1: Run the static harness test**

Run:

```bash
bash zluda/tests/sass_roundtrip_bench/test_roundtrip_harness.sh
```

Expected: exit 0.

- [ ] **Step 2: Run a dry-run Kimi-only CSV check**

Run:

```bash
dry_dir="$(mktemp -d /tmp/hetgpu-kimi-dry.XXXXXX)"
HETGPU_ROUNDTRIP_WORKDIR="${dry_dir}" \
HETGPU_ROUNDTRIP_SM=120 \
HETGPU_ROUNDTRIP_CASES=kimi_iq1m_unpack,kimi_rmsnorm_bits,kimi_swiglu_mix,kimi_rope_mix,kimi_attention_mask \
    zluda/tests/sass_roundtrip_bench/run.sh --dry-run
cat "${dry_dir}/bench.csv"
```

Expected: CSV header plus five `dry_run` rows, all using `sm_120`.

- [ ] **Step 3: Collect synthetic runtime results without stopping at first lifter gap**

Run:

```bash
HETGPU_ROUNDTRIP_KEEP=1 \
HETGPU_ROUNDTRIP_ALLOW_FAILURES=1 \
HETGPU_ROUNDTRIP_CASES=kimi_iq1m_unpack,kimi_rmsnorm_bits,kimi_swiglu_mix,kimi_rope_mix,kimi_attention_mask \
    zluda/tests/sass_roundtrip_bench/run.sh
```

Expected: script prints `[sass-roundtrip] CSV: <path>`. The CSV contains one row per Kimi case with status `pass`, `mismatch`, `missing_lifter_marker`, `missing_lifter_dump_marker`, `load_ptx_failed`, `run_ptx_failed`, or another existing roundtrip status from `roundtrip_runner.c`.

- [ ] **Step 4: Run the strict synthetic gate**

Run:

```bash
HETGPU_ROUNDTRIP_CASES=kimi_iq1m_unpack,kimi_rmsnorm_bits,kimi_swiglu_mix,kimi_rope_mix,kimi_attention_mask \
    zluda/tests/sass_roundtrip_bench/run.sh
```

Expected: exit 0 only when every Kimi synthetic case has status `pass`. If this exits nonzero, use the CSV from Step 3 as the lifter-correctness defect report.

- [ ] **Step 5: Verify e2e skip behavior on this host**

Run:

```bash
e2e_dir="$(mktemp -d /tmp/hetgpu-kimi-e2e-skip.XXXXXX)"
HETGPU_KIMI_E2E_WORKDIR="${e2e_dir}" \
BITNET_LLAMA_CLI="${e2e_dir}/missing-llama-cli" \
MODEL_DIR="${e2e_dir}/missing-model" \
    zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh
cat "${e2e_dir}/bench_kimi_k26_e2e.csv"
```

Expected: CSV header plus one `skipped_missing_runner` row.

- [ ] **Step 6: Run actual Kimi K2.6 e2e capture when local inputs exist**

Run:

```bash
HETGPU_KIMI_E2E_KEEP=1 \
HETGPU_KIMI_E2E_ALLOW_FAILURES=1 \
BITNET_LLAMA_CLI=/root/hetGPU/BitNet-work/build/bin/llama-cli \
MODEL_DIR=/root/hetGPU/models/bartowski/moonshotai_Kimi-K2.6-GGUF/moonshotai_Kimi-K2.6-IQ1_M \
NO_WARMUP=1 \
N_PREDICT=8 \
CTX_SIZE=1024 \
THREADS=8 \
    zluda/tests/sass_roundtrip_bench/run_kimi_k26_e2e.sh
```

Expected when the runner and all six shards exist: the command runs through `LD_PRELOAD=target/debug/libnvcuda.so`, writes a CSV row, and records nonzero `stdout_bytes`. A correctness-ready e2e run has status `pass`, nonzero `lifter_markers`, and nonzero `lifted_ptx_bytes`.

- [ ] **Step 7: Summarize benchmark evidence**

Report:

```text
synthetic_csv=<path from Step 3 or Step 4>
synthetic_pass_count=<number of Kimi rows with pass>
synthetic_failure_rows=<non-pass Kimi rows>
e2e_csv=<path from Step 5 or Step 6>
e2e_status=<status from Kimi e2e CSV row>
leftover_user_changes=zluda/Cargo.toml if it is still modified
```

Expected: the summary distinguishes harness implementation success from lifter correctness failures found by the new benchmarks.
