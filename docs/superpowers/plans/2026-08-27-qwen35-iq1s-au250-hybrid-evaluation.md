# Qwen3.5 Mixed-Quant IQ1_S AU250 Hybrid Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify and benchmark the fixed Qwen3.5-397B-A17B mixed-quant GGUF with CUDA attention/native mixed-format experts and fail-closed exact IQ1_S expert execution on all four AU250 compute units.

**Architecture:** Preserve the current llama.cpp binary and existing IQ1_S interception path. Add a standard-library GGUF contract audit, expose complete IQ1_S physical completion evidence, select only exact type-19 MMQ/MMVQ kernels through a Qwen route manifest, and adapt the existing deterministic evaluator to aggregate route/XRT evidence. A new validator and runner keep the superseded type-34 proof intact while rejecting any missing route, fallback, token mismatch, inactive CU, or model-type drift.

**Tech Stack:** Rust 2021, current pinned llama.cpp/CUDA 13, Python 3 standard library, pytest, Bash, Xilinx XRT, Alveo U250, repository tmatmul assembler and four-BO executor.

---

## File Structure

- Create `tools/qwen35_gguf_audit.py`: bounded GGUF header parser and fixed model tensor-contract validator.
- Create `tools/qwen35-iq1s-route-manifest.json`: exact type-19-to-XRT routing policy with native-CUDA default.
- Modify `zluda/src/impl/iq1s_xrt.rs`: record submission and completion ownership separately, including physical request IDs.
- Modify `tools/qwen35_au250_eval.py`: support IQ1_S route/XRT logs, audit binding, and a pre-timing semantic hardware gate.
- Create `tools/run_qwen35_iq1s_au250_hybrid.sh`: revised CUDA/IQ1_S A/B orchestration.
- Create `zluda/tests/validate_qwen35_iq1s_au250_proof.py`: fail-closed mixed-quant proof validator.
- Create `zluda/tests/test_qwen35_gguf_audit.py`: synthetic GGUF parser and contract tests.
- Modify `zluda/tests/test_qwen35_au250_eval.py`: IQ1_S evidence aggregation and semantic-gate tests.
- Modify `zluda/tests/test_au250_qwen35_runtime_static.sh`: revised runner and strict environment assertions.
- Create `zluda/tests/test_validate_qwen35_iq1s_au250_proof.py`: acceptance and mutation-rejection tests.
- Create `zluda/evaluation/2026-08-27-qwen35-iq1s-au250-evaluation.md`: generated only from a passing normalized proof.

### Task 1: Add the fixed-model GGUF tensor audit

**Files:**
- Create: `tools/qwen35_gguf_audit.py`
- Create: `zluda/tests/test_qwen35_gguf_audit.py`

- [ ] **Step 1: Write synthetic GGUF tests that fail before the parser exists**

Create a fixture writer that emits GGUF v3 metadata and tensor descriptors without payload bytes. The tests must cover the exact accepted contract and these independent failures: wrong architecture, wrong expert count, wrong expert type distribution, any type-34 tensor, any non-expert IQ1_S tensor, malformed string length, unsupported metadata value type, duplicate tensor name, and trailing descriptor truncation.

```python
EXPECTED_EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}

def test_fixed_contract_accepts_only_expected_mixed_experts(tmp_path):
    model = tmp_path / "model.gguf"
    write_gguf(
        model,
        architecture="qwen35moe",
        tensors=expert_tensors(EXPECTED_EXPERT_TYPES),
    )
    audit = load_auditor().audit_model(model, expected_size=model.stat().st_size,
                                       expected_sha256="a" * 64, verify_hash=False)
    assert audit["status"] == "pass"
    assert audit["routed_expert_count"] == 180
    assert audit["routed_expert_types"] == EXPECTED_EXPERT_TYPES
    assert audit["tq1_0_total"] == 0
    assert audit["non_expert_iq1s"] == []

@pytest.mark.parametrize("mutation", [
    "wrong_architecture", "missing_iq1s_expert", "add_tq1_tensor",
    "add_nonexpert_iq1s", "duplicate_tensor",
])
def test_fixed_contract_rejects_model_drift(tmp_path, mutation):
    model = build_mutated_fixture(tmp_path, mutation)
    with pytest.raises(AuditError):
        load_auditor().audit_model(model, expected_size=model.stat().st_size,
                                   expected_sha256="a" * 64, verify_hash=False)
```

- [ ] **Step 2: Run the audit tests and verify the intended red state**

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_gguf_audit.py
```

Expected: FAIL because `tools/qwen35_gguf_audit.py` does not exist.

- [ ] **Step 3: Implement the bounded standard-library GGUF parser**

Implement exact little-endian readers with checked file bounds. Parse GGUF metadata types 0 through 12, retain `general.architecture`, and skip other metadata recursively without allocating attacker-controlled arrays. Parse every tensor name, rank, dimensions, numeric GGML type, and offset. Map only the types required by the pinned model:

```python
GGML_TYPES = {
    0: "F32", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    21: "IQ3_S", 23: "IQ4_XS", 34: "TQ1_0", 39: "MXFP4",
}
EXPERT_NAME = re.compile(
    r"^blk\.(?:0|[1-9][0-9]*)\.ffn_(?:gate|up|down|gate_up)_exps\.weight$"
)
EXPECTED_MODEL_SIZE = 94_155_830_880
EXPECTED_MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
EXPECTED_EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}

class AuditError(ValueError):
    pass

def validate_contract(architecture, tensors):
    if architecture != "qwen35moe":
        raise AuditError(f"architecture is {architecture!r}, expected 'qwen35moe'")
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        raise AuditError("GGUF contains duplicate tensor names")
    experts = [tensor for tensor in tensors if EXPERT_NAME.fullmatch(tensor.name)]
    distribution = Counter(tensor.type_name for tensor in experts)
    non_expert_iq1s = sorted(
        tensor.name for tensor in tensors
        if tensor.type_name == "IQ1_S" and not EXPERT_NAME.fullmatch(tensor.name)
    )
    tq1_total = sum(tensor.type_name == "TQ1_0" for tensor in tensors)
    if len(experts) != 180 or dict(distribution) != EXPECTED_EXPERT_TYPES:
        raise AuditError(f"routed expert distribution is {dict(distribution)!r}")
    if tq1_total != 0:
        raise AuditError(f"model contains {tq1_total} unexpected TQ1_0 tensors")
    if non_expert_iq1s:
        raise AuditError(f"non-expert IQ1_S tensors: {non_expert_iq1s}")
    return experts, distribution, non_expert_iq1s
```

The CLI must accept `MODEL --model-verification RECORD --output OUTPUT`. It must compare path, size, device, inode, mtime, ctime, and SHA from the existing independently hashed record before auditing the header, then atomically write schema version 1 with the full all-tensor and routed-expert distributions.

- [ ] **Step 4: Run focused and live-header audit tests**

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_gguf_audit.py
python3 tools/qwen35_gguf_audit.py \
  /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf \
  --model-verification .proof/qwen35-tq1-au250-20260827T022424Z/model-verification.json \
  --output /tmp/qwen35-model-tensor-audit.json
jq '{status,routed_expert_count,routed_expert_types,tq1_0_total,non_expert_iq1s}' \
  /tmp/qwen35-model-tensor-audit.json
```

Expected: tests PASS; live audit reports `180`, `141/24/4/11`, `0`, and an empty list.

- [ ] **Step 5: Commit the audit tool and tests**

```bash
git add tools/qwen35_gguf_audit.py zluda/tests/test_qwen35_gguf_audit.py
git commit -m "test: audit Qwen mixed expert tensor contract"
```

### Task 2: Make IQ1_S XRT evidence prove physical request ownership

**Files:**
- Modify: `zluda/src/impl/iq1s_xrt.rs`

- [ ] **Step 1: Add failing evidence-accounting tests**

Extend the mock-wave tests so reordered completions must produce distinct submission and completion arrays and preserve every physical request ID:

```rust
assert_eq!(result.evidence.submission_count, 4);
assert_eq!(result.evidence.completion_count, 4);
assert_eq!(result.evidence.per_cu_submissions, vec![1, 1, 1, 1]);
assert_eq!(result.evidence.per_cu_completions, vec![1, 1, 1, 1]);
assert_eq!(result.evidence.request_ids.len(), 4);
assert_eq!(result.evidence.request_ids.iter().copied().collect::<HashSet<_>>().len(), 4);
assert_eq!(result.evidence.stall_codes.len(), result.evidence.completion_count as usize);
```

Add rejection tests for a duplicated completion request ID and a completion assigned to the wrong CU.

- [ ] **Step 2: Verify the evidence tests fail**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_xrt -- --nocapture
```

Expected: FAIL because `completion_count`, `per_cu_completions`, and `request_ids` are absent.

- [ ] **Step 3: Extend `XrtIq1sEvidence` and accounting**

Add:

```rust
pub(crate) completion_count: u64,
pub(crate) per_cu_completions: Vec<u64>,
pub(crate) request_ids: Vec<u64>,
```

Count planned jobs before `run_wave` as submissions. Validate every completion against the planned request/CU table, reject duplicates globally, then count it as a completion:

```rust
submission_count = submission_count
    .checked_add(u64::try_from(planned_wave.len()).map_err(|_| "wave size overflow")?)
    .ok_or("AU250 submission count overflow")?;
for planned in &planned_wave {
    per_cu_submissions[planned.cu_index] = per_cu_submissions[planned.cu_index]
        .checked_add(1).ok_or("AU250 per-CU submission count overflow")?;
}

if !all_request_ids.insert(completion.request_id) {
    return Err(format!("AU250 completion duplicated request id {}", completion.request_id));
}
completion_count = completion_count.checked_add(1)
    .ok_or("AU250 completion count overflow")?;
per_cu_completions[completion.cu_index] = per_cu_completions[completion.cu_index]
    .checked_add(1).ok_or("AU250 per-CU completion count overflow")?;
request_ids.push(completion.request_id);
```

Before returning, require submission/completion equality and identical per-CU arrays. Preserve Kimi JSON compatibility by only adding fields.

- [ ] **Step 4: Run IQ1_S, XRT, and Kimi validator regressions**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_xrt -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_tmatmul -- --nocapture
python3 zluda/tests/validate_au250_hybrid_proof.py \
  /home/victoryang00/hetGPU/.proof/kimi-au250-20260826T015154Z
```

Expected: Rust tests PASS; the retained Kimi proof remains valid because the validator tolerates additive fields.

- [ ] **Step 5: Commit physical evidence accounting**

```bash
git add zluda/src/impl/iq1s_xrt.rs
git commit -m "feat: record IQ1_S XRT completion ownership"
```

### Task 3: Add exact Qwen IQ1_S route selection and evidence aggregation

**Files:**
- Create: `tools/qwen35-iq1s-route-manifest.json`
- Modify: `tools/qwen35_au250_eval.py`
- Modify: `zluda/tests/test_qwen35_au250_eval.py`

- [ ] **Step 1: Write failing route and evidence tests**

Add a route manifest fixture and evaluator tests requiring only exact type-19 MMQ/MMVQ records to be eligible. Include native IQ2, MXFP4, attention, stream-fixup, malformed JSON, duplicate request ID, wrong-CU accounting, zero STALL, missing XRT operation, and XRT-without-route cases.

```python
routes = [
    route("_Z9mul_mat_qIL9ggml_type19E...", "cxl_tmatmul", "xrt"),
    route("_Z9mul_mat_qIL9ggml_type16E...", "gpu", "xrt"),
    route("flash_attn_f32", "gpu", "xrt"),
]
xrt = [xrt_record(request_ids=[9, 7], per_cu=[1, 1, 0, 0])]
parsed_routes, parsed_xrt, attention = evaluator.parse_iq1s_routing(routes, xrt)
assert parsed_routes == {"eligible": 1, "handled": 1, "fallback": 0, "error": 0}
assert parsed_xrt["submission_count"] == 2
assert parsed_xrt["completion_count"] == 2
assert attention == 1
```

Add a semantic-gate test proving the evaluator checks nonempty IQ1_S evidence and all four CUs immediately after exact `OK` and before warm-up requests.

- [ ] **Step 2: Run focused evaluator tests and verify failure**

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
```

Expected: FAIL because IQ1_S evidence mode and semantic hardware gating do not exist.

- [ ] **Step 3: Create the repository-owned route manifest**

Create exactly:

```json
{
  "version": 1,
  "default": "gpu",
  "routes": [
    {
      "match": "ggml_type19",
      "route": "cxl_tmatmul",
      "reason": "verified Qwen IQ1_S routed-expert tensor"
    }
  ]
}
```

The native-CUDA default ensures IQ2_XXS, IQ3_S, MXFP4, attention, and all other kernels are explicitly recorded as GPU work rather than rejected or ambiguously falling through.

- [ ] **Step 4: Implement IQ1_S evidence parsing in the evaluator**

Add `--evidence-kind tq1|iq1s`, `--route-evidence`, `--xrt-evidence`, and `--model-audit`. Keep `tq1` behavior unchanged. For IQ1_S:

```python
def is_iq1s_matmul(kernel):
    name = str(kernel).lower()
    return (
        "ggml_type19" in name
        and "stream_k_fixup" not in name
        and ("mul_mat_q" in name or "mul_mat_vec_q" in name)
    )

def parse_iq1s_routing(route_records, xrt_records):
    eligible = [record for record in route_records if is_iq1s_matmul(record.get("kernel"))]
    handled = [record for record in eligible if record.get("route") == "cxl_tmatmul"
               and record.get("backend") == "xrt" and record.get("xrt_enabled") is True
               and record.get("hardware_matmul_enabled") is True]
    fallback = [record for record in eligible if record.get("route") in ("gpu", "fallback")]
    errors = [record for record in eligible if record.get("route") == "reject"]
    if len(handled) != len(xrt_records):
        raise EvaluationError("IQ1_S route/XRT operation counts differ")
    # Validate and aggregate completion_count, per-CU arrays, local request IDs,
    # nonzero STALL, raw bounds [-4096, 4096], and comparison_status='pass'.
```

Namespace each operation's already validated local request IDs as `(operation_index + 1) << 32 | local_id`, rejecting local IDs outside `0..0xffffffff`. The existing physical executor numbers the first request in each captured operation as zero; the nonzero operation namespace makes the aggregate identifier unique without changing that device contract. Bind each mode record to the SHA-256 of `model-tensor-audit.json` and use schema version 2 for IQ1_S mode records.

After the semantic exact-`OK` request in hybrid mode, flush/read the evidence files and call the same parser. Require one semantic token ID, at least one eligible handled operation, zero fallback/error, and positive work on all four CUs before issuing warm-up or measured requests. Parse final evidence again after all requests.

- [ ] **Step 5: Run evaluator tests and preserve the TQ1 suite**

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
python3 -m pytest -q zluda/tests/test_validate_qwen35_tq1_au250_proof.py
bash zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: evaluator and superseded TQ1 validator tests PASS. The static test may remain red only for the not-yet-created revised runner assertions added in Task 4.

- [ ] **Step 6: Commit routing and evaluator support**

```bash
git add tools/qwen35-iq1s-route-manifest.json tools/qwen35_au250_eval.py \
  zluda/tests/test_qwen35_au250_eval.py
git commit -m "feat: evaluate exact Qwen IQ1_S XRT routes"
```

### Task 4: Add the fail-closed mixed-quant proof validator

**Files:**
- Create: `zluda/tests/validate_qwen35_iq1s_au250_proof.py`
- Create: `zluda/tests/test_validate_qwen35_iq1s_au250_proof.py`

- [ ] **Step 1: Build a complete passing fixture and mutation matrix**

Base the fixture on schema version 2 mode records. Require a bound model audit, five measurements, exact prompt/semantic/generated token equality, CUDA zero routes/XRT, hybrid 100% IQ1_S handling, all four CUs, physical request IDs, nonzero STALL, clean health, and passing TQ1 numerical qualification.

Mutation tests must independently reject: audit hash mismatch; wrong `141/24/4/11`; non-expert IQ1_S; any TQ1 tensor; wrong binary/model/revision; CPU layer placement; prompt or token mismatch; missing semantic route; fallback/reject; duplicated request ID; submission/completion mismatch; inactive CU; zero STALL; raw overflow; numerical failure; non-finite/negative timing; wrong measurement count; process failure; and bad firewall/fatal text.

```python
def test_accepts_complete_mixed_quant_proof(tmp_path):
    proof = write_passing_proof(tmp_path)
    normalized = validator.validate(proof)
    assert normalized["status"] == "pass"
    assert normalized["token_ids_match"] is True
    assert normalized["eligible_route_coverage"] == 1.0
    assert normalized["tensor_eligibility_coverage"] == pytest.approx(141 / 180)
    assert normalized["all_cus_active"] is True

def test_rejects_native_fallback_for_eligible_iq1s(tmp_path):
    proof = write_passing_proof(tmp_path)
    mutate(proof / "hybrid.json", lambda value: value["routes"].update(fallback=1))
    with pytest.raises(validator.ProofInvalid):
        validator.validate(proof)
```

- [ ] **Step 2: Run validator tests and verify the red state**

Run:

```bash
python3 -m pytest -q zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
```

Expected: FAIL because the validator does not exist.

- [ ] **Step 3: Implement schema-2 validation and normalization**

Start from the existing TQ1 validator's strict numeric and device-health helpers, but read and validate `model-tensor-audit.json`. Use IQ1_S raw bounds `[-4096, 4096]`, accept any nonzero integer STALL code, and require physical request IDs plus explicit completion accounting. Normalize:

```python
result = {
    "schema_version": 2,
    "status": "pass",
    "token_ids_match": True,
    "eligible_route_coverage": hybrid_routes["handled"] / hybrid_routes["eligible"],
    "tensor_eligibility_coverage": audit["routed_expert_types"]["IQ1_S"] / audit["routed_expert_count"],
    "all_cus_active": all(value > 0 for value in hybrid_xrt["per_cu_completions"]),
    "model_audit": audit,
    "modes": {"cuda": cuda_normalized, "hybrid": hybrid_normalized},
    "numerical": numerical_normalized,
}
```

On failure print `QWEN_IQ1S_PROOF_INVALID: <reason>` to stderr and emit no throughput JSON.

- [ ] **Step 4: Run all proof-validator suites**

Run:

```bash
python3 -m pytest -q \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py \
  zluda/tests/test_validate_qwen35_tq1_au250_proof.py \
  zluda/tests/test_validate_au250_hybrid_proof.py
```

Expected: all validator tests PASS.

- [ ] **Step 5: Commit the revised validator**

```bash
git add zluda/tests/validate_qwen35_iq1s_au250_proof.py \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
git commit -m "test: validate mixed Qwen IQ1_S hybrid proof"
```

### Task 5: Orchestrate CUDA-only and strict IQ1_S hybrid modes

**Files:**
- Create: `tools/run_qwen35_iq1s_au250_hybrid.sh`
- Modify: `zluda/tests/test_au250_qwen35_runtime_static.sh`

- [ ] **Step 1: Add static assertions for the revised runner**

Require the new runner to invoke the GGUF audit before either server, use the route manifest, disable the type-34 bridge in both modes, enable BitNet/XRT only in hybrid, retain the TQ1 live numerical gate, invoke the new validator last, and never accept an existing output directory.

```bash
grep -Fq 'qwen35_gguf_audit.py' "${iq1s_runner}"
grep -Fq 'qwen35-iq1s-route-manifest.json' "${iq1s_runner}"
test "$(grep -Fc 'HETGPU_QWEN_TQ1_XRT=0' "${iq1s_runner}")" -eq 2
grep -Fq 'HETGPU_TMATMUL_BACKEND=xrt' "${iq1s_runner}"
grep -Fq 'HETGPU_BITNET_DISAGGREGATE=1' "${iq1s_runner}"
grep -Fq 'HETGPU_BITNET_DISAGG_STRICT=1' "${iq1s_runner}"
grep -Fq 'HETGPU_TMATMUL_HARDWARE_MATMUL=1' "${iq1s_runner}"
grep -Fq 'run_au250_xrt_tq1.sh' "${iq1s_runner}"
grep -Fq 'validate_qwen35_iq1s_au250_proof.py' "${iq1s_runner}"
```

- [ ] **Step 2: Run the static test and verify failure**

Run:

```bash
bash zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: FAIL because `tools/run_qwen35_iq1s_au250_hybrid.sh` is absent.

- [ ] **Step 3: Implement the revised runner**

Copy only the common integrity, device, and model-preflight structure from the TQ1 runner. Use proof directories matching `.proof/qwen35-iq1s-au250-*`. After the one independent model hash, atomically create `model-verification.json`, then run the audit before CUDA-only mode.

CUDA environment:

```bash
export HETGPU_QWEN_TQ1_XRT=0 HETGPU_QWEN_TQ1_STRICT=0
export HETGPU_BITNET_DISAGGREGATE=0 HETGPU_BITNET_DISAGG_STRICT=0
unset HETGPU_TMATMUL_BACKEND HETGPU_TMATMUL_HARDWARE_MATMUL
unset HETGPU_BITNET_ROUTE_MANIFEST HETGPU_BITNET_ROUTE_LOG HETGPU_XRT_EXECUTION_LOG
```

Hybrid environment:

```bash
export HETGPU_QWEN_TQ1_XRT=0 HETGPU_QWEN_TQ1_STRICT=0
export HETGPU_TMATMUL_BACKEND=xrt
export HETGPU_BITNET_DISAGGREGATE=1 HETGPU_BITNET_DISAGG_STRICT=1
export HETGPU_TMATMUL_HARDWARE_MATMUL=1
export HETGPU_BITNET_ROUTE_MANIFEST=/work/tools/qwen35-iq1s-route-manifest.json
export HETGPU_BITNET_GPU_KERNELS=attention,attn,flash,softmax,soft_max,rope,kq,qk,qkv,query,key,value,kv_cache
export HETGPU_BITNET_CXL_KERNELS=ggml_type19
export HETGPU_BITNET_ROUTE_LOG="${proof_dir}/hybrid-mode/routes.jsonl"
export HETGPU_XRT_EXECUTION_LOG="${proof_dir}/hybrid-mode/xrt.jsonl"
export HETGPU_XRT_COMPARE_MAX_LAUNCHES=0
```

Pass `--evidence-kind iq1s`, both evidence paths, the model audit, and `--require-routing-evidence` to the evaluator. Preserve the same server, preload library, port pattern, threads, model verification, prompt, and all benchmark arguments. Run the existing live TQ1 single/tiled qualification between modes as a shared-hardware gate. Run the new validator last and write `summary.json` only from its passing output.

- [ ] **Step 4: Run shell/static/fake-server tests**

Run:

```bash
bash -n tools/run_qwen35_iq1s_au250_hybrid.sh
bash zluda/tests/test_au250_qwen35_runtime_static.sh
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
```

Expected: all commands PASS.

- [ ] **Step 5: Commit revised orchestration**

```bash
git add tools/run_qwen35_iq1s_au250_hybrid.sh \
  zluda/tests/test_au250_qwen35_runtime_static.sh
git commit -m "feat: orchestrate Qwen IQ1_S CUDA AU250 evaluation"
```

### Task 6: Run regression, build, diagnostic, and full live proof

**Files:**
- Runtime only: `.proof/qwen35-iq1s-au250-<UTC>/`

- [ ] **Step 1: Run focused Python and static suites**

Run:

```bash
python3 -m pytest -q \
  zluda/tests/test_qwen35_gguf_audit.py \
  zluda/tests/test_qwen35_au250_eval.py \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py \
  zluda/tests/test_validate_qwen35_tq1_au250_proof.py \
  zluda/tests/test_validate_au250_hybrid_proof.py
bash zluda/tests/test_fetch_qwen35_tq1_model.sh
bash zluda/tests/test_prepare_au250_qwen35_source.sh
bash zluda/tests/test_au250_hybrid_runtime_static.sh
bash zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: every command exits 0.

- [ ] **Step 2: Run Rust regression suites**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation -- --nocapture
```

Expected: all non-ignored tests pass. Record unrelated pre-existing warnings separately; do not broaden scope to mass-format or lint unrelated modules.

- [ ] **Step 3: Check only changed-file formatting and diffs**

Run:

```bash
cargo fmt --all -- --check
git diff --check -- \
  tools/qwen35_gguf_audit.py tools/qwen35_au250_eval.py \
  tools/run_qwen35_iq1s_au250_hybrid.sh \
  tools/qwen35-iq1s-route-manifest.json \
  zluda/src/impl/iq1s_xrt.rs zluda/tests/test_qwen35_gguf_audit.py \
  zluda/tests/test_qwen35_au250_eval.py \
  zluda/tests/validate_qwen35_iq1s_au250_proof.py \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py \
  zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: changed files have no whitespace errors. If global `cargo fmt --check` still fails on known unrelated baseline files, record the exact paths and verify `rustfmt --check zluda/src/impl/iq1s_xrt.rs` separately.

- [ ] **Step 4: Rebuild the pinned runtime and bind the manifest to HEAD**

Run:

```bash
tools/au250_qwen35_run.sh /work/tools/build_au250_qwen35_runtime.sh
jq -r '.hetgpu_commit, .artifacts.llama_server.sha256, .artifacts.libnvcuda.sha256' \
  /root/qwen35-au250-build/manifest.json
git rev-parse HEAD
```

Expected: build exits 0 and manifest commit equals `HEAD`.

- [ ] **Step 5: Require an idle GPU and clean AU250 preflight**

Run:

```bash
nvidia-smi --query-gpu=memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
source /au250_xrt/env.sh
au250-status
```

Expected: no unrelated compute process, at least model bytes plus 2 GiB free GPU memory, four listed CUs DONE, firewall GOOD, and temperature below 85 C. Do not kill other users' processes; wait for capacity.

- [ ] **Step 6: Run the full fail-closed proof**

Run:

```bash
proof="$(pwd -P)/.proof/qwen35-iq1s-au250-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "${proof}" | tee target/qwen35-iq1s-last-proof-path
tools/run_qwen35_iq1s_au250_hybrid.sh \
  /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf \
  /root/qwen35-au250-build/manifest.json \
  /au250_xrt/example/MaxCores_370M.xclbin \
  "${proof}"
python3 zluda/tests/validate_qwen35_iq1s_au250_proof.py "${proof}" \
  > "${proof}/normalized.json"
```

Expected: runner and validator exit 0; normalized JSON reports status pass, exact token equality, eligible route coverage 1.0, tensor eligibility coverage `141/180`, and all four CUs active. On failure, retain the proof and report the exact first-failure boundary without throughput.

- [ ] **Step 7: Inspect physical and semantic evidence manually**

Run:

```bash
jq '{audit:.model_audit|{routed_expert_count,routed_expert_types,tq1_0_total},routes:.modes.hybrid.routes,xrt:.modes.hybrid.xrt,token_ids_match,all_cus_active}' \
  "${proof}/normalized.json"
rg -n 'fallback|reject|ERROR|timeout|poison|non-finite|mismatch|BAD|FATAL' "${proof}"
```

Expected: `141/24/4/11`, zero TQ1, eligible equals handled, fallback/error zero, every CU positive, tokens match, and no unaccounted fatal line.

### Task 7: Generate and commit the evidence-backed report

**Files:**
- Modify: `tools/qwen35_au250_eval.py`
- Modify: `zluda/tests/test_qwen35_au250_eval.py`
- Create: `zluda/evaluation/2026-08-27-qwen35-iq1s-au250-evaluation.md`

- [ ] **Step 1: Add a failing report-content test**

Require report generation to accept only schema-2 passing normalized proofs and state the mixed-format boundary explicitly:

```python
report = evaluator.render_iq1s_report(normalized, Path("proof"))
assert "141/180 routed-expert tensors eligible" in report
assert "IQ2_XXS, IQ3_S, and MXFP4 remained on CUDA" in report
assert "Eligible IQ1_S operations handled by AU250: 100%" in report
assert "Active CUs: 4/4" in report
assert "pure TQ1_0" not in report
```

Add rejection tests for a nonpassing proof, incomplete route coverage, inactive CU, missing audit, or non-finite metric.

- [ ] **Step 2: Verify report tests fail**

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py -k iq1s_report
```

Expected: FAIL because `render_iq1s_report` does not exist.

- [ ] **Step 3: Implement report rendering from normalized evidence only**

Render CUDA median, hybrid median, hybrid/CUDA ratio, min-max, and population standard deviation for prompt TPS, generation TPS, TTFT, and end-to-end latency. Include model/hash/revision, audit distribution, operation coverage, tensor eligibility coverage, per-CU completions, numerical maximum errors, and the proof path. Do not parse raw logs or recompute rejected data during rendering.

- [ ] **Step 4: Generate, inspect, and commit the report**

Run:

```bash
proof="$(cat target/qwen35-iq1s-last-proof-path)"
python3 tools/qwen35_au250_eval.py render-iq1s-report \
  --normalized "${proof}/normalized.json" --proof-path "${proof}" \
  --output zluda/evaluation/2026-08-27-qwen35-iq1s-au250-evaluation.md
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
rg -n 'TBD|TODO|nan|inf|pure TQ1_0|100%.*180' \
  zluda/evaluation/2026-08-27-qwen35-iq1s-au250-evaluation.md
git diff --check -- tools/qwen35_au250_eval.py \
  zluda/tests/test_qwen35_au250_eval.py \
  zluda/evaluation/2026-08-27-qwen35-iq1s-au250-evaluation.md
git add tools/qwen35_au250_eval.py zluda/tests/test_qwen35_au250_eval.py
git add -f zluda/evaluation/2026-08-27-qwen35-iq1s-au250-evaluation.md
git commit -m "docs: report Qwen IQ1_S AU250 hybrid evaluation"
```

Expected: tests pass, forbidden search has no output, diff check passes, and the commit contains source/tests/report but no `.proof`, model, build, or binary artifact.

## Final Verification Checklist

- [ ] Run every focused Python, shell, and Rust command from Task 6 again after the report commit.
- [ ] Confirm the retained proof's build manifest commit equals the source/runtime commit recorded in that proof. A later report-only commit may advance `HEAD`; record both commits and do not rewrite the already validated proof or rebuild unchanged runtime binaries merely to bind generated documentation.
- [ ] Re-run the proof validator on the retained proof and verify it still emits status pass.
- [ ] Confirm `git status --short` contains only the pre-existing `ext/nvidia_runtime-sys/src/lib.rs` overlay and ignored `.proof/` data.
- [ ] Confirm no downloaded model, generated binary, build directory, or proof file is staged.
- [ ] Use the verification-before-completion skill before reporting benchmark numbers.
- [ ] Use the finishing-a-development-branch skill to decide merge, push, or worktree retention only after every gate passes.
