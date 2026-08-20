# FPGA CXL v3 Batch Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute capability-bounded IQ1_S batches through the real 16-instance CXL v3 UAPI and evaluate them with correctness and live-device evidence.

**Architecture:** A pure planner slices logical batches by `CapsV3.max_batch`; IQ1_S staging emits one `TaskV3` per component and slice with `LANE_ANY`; existing v3 submission/wait code validates and demultiplexes completions. Completion-derived reports measure actual lane use and work rather than queue bookkeeping.

**Tech Stack:** Rust 2021, CUDA/ZLUDA launch interception, Linux CXL v3 ioctls, device DAX, existing fake-v3 test backend.

---

### Task 1: Batch planner and metrics

**Files:**
- Create: `zluda/src/impl/batch_scheduler.rs`
- Modify: `zluda/src/impl/mod.rs`

- [ ] Write tests for zero batch rejection, exact and remainder slices, environment limits that may only lower live capability, and completion-derived lane/submission metrics.
- [ ] Run `cargo test -p zluda --no-default-features --features nvidia batch_scheduler -- --nocapture` and verify the new tests fail because the module/API is absent.
- [ ] Implement `BatchSlice`, `BatchPlan`, `BatchSchedulerConfig`, and `SchedulerReport`; use checked arithmetic and preserve slice order.
- [ ] Re-run the focused test and require all planner tests to pass.

### Task 2: Real GGML batch layout

**Files:**
- Modify: `zluda/src/impl/iq1s_tmatmul.rs`
- Modify: `zluda/src/impl/function.rs`

- [ ] Add failing producer-faithful tests for two K records, active batch 2, and `stride11 = 3`; require active record indices `0, 1, 3, 4`, ignored pitch padding, and correct `q8_group(batch, group)` selection.
- [ ] Treat `stride11` as GGML's pitch in 144-byte Q8_1 MMQ records. Validate `stride11 >= ne11` and compute the checked capture extent as `(((ne10 / 128 - 1) * stride11) + ne11) * 144`.
- [ ] Reject negative signed MMQ dimensions before the NVIDIA bridge converts them to unsigned signature fields; preserve batch-one behavior.
- [ ] Run the focused IQ1_S tests and require the new layout tests to pass.

### Task 3: Component-by-slice v3 execution

**Files:**
- Modify: `zluda/src/impl/iq1s_tmatmul.rs`
- Modify: `zluda/src/impl/cxl_tmatmul_v3.rs` only if a read-only capability accessor is required

- [ ] Add a failing fake-v3 test with logical batch 4 and live max batch 2; require two ordered slices per component, `TaskV3.batch == 2`, unique IDs, lane-any submission, and bit-exact outputs for all four rows.
- [ ] Stage input/output regions with per-component batch strides, build slice descriptors from the planner, and reconstruct outputs by `(component, batch_index, row)`.
- [ ] Add `SchedulerReport` to `ExecutionResult` and derive it only from validated completions.
- [ ] Re-run `iq1s_tmatmul` and `cxl_tmatmul_v3` suites.

### Task 4: NVIDIA integration and evidence output

**Files:**
- Modify: `zluda/src/impl/function.rs`

- [ ] Add a failing route test that requires batch size, descriptor count, submission count, and lane mask in the completed IQ1_S log record.
- [ ] Emit those fields only after `execute_captured` and CUDA output copy succeed; keep strict fallback behavior unchanged.
- [ ] Run the NVIDIA named-route tests with incremental compilation disabled.

### Task 5: Evaluation harness

**Files:**
- Replace: `zluda/tests/batch_scheduler_integration_test.rs`
- Create: `zluda/tests/run_v3_batch_evaluation.sh`
- Preserve: unrelated untracked tests and DAX-pool work

- [ ] Replace the trivial arithmetic assertions with contract tests that invoke the real planner-facing APIs available to integration tests or a narrow exported evaluation entrypoint.
- [ ] Add a fail-closed shell harness that records baseline/enabled commands, device paths, capability JSON, test output, live fixture output, and workload timing artifacts under a caller-selected result directory.
- [ ] Make live execution opt-in through `HETGPU_RUN_LIVE_V3_BATCH=1`; software checks remain mandatory.

### Task 6: Verification and review

**Files:**
- Verify all files changed by Tasks 1-5

- [ ] Run formatter only on touched Rust files and inspect the diff for unrelated changes.
- [ ] Run planner, IQ1_S, v3, and NVIDIA route tests; run `cargo check -p zluda --no-default-features --features nvidia`.
- [ ] Query the live capability block and, when compatible, run the live batch fixture and the shortest clean Kimi/MatMulFreeLM gates before longer TPS samples.
- [ ] Report software, live hardware, and workload proof as separate acceptance boundaries; do not claim target TPS without measured clean timing output.
