# Qwen3.5-397B U250 Layer-Persistent IQ1_S FFN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Replace per-component U250 execution with a four-CU persistent, raw-IQ1_S layer engine and produce a fail-closed, measured Qwen3.5-397B E2E TPS result.

**Architecture:** CUDA launch interception continues to capture exact matrix arguments while a versioned llama.cpp sideband ABI supplies layer and phase boundaries. All 55.078125 GiB of IQ1_S expert weights are row-sharded across four U250 DDR banks, and handwritten or AlgorithmTree-generated programs are executed through bank-local persistent command rings. Attention and all non-IQ1_S work stay on GPU.

**Tech Stack:** Rust 2021, C/C++, CUDA 13, Python 3, SystemVerilog, cocotb, Verilator, XRT 2.21, Vitis/Vivado 2026.1, Alveo U250, Bash, pytest, Cargo.

---

## Execution roots and fixed inputs

Software worktree:

    /home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826

Hardware source repository:

    /home/victoryang00/hetGPU/ternary_matmul

Hardware implementation worktree to create:

    /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902

Hardware base commit:

    45d1de5abcb8e35284d56444cf4fa3fc5c925a46

Approved design:

    docs/superpowers/specs/2026-09-02-qwen397b-u250-layer-persistent-design.md

Fixed model:

    /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf

Toolchain setup:

    source /mnt/disk0/2026.1/settings64.sh

Every Cargo command must set CARGO_BUILD_JOBS=32. Every Qwen build must set
QWEN35_BUILD_JOBS=32. Every Vitis link uses --jobs 32,
--vivado.synth.jobs 32, and --vivado.impl.jobs 32. Vivado Tcl entry points set
general.maxThreads to 32.

## File map

### Software repository

- Create tools/qwen35-iq1s-layer-abi.json: canonical register, descriptor, and completion schema.
- Create tools/generate_qwen35_iq1s_layer_abi.py: generate C, Rust, and SystemVerilog ABI definitions.
- Create tools/qwen35-iq1s-layer-generated.h: generated C ABI declarations.
- Create zluda/src/impl/iq1s_layer_abi.rs: generated Rust ABI types and constants.
- Create zluda/src/impl/iq1s_layer.rs: sideband FFI, transaction state machine, phase validation, and capture coordination.
- Create zluda/src/impl/iq1s_weight_arena.rs: row-shard planning, GGUF reads, hashes, and bank image construction.
- Create zluda/src/impl/iq1s_layer_trace.rs: handwritten and AlgorithmTree layer trace builders, relocations, and expanded counts.
- Create zluda/src/impl/xrt_iq1s_persistent.rs: four-CU persistent XRT rings, arena BOs, DMA ranges, and completions.
- Create tools/tune_qwen35_iq1s_persistent.py: correctness-gated knob search.
- Create tools/plot_qwen35_iq1s_persistent.py: reproducible tuning and E2E figure.
- Create zluda/tests/test_qwen35_layer_abi_generator.py: schema generation and drift checks.
- Create zluda/tests/test_qwen35_iq1s_persistent_proof.py: validator mutation tests.
- Modify tools/qwen35-tq1-bridge.h: include generated layer ABI.
- Modify tools/llama-qwen35-tq1-hetgpu.patch: emit sideband lifecycle calls around routed-expert projections.
- Modify zluda/src/lib.rs and zluda/src/impl/mod.rs: export and register new modules and FFI.
- Modify zluda/src/impl/function.rs: buffer captured IQ1_S launches into layer phases.
- Modify zluda/src/impl/iq1s_weight_registry.rs: expose immutable tensor/layer/role iteration.
- Modify zluda/src/impl/iq1s_trace.rs: retain legacy single-launch traces and share checked assembly helpers.
- Modify zluda/src/impl/iq1s_xrt.rs: preserve legacy Kimi execution and route Qwen layer phases to the persistent executor.
- Modify zluda/src/impl/xrt_tmatmul.rs: expose the existing XRT operations abstraction and range-aware fake events.
- Modify ptx/src/pass/tmatmul_algorithm_tree.rs: add checked, non-panicking assembly generation.
- Modify tools/build_au250_qwen35_runtime.sh: build and hash the v2 layer ABI runtime.
- Modify tools/run_qwen35_iq1s_au250_hybrid.sh: bind the new xclbin, preload arenas, run tuning, and run E2E.
- Modify tools/qwen35_au250_eval.py: collect phase-level metrics and compute reportable TPS.
- Modify zluda/tests/validate_qwen35_iq1s_au250_proof.py: enforce persistent-kernel, residency, correctness, and throughput gates.
- Modify existing Qwen static and evaluator tests for the new contract.
- Create zluda/evaluation/2026-09-02-qwen397b-u250-layer-persistent.md: generated final report.

### Hardware repository

- Create rtl/iq1s_layer_abi_pkg.sv: generated ABI constants and byte offsets.
- Create rtl/iq1s_command_ring.sv: persistent descriptor fetch and ring ownership.
- Create rtl/iq1s_descriptor_check.sv: magic/version/length/generation/bounds/CRC checks.
- Create rtl/iq1s_program_fetch.sv: cached program and relocation reads.
- Create rtl/iq1s_trace_relocator.sv: checked address binding into 128-bit TernIP instructions.
- Create rtl/iq1s_block_decoder.sv: native 50-byte IQ1_S decode and grid/delta stream generation.
- Create rtl/iq1s_scale_reconstruct.sv: exact integer partials and FP32 scale reconstruction.
- Create rtl/iq1s_result_writer.sv: coalesced row-shard writes.
- Create rtl/iq1s_completion_ring.sv: completion publication after result visibility.
- Create rtl/iq1s_fault_latch.sv: first-fault capture and quiescent reset protocol.
- Create rtl/axi_iq1s_layer_persistent.sv: persistent CU integration and AXI arbitration.
- Create rtl/axi_iq1s_layer_persistent_vivado_bd_wrapper.sv: concrete-width Vivado wrapper.
- Create dv/config/qwen397b_iq1s_1cu.json: small one-CU simulation target.
- Create dv/cocotb/iq1s_layer_persistent/Makefile and test_iq1s_layer_persistent.py: end-to-end RTL tests.
- Create synth/pynqvivado_au250/iq1s_bd.tcl: one-bank persistent-kernel block design.
- Create synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json: three-big/one-small target.
- Modify rtl/rtl.f: include persistent modules.
- Modify sw_utils/lib/target.py and sw_utils/target/resolve_target.py: expose kernel_kind and BD/kernel-ABI selection.
- Create sw_utils/tests/test_target.py: descriptor validation and resolver tests.
- Modify synth/pynqvivado_common/create_bd_xpr.tcl: select the target's BD script.
- Modify synth/pynqvivado_common/generate_kernel_xml.tcl: emit persistent ring pointer/register arguments.
- Modify synth/pynqvivado_common/package_ip.tcl: package IQ1_S grid memory.
- Modify synth/pynqvivado_common/generate_kernel_cfg.tcl and Makefile: enforce four-bank mapping and the 32-thread ceiling.

### Task 1: Create the isolated RTL worktree and prove both baselines

**Files:**
- Read: docs/superpowers/specs/2026-09-02-qwen397b-u250-layer-persistent-design.md
- Read: /home/victoryang00/hetGPU/ternary_matmul/synth/pynqvivado_au250/targets/pynqvivado_au250_MaxCores_370M.json

- [ ] **Step 1: Invoke the worktree setup skill**

At execution time, invoke superpowers:using-git-worktrees before changing the
hardware repository. Preserve the current main checkout.

- [ ] **Step 2: Create the hardware worktree from the frozen upstream base**

Run:

    git -C /home/victoryang00/hetGPU/ternary_matmul fetch origin
    git -C /home/victoryang00/hetGPU/ternary_matmul cat-file -e 45d1de5abcb8e35284d56444cf4fa3fc5c925a46^{commit}
    git -C /home/victoryang00/hetGPU/ternary_matmul worktree add \
      -b codex/qwen397b-iq1s-persistent-20260902 \
      /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902 \
      45d1de5abcb8e35284d56444cf4fa3fc5c925a46
    git -C /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902 \
      submodule update --init --recursive

Expected: the new worktree is clean at 45d1de5 and all four submodules have
non-prefixed entries in git submodule status.

- [ ] **Step 3: Prove the software baseline**

Run:

    cd /home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s -- --nocapture
    python3 -m pytest -q \
      zluda/tests/test_qwen35_au250_eval.py \
      zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
    bash zluda/tests/test_au250_qwen35_runtime_static.sh

Expected: all focused tests pass. Record any pre-existing failure verbatim
before changing source.

- [ ] **Step 4: Prove the upstream hardware simulation baseline**

Run:

    cd /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902
    python3 -m pip install --user -r sw_utils/requirements.txt cocotb cocotbext-axi
    make -C dv/cocotb/axi_ternip_batched SIM=verilator \
      TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_MaxCores_370M.json \
      KERNEL=ternip_big

Expected: cocotb returns zero with no fatal AXI assertion.

- [ ] **Step 5: Prove toolchain, platform, and live U250 identity**

Run:

    source /mnt/disk0/2026.1/settings64.sh
    v++ --version
    vivado -version
    platforminfo --list | rg xilinx_u250_gen3x16_xdma_4_1_202210_1
    sudo xrt-smi examine --device 0000:64:00.1
    sudo xbmgmt examine --device 0000:64:00.0

Expected: Vitis and Vivado are 2026.1, the legacy U250 platform is listed, the
user PF is ready, and the management PF reports Device Ready: Yes. This task
does not alter the card.

### Task 2: Generate one shared layer ABI for C, Rust, and RTL

**Files:**
- Create: tools/qwen35-iq1s-layer-abi.json
- Create: tools/generate_qwen35_iq1s_layer_abi.py
- Create: tools/qwen35-iq1s-layer-generated.h
- Create: zluda/src/impl/iq1s_layer_abi.rs
- Create: zluda/tests/test_qwen35_layer_abi_generator.py
- Create: /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/rtl/iq1s_layer_abi_pkg.sv

- [ ] **Step 1: Write failing generator tests**

Add tests that run the generator into a temporary directory and assert:

    assert schema["abi_version"] == 2
    assert schema["command"]["size"] == 128
    assert schema["completion"]["size"] == 128
    assert field("command", "transaction_id") == (24, "u64")
    assert field("command", "arena_offset") == (64, "u64")
    assert field("completion", "status") == (8, "u32")
    assert "#define HETGPU_IQ1S_COMMAND_BYTES 128u" in c_header
    assert "pub const IQ1S_COMMAND_BYTES: usize = 128;" in rust_source
    assert "localparam int IQ1S_COMMAND_BYTES = 128;" in sv_source

Also mutate one offset and assert --check exits nonzero with
ABI output differs from canonical schema.

- [ ] **Step 2: Run the test and verify the failure**

Run:

    cd /home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826
    python3 -m pytest -q zluda/tests/test_qwen35_layer_abi_generator.py

Expected: FAIL because the schema and generator do not exist.

- [ ] **Step 3: Define the canonical schema**

The JSON must define:

    {
      "abi_version": 2,
      "endian": "little",
      "constants": {
        "register_magic": "0x324c5149",
        "command_magic": "0x32435149",
        "completion_magic": "0x32445149"
      },
      "enums": {
        "phase": {"A": 1, "B": 2},
        "role": {"GATE": 1, "UP": 2, "DOWN": 3},
        "weight_format": {"IQ1_S": 19},
        "completion_status": {"OK": 0, "FAULT": 1, "ABORTED": 2},
        "fault_code": {
          "NONE": 0, "BAD_MAGIC": 1, "BAD_VERSION": 2,
          "BAD_LENGTH": 3, "BAD_GENERATION": 4, "BAD_CRC": 5,
          "BAD_BOUNDS": 6, "BAD_RELOCATION": 7, "AXI": 8,
          "TIMEOUT": 9, "NONFINITE": 10, "RING_OVERFLOW": 11
        }
      },
      "registers": {
        "abi_magic": 0,
        "abi_version": 4,
        "control": 8,
        "status": 12,
        "session_generation_lo": 16,
        "session_generation_hi": 20,
        "command_base_lo": 24,
        "command_base_hi": 28,
        "command_capacity": 32,
        "command_producer": 36,
        "command_consumer": 40,
        "doorbell": 44,
        "completion_base_lo": 48,
        "completion_base_hi": 52,
        "completion_capacity": 56,
        "completion_producer": 60,
        "completion_consumer": 64,
        "fault_code": 68,
        "fault_detail_lo": 72,
        "fault_detail_hi": 76,
        "quiescent": 80
      },
      "command": {
        "size": 128,
        "fields": [
          ["magic", "u32", 0], ["abi_version", "u16", 4],
          ["descriptor_bytes", "u16", 6], ["crc32", "u32", 8],
          ["flags", "u32", 12], ["session_generation", "u64", 16],
          ["transaction_id", "u64", 24], ["program_id", "u64", 32],
          ["trace_id", "u64", 40], ["layer_id", "u32", 48],
          ["phase", "u16", 52], ["role", "u16", 54],
          ["expert_id", "u16", 56], ["lane_mask", "u16", 58],
          ["lane_count", "u16", 60], ["weight_format", "u16", 62],
          ["arena_offset", "u64", 64], ["input_offset", "u64", 72],
          ["output_offset", "u64", 80], ["row_start", "u32", 88],
          ["row_count", "u32", 92], ["input_bytes", "u32", 96],
          ["output_bytes", "u32", 100], ["token_map_offset", "u64", 104],
          ["dependency_fence", "u64", 112],
          ["completion_slot", "u32", 120], ["reserved", "u32", 124]
        ]
      },
      "completion": {
        "size": 128,
        "fields": [
          ["magic", "u32", 0], ["abi_version", "u16", 4],
          ["completion_bytes", "u16", 6], ["status", "u32", 8],
          ["fault_code", "u32", 12], ["session_generation", "u64", 16],
          ["transaction_id", "u64", 24], ["program_id", "u64", 32],
          ["trace_id", "u64", 40], ["layer_id", "u32", 48],
          ["phase", "u16", 52], ["role", "u16", 54],
          ["cu_id", "u16", 56], ["expert_id", "u16", 58],
          ["lane_mask", "u16", 60], ["rows_completed", "u16", 62],
          ["descriptor_crc32", "u32", 64], ["command_index", "u32", 68],
          ["cycles", "u64", 72], ["ddr_read_bytes", "u64", 80],
          ["ddr_write_bytes", "u64", 88], ["iq1s_blocks", "u64", 96],
          ["grid_passes", "u32", 104], ["delta_passes", "u32", 108],
          ["result_fence", "u64", 112], ["fault_detail", "u64", 120]
        ]
      }
    }

- [ ] **Step 4: Implement deterministic generation**

The generator must validate non-overlap, natural alignment, exact record size,
known integer types, and zero gaps not named reserved. Generate repr(C) Rust
records with const size/offset assertions, packed C records with static
assertions, and SystemVerilog localparams for every byte offset. Include the
canonical schema SHA-256 in all outputs.

The command CRC is IEEE CRC32 over all 128 bytes with bytes 8..11 treated as
zero.

- [ ] **Step 5: Generate outputs and pass the drift test**

Run:

    python3 tools/generate_qwen35_iq1s_layer_abi.py \
      --schema tools/qwen35-iq1s-layer-abi.json \
      --c-out tools/qwen35-iq1s-layer-generated.h \
      --rust-out zluda/src/impl/iq1s_layer_abi.rs \
      --sv-out /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/rtl/iq1s_layer_abi_pkg.sv
    python3 tools/generate_qwen35_iq1s_layer_abi.py \
      --schema tools/qwen35-iq1s-layer-abi.json \
      --c-out tools/qwen35-iq1s-layer-generated.h \
      --rust-out zluda/src/impl/iq1s_layer_abi.rs \
      --sv-out /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/rtl/iq1s_layer_abi_pkg.sv \
      --check
    python3 -m pytest -q zluda/tests/test_qwen35_layer_abi_generator.py

Expected: PASS and all three generated files contain the same schema hash.

- [ ] **Step 6: Commit both repository changes**

Software:

    git add tools/qwen35-iq1s-layer-abi.json \
      tools/generate_qwen35_iq1s_layer_abi.py \
      tools/qwen35-iq1s-layer-generated.h \
      zluda/src/impl/iq1s_layer_abi.rs \
      zluda/tests/test_qwen35_layer_abi_generator.py
    git commit -m "feat: define Qwen IQ1S persistent ABI"

Hardware:

    git add rtl/iq1s_layer_abi_pkg.sv
    git commit -m "feat: import Qwen IQ1S persistent ABI"

### Task 3: Implement the fail-closed layer transaction coordinator

**Files:**
- Create: zluda/src/impl/iq1s_layer.rs
- Modify: zluda/src/impl/mod.rs
- Modify: zluda/src/lib.rs
- Modify: tools/qwen35-tq1-bridge.h

- [ ] **Step 1: Add failing transaction-state tests**

Define tests for:

    begin -> routes -> gate capture -> up capture -> phase A commit
          -> phase A done -> down capture -> layer commit -> closed

Also assert errors for duplicate transaction IDs, batch 0/17, wrong stream,
expert 512, duplicate roles, phase A commit without gate/up, down before Phase
A completion, stale generation, and a layer with GPU-native down closing
without a Phase B submission.

- [ ] **Step 2: Run the tests and verify they fail**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_layer -- --nocapture

Expected: FAIL because module iq1s_layer is absent.

- [ ] **Step 3: Implement the state and normalized records**

Use these core types:

    enum LayerState {
        Open,
        RoutesPending,
        PhaseACapture,
        PhaseACommitted,
        PhaseADone,
        PhaseBCapture,
        PhaseBCommitted,
        Closed,
        Aborted,
    }

    struct LayerKey {
        session_generation: u64,
        transaction_id: u64,
        layer_id: u32,
        stream: usize,
    }

    struct RouteAssignment {
        token_id: u32,
        expert_id: u16,
        route_weight: f32,
    }

    struct CapturedProjection {
        role: Iq1sExpertRole,
        weight: ResolvedIq1sWeight,
        launches: Vec<CapturedLaunch>,
    }

    struct LayerTransaction {
        key: LayerKey,
        state: LayerState,
        batch_count: u16,
        routes: Vec<RouteAssignment>,
        projections: HashMap<Iq1sExpertRole, CapturedProjection>,
        expected_iq1s_roles: BTreeSet<Iq1sExpertRole>,
    }

Use a Mutex<HashMap<(u64, u64), LayerTransaction>>. Every transition returns a
Result and sets Aborted before returning an error.

- [ ] **Step 4: Implement the FFI boundary**

Export exactly:

    hetgpu_iq1s_layer_begin_v2
    hetgpu_iq1s_layer_set_routes_v2
    hetgpu_iq1s_layer_phase_commit_v2
    hetgpu_iq1s_layer_commit_v2
    hetgpu_iq1s_layer_abort_v2

Catch panics at each extern C boundary. Return HETGPU_IQ1S_HANDLED on success
and HETGPU_IQ1S_ERROR on any error. Enqueue one route-table D2H copy on the
bound CUDA stream and wait on its event only at Phase A commit.

- [ ] **Step 5: Run focused tests**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_layer -- --nocapture

Expected: all layer-state and FFI mutation tests pass.

- [ ] **Step 6: Commit**

    git add zluda/src/impl/iq1s_layer.rs zluda/src/impl/mod.rs \
      zluda/src/lib.rs tools/qwen35-tq1-bridge.h
    git commit -m "feat: coordinate Qwen IQ1S layer phases"

### Task 4: Add sideband calls to the pinned Qwen CUDA backend

**Files:**
- Modify: tools/llama-qwen35-tq1-hetgpu.patch
- Modify: zluda/tests/test_prepare_au250_qwen35_source.sh
- Modify: zluda/tests/test_au250_qwen35_runtime_static.sh

- [ ] **Step 1: Add failing patch/static checks**

Require the prepared Qwen source to resolve all five v2 symbols with dlsym and
to keep a per-CUDA-stream state containing layer, transaction, gate_seen,
up_seen, and down_seen. Require the source to call Phase A commit only after
both gate and up have been observed, and to call layer commit after down.

Reject source that calls a v2 hook without checking the return value.

- [ ] **Step 2: Run and verify the failure**

Run:

    bash zluda/tests/test_prepare_au250_qwen35_source.sh
    bash zluda/tests/test_au250_qwen35_runtime_static.sh

Expected: FAIL because the patch does not contain the v2 lifecycle.

- [ ] **Step 3: Patch ggml_cuda_mul_mat_id**

In the pinned llama.cpp CUDA backend, parse layer and role from the registered
tensor name. For every routed-expert tensor, including a GPU-native down role:

    if first gate or up for stream/layer:
        layer_begin_v2(...)
        layer_set_routes_v2(...)
    call the native or intercepted projection
    if gate_seen and up_seen and phase_a_not_committed:
        layer_phase_commit_v2(..., HETGPU_IQ1S_PHASE_A)
    if role is down:
        layer_commit_v2(...)

Use a monotonic atomic uint64_t transaction counter. Abort the open transaction
on exceptions, CUDA errors, graph cancellation, stream/layer mismatch, or a
hook return other than HETGPU_IQ1S_HANDLED.

- [ ] **Step 4: Re-run patch and static tests**

Run the commands from Step 2.

Expected: PASS, including a clean apply of the patch to the pinned source
revision.

- [ ] **Step 5: Commit**

    git add tools/llama-qwen35-tq1-hetgpu.patch \
      zluda/tests/test_prepare_au250_qwen35_source.sh \
      zluda/tests/test_au250_qwen35_runtime_static.sh
    git commit -m "feat: mark Qwen expert layer phases"

### Task 5: Buffer intercepted launches instead of executing components

**Files:**
- Modify: zluda/src/impl/function.rs
- Modify: zluda/src/impl/iq1s_xrt.rs
- Modify: zluda/src/impl/iq1s_layer.rs

- [ ] **Step 1: Write failing interception tests**

Add tests proving that an open Qwen v2 transaction:

- captures all ten expert launches for a role;
- returns success without calling execute_captured;
- rejects an unregistered pointer or wrong expert ID;
- preserves legacy Kimi and non-transaction Qwen behavior;
- never claims attention or non-IQ1_S kernels.

Use a fake phase sink whose execute counter must remain zero during capture.

- [ ] **Step 2: Run and verify the failure**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation nvidia_modern_iq1s -- --nocapture

Expected: at least the buffering assertion fails because function.rs calls
execute_captured for every component.

- [ ] **Step 3: Replace the Qwen transaction loop**

After nvidia_capture_modern_iq1s_xrt_moe_mmvq returns, use:

    if iq1s_layer::has_open_transaction(cuda_stream) {
        iq1s_layer::capture_projection(
            cuda_stream,
            resolved_role,
            captured,
        )?;
        return Some(Ok(()));
    }

Retain the current synchronous loop only for the legacy Kimi/single-launch
path. Once capture_projection accepts an eligible launch, any fallback is an
error.

- [ ] **Step 4: Run focused and regression tests**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation nvidia_modern_iq1s -- --nocapture
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_xrt -- --nocapture

Expected: PASS and legacy XRT tests retain their prior counts.

- [ ] **Step 5: Commit**

    git add zluda/src/impl/function.rs zluda/src/impl/iq1s_xrt.rs \
      zluda/src/impl/iq1s_layer.rs
    git commit -m "feat: capture Qwen IQ1S work by layer"

### Task 6: Plan and build the four-bank raw IQ1_S arenas

**Files:**
- Create: zluda/src/impl/iq1s_weight_arena.rs
- Modify: zluda/src/impl/iq1s_weight_registry.rs
- Modify: zluda/src/impl/mod.rs

- [ ] **Step 1: Add failing arena tests**

Cover:

- exact 50/50/41 tensor role counts;
- exact raw total 59,139,686,400 bytes;
- 1024-row and 4096-row output shapes;
- complete, non-overlapping row coverage across four banks;
- 4 KiB alignment and 512 MiB superblock boundaries;
- no bank above 16 GiB after rings/slabs/reserve are included;
- changed source bytes causing a shard hash mismatch;
- an in-flight arena generation preventing eviction or replacement.

- [ ] **Step 2: Run and verify the failure**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_weight_arena -- --nocapture

Expected: FAIL because the arena module is absent.

- [ ] **Step 3: Expose immutable registry iteration**

Add:

    pub(crate) fn registered_sources(
        &self,
    ) -> Result<Vec<Arc<Iq1sTensorSource>>, String>

Return sources sorted by layer, role, and tensor name. Reject the arena plan
unless the set is exactly 141 IQ1_S tensors from one nonzero model hash.

- [ ] **Step 4: Implement checked row sharding**

Define:

    struct ArenaShard {
        tensor: Iq1sTensorIdentity,
        expert: u16,
        bank: u8,
        row_start: u32,
        row_count: u32,
        superblock: u16,
        offset: u64,
        bytes: u64,
        sha256: [u8; 32],
    }

    struct ArenaPlan {
        generation: u64,
        model_sha256: [u8; 32],
        bank_bytes: [u64; 4],
        shards: Vec<ArenaShard>,
    }

For each expert, divide output rows into aligned contiguous ranges. Start with
equal quarters. The tuner may later select another checked ratio. Compute row
bytes as ne[0] / 256 * 50 and copy only the selected rows from the registered
GGUF extent.

- [ ] **Step 5: Implement bounded superblock streaming**

Use 512 MiB maximum superblocks, 4 KiB alignment, FileExt::read_exact_at, and
incremental SHA-256. Never allocate a 55 GiB host vector. The backend API is:

    trait ArenaBackend {
        type Handle;
        fn allocate(&mut self, bank: u8, bytes: usize)
            -> Result<Self::Handle, String>;
        fn write_range(&mut self, handle: &Self::Handle, offset: usize,
            bytes: &[u8]) -> Result<(), String>;
        fn sync_to_device(&mut self, handle: &Self::Handle, offset: usize,
            bytes: usize) -> Result<(), String>;
        fn device_address(&self, handle: &Self::Handle)
            -> Result<u64, String>;
    }

- [ ] **Step 6: Run tests and a metadata-only real-model plan**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_weight_arena -- --nocapture
    HETGPU_QWEN_MODEL_SHA256=0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568 \
      CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation qwen_real_model_arena_plan -- --ignored --nocapture

Expected: PASS, 141 tensors, 59,139,686,400 raw bytes, four in-capacity banks,
and no device allocation in the metadata-only test.

- [ ] **Step 7: Commit**

    git add zluda/src/impl/iq1s_weight_arena.rs \
      zluda/src/impl/iq1s_weight_registry.rs zluda/src/impl/mod.rs
    git commit -m "feat: plan resident Qwen IQ1S arenas"

### Task 7: Compile real layer traces through AlgorithmTree

**Files:**
- Create: zluda/src/impl/iq1s_layer_trace.rs
- Modify: zluda/src/impl/iq1s_trace.rs
- Modify: zluda/src/impl/mod.rs
- Modify: ptx/src/pass/tmatmul_algorithm_tree.rs

- [ ] **Step 1: Add failing checked-compiler tests**

Test handwritten and compiler modes for gate/up/down, batch 1/6/9/16, repeated
and distinct experts, all four row shards, and the 262,144 context limit.
Assert semantic coverage equality and exact expanded counts:

    expected_blocks = row_count * (input_columns / 256) * lane_count
    expected_grid_passes = expected_blocks * 8
    expected_delta_passes = expected_blocks * 8

Mutate a relocation, row range, or expert and require a semantic-hash mismatch.

- [ ] **Step 2: Run and verify the failure**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_layer_trace -- --nocapture

Expected: FAIL because layer traces do not exist.

- [ ] **Step 3: Add non-panicking AlgorithmTree APIs**

Add Result-returning construction and assembly entry points:

    pub fn try_new_abstract_operation(
        &mut self,
        op: AbstractOperation,
        inputs: Vec<AbstractVector>,
        outputs: Vec<AbstractVector>,
        info: Option<HashMap<String, OperationInfo>>,
    ) -> Result<usize, TMatmulTreeError>

    pub fn try_generate_assembly(&self) -> Result<String, TMatmulTreeError>

Keep existing APIs as wrappers for current callers. New Qwen code must use only
the checked APIs.

- [ ] **Step 4: Implement semantic programs and relocations**

Define:

    enum RelocationSource {
        ArenaOffset,
        InputOffset,
        OutputOffset,
        TokenMapOffset,
    }

    struct ActivationRange {
        cuda_ptr: usize,
        slab_offset: u64,
        bytes: u32,
        stream: usize,
    }

    struct SemanticIq1sCommand {
        layer_id: u32,
        phase: LayerPhase,
        role: Iq1sExpertRole,
        expert_id: u16,
        lane_mask: u16,
        token_ids: Vec<u32>,
        row_shard: ArenaShard,
    }

    struct LayerProgram {
        kind: TraceKind,
        assembly: String,
        encoded: Vec<u8>,
        relocations: Vec<Relocation>,
        semantic_sha256: [u8; 32],
        assembly_sha256: [u8; 32],
        expanded: ExpandedIq1sCounts,
    }

    struct LayerPhasePlan {
        transaction_id: u64,
        phase: LayerPhase,
        commands: Vec<SemanticIq1sCommand>,
        activations: Vec<ActivationRange>,
    }

    struct CompiledLayerPhase {
        transaction_id: u64,
        phase: LayerPhase,
        programs: [LayerProgram; 4],
        commands: [Vec<Iq1sCommand>; 4],
        activations: Vec<ActivationRange>,
        semantic_sha256: [u8; 32],
    }

Model every operation as AlgorithmTree Ldv, Tmatmul, and Sv dependencies. Mark
the matrix operation's weight_format as IQ1_S in the semantic manifest. The
shared assembler encodes the existing tmatmul_go instruction; the descriptor
selects the native decoder and binds checked addresses.

- [ ] **Step 5: Run all trace tests**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p ptx tmatmul_algorithm_tree -- --nocapture
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_trace -- --nocapture
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_layer_trace -- --nocapture

Expected: PASS; compiler and handwritten modes have identical semantic
coverage and physically encodable programs.

- [ ] **Step 6: Commit**

    git add ptx/src/pass/tmatmul_algorithm_tree.rs \
      zluda/src/impl/iq1s_trace.rs zluda/src/impl/iq1s_layer_trace.rs \
      zluda/src/impl/mod.rs
    git commit -m "feat: compile Qwen IQ1S layer traces"

### Task 8: Implement the four-CU persistent XRT executor

**Files:**
- Create: zluda/src/impl/xrt_iq1s_persistent.rs
- Modify: zluda/src/impl/xrt_tmatmul.rs
- Modify: zluda/src/impl/mod.rs

- [ ] **Step 1: Add failing fake-XRT tests**

Require:

- one xclbin load and four CU/context opens per pool;
- one persistent start per CU;
- 512 MiB-or-smaller bank-local arena allocations;
- command and activation syncs coalesced by range;
- zero weight syncs after measurement_begin;
- completion matching by transaction, program, trace, generation, CU, and CRC;
- ring wrap without overwriting an unconsumed entry;
- first fault poisoning later submissions;
- graceful shutdown only after quiescent=1.

- [ ] **Step 2: Make the low-level XRT abstraction shareable**

Change XrtOps and RealXrt visibility to pub(crate). Extend the fake event:

    Event::BoSync {
        bo: usize,
        direction: i32,
        offset: usize,
        bytes: usize,
    }

Do not change legacy xrt_tmatmul behavior.

- [ ] **Step 3: Run and verify the failure**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation xrt_iq1s_persistent -- --nocapture

Expected: FAIL because the persistent pool is absent.

- [ ] **Step 4: Implement pool construction**

Define:

    struct PersistentIq1sPool<O: XrtOps> {
        ops: O,
        device: Handle,
        xclbin_uuid: [u8; 16],
        generation: u64,
        cus: [PersistentCu; 4],
        arena: ResidentArena,
        poisoned: Option<PersistentFault>,
        measured: bool,
    }

Open iq1s_layer_big instances 1..3 and iq1s_layer_small instance 1 with memory
groups 0, 3, 2, and 1. Validate the generated ABI magic/version before any BO
allocation.

- [ ] **Step 5: Implement range DMA and ring submission**

The public phase API is:

    fn submit_phase(
        &mut self,
        phase: &CompiledLayerPhase,
        activations: &[ActivationRange],
    ) -> Result<CompletedLayerPhase, PersistentError>

It validates every range, writes all descriptors before the producer head,
performs merged range syncs, rings each CU once, polls completions with bounded
backoff, validates all records, syncs merged result ranges, and only then
returns outputs. Set poisoned before returning any hardware/ABI error.

- [ ] **Step 6: Pass fake-XRT and legacy tests**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation xrt_iq1s_persistent -- --nocapture
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation xrt_tmatmul -- --nocapture

Expected: PASS and the legacy executor event sequences remain valid.

- [ ] **Step 7: Commit**

    git add zluda/src/impl/xrt_iq1s_persistent.rs \
      zluda/src/impl/xrt_tmatmul.rs zluda/src/impl/mod.rs
    git commit -m "feat: add persistent four-CU XRT rings"

### Task 9: Build and verify the RTL command/completion control plane

**Files:**
- Create: rtl/iq1s_command_ring.sv
- Create: rtl/iq1s_descriptor_check.sv
- Create: rtl/iq1s_program_fetch.sv
- Create: rtl/iq1s_trace_relocator.sv
- Create: rtl/iq1s_completion_ring.sv
- Create: rtl/iq1s_fault_latch.sv
- Create: dv/config/qwen397b_iq1s_1cu.json
- Create: dv/cocotb/iq1s_layer_persistent/Makefile
- Create: dv/cocotb/iq1s_layer_persistent/test_iq1s_layer_persistent.py
- Modify: rtl/rtl.f

- [ ] **Step 1: Write failing cocotb control-plane tests**

Drive a bank-local AXI RAM and AXI-Lite registers. Test valid command fetch,
doorbell, completion publication, ring wrap, backpressure, bad magic, bad
version, bad length, bad generation, bad CRC, invalid offsets, and reset while
non-quiescent.

For a valid descriptor, assert:

    assert completion.transaction_id == command.transaction_id
    assert completion.descriptor_crc32 == command.crc32
    assert completion.command_index == expected_index
    assert dut.command_consumer.value == 1
    assert dut.completion_producer.value == 1

- [ ] **Step 2: Run and verify the failure**

Run from the hardware worktree:

    make -C dv/cocotb/iq1s_layer_persistent SIM=verilator \
      TARGET=dv/config/qwen397b_iq1s_1cu.json

Expected: FAIL because the RTL and cocotb target do not exist.

- [ ] **Step 3: Implement descriptor validation and fault ownership**

iq1s_descriptor_check must validate all descriptor metadata before asserting a
memory-operation valid signal. iq1s_fault_latch captures only the first nonzero
fault and blocks command consumption until control reset is acknowledged while
quiescent.

- [ ] **Step 4: Implement ring fetch, relocation, and completion ordering**

Use monotonically wrapping u32 producer/consumer counters with capacity a
nonzero power of two. Fetch exactly 128 bytes per command. Publish a completion
only after the result-visible input is asserted. Relocation rejects address
addition overflow and an address outside the descriptor's declared slab.

- [ ] **Step 5: Run lint and cocotb**

Run:

    make lint TARGET=dv/config/qwen397b_iq1s_1cu.json
    make -C dv/cocotb/iq1s_layer_persistent SIM=verilator \
      TARGET=dv/config/qwen397b_iq1s_1cu.json

Expected: PASS with all AXI protocol checks enabled.

- [ ] **Step 6: Commit**

    git add rtl/iq1s_command_ring.sv rtl/iq1s_descriptor_check.sv \
      rtl/iq1s_program_fetch.sv rtl/iq1s_trace_relocator.sv \
      rtl/iq1s_completion_ring.sv rtl/iq1s_fault_latch.sv rtl/rtl.f \
      dv/config/qwen397b_iq1s_1cu.json \
      dv/cocotb/iq1s_layer_persistent
    git commit -m "feat: add IQ1S persistent command plane"

### Task 10: Implement native IQ1_S decode and scale reconstruction

**Files:**
- Create: rtl/iq1s_block_decoder.sv
- Create: rtl/iq1s_scale_reconstruct.sv
- Create: rtl/iq1s_result_writer.sv
- Create: rtl/mem/IQ1S_GRID.memh
- Create: rtl/mem/IQ1S_GRID.sha256
- Modify: zluda/src/impl/iq1s_tmatmul.rs
- Modify: dv/cocotb/iq1s_layer_persistent/test_iq1s_layer_persistent.py

- [ ] **Step 1: Add a deterministic Rust RTL-vector emitter test**

Under explicit HETGPU_RTL_VECTOR_OUT, HETGPU_IQ1S_GRID_OUT, and
HETGPU_IQ1S_GRID_SHA256_OUT paths, emit JSON fixtures plus the complete
2048-by-8 signed grid table in memh form and its SHA-256. Fixtures contain the
50 input bytes, all 8 groups, odd_scale,
delta_sign, four grid indices, 32 grid values, Q8_1 values/scales, raw
grid/delta dots, and reconstructed FP32 bits. With the variables unset, the
test only validates fixtures and the grid in memory.

- [ ] **Step 2: Generate fixtures and write failing decoder tests**

Run:

    export HETGPU_IQ1S_GRID_OUT=/home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/rtl/mem/IQ1S_GRID.memh
    export HETGPU_IQ1S_GRID_SHA256_OUT=/home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/rtl/mem/IQ1S_GRID.sha256
    HETGPU_RTL_VECTOR_OUT=/tmp/qwen-iq1s-rtl-vectors.json \
      CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation emit_iq1s_rtl_vectors -- --ignored --nocapture

Make cocotb load that file. Cover all odd scales 1..15, both delta signs, low
and high grid-index bits, zero and extreme signed Q8 values, and non-finite
scale rejection.

- [ ] **Step 3: Run and verify the failure**

Run:

    HETGPU_IQ1S_RTL_VECTORS=/tmp/qwen-iq1s-rtl-vectors.json \
      make -C dv/cocotb/iq1s_layer_persistent SIM=verilator \
      TARGET=dv/config/qwen397b_iq1s_1cu.json

Expected: FAIL because the decoder and reconstruction pipeline are absent.

- [ ] **Step 4: Implement the 50-byte decoder**

For each group g:

    qh = little_endian_u16(block[34 + 2*g : 36 + 2*g])
    odd_scale = (((qh >> 12) & 7) * 2) + 1
    delta_sign = (qh & 0x8000) ? -1 : 1
    grid_index[p] = block[2 + 4*g + p] | (((qh >> (3*p)) & 7) << 8)

Read the 2048-by-8 signed grid table from IQ1S_GRID.memh. Stream grid and delta
symbols directly to the existing TernIP input path; never expose an expanded
DDR address.

- [ ] **Step 5: Implement bit-accurate reconstruction**

Match iq1s_tmatmul.rs:

    d1q = iq1s_d * odd_scale
    wd = round_to_fp16_then_fp32(d1q)
    delta_factor = -1.0 + delta_sign * 0.125
    wdelta = round_to_fp16_then_fp32(d1q * delta_factor)
    unsigned_grid = grid_dot + delta_sign * delta_dot
    result = wd * q8_d * unsigned_grid + wdelta * q8_s

Integer dots are exact. FP operations are IEEE single precision with the two
explicit FP16 rounding points above. Non-finite input or output latches a
decoder fault.

- [ ] **Step 6: Pass exhaustive fixture and backpressure tests**

Run:

    HETGPU_IQ1S_RTL_VECTORS=/tmp/qwen-iq1s-rtl-vectors.json \
      make -C dv/cocotb/iq1s_layer_persistent SIM=verilator \
      TARGET=dv/config/qwen397b_iq1s_1cu.json
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s_tmatmul -- --nocapture

Expected: every reconstructed FP32 result is within atol 1e-4 and rtol 1e-3,
and all injected stalls preserve counts and order.

- [ ] **Step 7: Commit both repositories**

Software:

    git add zluda/src/impl/iq1s_tmatmul.rs
    git commit -m "test: export native IQ1S RTL vectors"

Hardware:

    git add rtl/iq1s_block_decoder.sv rtl/iq1s_scale_reconstruct.sv \
      rtl/iq1s_result_writer.sv rtl/mem \
      dv/cocotb/iq1s_layer_persistent/test_iq1s_layer_persistent.py
    git commit -m "feat: decode native IQ1S weights in RTL"

### Task 11: Integrate the persistent CU and descriptor-driven AU250 build

**Files:**
- Create: rtl/axi_iq1s_layer_persistent.sv
- Create: rtl/axi_iq1s_layer_persistent_vivado_bd_wrapper.sv
- Create: synth/pynqvivado_au250/iq1s_bd.tcl
- Create: synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json
- Modify: sw_utils/lib/target.py
- Modify: sw_utils/target/resolve_target.py
- Create: sw_utils/tests/test_target.py
- Modify: synth/pynqvivado_common/create_bd_xpr.tcl
- Modify: synth/pynqvivado_common/generate_kernel_xml.tcl
- Modify: synth/pynqvivado_common/package_ip.tcl
- Modify: synth/pynqvivado_common/generate_kernel_cfg.tcl
- Modify: Makefile

- [ ] **Step 1: Add failing target-resolver tests**

Extend Target tests to require:

    target.kernel_kind("iq1s_layer_big") == "iq1s_persistent"
    target.bd_script("iq1s_layer_big") == "iq1s_bd.tcl"
    target.kernel_abi("iq1s_layer_big") == "iq1s_layer_v2"
    target.kernel_count_spec() == "iq1s_layer_big:3 iq1s_layer_small:1"

Also assert memory groups/SLRs are 0/3/2/1 and any target with more than four
instances is rejected.

- [ ] **Step 2: Create the AU250 target**

Start from the upstream MaxCores 370M descriptor, use D=1024, three big
instances and one small instance, and set:

    "kernel_kind": "iq1s_persistent"
    "bd_script": "iq1s_bd.tcl"
    "kernel_abi": "iq1s_layer_v2"

Begin with big BatchSize 9 and small BatchSize 6. Later tuning may reduce these
values but may not exceed the post-route resource/timing-qualified values.

- [ ] **Step 3: Integrate one persistent CU**

axi_iq1s_layer_persistent instantiates the command path, decoder, existing
TernIP datapath, reconstruction, result path, completion path, and first-fault
latch. It exposes one AXI4 master for the bank and one 32-bit AXI-Lite slave for
the generated register map.

iq1s_bd.tcl connects that single master to M_AXI_0 and the control slave
directly to S_AXI. It has no instruction DMA and no per-command XRT launch
controller.

- [ ] **Step 4: Generate persistent kernel metadata**

For kernel_abi iq1s_layer_v2, kernel.xml declares user_managed control and the
six bank-local pointer arguments for command ring, completion ring, program,
arena manifest, activation slab, and result slab. generate_kernel_cfg.tcl keeps
the existing one-CU-per-DDR/SLR mapping.

- [ ] **Step 5: Enforce the 32-thread ceiling**

Set:

    VPP_JOBS ?= 32

Reject zero or values above 32 in Makefile. Add to both v++ link commands:

    --jobs $(VPP_JOBS)
    --vivado.synth.jobs $(VPP_JOBS)
    --vivado.impl.jobs $(VPP_JOBS)

Add set_param general.maxThreads 32 to every Vivado entry Tcl used by this
target.

- [ ] **Step 6: Run resolver, lint, and cocotb gates**

Run:

    python3 -m pytest -q sw_utils/tests/test_target.py
    PYTHONPATH=.:sw_utils python3 -m sw_utils resolve_target \
      --plan synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      pynqvivado_au250
    make lint TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      KERNEL=iq1s_layer_big
    make -C dv/cocotb/iq1s_layer_persistent SIM=verilator \
      TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      KERNEL=iq1s_layer_big

Expected: PASS and the resolved plan lists exactly four CUs on four banks.

- [ ] **Step 7: Commit**

    git add rtl/axi_iq1s_layer_persistent.sv \
      rtl/axi_iq1s_layer_persistent_vivado_bd_wrapper.sv \
      synth/pynqvivado_au250/iq1s_bd.tcl \
      synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      sw_utils/lib/target.py sw_utils/target/resolve_target.py \
      sw_utils/tests/test_target.py \
      synth/pynqvivado_common/create_bd_xpr.tcl \
      synth/pynqvivado_common/generate_kernel_xml.tcl \
      synth/pynqvivado_common/package_ip.tcl \
      synth/pynqvivado_common/generate_kernel_cfg.tcl Makefile
    git commit -m "feat: integrate persistent IQ1S AU250 target"

### Task 12: Build and validate hardware emulation and the physical xclbin

**Files:**
- Create: sw_utils/target/test_iq1s_persistent.py
- Modify: Makefile
- Proof output: hardware build logs and xclbin metadata outside git

- [ ] **Step 1: Add the hw_emu host test**

The test allocates command/completion/program/activation/result BOs, writes one
valid gate descriptor, rings the doorbell, waits with a fixed timeout, and
checks the completion identity and finite result. A second descriptor verifies
ring reuse without re-opening the kernel.

Makefile selects test_iq1s_persistent for kernel_kind iq1s_persistent and keeps
test_pynqvivado for the legacy ternip targets.

- [ ] **Step 2: Re-run RTL gates before every Vitis build**

Run:

    make lint TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      KERNEL=iq1s_layer_big
    make -C dv/cocotb/iq1s_layer_persistent SIM=verilator \
      TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      KERNEL=iq1s_layer_big

Expected: PASS.

- [ ] **Step 3: Build and run hardware emulation**

Run:

    source /mnt/disk0/2026.1/settings64.sh
    VPP_JOBS=32 make pynqvivado_au250_hw_emu \
      TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      MODEL=MMfreeLM-370M

Expected: v++ exits zero and test_iq1s_persistent completes both descriptors.
Do not accept a build whose host test was skipped.

- [ ] **Step 4: Commit the emulation host test**

    git add sw_utils/target/test_iq1s_persistent.py Makefile
    git commit -m "test: exercise persistent IQ1S hardware emulation"

- [ ] **Step 5: Build the physical xclbin**

Run:

    source /mnt/disk0/2026.1/settings64.sh
    set -o pipefail
    VPP_JOBS=32 make pynqvivado_au250_hw \
      TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      MODEL=MMfreeLM-370M 2>&1 | tee /tmp/qwen397b-iq1s-u250-vpp.log

Expected: v++ and the post-build host smoke both exit zero.

- [ ] **Step 6: Verify timing, metadata, and hashes**

Run:

    xclbinutil --info --input /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin \
      > /tmp/qwen397b-iq1s-u250-xclbin-info.txt
    sha256sum /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin
    rg "iq1s_layer_big|iq1s_layer_small|bank0|bank1|bank2|bank3" \
      /tmp/qwen397b-iq1s-u250-xclbin-info.txt
    rg "Timing constraints are met|Slack" /tmp/qwen397b-iq1s-u250-vpp.log

Expected: three big instances, one small instance, all four DDR banks, a
nonzero SHA-256, and passing timing. Resolve the concrete xclbin path using:

    PYTHONPATH=.:sw_utils python3 -m sw_utils resolve_target \
      --plan synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      pynqvivado_au250

### Task 13: Connect phase commits to arenas, traces, and persistent XRT

**Files:**
- Modify: zluda/src/impl/iq1s_layer.rs
- Modify: zluda/src/impl/iq1s_xrt.rs
- Modify: zluda/src/impl/xrt_iq1s_persistent.rs
- Modify: tools/build_au250_qwen35_runtime.sh
- Modify: tools/run_qwen35_iq1s_au250_hybrid.sh
- Modify: zluda/tests/test_au250_qwen35_runtime_static.sh

- [ ] **Step 1: Add failing integration tests**

Use fake CUDA copies and fake XRT to prove:

- Phase A emits one four-CU submission set for all gate/up assignments;
- GPU-native down closes without Phase B;
- IQ1_S down emits one second four-CU submission set;
- outputs become visible only after all four completions;
- measurement mode rejects any arena sync;
- attention remains classified GPU-native;
- handwritten and compiler modes use the same ResidentArena generation.

- [ ] **Step 2: Run and verify the failure**

Run:

    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation qwen_iq1s_layer_integration -- --nocapture

Expected: FAIL because commits do not call the persistent executor.

- [ ] **Step 3: Wire Phase A and Phase B**

At Phase A commit:

    let plan = LayerPhasePlan::phase_a(transaction, arena.manifest())?;
    let program = compiler.compile(mode, &plan)?;
    let done = pool.submit_phase(&program, &plan.activations)?;
    publish_phase_a_to_cuda(done)?;

At layer commit, execute the same sequence for IQ1_S down or validate the
GPU-native down marker. Remove no legacy Kimi entry point.

- [ ] **Step 4: Bind runtime artifacts and hashes**

The build manifest records the v2 ABI schema hash. The runner requires the new
xclbin filename, SHA-256, kernel names, and memory map. It preloads the arena
before warm-up and calls measurement_begin only after preload and one complete
warm-up.

- [ ] **Step 5: Run all focused software gates**

Run:

    python3 -m pytest -q \
      zluda/tests/test_qwen35_layer_abi_generator.py \
      zluda/tests/test_qwen35_au250_eval.py
    bash zluda/tests/test_au250_qwen35_runtime_static.sh
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s -- --nocapture
    QWEN35_BUILD_JOBS=32 tools/build_au250_qwen35_runtime.sh

Expected: PASS and the build log reports no more than 32 jobs.

- [ ] **Step 6: Commit**

    git add zluda/src/impl/iq1s_layer.rs zluda/src/impl/iq1s_xrt.rs \
      zluda/src/impl/xrt_iq1s_persistent.rs \
      tools/build_au250_qwen35_runtime.sh \
      tools/run_qwen35_iq1s_au250_hybrid.sh \
      zluda/tests/test_au250_qwen35_runtime_static.sh
    git commit -m "feat: execute Qwen FFN as persistent layers"

### Task 14: Add correctness-gated tuning, proof validation, and plotting

**Files:**
- Create: tools/tune_qwen35_iq1s_persistent.py
- Create: tools/plot_qwen35_iq1s_persistent.py
- Create: zluda/tests/test_qwen35_iq1s_persistent_proof.py
- Modify: tools/qwen35_au250_eval.py
- Modify: zluda/tests/validate_qwen35_iq1s_au250_proof.py
- Modify: zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
- Modify: zluda/tests/test_qwen35_au250_eval.py

- [ ] **Step 1: Add failing proof mutation tests**

Start from one valid synthetic proof bundle and independently mutate:

- xclbin hash or kernel count;
- arena byte count, bank overflow, or resident hash;
- measured-window weight DMA;
- persistent-start count greater than four;
- more than two phase fences in a layer;
- missing CU completion or duplicate transaction;
- attention routed to U250;
- non-IQ1_S routed to U250;
- FFN tolerance or token equality;
- incomplete 2,048-token workload;
- a hybrid pass below 15 tok/s.

The last mutation must produce a completed-but-performance-failed result, not a
reportable pass.

- [ ] **Step 2: Run and verify the failure**

Run:

    python3 -m pytest -q \
      zluda/tests/test_qwen35_iq1s_persistent_proof.py \
      zluda/tests/test_validate_qwen35_iq1s_au250_proof.py \
      zluda/tests/test_qwen35_au250_eval.py

Expected: FAIL because persistent proof fields are not validated.

- [ ] **Step 3: Implement the fixed knob search**

Search only:

    row_shards
    ring_depth
    dma_coalesce_bytes
    descriptors_per_doorbell
    gate_up_order
    gpu_overlap
    completion_backoff_us

Each candidate first runs a deterministic sampled-layer comparison. Reject it
on any token, tolerance, route, residency, or hardware fault. Rank remaining
candidates by completed E2E TPS and write the frozen winner with a SHA-256 over
the canonical JSON.

Do not include attention placement; force flash attention on GPU.

- [ ] **Step 4: Extend evaluator and validator**

Require primary records for four persistent starts, all 141 resident tensors,
59,139,686,400 raw bytes, zero measured weight DMA, at most two phase fences,
all four CUs, exact tokens, numerical tolerance, and 2,048 completions.

Compute:

    tps = 2048 / (last_completion_ns - first_enqueue_ns) * 1_000_000_000

Report every pass plus minimum, median, and mean. An incomplete run is
NOT_REPORTABLE. A complete run below 15 tok/s is BELOW_TARGET.

- [ ] **Step 5: Generate the architecture/performance figure**

plot_qwen35_iq1s_persistent.py reads only validated JSON and writes PNG and SVG
showing CUDA vs handwritten/compiler TPS, per-CU utilization, phase latency,
DMA bytes, and resident hit rate. It labels incomplete or below-target runs
without substituting zero.

- [ ] **Step 6: Run proof tests**

Run:

    python3 -m pytest -q \
      zluda/tests/test_qwen35_iq1s_persistent_proof.py \
      zluda/tests/test_validate_qwen35_iq1s_au250_proof.py \
      zluda/tests/test_qwen35_au250_eval.py

Expected: PASS for the valid fixture and exact fail-closed errors for every
mutation.

- [ ] **Step 7: Commit**

    git add tools/tune_qwen35_iq1s_persistent.py \
      tools/plot_qwen35_iq1s_persistent.py tools/qwen35_au250_eval.py \
      zluda/tests/validate_qwen35_iq1s_au250_proof.py \
      zluda/tests/test_qwen35_iq1s_persistent_proof.py \
      zluda/tests/test_validate_qwen35_iq1s_au250_proof.py \
      zluda/tests/test_qwen35_au250_eval.py
    git commit -m "test: qualify persistent Qwen hybrid throughput"

### Task 15: Run live qualification, tune, and report E2E TPS

**Files:**
- Create: zluda/evaluation/2026-09-02-qwen397b-u250-layer-persistent.md
- Proof output: .proof/qwen35-iq1s-persistent-final/

- [ ] **Step 1: Run fail-closed live preflight**

Run:

    sudo xrt-smi examine --device 0000:64:00.1
    sudo xbmgmt examine --device 0000:64:00.0
    nvidia-smi
    sha256sum /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf
    xclbinutil --info --input /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin

Expected: healthy U250, healthy GPU, exact model hash, and exactly three
iq1s_layer_big plus one iq1s_layer_small CU on four banks.

- [ ] **Step 2: Run one-layer numerical smoke**

Run the runner with a single sampled layer, both trace modes, compare-max set
to cover all four CUs, and no throughput claim:

    QWEN35_BUILD_JOBS=32 tools/run_qwen35_iq1s_au250_hybrid.sh \
      --xclbin /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin \
      --smoke-layer 0 --trace-mode handwritten
    QWEN35_BUILD_JOBS=32 tools/run_qwen35_iq1s_au250_hybrid.sh \
      --xclbin /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin \
      --smoke-layer 0 --trace-mode compiler

Expected: both modes complete, all four CUs report work, and sampled output
passes atol 1e-4 and rtol 1e-3.

- [ ] **Step 3: Preload all weights and prove steady-state residency**

Run:

    QWEN35_BUILD_JOBS=32 tools/run_qwen35_iq1s_au250_hybrid.sh \
      --xclbin /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin \
      --preload-only

Expected: 141 tensors, 59,139,686,400 raw bytes, four in-capacity banks, all
hashes valid, U250 remains healthy, and the second residency probe transfers
zero weight bytes.

- [ ] **Step 4: Run the bounded knob search**

Run:

    python3 tools/tune_qwen35_iq1s_persistent.py \
      --model /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf \
      --xclbin /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin \
      --requests 64 --active 16 --tokens 32 --context 512 \
      --trace-modes handwritten,compiler \
      --output .proof/qwen35-iq1s-persistent-tuning

Expected: every retained candidate passes correctness, rejected candidates
retain reasons, and frozen-config.json has a reproducible hash.

- [ ] **Step 5: Run the full fixed E2E qualification**

Run:

    QWEN35_BUILD_JOBS=32 tools/run_qwen35_iq1s_au250_hybrid.sh \
      --xclbin /home/victoryang00/hetGPU/.worktrees/ternary-qwen-iq1s-persistent-20260902/synth/pynqvivado_au250/build/pynqvivado_au250_Qwen397B_IQ1S/hw/kernel.xclbin \
      --frozen-config .proof/qwen35-iq1s-persistent-tuning/frozen-config.json \
      --requests 64 --active 16 --tokens 32 --context 512 \
      --passes 3 --modes cuda,handwritten,compiler \
      --proof-dir .proof/qwen35-iq1s-persistent-final

Expected: CUDA plus three handwritten and three compiler passes each complete
2,048 generated tokens. Every hybrid pass must be at least 15 aggregate tok/s
to pass the performance gate.

- [ ] **Step 6: Validate proof and generate the figure**

Run:

    python3 zluda/tests/validate_qwen35_iq1s_au250_proof.py \
      .proof/qwen35-iq1s-persistent-final
    python3 tools/plot_qwen35_iq1s_persistent.py \
      --proof .proof/qwen35-iq1s-persistent-final \
      --png .proof/qwen35-iq1s-persistent-final/e2e.png \
      --svg .proof/qwen35-iq1s-persistent-final/e2e.svg

Expected: validator PASS only if every fixed gate passes. Otherwise preserve
the bundle and report the exact NOT_REPORTABLE or BELOW_TARGET boundary.

- [ ] **Step 7: Generate and commit the evidence-backed report**

Render the evaluation Markdown from validated proof fields. It must state every
pass, minimum, median, mean, exact token equality, numerical error, resident
bytes, measured weight DMA, phase fences, four-CU utilization, xclbin hash, and
whether the 15 tok/s gate passed.

Run:

    git add zluda/evaluation/2026-09-02-qwen397b-u250-layer-persistent.md
    git commit -m "docs: report Qwen persistent hybrid E2E TPS"

Do not commit .proof artifacts unless the user explicitly requests it.

## Final verification

After all implementation commits, invoke superpowers:verification-before-completion
and run:

    cd /home/victoryang00/hetGPU/.worktrees/qwen35-tq1-au250-20260826
    git status --short
    python3 -m pytest -q \
      zluda/tests/test_qwen35_layer_abi_generator.py \
      zluda/tests/test_qwen35_iq1s_persistent_proof.py \
      zluda/tests/test_validate_qwen35_iq1s_au250_proof.py \
      zluda/tests/test_qwen35_au250_eval.py
    bash zluda/tests/test_prepare_au250_qwen35_source.sh
    bash zluda/tests/test_au250_qwen35_runtime_static.sh
    CARGO_BUILD_JOBS=32 cargo test -p ptx tmatmul_algorithm_tree -- --nocapture
    CARGO_BUILD_JOBS=32 cargo test -p zluda --no-default-features \
      --features nvidia,evaluation iq1s -- --nocapture
    python3 zluda/tests/validate_qwen35_iq1s_au250_proof.py \
      .proof/qwen35-iq1s-persistent-final

Then in the hardware worktree:

    make lint TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      KERNEL=iq1s_layer_big
    HETGPU_IQ1S_RTL_VECTORS=/tmp/qwen-iq1s-rtl-vectors.json \
      make -C dv/cocotb/iq1s_layer_persistent SIM=verilator \
      TARGET=synth/pynqvivado_au250/targets/pynqvivado_au250_Qwen397B_IQ1S.json \
      KERNEL=iq1s_layer_big
    git status --short

Completion requires clean intended source state in both worktrees, preserved
pre-existing user changes in the software worktree, a passing proof validator,
and six completed hybrid passes at or above 15 aggregate tok/s. If hardware
measures below target, report the completed measured TPS and keep the goal
open; do not relabel it as passing.
