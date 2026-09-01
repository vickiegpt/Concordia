# Qwen3.5-397B IQ1_S Compiler Traces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the existing Kimi 2.6 IQ1_S CUDA-launch interception path for the fixed Qwen3.5-397B-A17B mixed-quant GGUF, execute all and only its 141 IQ1_S routed-expert tensors on four U250 compute units through handwritten and `AlgorithmTree` compiler traces, and produce a fail-closed CUDA-equivalence and continuous-batch throughput proof.

**Architecture:** llama.cpp registers immutable IQ1_S GGUF tensor metadata and binds each live CUDA weight pointer to that identity, but execution remains selected at intercepted CUDA MMVQ/MMQ launches. Qualified launches flow through one validated semantic contract, either the handwritten or compiler trace builder, one host packed-tile cache, one bank-aware resident U250 weight cache, the existing assembler, and one persistent four-CU XRT executor. Attention and the 39 IQ2_XXS/IQ3_S/MXFP4 expert tensors remain GPU-native. Every strict-mode transition is evidence-bound and aborts on ambiguity, fallback after eligibility, cache/program mismatch, incomplete XRT completion, or correctness drift.

**Tech Stack:** Rust 2021, CUDA 13 launch interception, pinned Qwen llama.cpp/BitNet build, `ptx::pass::tmatmul_algorithm_tree`, repository tmatmul assembler, Xilinx XRT, Alveo U250, Python 3, Bash, pytest, Cargo tests, JSONL proof records.

---

## Fixed acceptance contract

- Model: `/root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf`, exactly `94,155,830,880` bytes, SHA-256 `0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568`.
- Routed-expert tensors: exactly 141 IQ1_S on U250; 24 IQ2_XXS, 4 IQ3_S, and 11 MXFP4 on GPU; zero literal TQ1_0.
- GPU owns attention, DeltaNet, KV/recurrent state, embeddings, routing, normalization, sampling, and every non-qualified operation.
- U250 topology: four CUs with lane counts 9/9/9/6 and memory groups 0/3/2/1 using `MaxCores_370M.xclbin`.
- Compiler arithmetic limit: 262,144 tokens. Benchmark context: 512 tokens.
- Workload: 64 requests, maximum 16 active requests, exactly 32 generated tokens per request, greedy deterministic decoding.
- Correctness: identical token IDs across CUDA/handwritten/compiler and sampled FFN tolerance `atol=1e-4`, `rtol=1e-3`.
- Performance: after one warm-up, each of three measured passes in each hybrid mode must reach at least 15 aggregate generated tokens/s.

---

## File map

- Modify `tools/build_au250_qwen35_runtime.sh`: include `libggml.so` in the immutable build manifest.
- Create `tools/qwen35_build_preflight.py`: verify manifest artifacts, canonical build-root containment, hashes, and required IQ1_S symbols before model load.
- Modify `tools/run_qwen35_iq1s_au250_hybrid.sh`: invoke strict preflight; run CUDA, handwritten, and compiler modes with the fixed workload.
- Create `zluda/tests/test_qwen35_build_preflight.py`: reject missing, escaped, changed, or symbol-incomplete `libggml.so` artifacts.
- Modify `tools/qwen35-tq1-bridge.h`: add versioned IQ1_S registration and live-pointer binding records without changing the TQ1 ABI.
- Modify `tools/llama-qwen35-tq1-hetgpu.patch`: register all 141 IQ1_S expert tensors and bind their CUDA pointers immediately before eligible kernel launch.
- Create `zluda/src/impl/iq1s_weight_registry.rs`: validate IQ1_S GGUF ranges, roles, shapes, per-tensor SHA-256 identities, CUDA pointer generations, and lookups.
- Modify `zluda/src/lib.rs` and `zluda/src/impl/mod.rs`: export and register the IQ1_S ABI entry points/module.
- Modify `zluda/Cargo.toml`: add the SHA-256 dependency used for immutable weight identities.
- Modify `zluda/src/impl/function.rs`: finish explicit CUDA 13 Qwen IQ1_S MMVQ/MMQ decoders and route only validated, registered expert launches.
- Modify `zluda/src/impl/bitnet_disagg.rs`: preserve attention/GPU precedence and fail closed after IQ1_S eligibility.
- Modify `zluda/src/impl/iq1s_tmatmul.rs`: consume registered weight identity, remove strict-Qwen Kimi library fallback, and expose a normalized launch contract.
- Modify `ptx/src/pass/tmatmul_algorithm_tree.rs`: add checked, non-panicking runtime construction and assembly generation APIs.
- Create `zluda/src/impl/iq1s_trace_compiler.rs`: define handwritten/compiler trace modes, lower real IQ1_S tile operations through `AlgorithmTree`, and compute trace/assembly identities.
- Modify `zluda/src/impl/iq1s_xrt.rs`: use the normalized trace contract, shared cache keys, and expanded route/execution evidence.
- Modify `zluda/src/impl/xrt_tmatmul.rs`: add per-bank resident matrix BOs, bound program BO caching, and trace-to-program completion validation.
- Modify `tools/qwen35_au250_eval.py`: issue the 64-request/16-active/32-token continuous batch in all three modes and collect correctness/performance evidence.
- Modify `zluda/tests/run_au250_xrt_iq1s.sh`: execute and validate the standalone four-CU fixture in either handwritten or compiler mode.
- Modify `zluda/tests/validate_qwen35_iq1s_au250_proof.py`: require three-mode equality, trace binding, cache reuse, four-CU work, and throughput gates.
- Modify `zluda/tests/test_validate_qwen35_iq1s_au250_proof.py`, `zluda/tests/test_qwen35_au250_eval.py`, `zluda/tests/test_au250_qwen35_runtime_static.sh`, and `zluda/tests/test_prepare_au250_qwen35_source.sh`: cover the new contracts and mutation failures.
- Create `zluda/evaluation/2026-09-01-qwen397b-iq1s-compiler-traces.md`: generate the final result only from a passing proof bundle.

The worktree already contains relevant uncommitted Qwen fixes. Every commit below must stage only the named files, inspect `git diff --cached`, and preserve `.proof/` plus unrelated dirty changes.

### Task 1: Make Qwen `libggml.so` selection deterministic and fail closed

**Files:**
- Modify: `tools/build_au250_qwen35_runtime.sh`
- Create: `tools/qwen35_build_preflight.py`
- Create: `zluda/tests/test_qwen35_build_preflight.py`
- Modify: `tools/run_qwen35_iq1s_au250_hybrid.sh`
- Modify: `zluda/tests/test_au250_qwen35_runtime_static.sh`
- Modify: `zluda/src/impl/iq1s_tmatmul.rs`

- [ ] **Step 1: Add failing preflight tests**

Build temporary manifest/build-root fixtures and test: valid regular `libggml.so`; missing manifest entry; wrong SHA-256; canonical path outside the build root; symlink escape; directory instead of file; missing `dequantize_row_iq1_s`, `ggml_init`, or `ggml_free`; and schema/revision mismatch. Assert failure occurs without invoking the server.

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_build_preflight.py
```

Expected: FAIL because `tools/qwen35_build_preflight.py` does not exist.

- [ ] **Step 2: Add `libggml` to the build manifest**

In `tools/build_au250_qwen35_runtime.sh`, resolve `/qwen-build/llama-build/bin/libggml.so`, require a nonempty regular file, hash it, and write an `artifacts.libggml` record next to `llama_server` and `libnvcuda`. Keep manifest schema 1 because the artifact addition is backward-compatible, and update Qwen consumers to require the new entry.

- [ ] **Step 3: Implement the strict preflight helper**

`qwen35_build_preflight.py` must accept `--manifest`, `--build-root`, `--llama-revision`, and `--output`. It must canonicalize both root and artifact, require `artifact.is_file()`, use `Path.is_relative_to(canonical_root)`, compare the manifest SHA-256 to a fresh digest, load the exact library with `ctypes.CDLL`, resolve the three required symbols, and atomically write the verified canonical path and hash. It must never consult the hard-coded Kimi path.

- [ ] **Step 4: Wire the runner and strict decoder**

Invoke the helper before GPU-memory preflight and before starting the 94 GB model load. Export `HETGPU_LIBGGML` from the generated verification record, export the audited model hash as `HETGPU_QWEN_MODEL_SHA256`, and include both in `artifact-hashes.txt` and proof environment. In `validated_grid`, require `HETGPU_LIBGGML` whenever `HETGPU_QWEN_IQ1S_STRICT=1`; retain `DEFAULT_LIBGGML` only for non-Qwen/Kimi execution.

- [ ] **Step 5: Verify focused behavior and commit**

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_build_preflight.py
bash zluda/tests/test_au250_qwen35_runtime_static.sh
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_tmatmul -- --nocapture
```

Expected: PASS, including exact rejection text for an unset strict-Qwen library.

Commit:

```bash
git add tools/build_au250_qwen35_runtime.sh tools/qwen35_build_preflight.py \
  tools/run_qwen35_iq1s_au250_hybrid.sh zluda/tests/test_qwen35_build_preflight.py \
  zluda/tests/test_au250_qwen35_runtime_static.sh zluda/src/impl/iq1s_tmatmul.rs
git commit -m "fix: bind Qwen IQ1S decoder to verified libggml"
```

### Task 2: Register the actual 141 IQ1_S Qwen expert tensors

**Files:**
- Modify: `tools/qwen35-tq1-bridge.h`
- Modify: `tools/llama-qwen35-tq1-hetgpu.patch`
- Create: `zluda/src/impl/iq1s_weight_registry.rs`
- Modify: `zluda/src/lib.rs`
- Modify: `zluda/src/impl/mod.rs`
- Modify: `zluda/Cargo.toml`
- Modify: `zluda/tests/test_prepare_au250_qwen35_source.sh`
- Modify: `zluda/tests/test_au250_qwen35_runtime_static.sh`

- [ ] **Step 1: Write Rust registry and source-overlay tests first**

The Rust tests must accept a valid `blk.N.ffn_{gate,up,down,gate_up}_exps.weight` IQ1_S record and reject TQ1_0, IQ2_XXS, invalid names, role mismatch, short file spans, shape/stride overflow, changed inode/mtime/content, duplicate conflicting registration, null pointer binding, overlapping live pointer spans, stale allocation generation, and a zero content hash. Static overlay tests must require both IQ1_S registration and live pointer binding calls.

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_weight_registry -- --nocapture
bash zluda/tests/test_prepare_au250_qwen35_source.sh
```

Expected: FAIL because the module and IQ1_S ABI are absent.

- [ ] **Step 2: Add a separate versioned IQ1_S ABI**

Keep `hetgpu_tq1_*_v1` unchanged. Add `hetgpu_iq1s_register_tensor_v1` for canonical file path, offset, byte span, tensor name, GGML type 19, role, dimensions, and strides; add `hetgpu_iq1s_bind_device_v1` for tensor name, live CUDA base pointer, allocated span, and allocation generation. Export both through `zluda/src/lib.rs` with panic containment and fail-closed return codes.

- [ ] **Step 3: Implement immutable registry identity**

The registry must canonicalize and open the GGUF read-only, validate checked layout/range arithmetic, stream SHA-256 over the exact tensor byte range once, and store model/tensor identity. A live binding maps a nonoverlapping CUDA allocation range and generation to that identity. Lookup must require the intercepted matrix extent to lie wholly within one current binding and return tensor name, layer, role, expert coordinate, model identity, allocation generation, and the full 32-byte tensor hash.

- [ ] **Step 4: Patch llama.cpp at metadata and dispatch boundaries**

During model loading, register every routed-expert tensor whose type is exactly `GGML_TYPE_IQ1_S`; the fixed audit later proves there are 141. Immediately before the native IQ1_S MMVQ/MMQ launch, bind `src0->data` and its backend allocation span to the registered name. The hook annotates identity only: it does not execute the operation or bypass CUDA launch interception. Non-IQ1_S tensors make no IQ1_S calls.

- [ ] **Step 5: Verify ABI layout, registration count behavior, and commit**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_weight_registry -- --nocapture
bash zluda/tests/test_prepare_au250_qwen35_source.sh
bash zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: PASS; generated overlay contains the two new hooks and retains the old TQ1 hooks.

Commit:

```bash
git add tools/qwen35-tq1-bridge.h tools/llama-qwen35-tq1-hetgpu.patch \
  zluda/src/impl/iq1s_weight_registry.rs zluda/src/lib.rs zluda/src/impl/mod.rs \
  zluda/Cargo.toml Cargo.lock zluda/tests/test_prepare_au250_qwen35_source.sh \
  zluda/tests/test_au250_qwen35_runtime_static.sh
git commit -m "feat: register Qwen IQ1S expert weight identities"
```

### Task 3: Finish exact CUDA 13 launch decoding and strict Qwen routing

**Files:**
- Modify: `zluda/src/impl/function.rs`
- Modify: `zluda/src/impl/bitnet_disagg.rs`
- Modify: `zluda/src/impl/iq1s_tmatmul.rs`
- Modify: `zluda/tests/test_au250_qwen35_runtime_static.sh`

- [ ] **Step 1: Add decoder/routing red tests around the observed Qwen kernels**

Cover each supported modern CUDA 13 `ggml_type19` MMVQ/MMQ argument layout, vector and batched shapes, row/K strides, batch widths 1 through 16, pointer bounds, attention-marker precedence, stream-K/fixup exclusion, registered expert identity, all 39 non-IQ1_S GPU cases, unknown template variants, and the distinction between unqualified GPU-native and eligible-but-failed fatal routing.

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation nvidia_plan_modern_iq1s -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation bitnet_disagg -- --nocapture
```

Expected: at least the identity and strict post-eligibility assertions FAIL.

- [ ] **Step 2: Define one normalized launch contract**

Extend `LogicalLaunch` or introduce `ValidatedIq1sLaunch` containing kernel family, ABI revision, rows, K, active batch, all physical strides/extents, CUDA stream, matrix/input/output pointers, tensor/layer/role/expert identity, allocation generation, and full content hash. Construction must use checked arithmetic and the registry lookup from Task 2. Remove the current `DefaultHasher` fallback; strict Qwen rejects `[0; 32]`.

- [ ] **Step 3: Route on contract validity, not mangled-name substring alone**

Apply GPU markers first. Decode only an enumerated supported ABI. Resolve the weight binding and prove it is one of the audited IQ1_S routed experts. Once this succeeds, any capture/XRT/compiler/publication error returns a CUDA launch error and terminates strict mode; it must not invoke the original GPU kernel. Unsupported or non-IQ1_S launches remain GPU-native and are logged as unqualified.

- [ ] **Step 4: Run launch, routing, and Kimi regressions and commit**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation function::tests -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation bitnet_disagg -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_tmatmul -- --nocapture
bash zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: PASS with explicit GPU-native records for IQ2_XXS/IQ3_S/MXFP4 and no permissive fallback after eligibility.

Commit:

```bash
git add zluda/src/impl/function.rs zluda/src/impl/bitnet_disagg.rs \
  zluda/src/impl/iq1s_tmatmul.rs zluda/tests/test_au250_qwen35_runtime_static.sh
git commit -m "fix: qualify Qwen CUDA13 IQ1S launches exactly"
```

### Task 4: Add checked `AlgorithmTree` runtime APIs and the two trace builders

**Files:**
- Modify: `ptx/src/pass/tmatmul_algorithm_tree.rs`
- Create: `zluda/src/impl/iq1s_trace_compiler.rs`
- Modify: `zluda/src/impl/mod.rs`
- Modify: `zluda/src/impl/iq1s_xrt.rs`

- [ ] **Step 1: Add compiler contract tests before implementation**

Create fixtures for one real 64x64 IQ1_S component tile, a K-tiled projection, active batches 1 and 16, lane capacities 9/9/9/6, vector-register counts 2 and 4, and the 262,144-token model limit. Assert handwritten and compiler traces have identical ordered semantic coverage, terminal `stall`, real matrix/input/output labels, no unresolved labels, and assembly accepted by the repository assembler. Add overflow, dependency-cycle, register-exhaustion, zero-dimension, token-limit-plus-one, and incomplete-component rejection tests.

Run:

```bash
cargo test -p ptx tmatmul_algorithm_tree -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_trace_compiler -- --nocapture
```

Expected: FAIL because checked runtime APIs and the compiler module do not exist.

- [ ] **Step 2: Make AlgorithmTree runtime construction return errors**

Add checked constructors/emission/scheduling/register-allocation/assembly methods that return `Result` for untrusted runtime dimensions. Preserve existing convenience methods for offline callers, but the Qwen compiler must call only checked APIs. Replace lossy `usize as i64` conversions on this path with `try_from`; use checked `u64`/`u128` intermediates for token, batch, tile, byte-offset, operation-count, and duration calculations.

- [ ] **Step 3: Implement explicit trace modes**

Define `Iq1sTraceMode::{Handwritten, Compiler}` parsed only from `HETGPU_IQ1S_TRACE_MODE=handwritten|compiler`. Both builders consume the same validated launch/tile/component contract and produce `CompiledIq1sTrace` with ordered semantic operations, assembly, semantic SHA-256, assembly SHA-256, labels, vector-register count, lane capacity, and expected component coverage.

- [ ] **Step 4: Lower actual work through AlgorithmTree**

For compiler mode, create abstract vectors sized from the active tile, emit each actual IQ1_S grid/delta tmatmul operation with its true row/K/component coordinates, generate dependencies/order/register assignment through `AlgorithmTree`, append required import/export/accumulation and one terminal `stall`, then pass the emitted assembly to the existing assembler. Do not synthesize a 262,144-token trace: the limit test validates arithmetic while runtime emission uses the actual batch.

- [ ] **Step 5: Verify semantic parity and commit**

Run:

```bash
cargo test -p ptx tmatmul_algorithm_tree -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_trace_compiler -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_xrt -- --nocapture
```

Expected: PASS; compiler assembly contains operations derived from fixture dimensions and assembles to complete 16-byte words ending in `stall`.

Commit:

```bash
git add ptx/src/pass/tmatmul_algorithm_tree.rs zluda/src/impl/iq1s_trace_compiler.rs \
  zluda/src/impl/mod.rs zluda/src/impl/iq1s_xrt.rs
git commit -m "feat: compile IQ1S launches through AlgorithmTree"
```

### Task 5: Add the shared bank-aware resident weight and program caches

**Files:**
- Modify: `zluda/src/impl/xrt_tmatmul.rs`
- Modify: `zluda/src/impl/iq1s_xrt.rs`
- Modify: `zluda/src/impl/tq1_xrt.rs`

- [ ] **Step 1: Add mock-XRT cache tests first**

Using the existing injected XRT operations, assert: first matrix use allocates/writes/syncs once; a second same-key job in the same bank reuses the BO with zero matrix bytes transferred; the same tile in another bank gets a distinct BO; changed content/generation/tile misses; byte limits are enforced per bank; in-flight entries cannot be evicted; quiescent LRU entries can; allocation/write/sync/address failures poison strict execution. Add the same matrix for handwritten/compiler traces to prove one shared weight cache.

Add program-cache tests asserting a hit only when semantic hash, assembly hash, bound label addresses, register count, memory group, and lane capacity all match. Reject an encoded length not divisible by 16, a missing terminal STALL, and a job whose expected program hash differs from the bound BO.

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation xrt_tmatmul -- --nocapture
```

Expected: FAIL because `ReusableCu` still owns only one rewritten matrix BO and one fixed program BO.

- [ ] **Step 2: Introduce immutable cache/job keys**

Replace raw `XrtWaveJob.matrix` semantics with an immutable packed-tile key plus bytes and an unbound trace program specification. Include model/tensor identity, allocation generation, shape/strides, quant component, row/K/scale-group tile, content hash, and memory group in the matrix key. Program identity must include compiler semantic/assembly hashes plus the actual bound matrix/input/output addresses, register count, CU group, and lane capacity.

- [ ] **Step 3: Implement per-bank resident matrix BO ownership**

Move matrix BO ownership into bounded per-bank LRU maps inside the persistent pool. On miss allocate in the target group, write/sync once, record address, and pin until every referencing request completes. On hit reuse the address without transfer. Parse `HETGPU_XRT_RESIDENT_MATRIX_CACHE_BYTES`; reject zero/overflow/undersized limits. Release only quiescent entries and all remaining BOs during orderly pool destruction.

- [ ] **Step 4: Implement bound encoded-program caching**

After resolving the resident matrix and reusable input/output addresses, bind labels, assemble through `assemble_tmatmul_program_for_vector_registers`, compact/validate, hash the exact encoded bytes, and cache a program BO per CU. Program the MM2S address/length from that cached entry for the job. Completion must carry matrix key hash, trace semantic hash, assembly hash, encoded program hash/address/bytes, and nonzero STALL.

- [ ] **Step 5: Verify transfers, eviction safety, and commit**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation xrt_tmatmul -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_xrt -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation tq1_xrt -- --nocapture
```

Expected: PASS; legacy TQ1 callers adapt to the explicit handwritten program specification without changing their numerical behavior.

Commit:

```bash
git add zluda/src/impl/xrt_tmatmul.rs zluda/src/impl/iq1s_xrt.rs zluda/src/impl/tq1_xrt.rs
git commit -m "feat: cache U250 IQ1S weights and bound programs"
```

### Task 6: Bind execution evidence to the exact trace and shared cache

**Files:**
- Modify: `zluda/src/impl/iq1s_xrt.rs`
- Modify: `zluda/src/impl/function.rs`
- Modify: `zluda/src/impl/bitnet_disagg.rs`
- Modify: `tools/qwen35-iq1s-route-manifest.json`

- [ ] **Step 1: Add mutation-style evidence tests**

Require every eligible launch to emit one route record and one execution record. Tests must reject missing/duplicate request IDs, wrong CU, zero STALL, trace/program hash mismatch, program-address mismatch, incomplete component coverage, missing host or resident cache accounting, XRT work without an eligible route, and eligible route without XRT work.

- [ ] **Step 2: Extend evidence records with checked counters**

Record kernel/ABI/tensor/layer/expert/projection, actual shapes/strides/batch, trace mode, semantic operations/hash, full ordered assembly/hash, encoded program hash/address/bytes, host-pack and resident hit/miss/bytes/evictions, per-CU submissions/completions/request IDs/STALLED codes/timing, raw bounds, reconstruction time, and comparison metrics. Repeated assembly bodies may be interned by hash, but each launch must reference the stored body and exact program binding.

- [ ] **Step 3: Enforce route-manifest precedence**

Keep manifest default `gpu`. The only accelerator entry remains exact `ggml_type19`; final qualification still requires registered routed-expert identity and supported ABI. Count the 39 non-IQ1_S experts as expected GPU-native operations, never as fallback.

- [ ] **Step 4: Run evidence regressions and commit**

Run:

```bash
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s_xrt -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation bitnet_disagg -- --nocapture
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
```

Expected: PASS with `handled == eligible`, zero strict fallback/error, and trace/program equality checked per physical completion.

Commit:

```bash
git add zluda/src/impl/iq1s_xrt.rs zluda/src/impl/function.rs \
  zluda/src/impl/bitnet_disagg.rs tools/qwen35-iq1s-route-manifest.json
git commit -m "feat: prove Qwen IQ1S trace execution binding"
```

### Task 7: Implement the fixed continuous-batch three-mode evaluator

**Files:**
- Modify: `tools/qwen35_au250_eval.py`
- Modify: `zluda/tests/test_qwen35_au250_eval.py`
- Modify: `tools/run_qwen35_iq1s_au250_hybrid.sh`
- Modify: `zluda/tests/run_au250_xrt_iq1s.sh`
- Modify: `zluda/tests/test_au250_qwen35_runtime_static.sh`

- [ ] **Step 1: Add scheduler/timing red tests**

Use a fake HTTP server/client to prove exactly 64 stable request IDs, at most 16 active requests, exactly 32 returned generated token IDs per request, deterministic prompt construction within a 512-token context, greedy decoding, and throughput measured from first measured enqueue to last measured completion. Reject missing/extra/duplicate requests, fewer/more tokens, HTTP failures, non-finite timing, and prompt-token contamination of generated throughput.

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
```

Expected: FAIL because the evaluator still starts `--parallel 1` and sends sequential requests.

- [ ] **Step 2: Add bounded continuous submission**

Start llama-server with `--ctx-size 512 --parallel 16`, deterministic seed/temperature settings, and enough server slots for 16 active sequences. Submit 64 requests through a bounded executor or async semaphore of 16. Store completion results by request ID rather than arrival order. Require `completion_tokens == 32` and preserve the exact token-ID list.

- [ ] **Step 3: Add three explicit modes**

Support `cuda`, `handwritten`, and `compiler`. CUDA disables interception routing. Both hybrid modes enable identical strict Qwen routing/cache/executor settings and differ only in `HETGPU_IQ1S_TRACE_MODE`. Run one unmeasured warm-up batch, then three measured batches for each hybrid mode. Record queue, service, first-token, end-to-end, aggregate generated-token throughput, and single-stream decode latency separately.

Update `run_au250_xrt_iq1s.sh` to accept `--trace-mode handwritten|compiler`, propagate the mode into its container, run a fixture large enough to submit work to all four configured CUs, and reject any inactive CU or trace/program mismatch. Replace the Qwen runner's stale `run_au250_xrt_tq1.sh` numerical call with one IQ1_S run per trace mode.

- [ ] **Step 4: Add sampled FFN output capture**

Use a deterministic launch/request sampling policy. Before timed batches, run an unmeasured comparison probe for each hybrid trace mode through the existing native-CUDA shadow/reference mechanism, then execute the U250 result and retain both output vectors. Require finiteness and compute elementwise `abs_error <= 1e-4 + 1e-3 * abs(reference)`. Disable shadow comparison for measured batches so comparison work and synchronization cannot enter throughput.

- [ ] **Step 5: Verify evaluator and runner contracts and commit**

Run:

```bash
python3 -m pytest -q zluda/tests/test_qwen35_au250_eval.py
bash -n zluda/tests/run_au250_xrt_iq1s.sh
bash zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: PASS; static assertions find `--parallel 16`, `64`, `32`, `512`, both trace modes, and three measured hybrid passes.

Commit:

```bash
git add tools/qwen35_au250_eval.py tools/run_qwen35_iq1s_au250_hybrid.sh \
  zluda/tests/run_au250_xrt_iq1s.sh \
  zluda/tests/test_qwen35_au250_eval.py zluda/tests/test_au250_qwen35_runtime_static.sh
git commit -m "feat: benchmark Qwen in three continuous batch modes"
```

### Task 8: Strengthen the fail-closed proof validator

**Files:**
- Modify: `zluda/tests/validate_qwen35_iq1s_au250_proof.py`
- Modify: `zluda/tests/test_validate_qwen35_iq1s_au250_proof.py`

- [ ] **Step 1: Expand the passing fixture and mutation matrix**

The passing fixture must contain CUDA, handwritten, and compiler records; 64x32 tokens per mode; identical greedy token IDs by request; sampled FFN tolerance; exactly 141 audited IQ1_S and 39 GPU-native expert tensors; all four CUs active; nonzero resident-cache hits after warm-up; complete trace/program binding; healthy firewall; canonical artifact hashes; and three hybrid measurements each at least 15 generated tok/s.

Independently mutate every required field: model/build/source hash, context limit, workload count/concurrency/token count, token IDs, FFN values/tolerance, tensor distribution, eligible/handled/fallback/error counts, trace mode, AlgorithmTree hash, assembly body/hash, program hash/address/bytes, resident hit/transfer/eviction accounting, request/CU/STALL completion, firewall, PCIe record, and throughput.

- [ ] **Step 2: Implement three-way validation**

Require exact request-ID and token-ID equality `CUDA == handwritten == compiler`. Require both hybrid modes to satisfy the same eligible tensor/launch set and semantic tile/component coverage, while allowing safe assembly ordering differences. Independently recompute evidence hashes and reject logged-only compiler traces whose encoded program hash is not the completion's bound program hash.

- [ ] **Step 3: Enforce performance and topology gates**

For each hybrid measured pass, compute `64 * 32 / measured_wall_seconds` and require at least 15.0. Require the PCIe link record and report the observed Gen3 x4 downgrade as a risk, but do not substitute a projection or standalone tile result for the measured gate.

- [ ] **Step 4: Run validator tests and commit**

Run:

```bash
python3 -m pytest -q zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
```

Expected: PASS; every single-field mutation is rejected with a specific reason.

Commit:

```bash
git add zluda/tests/validate_qwen35_iq1s_au250_proof.py \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
git commit -m "test: require fail closed Qwen compiler trace proof"
```

### Task 9: Run software regressions and live one-token hardware gates

**Files:**
- Modify only if a failing test identifies a scoped defect in files from Tasks 1-8.

- [ ] **Step 1: Run the complete focused software suite**

```bash
cargo test -p ptx tmatmul_algorithm_tree -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation iq1s -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation xrt_tmatmul -- --nocapture
cargo test -p zluda --no-default-features --features nvidia,evaluation tq1 -- --nocapture
python3 -m pytest -q \
  zluda/tests/test_qwen35_build_preflight.py \
  zluda/tests/test_qwen35_gguf_audit.py \
  zluda/tests/test_qwen35_au250_eval.py \
  zluda/tests/test_validate_qwen35_iq1s_au250_proof.py
bash zluda/tests/test_prepare_au250_qwen35_source.sh
bash zluda/tests/test_au250_qwen35_runtime_static.sh
```

Expected: all commands PASS. Fix causes, add a regression assertion, rerun the failing command, and make a narrowly scoped commit before proceeding.

- [ ] **Step 2: Rebuild the pinned Qwen runtime and verify manifest artifacts**

```bash
bash tools/au250_qwen35_run.sh \
  bash /work/tools/build_au250_qwen35_runtime.sh
bash tools/au250_qwen35_run.sh \
  python3 /work/tools/qwen35_build_preflight.py \
    --manifest /qwen-build/manifest.json \
    --build-root /qwen-build \
    --llama-revision 925e1179947ea0c0ebfb0032df18af3a729822be \
    --output /qwen-build/qwen35-build-preflight.json
```

Expected: PASS and the output names the Qwen build's canonical `libggml.so`, not `/home/eabban/BitNet/...`.

- [ ] **Step 3: Run live standalone fixtures in both trace modes**

Run:

```bash
bash zluda/tests/run_au250_xrt_iq1s.sh --trace-mode handwritten
bash zluda/tests/run_au250_xrt_iq1s.sh --trace-mode compiler
```

Expected: both standalone runs use all four CUs, report nonzero STALL, demonstrate resident cache reuse, bind the expected trace/program hash, produce finite results, and satisfy `atol=1e-4`, `rtol=1e-3`.

- [ ] **Step 4: Run one-token Qwen smoke tests**

Run CUDA, handwritten hybrid, and compiler hybrid with one deterministic generated token. Require identical token IDs; for each hybrid require at least one eligible/handled IQ1_S launch, zero fallback/error, physical work on all four CUs, and a verified Qwen `libggml.so` record. If either hybrid fails, preserve its proof directory and exact stderr, fix the cause test-first, and repeat all three modes.

### Task 10: Run the fixed acceptance benchmark and publish only a passing result

**Files:**
- Create: `zluda/evaluation/2026-09-01-qwen397b-iq1s-compiler-traces.md`

- [ ] **Step 1: Capture immutable preflight and launch the full run**

```bash
proof_dir=".proof/qwen35-iq1s-au250-$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "${proof_dir}" > /tmp/qwen35-iq1s-proof-dir
bash tools/run_qwen35_iq1s_au250_hybrid.sh \
  /root/models/qwen35-tq1/Qwen3.5-397B-A17B-UD-TQ1_0.gguf \
  /root/qwen35-au250-build/manifest.json \
  /au250_xrt/example/MaxCores_370M.xclbin \
  "${proof_dir}"
```

Expected workload: 64 requests, maximum 16 active, 32 generated tokens each, 512-token context, CUDA plus both hybrid trace modes, one warm-up and three measured hybrid passes.

- [ ] **Step 2: Validate the proof independently**

```bash
proof_dir="$(cat /tmp/qwen35-iq1s-proof-dir)"
python3 zluda/tests/validate_qwen35_iq1s_au250_proof.py \
  "${proof_dir}"
```

Expected: PASS only if token IDs are identical, sampled FFN outputs satisfy tolerance, all 141 IQ1_S experts are the only eligible offloads, the 39 other experts remain GPU-native, both trace modes bind their actual programs, all four CUs execute, resident weight hits are nonzero, and every measured hybrid pass reaches 15 aggregate generated tok/s.

- [ ] **Step 3: Diagnose measured shortfalls without weakening gates**

If correctness passes but throughput is below 15, use the proof's queue/service/XRT/reconstruction timing and cache/PCIe byte counters to identify the dominant measured cost. Optimize only evidence-supported redundant matrix packing/transfers, program reassembly, wave underfill, or host/device serialization; add a regression/performance counter assertion for the changed path, rerun Tasks 9 and 10, and retain the failed bundle as non-qualifying evidence. Do not reduce requests, generated tokens, concurrency, measurements, correctness checks, or physical-work requirements.

- [ ] **Step 4: Generate the evaluation report from normalized proof**

Write the report with artifact hashes, source commit/diff identity, GPU/U250 topology including Gen3 x4, model audit, weight/cache counts, trace/assembly/program identities, per-CU work, CUDA/handwritten/compiler token and FFN comparison, each measured pass, aggregate throughput, latency breakdown, and explicit acceptance verdict. If the proof validator fails, the report must say `NOT QUALIFIED` and must not claim 15 tok/s or GPU-equivalent results.

- [ ] **Step 5: Run final verification and commit the report**

```bash
proof_dir="$(cat /tmp/qwen35-iq1s-proof-dir)"
python3 zluda/tests/validate_qwen35_iq1s_au250_proof.py \
  "${proof_dir}"
git status --short
```

Expected: validator PASS and only intended source/report changes plus preserved pre-existing `.proof/` state. Force-add the Markdown report because this repository ignores `*.md`:

```bash
git add -f zluda/evaluation/2026-09-01-qwen397b-iq1s-compiler-traces.md
git commit -m "docs: report Qwen IQ1S compiler trace evaluation"
```
