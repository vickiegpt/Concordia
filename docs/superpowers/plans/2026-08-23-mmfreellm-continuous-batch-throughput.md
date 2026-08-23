# MatMulFreeLM Continuous-Batch Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, reproducible MatMulFreeLM 2.7B CUDA benchmark that sustains at least 200 aggregate generated tokens per second with no more than 16 concurrent requests.

**Architecture:** A condition-protected FIFO feeds one GPU worker with windowed microbatches dispatched at 16 requests, at the oldest-request timeout, or when the producer closes. The model is loaded once and warmed outside measurement; two qualification runs and a separate proof harness preserve correctness, latency, provenance, and immutable artifacts.

**Tech Stack:** Python 3 standard library, PyTorch CUDA, Hugging Face Transformers, local `/root/matmulfreellm`, `unittest`, Bash, SHA-256.

**Spec:** `docs/superpowers/specs/2026-08-21-mmfreellm-continuous-batch-throughput-design.md`

## Global Constraints

- CUDA is mandatory; CPU execution and a missing CUDA device fail before warmup.
- The qualified profile is exactly 64 requests, 8 generated tokens per request, greedy decoding, and at most 16 requests per microbatch.
- Dispatch is windowed continuous batching around complete `model.generate` calls, not iteration-level sequence insertion.
- There is exactly one model worker and no concurrent call into the model object.
- Aggregate TPS is validated generated tokens divided by first-enqueue-to-last-completion wall time; model load and warmup are excluded.
- Every output retains its non-empty prompt and appends exactly the requested number of generated token IDs.
- Two identical batch-16 runs produce identical generated token IDs per request, and each run reaches at least 200 aggregate generated tok/s.
- Queue, service, and end-to-end latency are reported separately from aggregate throughput.
- `MMFREELM_BITLINEAR_BACKEND=default`; TernIP/CXL is disabled; JSON records `fpga_tps_reported=false`.
- Existing FPGA `MAX_LANES = 4` is unchanged, and no GPU result is attributed to Agilex.
- Result files and proof directories use create-only semantics and never overwrite previous evidence.

---

## File structure

- Create `zluda/tests/mmfreellm_continuous_batch.py`: records, queue policy, one-producer/one-worker runner, metrics, and validation.
- Create `zluda/tests/test_mmfreellm_continuous_batch.py`: controlled-policy and fake-backend unit tests.
- Create `zluda/tests/run_mmfreellm_continuous_batch_benchmark.py`: CUDA backend, warmup, two-run qualification, CLI, and canonical JSON.
- Create `zluda/tests/test_run_mmfreellm_continuous_batch_benchmark.py`: fake CUDA/model and qualification tests.
- Modify `zluda/tests/run_mmfreellm_2p7b_benchmark.py`: expose the control run's generated token IDs for a non-gating cross-batch comparison.
- Modify `zluda/tests/test_run_mmfreellm_2p7b_benchmark.py`: verify exact control token-ID serialization.
- Create `zluda/tests/run_mmfreellm_continuous_batch_evaluation.sh`: immutable batch-1 control plus batch-16 proof harness.
- Create `zluda/tests/test_run_mmfreellm_continuous_batch_evaluation_static.sh`: fixture-driven harness tests.
- Create `zluda/evaluation/2026-08-23-mmfreellm-continuous-batch-evaluation.md`: live results and proof index.

### Task 1: Core data contract, metrics, and validation

**Files:**
- Create: `zluda/tests/mmfreellm_continuous_batch.py`
- Create: `zluda/tests/test_mmfreellm_continuous_batch.py`

**Interfaces:**
- Consumes: Python standard library only.
- Produces: `BenchmarkConfig`, `RequestSpec`, `QueuedRequest`, `GeneratedOutput`, `BackendBatchResult`, `RequestResult`, `MicrobatchResult`, `RunResult`, `batch_size_if_ready()`, `percentile()`, `validate_run()`, and `summarize_run()`.
- `BenchmarkConfig.validate()` enforces positive request/token counts, `max_batch_size` in `1..16`, finite non-negative queue/interarrival times, and finite positive acceptance TPS.
- `validate_run(run, config)` raises `ValueError` unless IDs, counts, output semantics, timing, and microbatch membership are complete and consistent.

- [ ] **Step 1: Write failing contract tests**

Create `test_mmfreellm_continuous_batch.py` with this fixture and test set:

```python
#!/usr/bin/env python3
import math
import unittest

from mmfreellm_continuous_batch import (
    BenchmarkConfig, MicrobatchResult, RequestResult, RunResult,
    batch_size_if_ready, percentile, summarize_run, validate_run,
)


def valid_run():
    requests = tuple(
        RequestResult(index, "The quick brown fox",
                      "The quick brown fox jumps", (10, 11),
                      1.0, 1.1, 1.3)
        for index in range(2)
    )
    return RunResult(1, 1.0, 1.3, requests,
                     (MicrobatchResult((0, 1), 1.1, 1.3, 0.2, 1024),))


class CoreContractTests(unittest.TestCase):
    def test_batch_size_is_capped_at_sixteen(self):
        with self.assertRaisesRegex(ValueError, "1..16"):
            BenchmarkConfig(max_batch_size=17).validate()

    def test_dispatch_is_full_timeout_or_close_only(self):
        self.assertEqual(batch_size_if_ready(16, 0, 16, .002, False), 16)
        self.assertEqual(batch_size_if_ready(3, .003, 16, .002, False), 3)
        self.assertEqual(batch_size_if_ready(3, 0, 16, .002, True), 3)
        self.assertEqual(batch_size_if_ready(3, .001, 16, .002, False), 0)

    def test_percentile_is_linear_interpolation(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 95), 2.9)

    def test_summary_uses_tokens_over_end_to_end_time(self):
        summary = summarize_run(valid_run(),
                                BenchmarkConfig(request_count=2,
                                                max_new_tokens=2))
        self.assertEqual(summary["total_generated_tokens"], 4)
        self.assertAlmostEqual(summary["aggregate_tokens_per_second"], 4 / .3)
        self.assertEqual(summary["observed_batch_sizes"], [2])

    def test_validation_rejects_duplicate_ids(self):
        run = valid_run()
        broken = RunResult(1, 1.0, 1.3,
                           (run.requests[0], run.requests[0]), run.microbatches)
        with self.assertRaisesRegex(ValueError, "request IDs"):
            validate_run(broken, BenchmarkConfig(request_count=2,
                                                 max_new_tokens=2))

    def test_validation_rejects_bad_token_count_empty_output_and_time(self):
        config = BenchmarkConfig(request_count=2, max_new_tokens=2)
        run = valid_run()
        replacements = (
            ("generated-token count", RequestResult(1, run.requests[1].prompt,
             run.requests[1].output, (10,), 1.0, 1.1, 1.3)),
            ("empty decoded output", RequestResult(1, run.requests[1].prompt,
             "", (10, 11), 1.0, 1.1, 1.3)),
            ("timing", RequestResult(1, run.requests[1].prompt,
             run.requests[1].output, (10, 11), 1.4, 1.1, 1.3)),
        )
        for message, replacement in replacements:
            with self.subTest(message=message):
                broken = RunResult(1, 1.0, 1.3,
                                   (run.requests[0], replacement), run.microbatches)
                with self.assertRaisesRegex(ValueError, message):
                    validate_run(broken, config)

    def test_non_finite_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "acceptance"):
            BenchmarkConfig(min_aggregate_tps=math.nan).validate()
```

- [ ] **Step 2: Confirm the tests fail because the module is absent**

Run:

```bash
cd /home/victoryang00/hetGPU/zluda/tests
python3 -m unittest -v test_mmfreellm_continuous_batch.py
```

Expected: import failure for `mmfreellm_continuous_batch`.

- [ ] **Step 3: Implement immutable records and pure helpers**

Create the module with these public record fields:

```python
@dataclass(frozen=True)
class BenchmarkConfig:
    request_count: int = 64
    max_batch_size: int = 16
    max_new_tokens: int = 8
    queue_timeout_ms: float = 2.0
    interarrival_ms: float = 0.0
    min_aggregate_tps: float = 200.0

@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt: str
    max_new_tokens: int

@dataclass(frozen=True)
class QueuedRequest:
    spec: RequestSpec
    enqueued_at: float

@dataclass(frozen=True)
class GeneratedOutput:
    output: str
    generated_token_ids: tuple[int, ...]

@dataclass(frozen=True)
class BackendBatchResult:
    outputs: tuple[GeneratedOutput, ...]
    service_seconds: float
    peak_cuda_memory_bytes: int

@dataclass(frozen=True)
class RequestResult:
    request_id: int
    prompt: str
    output: str
    generated_token_ids: tuple[int, ...]
    enqueued_at: float
    dispatched_at: float
    completed_at: float

@dataclass(frozen=True)
class MicrobatchResult:
    request_ids: tuple[int, ...]
    dispatched_at: float
    completed_at: float
    service_seconds: float
    peak_cuda_memory_bytes: int

@dataclass(frozen=True)
class RunResult:
    run_index: int
    first_enqueued_at: float
    last_completed_at: float
    requests: tuple[RequestResult, ...]
    microbatches: tuple[MicrobatchResult, ...]
```

Implement readiness exactly as: return zero for empty; return `max_batch_size` when full; return all pending when producer is closed or oldest age meets timeout; otherwise return zero. Implement percentile at sorted fractional index `(n - 1) * percent / 100` with linear interpolation.

`validate_run()` checks, in order: finite positive run interval; IDs exactly `0..request_count-1`; result count equals request count; non-empty prompt; non-empty output beginning with prompt; generated-ID count equals `max_new_tokens`; finite ordering `first <= enqueue <= dispatch <= complete <= last`; non-empty microbatch list; every batch size in `1..max_batch_size`; every request appears in exactly one batch; finite positive service time; and non-negative peak memory.

`summarize_run()` first validates, then returns JSON-safe fields for schema
`matmulfreellm-continuous-batch-run-v1`; `run_index`;
`requested_requests`, `completed_requests`, and `failed_requests`;
`total_generated_tokens`; `end_to_end_seconds`;
`aggregate_tokens_per_second`; `configured_max_batch_size`;
`observed_batch_sizes`; `microbatch_count`; `microbatches`;
`peak_cuda_memory_bytes`; `requests`; and a `latency_seconds` object whose
`queue`, `service`, and `end_to_end` children each contain `p50`, `p95`, and
`max`.

- [ ] **Step 4: Run contract tests**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
python3 -m unittest -v test_mmfreellm_continuous_batch.py
```

Expected: all `CoreContractTests` pass.

- [ ] **Step 5: Commit the contract**

```bash
cd /home/victoryang00/hetGPU
git add zluda/tests/mmfreellm_continuous_batch.py \
        zluda/tests/test_mmfreellm_continuous_batch.py
git commit -m "test: define continuous batch benchmark contract"
```

### Task 2: Condition-protected FIFO and one-worker execution

**Files:**
- Modify: `zluda/tests/mmfreellm_continuous_batch.py`
- Modify: `zluda/tests/test_mmfreellm_continuous_batch.py`

**Interfaces:**
- Consumes: Task 1 records and a backend implementing `generate(requests: Sequence[RequestSpec]) -> BackendBatchResult`.
- Produces: `WindowedRequestQueue.submit()`, `close()`, `take_batch()`, and `run_windowed_batch(run_index, prompts, backend, config, clock=time.perf_counter, sleeper=time.sleep) -> RunResult`.
- `take_batch()` returns `None` only after closure and drain; `run_windowed_batch()` uses exactly one producer and one worker and propagates either thread's exception.

- [ ] **Step 1: Add failing scheduler tests**

Append this backend and cases:

```python
from mmfreellm_continuous_batch import (
    BackendBatchResult, GeneratedOutput, run_windowed_batch,
)


class RecordingBackend:
    def __init__(self):
        self.batches = []
        self.active = 0

    def generate(self, requests):
        self.assert_not_active()
        self.active = 1
        try:
            self.batches.append([request.request_id for request in requests])
            return BackendBatchResult(
                tuple(GeneratedOutput(request.prompt + " jumps", (10, 11))
                      for request in requests), .01, 2048)
        finally:
            self.active = 0

    def assert_not_active(self):
        if self.active:
            raise AssertionError("backend called concurrently")


class SchedulerTests(unittest.TestCase):
    def test_full_then_closed_partial_is_fifo(self):
        backend = RecordingBackend()
        config = BenchmarkConfig(request_count=6, max_batch_size=4,
                                 max_new_tokens=2, queue_timeout_ms=1000)
        result = run_windowed_batch(1, ["The quick brown fox"] * 6,
                                    backend, config)
        self.assertEqual(backend.batches, [[0, 1, 2, 3], [4, 5]])
        self.assertEqual([item.request_id for item in result.requests],
                         list(range(6)))

    def test_timeout_flushes_a_slow_arrival(self):
        backend = RecordingBackend()
        config = BenchmarkConfig(request_count=3, max_batch_size=16,
                                 max_new_tokens=2, queue_timeout_ms=2,
                                 interarrival_ms=15)
        run_windowed_batch(1, ["The quick brown fox"] * 3, backend, config)
        self.assertEqual(backend.batches, [[0], [1], [2]])

    def test_worker_exception_reaches_caller(self):
        class BrokenBackend:
            def generate(self, requests):
                raise RuntimeError("injected backend failure")
        with self.assertRaisesRegex(RuntimeError, "injected backend failure"):
            run_windowed_batch(1, ["The quick brown fox"], BrokenBackend(),
                BenchmarkConfig(request_count=1, max_batch_size=1,
                                max_new_tokens=2))
```

- [ ] **Step 2: Confirm scheduler symbols are missing**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
python3 -m unittest -v test_mmfreellm_continuous_batch.SchedulerTests
```

Expected: failure because `run_windowed_batch` is undefined.

- [ ] **Step 3: Implement queue waiting and FIFO removal**

Implement `WindowedRequestQueue` with `deque` and `threading.Condition`. Under the condition lock: wait while empty/open; return `None` when empty/closed; compute oldest age with the injected monotonic clock; use `batch_size_if_ready()`; pop the selected count from the left; otherwise wait for only `timeout - oldest_age` and re-evaluate. `submit()` rejects closure and notifies one waiter; `close()` marks closed and notifies all.

Use this backend boundary:

```python
class GenerationBackend(Protocol):
    def generate(self, requests: Sequence[RequestSpec]) -> BackendBatchResult:
        raise NotImplementedError
```

- [ ] **Step 4: Implement producer/worker lifecycle**

`run_windowed_batch()` performs these exact actions:

```text
validate config and prompt count
producer: stamp and submit IDs in input order; sleep only between arrivals; close in finally
worker: take one batch; stamp dispatch; call backend once; require equal output count;
        stamp completion; emit one RequestResult per row and one MicrobatchResult
main: start producer and worker; join both; re-raise first captured exception;
      sort results by ID; construct RunResult; validate; return
```

Protect result/error lists with a lock. On either thread failure close the queue so its peer terminates. Define first enqueue as the minimum actual enqueue timestamp and last completion as the maximum actual completion timestamp; never use thread-start time.

- [ ] **Step 5: Run scheduler tests twice to expose timeout flakes**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
python3 -m unittest -v test_mmfreellm_continuous_batch.py
python3 -m unittest -v test_mmfreellm_continuous_batch.py
```

Expected: both runs pass and timeout batches are `[[0], [1], [2]]`.

- [ ] **Step 6: Commit scheduler execution**

```bash
cd /home/victoryang00/hetGPU
git add zluda/tests/mmfreellm_continuous_batch.py \
        zluda/tests/test_mmfreellm_continuous_batch.py
git commit -m "feat: add bounded continuous batch scheduler"
```

### Task 3: CUDA backend and two-run qualification CLI

**Files:**
- Create: `zluda/tests/run_mmfreellm_continuous_batch_benchmark.py`
- Create: `zluda/tests/test_run_mmfreellm_continuous_batch_benchmark.py`
- Modify: `zluda/tests/run_mmfreellm_2p7b_benchmark.py`
- Modify: `zluda/tests/test_run_mmfreellm_2p7b_benchmark.py`

**Interfaces:**
- Consumes: Task 1-2 public interfaces.
- Produces: `CudaGenerationBackend`, `qualify_summaries()`, `format_json_record()`, `write_result_exclusive()`, `parse_args()`, `validate_cli()`, and `main()`.
- Canonical prefix: `MMFREELM_CONTINUOUS_BATCH_JSON=`.
- Exit `0` only when two runs are deterministic and each meets threshold; exit `1` for a valid failed qualification; exit `2` for setup/runtime/record errors.

- [ ] **Step 1: Write failing qualification and output tests**

Use `importlib.util` like the existing batch-1 test. Test these exact gates:

```python
def valid_summary(run_index, tps, token_ids):
    return {
        "run_index": run_index,
        "aggregate_tokens_per_second": tps,
        "requests": [
            {"request_id": 0, "generated_token_ids": list(token_ids)},
        ],
    }


def test_each_run_must_meet_threshold(self):
    result = benchmark.qualify_summaries(
        [valid_summary(1, 199.9, [10, 11]),
         valid_summary(2, 260.0, [10, 11])], 200.0)
    self.assertFalse(result["qualification_passed"])
    self.assertIn("run 1 aggregate TPS 199.900000 is below 200.000000",
                  result["failure_reasons"])

def test_generated_ids_must_match_between_runs(self):
    result = benchmark.qualify_summaries(
        [valid_summary(1, 250.0, [10, 11]),
         valid_summary(2, 260.0, [10, 12])], 200.0)
    self.assertFalse(result["deterministic_generated_token_ids"])
    self.assertIn("determinism", result["failure_reasons"])

def test_result_file_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        benchmark.write_result_exclusive(path, {"validated": True})
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            benchmark.write_result_exclusive(path, {"validated": True})

def test_cpu_and_batch_seventeen_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "CUDA is mandatory"):
        benchmark.validate_cli(benchmark.parse_args(["--device", "cpu"]))
    with self.assertRaisesRegex(ValueError, "1..16"):
        benchmark.validate_cli(
            benchmark.parse_args(["--max-batch-size", "17"]))

def test_json_marks_fpga_false(self):
    line = benchmark.format_json_record({"fpga_tps_reported": False})
    self.assertTrue(line.startswith("MMFREELM_CONTINUOUS_BATCH_JSON="))
    self.assertFalse(json.loads(line.split("=", 1)[1])["fpga_tps_reported"])
```

Construct `CudaGenerationBackend` with injected `torch_module`, `tokenizer`, and
`model` objects so the unit test does not import CUDA. The fake tokenizer records
`padding=True` and returns two CUDA-movable tensors with input width four; the
fake model records keyword arguments and returns two rows with four prompt IDs
plus IDs `(10, 11)`; the fake Torch object counts inference-mode entry and two
synchronizations. Assert those recorded values, both output-ID tuples, decoded
row count, service-time positivity, and peak-memory propagation.

- [ ] **Step 2: Confirm the CLI module is absent**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
python3 -m unittest -v test_run_mmfreellm_continuous_batch_benchmark.py
```

Expected: import failure for the new benchmark module.

- [ ] **Step 3: Implement CLI defaults and fail-closed setup**

Define these defaults:

```python
DEFAULT_MODEL = "/root/.cache/huggingface/hub/models--ridger--MMfreeLM-2.7B/snapshots/77deff0c1c9ac79aa51eb3ab7dd34fc375bf9324"
DEFAULT_REPO = Path("/root/matmulfreellm")
# request_count=64, max_batch_size=16, max_new_tokens=8
# queue_timeout_ms=2, interarrival_ms=0, runs=2, warmup_runs=1
# min_aggregate_tps=200, device="cuda", dtype="half"
```

Expose arguments for every value above plus `--model`, `--repo-root`, `--prompt`, `--online`, and `--result-json`. Require CUDA, exactly two runs, at least one warmup, valid local paths in offline mode, and valid `BenchmarkConfig`. Set `TOKENIZERS_PARALLELISM=false`; in offline mode set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Require `torch.cuda.is_available()` before loading weights.

Load `mmfreelm` from `--repo-root` before `AutoModelForCausalLM`; assign EOS as pad token only when pad is missing; load tokenizer/model once; call `.eval().to(device="cuda", dtype=torch.float16 or torch.bfloat16)`; count parameters once; record backend from `MMFREELM_BITLINEAR_BACKEND` and require it to be `default`.

- [ ] **Step 4: Add generated token IDs to the existing batch-1 control**

Extend the existing benchmark test to require the canonical JSON field:

```python
summary.update({"generated_token_ids": [101, 102, 103, 104]})
payload = json.loads(benchmark.format_json_record(summary).split("=", 1)[1])
self.assertEqual(payload["generated_token_ids"], [101, 102, 103, 104])
```

After the final measured `model.generate()` in
`run_mmfreellm_2p7b_benchmark.py`, slice
`output_ids[0, prompt_tokens:]`, convert every element to a Python integer, and
store the resulting list as `generated_token_ids`. Require its length to equal
`args.max_new_tokens` before emitting JSON. This field is evidence for the
cross-batch comparison and does not change batch-1 acceptance.

- [ ] **Step 5: Implement synchronized batch generation**

`CudaGenerationBackend.generate(requests)` requires equal positive token counts, tokenizes all prompts with padding, moves tensors to CUDA, and sets `input_width = encoded["input_ids"].shape[1]`. Under `torch.inference_mode()`: reset peak memory, synchronize, time one `model.generate`, synchronize, then stop time. Generate with:

```python
{
    "max_new_tokens": requested_tokens,
    "min_new_tokens": requested_tokens,
    "do_sample": False,
    "pad_token_id": tokenizer.pad_token_id,
}
```

For each output row, slice generated IDs at `input_width`, require exactly the requested count, decode the full row with special tokens skipped, and require the decoded text begins with the prompt. Return `BackendBatchResult(outputs, elapsed, torch.cuda.max_memory_allocated())`; reject non-finite/non-positive elapsed time or row mismatch.

Warmup constructs exactly `max_batch_size` identical requests and invokes the same backend once per configured warmup before measured enqueue.

- [ ] **Step 6: Implement two measured runs and qualification**

For each run, make 64 identical prompts, call `run_windowed_batch()`, and `summarize_run()`. Compare each request's generated-ID list with the same ID in run 1. Produce:

```python
{
  "schema": "matmulfreellm-continuous-batch-benchmark-v1",
  "validated": True,
  "qualification_passed": qualification_passed,
  "failure_reasons": failure_reasons,
  "model": args.model,
  "device": "cuda",
  "dtype": args.dtype,
  "parameter_count": backend.parameter_count,
  "request_count": 64,
  "max_batch_size": 16,
  "max_new_tokens": 8,
  "qualification_runs": 2,
  "min_aggregate_tps": 200.0,
  "deterministic_generated_token_ids": deterministic,
  "bitlinear_backend": "default",
  "ternip_adapter": "disabled",
  "fpga_tps_reported": False,
  "runs": run_summaries,
}
```

Write `--result-json` with mode `x`; print compact sorted JSON with the canonical prefix. Preserve and print valid below-target JSON before exit `1`. Print setup/runtime errors to stderr and exit `2` without claiming qualification.

- [ ] **Step 7: Run Python tests and CLI help**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
python3 -m unittest -v test_mmfreellm_continuous_batch.py \
  test_run_mmfreellm_continuous_batch_benchmark.py \
  test_run_mmfreellm_2p7b_benchmark.py
python3 run_mmfreellm_continuous_batch_benchmark.py --help
```

Expected: tests pass; help exposes all qualified dimensions and result path.

- [ ] **Step 8: Commit the benchmark**

```bash
cd /home/victoryang00/hetGPU
git add zluda/tests/run_mmfreellm_continuous_batch_benchmark.py \
        zluda/tests/test_run_mmfreellm_continuous_batch_benchmark.py \
        zluda/tests/run_mmfreellm_2p7b_benchmark.py \
        zluda/tests/test_run_mmfreellm_2p7b_benchmark.py
git commit -m "feat: qualify MatMulFreeLM continuous batching"
```

### Task 4: Immutable evaluation harness

**Files:**
- Create: `zluda/tests/run_mmfreellm_continuous_batch_evaluation.sh`
- Create: `zluda/tests/test_run_mmfreellm_continuous_batch_evaluation_static.sh`

**Interfaces:**
- Consumes: existing `run_mmfreellm_2p7b_benchmark.py`, Task 3 CLI, and one new result-directory argument.
- Produces: manifest, batch-1/batch-16 stdout/stderr and JSON, environment snapshots, source copies/inventory, source/artifact hashes, and `qualification-status.txt`.
- Command overrides are accepted only with `HETGPU_MMFREELM_STATIC_TEST_MODE=1`; live mode uses fixed argument arrays.

- [ ] **Step 1: Write fixture-driven harness tests**

Create fake executables in `mktemp -d`: batch-1 prints valid `MMFREELM_BENCHMARK_JSON`; batch-16-pass prints two deterministic 250-TPS runs with `qualification_passed=true`; batch-16-fail prints one 199-TPS run and exits `1`. Assert:

```bash
HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
HETGPU_MMFREELM_BATCH1_CMD="${fixture_dir}/batch1-pass" \
HETGPU_MMFREELM_BATCH16_CMD="${fixture_dir}/batch16-pass" \
  "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" "${pass_dir}"
test "$(cat "${pass_dir}/qualification-status.txt")" = pass
(cd "${pass_dir}" && sha256sum --check hashes/artifacts.sha256)

if HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
   HETGPU_MMFREELM_BATCH1_CMD="${fixture_dir}/batch1-pass" \
   HETGPU_MMFREELM_BATCH16_CMD="${fixture_dir}/batch16-fail" \
     "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" "${fail_dir}"; then
  echo "failed qualification unexpectedly passed" >&2; exit 1
fi
test "$(cat "${fail_dir}/qualification-status.txt")" = failed

if HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
   HETGPU_MMFREELM_BATCH1_CMD="${fixture_dir}/batch1-pass" \
   HETGPU_MMFREELM_BATCH16_CMD="${fixture_dir}/batch16-pass" \
     "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" "${pass_dir}"; then
  echo "existing proof unexpectedly overwritten" >&2; exit 1
fi
```

- [ ] **Step 2: Confirm the harness is absent**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
bash test_run_mmfreellm_continuous_batch_evaluation_static.sh
```

Expected: failure because the harness is missing.

- [ ] **Step 3: Implement path safety and fixed commands**

Follow `run_v3_batch_evaluation.sh`: `set -Eeuo pipefail`, `umask 077`, one non-empty path, existing parent, reject `/`, repo root, and script dir, reject existing paths/symlinks, then create mode-700 directories.

Live arrays are exactly:

```bash
batch1_command=(python3 "${script_dir}/run_mmfreellm_2p7b_benchmark.py"
  --device cuda --dtype half --max-new-tokens 8 --warmup-runs 1 --runs 3)
batch16_command=(python3 "${script_dir}/run_mmfreellm_continuous_batch_benchmark.py"
  --device cuda --dtype half --request-count 64 --max-batch-size 16
  --max-new-tokens 8 --queue-timeout-ms 2 --interarrival-ms 0
  --warmup-runs 1 --runs 2 --min-aggregate-tps 200
  --result-json "${result_dir}/batch16/result.json")
```

Reject a non-default BitLinear backend or enabled TernIP adapter. Export default backend, disabled adapter, offline Hugging Face variables, and disabled tokenizer parallelism.

- [ ] **Step 4: Implement parsing, provenance, and fail propagation**

Capture stdout/stderr separately and preserve each exit code. Extract the final canonical JSON line with Python. Validate batch-1 CUDA/8-token/non-empty output and exactly eight generated token IDs. Validate batch-16 schema, validated/qualified flags, `fpga_tps_reported=false`, exact dimensions, two runs, 64 completed/0 failed/512 tokens per run, each run TPS at least 200, outputs non-empty, and deterministic IDs. Compare the control IDs with request 0 of qualified run 1 and record `cross_batch_token_ids_equal` in the manifest, but never include that comparison in the pass/fail expression.

Manifest fields: UTC start/end, absolute repo, Git SHA/branch/status artifact, exact argv arrays, model and GPU identity, environment allowlist, and artifact paths. Capture `nvidia-smi` identity and Python/Torch/Transformers versions.

Copy and SHA-256 hash this exact inventory: the spec, this plan,
`mmfreellm_continuous_batch.py`, `test_mmfreellm_continuous_batch.py`,
`run_mmfreellm_continuous_batch_benchmark.py`,
`test_run_mmfreellm_continuous_batch_benchmark.py`, the existing
`run_mmfreellm_2p7b_benchmark.py`, and both new shell files. Finally hash every
proof artifact except `hashes/artifacts.sha256`, using sorted relative paths.
Write `pass` only after all gates; on any command/parser/gate failure write
`failed`, preserve artifacts, and exit non-zero.

- [ ] **Step 5: Run syntax and static evidence tests**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
bash -n run_mmfreellm_continuous_batch_evaluation.sh
bash -n test_run_mmfreellm_continuous_batch_evaluation_static.sh
bash test_run_mmfreellm_continuous_batch_evaluation_static.sh
```

Expected: all three commands pass.

- [ ] **Step 6: Commit the harness**

```bash
cd /home/victoryang00/hetGPU
git add zluda/tests/run_mmfreellm_continuous_batch_evaluation.sh \
        zluda/tests/test_run_mmfreellm_continuous_batch_evaluation_static.sh
git commit -m "test: add MatMulFreeLM throughput proof harness"
```

### Task 5: Live qualification and evaluation report

**Files:**
- Create: `zluda/evaluation/2026-08-23-mmfreellm-continuous-batch-evaluation.md`
- Verify: all Task 1-4 files.

**Interfaces:**
- Consumes: healthy CUDA device, local model/repository, and Task 4 harness.
- Produces: one immutable proof directory plus a checked-in report containing only values read from that proof.

- [ ] **Step 1: Run non-live regression gates**

```bash
cd /home/victoryang00/hetGPU/zluda/tests
python3 -m unittest -v test_run_mmfreellm_2p7b_benchmark.py \
  test_mmfreellm_continuous_batch.py \
  test_run_mmfreellm_continuous_batch_benchmark.py
bash test_run_mmfreellm_continuous_batch_evaluation_static.sh
```

Expected: all tests pass.

- [ ] **Step 2: Check live prerequisites**

```bash
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader
python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
test -d /root/matmulfreellm
test -d /root/.cache/huggingface/hub/models--ridger--MMfreeLM-2.7B/snapshots/77deff0c1c9ac79aa51eb3ab7dd34fc375bf9324
```

Expected: GPU 0 and both local paths are available.

- [ ] **Step 3: Run live proof into a new directory**

```bash
cd /home/victoryang00/hetGPU
proof_dir="/home/victoryang00/hetGPU/.proof/mmfreellm-continuous-batch-$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES=0 \
  zluda/tests/run_mmfreellm_continuous_batch_evaluation.sh "${proof_dir}"
printf '%s\n' "${proof_dir}"
```

Expected: exit 0 and `qualification-status.txt` is `pass`.

- [ ] **Step 4: Independently verify JSON and hashes**

Set `proof_dir` to the printed path and run:

```bash
python3 - "${proof_dir}/batch16/result.json" <<'PY'
import json, pathlib, sys
r = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert r["validated"] and r["qualification_passed"]
assert not r["fpga_tps_reported"]
assert (r["request_count"], r["max_batch_size"], r["max_new_tokens"]) == (64, 16, 8)
assert r["deterministic_generated_token_ids"] and len(r["runs"]) == 2
for run in r["runs"]:
    assert (run["completed_requests"], run["failed_requests"]) == (64, 0)
    assert run["total_generated_tokens"] == 512
    assert run["aggregate_tokens_per_second"] >= 200
    assert max(run["observed_batch_sizes"]) <= 16
    assert all(item["output"].strip() for item in run["requests"])
print([run["aggregate_tokens_per_second"] for run in r["runs"]])
print([run["latency_seconds"]["end_to_end"]["p95"] for run in r["runs"]])
PY
(cd "${proof_dir}" && sha256sum --check hashes/source.sha256)
(cd "${proof_dir}" && sha256sum --check hashes/artifacts.sha256)
```

Expected: all assertions and hash checks pass; two TPS and p95 values print.

- [ ] **Step 5: Write report from immutable artifacts**

Include Git SHA/status, GPU UUID/driver, PyTorch/CUDA/Transformers versions, model snapshot, dtype/backend, batch-1 mean/median TPS and semantic status, and a two-row batch-16 table with TPS, elapsed time, p50/p95/max queue/service/end-to-end latency, observed batches, peak memory, completed requests, and tokens. Record same-shape determinism, the non-gating `cross_batch_token_ids_equal` observation, and the threshold; explicitly state GPU-native BitLinear and `fpga_tps_reported=false`; and link the absolute proof path plus manifest, JSON, logs, inventory, and hashes.

Use only Task 5 proof values in the acceptance table; the preliminary 349.8 tok/s scaling probe is context, not qualification evidence.

- [ ] **Step 6: Run final verification and scoped diff checks**

```bash
cd /home/victoryang00/hetGPU
python3 -m unittest -v zluda/tests/test_run_mmfreellm_2p7b_benchmark.py \
  zluda/tests/test_mmfreellm_continuous_batch.py \
  zluda/tests/test_run_mmfreellm_continuous_batch_benchmark.py
bash zluda/tests/test_run_mmfreellm_continuous_batch_evaluation_static.sh
git diff --check
git status --short
git diff -- zluda/tests/mmfreellm_continuous_batch.py \
  zluda/tests/test_mmfreellm_continuous_batch.py \
  zluda/tests/run_mmfreellm_continuous_batch_benchmark.py \
  zluda/tests/test_run_mmfreellm_continuous_batch_benchmark.py \
  zluda/tests/run_mmfreellm_continuous_batch_evaluation.sh \
  zluda/tests/test_run_mmfreellm_continuous_batch_evaluation_static.sh \
  zluda/evaluation/2026-08-23-mmfreellm-continuous-batch-evaluation.md
```

Expected: tests pass, diff check is silent, and scoped diff contains no unrelated files.

- [ ] **Step 7: Commit measured evaluation**

```bash
cd /home/victoryang00/hetGPU
git add zluda/evaluation/2026-08-23-mmfreellm-continuous-batch-evaluation.md
git commit -m "docs: record MatMulFreeLM 200 TPS qualification"
```

The machine-specific `.proof` directory remains uncommitted; its absolute path and hashes are recorded in the report.
