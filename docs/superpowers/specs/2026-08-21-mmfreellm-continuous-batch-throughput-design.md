# MatMulFreeLM Continuous-Batch Throughput Design

## Goal

Reach at least 200 aggregate generated tokens per second for the local
`ridger/MMfreeLM-2.7B` model on the NVIDIA GPU with at most 16 concurrent
sequences. Preserve greedy-generation behavior, report per-request latency
separately from aggregate throughput, and fail closed when CUDA execution,
token accounting, or semantic output cannot be validated.

The measured starting point on the RTX PRO 6000 Blackwell Server Edition is
21.7 tokens per second at batch 1, 174.5 aggregate tokens per second at batch
8, and 349.8 aggregate tokens per second at batch 16 for eight generated
tokens per request. The design therefore reaches the requested aggregate
target through bounded batching rather than changing model weights or
quantization math.

## Scope and terminology

This design adds windowed continuous batching to the existing MatMulFreeLM
evaluation path. Requests may arrive continuously, enter a bounded queue, and
leave the queue in microbatches. A microbatch is dispatched when it reaches 16
requests or when the oldest queued request reaches the configured queue
timeout. New requests may fill the next microbatch while the GPU executes the
current one.

The current Hugging Face model executes one complete `model.generate` call per
microbatch. It does not admit new sequences into a generation already in
progress. The implementation and evidence must call this windowed continuous
batching, not iteration-level sequence insertion.

The default benchmark uses 64 requests, a maximum batch size of 16, eight
generated tokens per request, greedy decoding, and a short queue timeout.
Batch size may be lowered for latency experiments but may not exceed 16 in the
qualified profile.

## Architecture

`zluda/tests/run_mmfreellm_continuous_batch_benchmark.py` owns the executable
benchmark and its structured evidence. It loads the tokenizer and model once,
warms the selected batch shape outside measured intervals, starts one producer
and one GPU worker, and measures the interval from the first request enqueue to
the last completed response.

The producer creates request records containing a stable request ID, prompt,
requested output-token count, and enqueue timestamp. It supports immediate
arrival for peak-throughput qualification and a configurable inter-arrival
delay for latency experiments.

The scheduler owns a condition-protected FIFO queue. It selects at most 16
requests in enqueue order. It flushes a partial batch only after the oldest
request reaches the queue timeout or the producer closes the input stream.
There is exactly one model worker and therefore no concurrent calls into the
same model object.

The model worker tokenizes each selected prompt with padding, moves inputs to
CUDA, synchronizes before and after generation, and calls `model.generate`
under `torch.inference_mode()` with:

- `do_sample=False`;
- equal positive `min_new_tokens` and `max_new_tokens`;
- the tokenizer pad-token ID;
- the existing default BitLinear backend.

Each response records dispatch, completion, queue, service, and end-to-end
latencies. The worker decodes every response and verifies its generated-token
count before making the result visible to the benchmark summary.

## Metrics and evidence

Aggregate TPS is total validated generated tokens divided by measured
end-to-end wall time. It is never computed from parameter-count FLOP estimates,
prompt tokens, model-load time, or an incomplete request set.

The canonical JSON record includes:

- model path, device, dtype, and parameter count;
- requested, completed, and failed request counts;
- total generated tokens and end-to-end elapsed time;
- aggregate tokens per second;
- configured and observed batch sizes;
- microbatch count and per-microbatch GPU service times;
- queue, service, and end-to-end latency p50, p95, and maximum;
- peak CUDA memory allocation;
- every request ID, generated-token count, timing fields, and decoded output;
- the BitLinear backend and an explicit `fpga_tps_reported=false` field.

The benchmark writes its record to stdout with a stable prefix and optionally
to a caller-selected new result file. It refuses to overwrite an existing
result.

## Correctness and failure behavior

CUDA is mandatory. The benchmark rejects CPU execution, a missing CUDA device,
non-finite or non-positive timing, duplicate or missing request IDs, output
count mismatches, empty decoded output, early process failure, and aggregate
TPS below the caller-selected acceptance threshold.

Greedy output must be deterministic across two runs of the same qualified
batch shape. Cross-batch token equality is recorded but is not an acceptance
requirement because different GEMM batch shapes can select numerically distinct
but valid greedy tokens. Every output must retain the non-empty input prompt
and append exactly the requested number of generated token IDs.

The benchmark does not enable the CXL/FPGA backend. It explicitly records the
GPU-native BitLinear route and never attributes its throughput to Agilex. The
existing `MAX_LANES = 4` FPGA policy remains unchanged and outside this GPU
throughput acceptance boundary.

## Testing and acceptance

Unit tests use a fake tokenizer, fake CUDA/model boundary, and controlled clock
to cover FIFO ordering, full and timeout-triggered partial batches, producer
closure, exact token accounting, percentile calculation, result-file refusal,
and every fail-closed condition.

The live acceptance sequence is:

1. Run the existing batch-1 benchmark as a control and require valid semantic
   output.
2. Run two identical batch-16 qualification passes with 64 requests and eight
   generated tokens per request.
3. Require every request to complete, every output to be non-empty, and token
   IDs to be deterministic between the two qualified passes.
4. Require at least 200 aggregate generated tokens per second in each pass.
5. Report per-sequence and p95 end-to-end latency without treating aggregate
   throughput as batch-1 latency.
6. Preserve raw stdout, stderr, environment, JSON summary, source hashes, and
   artifact hashes under a new proof directory.

If the live target is missed, the result remains a valid measurement but the
qualification status is `failed`; the harness must not relabel partial or
estimated performance as 200 TPS.
