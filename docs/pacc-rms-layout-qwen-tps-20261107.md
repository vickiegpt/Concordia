# Qwen3.5-35B-A3B: strided RMSNorm and correct hybrid inference

## Scope and hardware state

This experiment uses lanxin2, PACC0 and the existing v50_q8_tiles image
(build marker 0x20032). Four XM workers are inside PACC0 for Q6; Q8 currently
uses one worker. This is not four PACC devices, GPU inference or 500 TPS.
The host was not rebooted, firmware was not reloaded and the unrelated CPU
llama-server PID 4732 was left untouched. No unowned DDR address was used.

The pinned allocation remains 0x2075d000000, size 4 MiB, with a 2 MiB payload.
The harness refuses a different host boot ID. The canonical vendor firmware
SHA256 remains 19cdacf4ddfa8f6861477daa2be79f3e1e52d1a11780c7a209a9419e4a2a83a8.

## Fix and verification

The Q8 tiled path exposed the next error: GGML RMSNorm input is not always a
flat rows-by-hidden array. The observed hidden=256 case has grid [16,2,1]
and element strides [512,8192,16384]. Treating those rows as contiguous reads
the wrong elements. The new layout ABI carries three row/channel/sample
strides plus weight shapes and strides. Gathering and weight repetition run
on the host; normalization arithmetic runs on PACC through the existing
RMSNorm job. Shared weights are batched; varying weights use individual rows.

The native dispatch recognizes the GGML F32 uint3 ABI, validates allocation
spans and rejects unsupported fused-add mode. Output is staged separately
and committed only after every device result succeeds and is finite. The
new entry point rejects host RMS fallback, skipped job waits, disabled shared
staging and assumed-success-on-output-timeout policies. No CPU matmul was
added as a failure fallback.

Two host unit tests cover packing, broadcast, shared-weight batching,
overflow, short inputs and failed/unwritten callback outputs. Four hardware
cases cover strided 256-wide rows, repeating weights, multi-channel/sample
layout and 2048-wide rows. Maximum absolute errors are respectively
1.15795692e-7, 1.15795692e-7, 8.93754595e-8 and 1.07935459e-7.
Unsafe policy checks must return errors without modifying the output.

## Placement and measurement

Model: Qwen3.5-35B-A3B-Q4_K_M.gguf. The working configuration is explicitly
mixed CPU/PACC: ngl=2 plus `--no-kv-offload` and
`--override-tensor 'blk\.[0-9]+\.ffn.*=CPU'`. KV attention, FFN and model
blocks outside the offload depth remain on CPU. PACC0 executes Q8 QKV and
attention-output projections, the Q6 final projection and supported other
device operations. Existing shim pointwise and activation-quantization host
operations remain. This must not be called full-layer PACC inference.

Each result is one synchronized concurrent cohort, with deterministic prompts,
temperature zero, seed 42 and no prompt caching. Aggregate TPS is the total
validated completion tokens divided by earliest-request-start to latest-finish
wall time. It includes prefill and excludes server startup. These are bounded
correctness/throughput tests, not arrival-rate saturation measurements.

The first working count16 cohort used four requests: all 64 tokens matched,
81.044775251 seconds, 0.789686933 TPS. Its job-5 capture observed 9349 complete
submissions with no ring-index gaps and the expected v50 build. Device phase
sums were 31.016635 s compute, 10.943292 s input copy, 1.540567 s output copy
and 0.047335 s completion publication. These are job-5 stage durations, not
all operations or hardware utilization. The initial candidate predates the
additional unsafe-policy guards; its binary hash was not retained separately.

The hardened candidate's count64 cohort used eight requests, batch 256,
microbatch 8, context 2048 and eight host threads. All eight responses passed
the exact-prefix, length-stop and 64-token checks: **512 valid output tokens
in 320.387549972 seconds = 1.598064594 aggregate TPS**. Token-capped text ends
partway through the next integer, which is an expected length stop, not a
claim of completing the requested 1..100 sequence.

The long capture observed 33940 complete job-5 submissions, no incomplete
phase sets and zero observed ring-index gaps. Build matching and capture
validation passed. Its device-stage totals are:

| Stage | Sum (s) | p99 (microseconds) |
| --- | ---: | ---: |
| Input copy | 39.045478 | 2936 |
| Compute, including unpacking and reduction | 126.842676 | 25973 |
| Output copy | 11.067520 | 356 |
| Completion publication only | 0.174519 | 7 |

Do not interpret the 7 us publication p99 as end-to-end completion latency.
The job-5 device-before-publish total is 184.546412 s; the difference from
cohort wall time also includes CPU/model work, other device jobs, transfers,
scheduling and polling. It cannot all be attributed to interrupts or DMA.
Compared with the earlier v49 ngl=1 eight-request count64 result of 1.968910
TPS, this deeper hybrid configuration is slower, despite executing more
matrix operations on PACC. The runtime and placement differ, so this is not
an isolated RMSNorm speedup experiment or a new peak-throughput claim.

## Remaining full-offload blocker

Without explicit CPU KV/FFN placement, ngl=2 completes Q8 and strided RMSNorm
but fails at `_ZL9mul_mat_fI7__half2Li32ELi2ELi2ELb0EEvPKT_PKfPKiPfiiiiiiiiiiiiiiii`.
The module parser rejects TF32 MMA and .noreturn constructs, and the runtime
correctly refuses the resulting empty ELF. The native `mul_mat_f` dispatch
does not yet cover this FP16 attention case. That run has null TPS and must
not be counted as valid inference. The hybrid placement avoids this unsupported
operation by explicit scheduling, not by claiming it was repaired.

The run named `ngl2-rms-cpu-kv-ffn-p4` started before its updated wrapper had
finished transferring. Its actual recorded argv lacks the placement flags;
exclude it from hybrid comparisons. `ngl2-rms-explicit-cpu-kv-ffn-p4` is the
correctly configured successful short run.

## Reproduction and provenance

Run from /srv/lanxin/qwen35-a3b on the already-qualified current host boot:

```sh
CONCORDIA_RELEASE=/srv/lanxin/qwen35-a3b/runtime-rms \
  bash bin/run-xm-qwen-benchmark.sh new-unique-prefix 0x20032 8 count64 2 cpu-kv-ffn
```

Do not overwrite an existing prefix or reload firmware merely to run this
test. Select the candidate directory explicitly; installed default libraries
were not replaced. The hardened libnvcuda SHA256 is
b7e77648f72712deefcf2f147b993676063086934f128cd72d4c4e4acc2c3ef2.
It was built with rustup Cargo, LLVM_ZLUDA_PREBUILT=/usr/lib/llvm-20,
release mode, and features sifive,embed_cudart in the original existing
checkout. That checkout has unrelated edits which were preserved. This is
not a clean build of the release branch, and the separately committed JIT
optimization changes must not be inferred to be in this binary.

The evidence-rms-layout manifest identifies source files, runtime binary,
firmware and full raw logs. Compact cohort results and phase summaries are
committed; large raw phase captures stay under the evaluation tree's logs/.
Telemetry uses repeated stable reads rather than an atomic source-domain
snapshot. For this new placement there is no independently established total
job count; capture validation checks observed phase completeness, build marker
and ring-index continuity. It cannot prove the absence of every possible
uncaptured device event.
