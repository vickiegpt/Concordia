# Four-PACC XM batching and phase profiling

## Scope and safety

This change targets real Qwen3.5-35B-A3B Q4_K_M inference on four physical
PACC devices. No host reboot, device reset/rebind, firmware replacement, or
default runtime installation was performed. Each worker retains a distinct
512 KiB payload partition and the v54 owned-slot firmware (build 0x20036).
The host boot ID is 89a6c374-059a-40da-8c4f-c00179fd48d4.

All quantized matrix dot products still execute xsfmm_native_bf16 on XM.
This is not a claim that every graph operation runs on XM: existing host
data movement, activation quantization and pointwise shims remain. The
virtual CUDA memory capacity is not physical PACC resident memory.

## Phase definitions

HETGPU_SIFIVE_ELF_PROFILE=1 emits successful host submissions:

- setup: locks, bootstrap checks, descriptor/status setup.
- stage: host buffers to shared DDR, including initialized output buffers.
- publish: image/descriptor preparation, writes, readbacks, fences and IRQ.
- wait: completion wait, INCLUDING device execution.
- readback: shared DDR outputs to host.

HETGPU_SIFIVE_QK_PROFILE=1 emits Q4/Q5/Q8 guest durations from
CLOCK_MONOTONIC: local payload copy, activation preparation, weight
unpacking, XM invocation, reduction, and total kernel duration.
Do not add host wait and device execution: they overlap. The paired
publish+wait minus kernel duration includes image publishing, guest
setup/cleanup, scheduling, polling and completion visibility. It is not
an isolated interrupt latency measurement. Other ELF kernels currently
have host-only instrumentation. Device clock calls perturb performance;
the end-to-end comparison disables detailed timing.

The existing jobd phase ring covers PRELOADED submissions, not all ELF
execution. Its completion_publish statistic alone is not end-to-end
completion latency. Ring double reads do not prove source-side atomicity.

## Measured first-four-layer ablation

The diagnostic preserves the first four original layers and stops after
the first prompt token's fourth layer. It intentionally exits 3. These
are measured durations in seconds, NOT token throughput or a full-model run.

| Variant | QK submissions | Host total | Device total | Weight unpack | XM | Local copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original algorithm + profiling | 280 | 22.142701 | 21.056895 | 19.607863 | 0.153525 | 0 |
| Local input copy + expert grouping | 268 | 7.432483 | 6.361476 | 4.787681 | 0.156948 | 0.101662 |
| Also RVV weight unpack | 268 | 4.128337 | 3.053475 | 1.456813 | 0.170110 | 0.102880 |
| Also M=32 alignment, zero IRQ delay probe | 278 | 4.143524 | 3.021885 | 1.493391 | 0.148911 | 0.104447 |

Baseline host stage/readback were 0.089539/0.012428 seconds. Weight
unpacking accounted for about 93% of the measured QK device duration.
The publish+wait residual remained about 0.95-0.99 seconds across variants.
Removing the 1000 us post-doorbell sleep shifted time from publish to wait,
without a demonstrated total-time gain. The final runner retains 1000 us.
M=32 alignment reduced native XM invocations from 130432 to 118784, but
does not itself establish a latency win. Bulk local copies and vector
unpacking account for the demonstrated improvement.

All four layer output tensors remained byte-identical to the original
PACC runtime. This is not a claim of bitwise identity with CPU execution.

## Implementation and batching boundary

- Bulk-copy only input payloads into bounded PACC-local stack storage
  using RVV, then unpack locally. Cache limit is 512 KiB; caller reserves
  output/diagnostic/image space and rejects oversized raw single tiles.
- Decode Q4_K/Q5_K/Q8_0 weights and form exact BF16 integer operands with
  RVV. Quantization scales, zero-point correction and reduction semantics
  are unchanged. Guest VLEN is 1024; host VLEN is not assumed to match.
- Group expert channels only when their activation pointer is identical.
  Concatenate their M rows into XM submissions and scatter back to the
  original expert ordering/strides. Align sufficiently large tiles to 32
  rows to avoid repeated partial native tiles.
- Preserve checks, bounds, completion waits and error propagation.

This is M-dimension expert batching, not cross-request N batching.
The real-model four-request runner retains ubatch=1. Native kernel tests
separately cover genuine N=8 XM matrix execution. Large-batch serving and
500 TPS are NOT established by this change.

## Validation

Q4/Q5/Q8 hardware tests cover non-square shapes, N=8, K=32768, gated
outputs, nonzero biases, reverse expert selection, output padding guards,
oversized tile rejection and distinct activation pointers. Final v5
tests pass. The separate distinct-input case requires two input groups;
it must never be merged as a shared-input batch.

The four-card initial launch failed before inference because the helper
driver uses a global mutex_trylock for the shared DDR base read. The
profiler now retries EBUSY for at most five seconds, still validates the
base, and fails immediately on short reads/other errors. No driver was
reloaded. Unit tests cover transient busy, deadline, I/O failure and
short-read rejection, as well as throughput and partition-audit gates.

## Reproduce

Build in Concordia-xm-release using the qualified LLVM20 guest compiler:

```sh
env PATH=/home/ubuntu/.cargo/bin:/usr/local/bin:/usr/bin:/bin \
  LLVM_ZLUDA_PREBUILT=/usr/lib/llvm-20 \
  CARGO_TARGET_DIR=/home/ubuntu/Concordia/target-pacc-compile \
  cargo build --locked -j4 --release -p zluda \
  --no-default-features --features sifive,embed_cudart
```

Candidate: /srv/lanxin/qwen35-a3b/runtime-xm-local-batch-v5.
libnvcuda.so SHA256:
4f342c23f756e75cff208f846c198f1d83737e3a462ef02d162b8ee46119117f.
The currently installed default runtime was not overwritten.

```sh
bash /srv/lanxin/qwen35-a3b/bin/run-four-pacc-batched.sh UNIQUE_PREFIX
python3 /srv/lanxin/qwen35-a3b/bin/summarize-elf-profile.py LOG SUMMARY.json
```

The launcher requires idle devices, the qualified boot/allocation, and
the unchanged CPU reference. It numerically qualifies each physical
device before inference. Throughput is null if any token, process,
firmware build or profile validity gate fails.

## End-to-end comparison

The previous identical four-request, two-token ngl99 baseline produced
[17,220] ("2 ") on all four devices: aggregate output 0.005046462 TPS,
aggregate decode-window 0.009212154 TPS. These exclude model loading;
output throughput includes prefill. Decode includes worker skew.

Final run: logs/four-ngl99-xm-batch-v5-retry. All four processes exited 0,
generated [17,220], and passed CPU-reference token checks and firmware /
capture validity gates. There were no observed ring index gaps.

| Metric | Previous runtime | Optimized v5 | Speedup |
| --- | ---: | ---: | ---: |
| Aggregate output TPS, includes prefill | 0.005046462 | 0.019416933 | 3.85x |
| Aggregate decode-window TPS | 0.009212154 | 0.041391351 | 4.49x |

Eight output tokens occupied a 412.011526 second request window. Four
decode tokens occupied a 96.638546 second window, including worker skew.
Outer elapsed time, including model load and profile collection, was
473.502352 seconds. These are one-run comparisons, not confidence intervals.

| Physical PACC | Prefill seconds | One decode step seconds | Token IDs |
| --- | ---: | ---: | --- |
| 0 | 315.288815 | 79.477684 | 17,220 |
| 1 | 322.394271 | 81.785072 | 17,220 |
| 2 | 328.762672 | 83.158886 | 17,220 |
| 3 | 322.434604 | 81.448874 | 17,220 |

The end-to-end run also captured 17408 complete PRELOADED submissions
per device (separate from the 19650 logged ELF submissions). Their
device-local phase sums are shown below in seconds. They exclude ELF
work and must not be treated as a full-request breakdown or summed
across concurrent devices to obtain wall time.

| PACC | Input copy | Compute | Output copy | Completion publish | Publish p99 (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 9.378843 | 82.922885 | 0.446156 | 0.086253 | 7 |
| 1 | 10.189697 | 86.595567 | 0.465174 | 0.078698 | 5 |
| 2 | 10.882337 | 90.088456 | 0.471093 | 0.086605 | 7 |
| 3 | 10.052810 | 85.483574 | 0.459389 | 0.084030 | 6 |

Publish p99 is the device's publication operation, NOT the time until
host observation or an end-to-end completion latency qualification.

The final v5 distinct-input test also passes with groups=2, gate both
disabled/enabled, nonzero biases, 512 rows, K=2048 and intact guards.
The archived evidence includes phase summaries, hardware tests, layer
tensor hashes/equality, runtime hashes, per-device results, qualification
captures and a logged-span partition audit. Large raw rings and model
weights remain on the host, not in git.

Two tokens are a correctness/performance smoke, not sustained-generation
qualification. Neither 500 TPS nor cross-request N batching is achieved.
The measured next optimization targets are remaining weight unpacking,
scale/reduction work, and per-tile submission costs; the data does not
justify blaming interrupt handling alone.
