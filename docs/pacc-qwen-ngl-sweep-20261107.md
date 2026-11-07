# Qwen offload-depth test, 2026-11-07

## Configuration

lanxin2, Qwen3.5-35B-A3B-Q4_K_M, PACC0 firmware v49_vlen1024 (0x20031).
Four concurrent requests, count16, CPU threads=8, batch=256, ubatch=4,
context=1024, flash attention off. Only ngl changes. Strict error propagation
and completion waits remain enabled. No host reboot, driver rebind, firmware
replacement, physical-address change or payload-size override was performed.

At preflight the host had approximately 246 GiB available. Device allocations
remain host-backed; the owned shared-DDR allocation is 4 MiB, with a 2 MiB
control reservation and 2 MiB payload. Host free RAM and shim VRAM reporting
are not PACC-local capacity or permission to enlarge this DMA aperture.

## Observed failures

| ngl | Four-request count16 outcome | Correct output tokens | TPS |
| --- | --- | ---: | ---: |
| 1 (regression after both failures) | Passed | 64 | 1.043038 |
| 2 | Startup abort, oversized Q8 submission | None generated | Not valid |
| 3 | Startup abort, oversized Q8 submission | None generated | Not valid |

The ngl=1 regression completed in 61.359227s with 8245/8245 job-5 submissions,
no phase gaps, and the expected v49 build marker. The previous 8-request,
64-token-per-request result was 1.968910 TPS; different cohorts must not be
compared as a change caused by ngl.

Both ngl=2 and ngl=3 exited with SIGABRT (-6) during the server's startup
context check, before readiness and before the concurrent token requests.
There is no valid TPS for either configuration.

The immediate fatal operation in both cases was:

```
mul_mat_vec_q<ggml_type=8, ncols=2, no fusion>
grid=(4096,1,1), block=(32,4,1)
ELF bytes: 5,652,232
Total required submission bytes: 23,549,120
Actual payload bytes: 2,097,152
```

Type 8 is Q8_0 weights and this MMVQ uses Q8_1 activations. Inspection of
the running runtime source shows the native tiled MMVQ interception accepts
only type 14 (Q6_K). Type 8 instead reaches generic ELF/argument staging,
which correctly rejects the oversized submission. It is not a model-weight
allocation failure, a new sf.vtzero SIGILL, or evidence of a completed Q8
kernel. Even the current ELF alone exceeds the payload, so just reducing
request batch size cannot fit this unchanged image.

There is also an earlier PTX parse warning for `.noreturn` and NVFP4-related
unavailable-device stubs in a module. Dispatch proceeds past that warning
and fails at the Q8 submission-size check. Keep these separate; enlarging
the payload would not prove parser support, ABI correctness or executable
warp semantics for this newly reached kernel.

The job-5 phase collector reports zero completed MMVF jobs for these failed
startup runs. RoPE submissions appear earlier in the server logs, so this
must not be described as zero total device activity. Failure JSON has
`all_correct=false` and `aggregate_output_tps=null`.

## Changes and next gate

The benchmark tools now accept ngl explicitly, preserve failed-startup JSON,
refuse existing output prefixes and allow bounded startup/request timeouts.
ngl=1 keeps its established expected job counts. Higher ngl values do not
reuse those counts; they still enforce per-observed-job completeness, build
identity, no index gaps and token correctness when a run reaches requests.

Do not increase the shared-DDR environment size without a new owned
allocation and matching firmware mappings. The preferred next implementation
is a bounded native Q8_0 x Q8_1 tiled submission path, tested against GGML
dequantization across shapes/strides/tails and then exact Qwen tokens. That
requires real XM execution and scale semantics, not enabling host matmul or
success-on-error. Additional MoE/attention kernels may become the next gate;
passing Q8 alone does not establish arbitrary ngl support.

Evidence lives in `tools/pacc-qwen35/evidence-ngl-sweep/`; complete raw profiles
remain under `/srv/lanxin/qwen35-a3b/logs` and are hashed in the manifest.
