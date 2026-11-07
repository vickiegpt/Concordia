# PACC0 XM Qwen3.5 Performance Follow-up

## Scope

Host: `lanxin2` (`wr5219`). Host boot ID stayed
`89a6c374-059a-40da-8c4f-c00179fd48d4`. No host reboot or helper-module unload.
Only PACC0 was rebound through its vendor driver during firmware experiments.
PACC1-3 were not changed. The canonical vendor image was restored after each load.

Model: Qwen3.5-35B-A3B Q4_K_M. SHA256:
`3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375`.
The existing `ngl=1` CPU/PACC split and CUDA shim were retained. This is not
whole-model XM, GPU+4PACC, or a 500 TPS capacity result. Q6_K matrix products
execute on real XM; RVV handles packing, copying and scaling, not a matmul fallback.

## Implemented

- Exact RVV Q6 unpacking: six-bit integers become exactly representable BF16
  operands. Q6 integer subblock scales remain outside the K=16 XM dot products.
- RVV payload copying, restricted to the owned shared-DDR data aperture.
  Control/MMIO access widths and existing coherence fences are unchanged.
- Four persistent guest worker threads with disjoint row partitions. These are
  four cores inside PACC0, not four PACC devices. Initialization failure fails closed.
- Exact RVV integer scale combination and separate FP32 multiply/add steps.
  Non-finite, fractional and out-of-range XM results remain errors.
- Success-path console logging no longer blocks every tile. Error reporting
  and shared-memory phase diagnostics remain enabled.
- A bounded 2 ms guest hot-scan window; normal idle behavior resumes afterward.
  Host busy polling and disabling response polling were tested without meaningful
  benefit and are not selected defaults.
- Per-job activation-pack, weight-unpack, XM-leaf, reduction and worker-count
  telemetry. Detailed times sum overlapping worker elapsed durations; they are
  not matrix-unit utilization or directly additive to device wall time.
- 1 ms optional host sampling and linear-time completion-record classification.
  A bounded four-ring de-duplication window and deferred cyclic collection avoid
  ever-growing hash-table/heap pauses during long captures.

## Matrix Results

All eight versions below passed the same five-shape numerical suite and 493
sequence/build-matched submissions. Copy and compute columns cover the entire
suite; the last column covers only the complete 248320x1x2048 output projection.
These are single observations, not confidence intervals.

| Version | Incremental change | Device input copy (s) | Device compute wall (s) | Full projection runtime (s) |
| --- | --- | ---: | ---: | ---: |
| v39 | Detailed instrumentation | 5.426310 | 8.949705 | 15.034053 |
| v40 | RVV unpack with lookup | 5.174893 | 4.569037 | 10.435118 |
| v41 | RVV payload copy | 0.593416 | 4.565507 | 6.221180 |
| v42 | Exact vector float packing | 0.605188 | 3.206986 | 6.218489 |
| v43 | Four XM workers | 0.595354 | 1.158567 | 6.488392 |
| v44 | Bounded hot scan | 0.536988 | 1.223923 | 6.192541 |
| v45 | Quiet successful dispatch | 0.548510 | 1.190374 | 2.474343 |
| v46 | Vector scale reduction | 0.501344 | 1.013595 | 2.252206 |

The prior v38 projection took 14.769347 s: v46 is 6.56x faster on that runtime
measurement. The test program's separate PASS elapsed time includes CPU reference
dequantization and comparison; it is not accelerator runtime.

The v44-to-v45 change isolated a large console-logging cost. The unexplained
host wait residual dropped from 4.073 s to 0.389 s over 485 tiles. This does not
prove that all remaining waiting is an interrupt or DMA defect. Pure completion
publication in the v46 suite had nearest-rank p99 5 us and max 7 us; it excludes
notification, diagnostics and host observation latency.

## Token Results

Throughput is total validated output tokens divided by the concurrent cohort's
wall time. It includes request prefill and excludes server startup. Temperature 0,
seed 42, eight CPU threads, batch 256, microbatch equal to parallel requests.

| Firmware | Requests | Output tokens/request | Correct total | Wall (s) | Aggregate TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| v38 baseline | 4 | 16 | 64 | 304.451023 | 0.210214 |
| v45 | 4 | 16 | 64 | 73.212614 | 0.874166 |
| v46 first run | 4 | 16 | 64 | 61.532657 | 1.040098 |
| v46 repeat, 1 ms sampling | 4 | 16 | 64 | 61.375528 | 1.042761 |
| v46 first long run | 8 | 64 | 512 | 260.330737 | 1.966729 |
| v46 long repeat, 1 ms sampling | 8 | 64 | 512 | 260.012564 | 1.969136 |
| v46 long repeat, bounded collector | 8 | 64 | 512 | 261.220700 | 1.960028 |
| CPU only, devices disabled | 8 | 64 | 512 | 60.506874 | 8.461848 |

The clean four-request repeat is 4.96x the v38 baseline. Count16 checks the exact
token-capped English sequence. Count64 checks an exact prefix of the requested
comma-separated integers, exactly 64 output tokens and a length stop.
The eight-request CPU-only baseline uses `--device none`, `ngl=0`, no preload,
and emitted no PACC launch/projection records. **The current CPU/PACC hybrid is
still about 4.3x slower than this CPU-only baseline.** Improvement over the old
XM path must not be represented as a model-level acceleration over CPU.

The initial four-request capture lost two internal detail events, although all
8,245 phase sets and output tokens were present. The first long capture lost
two complete phase sets (31,523/31,525); the 1 ms long repeat had the same count.
The client still verified completions
and all 512 generated tokens. Those captures are retained as incomplete telemetry,
not presented as complete fine-grained breakdowns. The four-request repeat has
8,245/8,245 phase sets and zero ring-index gaps.
After bounding the collector's de-duplication window and deferring cyclic GC,
the final long run captured **31,525/31,525** phase sets, matching `0x2002e`
on every job, with **zero ring-index gaps**. Maximum observed host sampling gap
was 6.495 ms. Its complete timing evidence supports the conservative 1.960 TPS
result. All three long runs produced 512 correct tokens and measured 1.960-1.969 TPS.
Changing polling frequency alone did not fix the earlier loss; the bounded
collector did in this run. This is not a general atomic-snapshot guarantee.

On the final long run, device input copy totaled 35.134 s, compute wall 86.804 s,
output copy 10.314 s, and completion publication 0.155 s. Publication p99 was
7 us, max 61 us; this is not end-to-end submission/completion latency.

The clean four-request run spent 42.676 s of 61.376 s in the 17 output
projections (about 69.5%). Within that path: device input copy 8.808 s,
compute wall 18.727 s, output copy 1.363 s, host packing 1.241 s, host writes
3.158 s, readback 0.554 s, and wait residual 7.010 s. These are nested stages,
not values to add to the 42.676 s. Other model work and request overhead account
for the remaining 18.700 s; that remainder is not an isolated CPU kernel time.
Across workers, weight unpack totals 45.072 s versus 10.998 s in the XM leaf;
these overlapping sums point to unpacking but are not device utilization.

## Validation and Limits

- Numerical reference is the actual GGML Q6_K dequantizer, with independent double
  accumulation, padded input/output strides and untouched output guards.
- Uneven-worker shapes 128x3x256, 129x5x2048, 513x7x2048 and 31x3x8192 passed.
- Copy tests passed 648 alignment/length cases and 514 guard-page tail cases.
- Fault injection rejected seven invalid XM values, leaf failure and invalid shape;
  valid zero output passed. This host-only fake leaf is never linked into firmware.
- Five diagnostic unit tests passed. Firmware and host harness builds passed.
  A clean rebuild of the entire Rust/LLVM/llama stack was not performed.
- Double-read ring stability is not a source-side atomic snapshot. Completeness,
  build markers and gap counts must be checked for each capture.
- Measurements used a live shared host, not an isolated performance lab. No clocks,
  topology or theoretical peak extrapolation is presented as measured throughput.

## Current Deployment and Reproduction

Running image: `v46_xm_reduce`, build marker `0x2002e`. Image SHA256:
`f502ba83c312cbe121cac3cc4144e63815537bb21592f5e84604fc036507bf59`.
Jobd source SHA256:
`2d467620cbaf853259c0d6c39a9d2df16b6b06a920c1afa611a2cfd477954560`.
Pinned owned DDR: `0x2075d000000`, 4 MiB. This address is valid only for the
checked host boot, not a persistent allocation.

Run from `/srv/lanxin/qwen35-a3b` after matching image/ownership qualification:

```sh
bash bin/run-xm-validation.sh unique-matrix-run 0x2002e
bash bin/run-xm-validation.sh unique-tail-run 0x2002e --hardware-workers
bash bin/run-xm-qwen-benchmark.sh unique-p4-run 0x2002e 4 count16
bash bin/run-xm-qwen-benchmark.sh unique-p8-long-run 0x2002e 8 count64
```

Sources and bounded evidence are under `tools/pacc-qwen35` in branch
`fix/pacc-xm-context-qwen35`. Large raw captures and logs remain in
`/srv/lanxin/qwen35-a3b/logs`; evidence manifests contain their hashes.

The next substantive targets are persistent validated weight/packed-tile reuse,
larger N batching with coordinated host/device ABI checks, fewer tiny XM leaf
calls, and profile-guided optimization of the remaining model graph. None is
claimed implemented by this release, and 500 TPS remains unachieved.
