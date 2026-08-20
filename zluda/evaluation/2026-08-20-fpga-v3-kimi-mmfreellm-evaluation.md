# FPGA CXL v3, Kimi IQ1_S, and MatMulFreeLLM 2.7B Evaluation

Date: 2026-08-20 UTC

## Software acceptance

The four-lane source-stable software harness completed at
`/tmp/hetgpu-v3-max-lanes4-software-20260820-final` with every mandatory gate at
status zero. The accepted Rust test counts are:

- integration contract: 3;
- batch planner: 13;
- IQ1_S capture, staging, reconstruction, and live-fixture contract: 26;
- CXL v3 UAPI/session validation: 34;
- NVIDIA completion logging: 4.

The harness also passed the NVIDIA/evaluation `cargo check`, strict workload
timing parser tests, completion-evidence validator tests, and the real
MatMulFreeLLM benchmark-driver tests. Its `summary.json` reports
`software_gates_passed=true`, `tps_claimed=false`, and no live fixture or live
workload success.

## Live FPGA boundary

`QUERY_CAPS_V3` succeeds on `/dev/cxl_tmatmul3b001` and reports version 3,
16 instances, dimension 2048, max batch 256, max descriptors 256, one in-flight
submission, a 32-GiB DAX aperture, and a 400-MHz accelerator clock.

The host-captured batch-four IQ1_S fixture reaches registration, commit,
submission, and wait, but request 1 completes with status `-5` (`EIO`). A BAR0
snapshot after the failure shows lane 0 direct-descriptor status `0xff` and
wide-DMA status `0xff`; lanes 1 through 3 remain `0x01`. Raw log:
`/tmp/hetgpu-v3-live-fixture-counter-20260820.log`, SHA-256
`d8a6cb94d93b899f710963c2ae2e3ab77f49b61d717037e903ceaea70aa1f9b7`.

The loaded driver clamps usable lanes to `TMATMUL_V3_MAX_LANES = 4` even though
capability discovery reports 16 instances. That four-lane maximum is now the
intentional userspace and evaluation policy: only lanes 0 through 3 are valid,
and completion evidence contains exactly four counters. It is no longer an
acceptance blocker. No live throughput claim is valid until the independent DMA
error is corrected and the bit-exact four-lane fixture passes.

## Pure MatMulFreeLLM 2.7B (1.58-bit BitLinear)

The benchmark uses actual deterministic `model.generate` calls, excludes model
load and warmup from measured intervals, counts generated tokens, and emits a
strict parser-compatible summary plus canonical JSON.

| Path | Conditions | Result |
| --- | --- | --- |
| CUDA reference | FP16, one warmup, 3 measured runs, 8 forced tokens/run | 19.78 tok/s mean; 24 tokens; semantic output present |
| Matched CPU reference | FP32, Triton disabled, no warmup, 1 measured forced token | 0.148 tok/s; semantic output `The quick brown fox in` |
| TernIP CXL v3 | Same CPU host path, FP32, one forced token | Failed closed at request 1 with `EIO (-5)`; no TPS |

Structured baseline records:

- `/tmp/mmfreellm-2p7b-cuda-baseline-20260820.timing.json`, SHA-256
  `df4765fceff670b239d7340545bebd7315872aa446de293c2d37cc18d5b94550`;
- `/tmp/mmfreellm-2p7b-cpu-baseline-short-20260820.timing.json`, SHA-256
  `20cd5b415e27cecdcc33127101af8318992c8dae751cb4ab295bef8b8a80621e`.

Enabled failure log:
`/tmp/mmfreellm-2p7b-fpga-short-after-msync-20260820.log`, SHA-256
`f56b52e151a84e2e3f0d36bcf76eef7c383ef002539156d18eb239c61e143a04`.

Before reaching hardware, the MatMulFreeLLM adapter incorrectly called
`mmap.flush()` on devdax and received `EINVAL`. The adapter now uses direct
shared-mapping stores; all 67 CXL-v3 adapter tests pass. The rerun then exposed
the real device `EIO` above.

## Hybrid Kimi K2.6 IQ1_S

There is no validated `capability-v2.json` under
`/root/matmulfreellm/.proof/kimi-k2.6-iq1s-v3`. The strict smoke entry exits
nonzero with `missing validated backend capability`, emits no route/completion
evidence, and therefore emits no TPS. Raw wrapper log:
`/tmp/kimi-k2p6-iq1s-hybrid-smoke-no-capability-20260820.log`, SHA-256
`34105aae15bd5d64848a8a7345d09640051552b19f45e15e6da72ccb57384225`.

Earlier non-proof Kimi attempts are not accepted: they fell back after Q8_1
metadata validation failures and reported zero evaluation milliseconds with
infinite TPS. Qualification must be regenerated only after a successful live,
bit-exact FPGA run. This is Kimi K2.6 IQ1_S scope, not Kimi K2.7.
