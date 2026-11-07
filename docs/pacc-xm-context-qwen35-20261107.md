# PACC XM Context Repair and Qwen35 Results

Measured on lanxin2 / wr5219, 2026-11-07 UTC. No host reboot was performed.
Host boot ID remained `89a6c374-059a-40da-8c4f-c00179fd48d4`.

## Outcome

The `sf.vtzero.t` SIGILL is resolved. Genuine XM BF16 batch arithmetic and
packed Q6_K x Q8_1 output projection now pass numerical tests. Qwen3.5-35B-A3B
produced 64 validated output tokens across four concurrent requests at
**0.2102144358 aggregate output TPS** (304.451023 seconds).

This is a correctness-qualified first XM implementation, not a speedup or a
500 TPS result. It is CPU model blocks/preprocessing plus **PACC0** output
projection/RMSNorm, `ngl=1`, not all-model XM, four-PACC, or GPU+PACC inference.

## Cause and Controlled Experiments

| Image | Change | Result |
|---|---|---|
| v35 | Configuration snapshots before tile access | BF16/F16/F32/I8 configuration accepted, vill=0; all fault at `0x43e06057` (`sf.vtzero.t`) |
| v36 | Only 80 DT bytes: existing xlxmm aliases normalized to xsfmm | All four tile-zero/discard probes pass; guest reports xsfmmbase; BF16 numerical probe still fails |
| v37 | FP32 output-store configuration corrected | Four shapes, 64 consecutive XM calls pass exact references and output sentinels |
| v38 | Exact-integer Q6_K x Q8_1 batch path through XM | Packed shape tests, full output projection and Qwen token validation pass |

The original guest DT contains `xlxmmbase`, `xlxmm32a8i`, `xlxmm32a16f`,
`xlxmm32a32f`, `xlxmm64t`. The actual vendor Linux 6.7.9+ kernel recognizes
their `xsf*` equivalents. Symbol recovery from the supplied guest image exposed
`riscv_mammoth_setup_size` and `riscv_mammoth_first_use_handler`; the latter
contains the normal tile-state allocation and saved-context initialization.
No kernel code or privileged status CSR was patched.

The DT-only A/B experiment distinguishes the first fault from an unsupported
vtzero encoding: changing only existing capability names makes the same
instruction succeed. It does not certify every instruction in the entire ISA.

The remaining arithmetic failure was the output-store context: the old leaf
retained e16alt/w2 while storing FP32 tile elements. Setting e32/w1 before
`sf.vste32` resolves the numerical tests. The encoding/type interpretation was
cross-checked against the [LLVM Xsfmm definitions](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/RISCV/RISCVInstrInfoXSfmm.td).

## Numerical Validation

- Native XM shapes: 1x1x2, 3x5x6, 32x8x32, 32x32x128, repeated to 64 calls.
  All 64 pass, no SIGILL, timeout or output sentinel damage. Aggregate probe
  elapsed 54.640 ms includes input preparation, CPU reference and scheduling;
  it is not a matrix-unit throughput measurement.
- Q6_K shapes: 17x1x256, 1031x2x2048, 3x1x8192, 1031x8x2048, and
  248320x1x2048, checked against GGML dequantization with padded strides.
- The large run has 493/493 completed submissions, matching build `0x20026`,
  no incomplete phase records and no observed event-index gaps.
- XM inputs are exact BF16 integer values. Two K=16 dots per Q8 group retain
  separate Q6 subblock scales; those scales are applied in integer arithmetic
  before the original floating scaling/accumulation order. The maximum error
  in the large projection was 0.0012552687 against the dequantized reference,
  consistent with the tested reference path's FP32 accumulation.
- No RVV/CPU matmul fallback is selected by the v38 Q6_K branch. The host-only
  layout test double is excluded from firmware linkage.

## Performance

Full 248320x1x2048 output projection, v38, 485 tiles:

| Phase | Time |
|---|---:|
| Host pack | 73.908 ms |
| Host shared-DDR write | 180.069 ms |
| Device input copy | 5281.013 ms |
| Device compute including unpack/pack/scales/XM | 8686.956 ms |
| Device output copy | 25.089 ms |
| Device completion publication, cumulative | 2.420 ms |
| Host readback | 20.345 ms |
| Host wait residual | 428.552 ms |
| Runtime total | 14769.347 ms |

The previous v34 RVV projection was 10398.785 ms; this first XM implementation
is slower. The 1031x8x2048 XM case took 104.035 ms versus the earlier RVV
103.375 ms. These are individual observations, not controlled repeated means.

Qwen count16, same four-request short correctness cohort:

| Backend | Valid tokens | Wall time | Aggregate TPS |
|---|---:|---:|---:|
| Previous CPU baseline | 64 | 14.026093 s | 4.562924 |
| Previous v34 RVV PACC0 | 64 | 272.786667 s | 0.234616 |
| v38 XM PACC0 | 64 | 304.451023 s | 0.210214 |

All four v38 responses exactly match
`one, two, three, four, five, six, seven, eight,` under a 16-token limit.
This is token-capped and does not claim completion of the count through ten.
TPS includes request prefill and excludes server startup; it is not pure decode
speed or a large-batch saturation benchmark. Model: Qwen3.5-35B-A3B-Q4_K_M.gguf,
SHA256 `3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375`.

Across the v38 Qwen cohort, all 8245 submissions have build `0x20026` and
complete phase records, with no event-index gaps. Device input copy totals
90.454698 s; compute totals 191.870330 s; completion publication totals
40.868 ms, nearest-rank p99 7 us and maximum 23 us. The primary optimization
targets are packing/scale reduction, small-tile invocation overhead and repeated
weight movement, not completion publication. Pure XM execution time has not
yet been separated from packing/scaling inside the compute phase.

## Reproduction, Scope and Provenance

See `tools/pacc-qwen35/README.md`. The release includes the Q6_K transport,
Q8_1/RoPE/get-rows ABI fixes, SiFive dp4a lowering, the tested legacy jobd
experiment source, exact alias patcher, numerical probes and bounded profilers.
It intentionally excludes unrelated CXL routing edits and build artifacts.

v38 firmware SHA256:
`45b7b88ca8447bebfc54dae84c1aed3475fe8a50acc3dff9af32fa93e4ce77fe`.
v38 jobd source SHA256:
`adb4dd45b6cae6658058386276b4526d9ee351881f3daa83f01bd4e5308b5e40`.

Raw logs remain in `/srv/lanxin/qwen35-a3b/logs`, notably
`profile-v35-xm-config.json`, `profile-v36-isa-alias.json`,
`profile-v37-xm-batch.json`, `profile-v38-q6k-xm.json`,
`profile-v38-xm-large.json`, `phases-v38-xm-count16-p4.json`, and
`concurrent-count16-v38-xm-p4.{json,server.log}`. Bounded summaries, actual
responses, test logs and a SHA256 manifest are committed under
`tools/pacc-qwen35/evidence`.

No source-side atomic telemetry snapshot is claimed: the reader accepts two
identical reads and verifies sequence completeness. Residual host wait is not
an isolated IRQ measurement. Archived micro-test p99 values predate the
nearest-rank fix; the Qwen v38 cohort uses the corrected calculation.

The running PACC0 image is temporary v38; the canonical vendor firmware file
was restored after loading. PACC1-3 and the host kernel were not reconfigured.
Five Python diagnostic tests and the host packed-layout test passed. Firmware
was compiled and exercised on hardware. The release branch's combined Rust
tree has not had a fresh clean rebuild; the related runtime/LLVM changes were
carried from the working runtime used by these tests, excluding unrelated CXL
dispatch code. This distinction matters when reproducing from a clean checkout.
