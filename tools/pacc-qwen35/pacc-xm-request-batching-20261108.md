# Cross-request XM batching on four PACC devices

## Scope and safety

Host: lanxin2 / wr5219. Model: Qwen3.5-35B-A3B-Q4_K_M.gguf.
The target is aggregate throughput across independent requests, not single-request
decode throughput. 500 TPS has NOT been achieved.

No host reboot, device reset, driver rebind, firmware reload or partition expansion
was performed in this work. Boot ID: 89a6c374-059a-40da-8c4f-c00179fd48d4.
Firmware build remains 0x20036. Each PACC owns its existing 512 KiB payload window
inside the pinned 4 MiB shared-DDR allocation at 0x2075d000000.
The unrelated CPU llama-server PID 4732 was left running.

The new matrix path performs real XM BF16 integer-dot products on each PACC.
Scaling, reduction, recurrent GDN and supported floating matrix operations run on
the device cores. Host address packing, scattering, quantization and existing
pointwise compatibility paths still exist. This is NOT an entirely device-resident
LLM graph, and ngl=99 does not establish otherwise. No CPU matrix fallback was added.

## Changes

- Group independent activation columns by shared dense weights or MoE expert;
  submit N=2/4/8 XM matrices and scatter to separate request outputs. MoE groups
  may be smaller when requests select different experts. Do not count replication
  of one activation as independent request throughput.
- Tile payloads within each device's owned 512 KiB window, with no overlap.
- Compact device concat, sparse KV cast/scatter, row reduction and bitonic argsort
  avoid oversized generic translated kernels. Floating mul_mat_f maps to existing
  device MMVF with checked heads, strides, broadcasting and padding.
- Explicit RVV copies reduce device GDN staging without reordering its arithmetic.
- Optional RVV quantized postprocessing preserves the original scalar codegen's
  two fused multiply-add operations and per-output accumulation order. It retains
  finite/integer/range checks on XM dots and rejects insufficient vector length.
- Reject high-bit firmware errors such as 0xffff5109 instead of aliasing their low
  bits to successful 0x5109. Completion must match the exact submission sequence.
- Fix physical ring-wrap event ordering in the streaming profiler. Late start
  records of already completed jobs are accepted only with an earlier index;
  duplicate phases, sequence reuse, missing phases, index gaps and wrong builds fail.
- Add independent sequence/KV state, CPU-reference token checks, bounded phase
  collection, hashes and shared-clock aggregate TPS calculation to the benchmark.

All new batch dispatch is opt-in via HETGPU_SIFIVE_QK_REQUEST_BATCH=1.
RVV scaling is separately opt-in via HETGPU_SIFIVE_QK_VECTOR_REDUCE=1.
Only supported templates/shapes are accepted; this is not universal CUDA support.

## Correctness gates

- Q4/Q5/Q8 dense and MoE: independent inputs, expert routing, N=2/4/8, nontrivial
  scales/minima, padding and output guards checked against numerical references.
- RVV/scalar A/B: all output bytes identical, including gated and K=32768 cases;
  nine launches per variant/shape, first launch excluded from timing.
- Layout/cast, row sum/mean reduction, sorting/ties and floating matrix ABI tests pass.
- GDN outputs match scalar baseline SHA256 and independent reference tolerances.
- Intentional unresolved device ELF symbol returns rc=-1 and firmware 0xffff5109.
- Four-device mixed XM/RMS stress passed 64 alternating submissions per device.
- Full model v16: four devices, two independent requests each, two generated tokens
  each, ngl=99. All 16 tokens match CPU; all phase profiles complete and correct build;
  each device reports 12,000 actual multi-column XM tiles, maximum N=2.
- Final v18 with RVV scaling enabled passes the same four-card full-model gate:
  all 16 tokens identical to CPU, four valid phase profiles and 12,000 multi-column
  XM tiles per device. The six runtime unit tests also pass.

## Measurements

These operator figures are instrumented wall times, NOT LLM tokens/sec.

Q4, M=256, K=2048, same weights and independent activation columns:

| N | Scalar postprocess call (ms) | RVV postprocess call (ms) | Speedup |
|---|---:|---:|---:|
| 1 | 17.045 | 16.487 | 1.034x |
| 2 | 19.831 | 16.885 | 1.174x |
| 4 | 25.735 | 17.931 | 1.435x |
| 8 | 36.610 | 19.264 | 1.900x |

For N=8, device scaling/reduction falls from 20.133 ms to 2.832 ms. Remaining
device phases include 6.267 ms weight unpack, 5.100 ms activation pack, 0.641 ms
XM leaf calls, and 0.363 ms local payload copy. The 2.302 ms non-device residual
includes host staging, transport, dispatch, completion observation and copyback;
it cannot be attributed solely to DMA or interrupts. Timing instrumentation adds overhead.
Q5 N=8 improves 1.843x; Q8 N=8 at M=128 improves 1.756x.

GDN isolated mean wall times, bit-identical output:

| Width / tokens | Before (ms) | After (ms) |
|---|---:|---:|
| 128 / 1 | 11.334 | 3.592 |
| 128 / 3 | 26.227 | 5.272 |
| 32 / 5, KDA | 4.517 | 2.302 |

Full-model v16 four-card short correctness run: aggregate decode 0.057148 TPS,
8 decode tokens / 139.986515 seconds measured on one shared host clock. The
16-token request window is 278.087080 seconds and includes prefill. This is a
two-token smoke result, not sustained long-token throughput or a 500 TPS result.

Final v18 repeats this test with RVV scaling enabled: aggregate decode
**0.062409 TPS** (8 / 128.186419 s), an observed **9.2%** improvement over v16.
Aggregate output including prefill is 0.063575 TPS (16 / 251.669363 s).
This is one short paired comparison, not a statistically characterized sustained rate.

The PRELOADED phase collector covers only that job family, not every ELF/XM call.
For v16 PACC0 it captured 3,378 complete PRELOADED jobs: 39.037 s compute,
1.078 s input copy, 0.097 s output copy, 0.016 s completion publish. Completion
publish p99 is 5 us, but that is NOT host-observed completion latency. Do not
extrapolate this subset into a full-model CPU/IRQ/DMA attribution.

## Known failures and evidence boundaries

- Earlier v12 4x8-request run with configured ubatch=2 produced 32 correct first
  tokens, then PACC3 failed RMSNorm during the next decode. The runner terminated
  its other workers and published null TPS. Actual microbatch sizes included N=3.
- v16 exact completion handling passed the subsequent 4x2-request run, but this
  alone does not prove the previous larger-batch RMSNorm failure is eliminated.
- N=8 generic RoPE staging previously needed 592,512 bytes, exceeding the owned
  524,288-byte window. A compact native RoPE experiment depended on unresolved
  libm symbols; it was removed from the final implementation, not silently enabled.
- v17 RVV postprocessing failed the byte-equality test because it did not reproduce
  scalar fused multiply-add rounding. v18 fixes this; v17 is not a release candidate.
- Initial Q8 A/B M=256 exceeded the partition and was correctly rejected. Valid
  Q8 timing uses M=128. The earlier non-isolated column benchmark is also excluded;
  use only the explicitly isolated column result and validated A/B results.
- The long case contains eight natural-language requests per card, 128 output
  tokens each, with complete CPU references. A CPU reference alone is not a PACC
  throughput measurement. Do not report long-run TPS unless all four reports pass.
- Final v18 actually attempted this 32-request / 128-token case with ubatch=8.
  It failed during prefill after 61.751 s including model load: all four devices
  reached N=8 XM matrices (532 multi-column tiles per device), then rope_multi
  required 592,512 bytes versus 524,288 available. No complete output was generated;
  both aggregate TPS fields are null. Raw run: logs/four-requests-long-v18/.
  The test did not hit its 900-second timeout, restart the host or enlarge partitions.

## Reproduction

Working directory: /srv/lanxin/qwen35-a3b. Snapshot runtime-v18 is built from the
release worktree; the existing default installed runtime was not replaced.
Raw logs live below logs/; bounded evidence and SHA256 provenance are archived in
tools/pacc-qwen35/evidence-xm-requests. Scripts refuse to overwrite existing runs.

```bash
cd /srv/lanxin/qwen35-a3b
env HETGPU_SIFIVE_QK_VECTOR_REDUCE=1 python3 bin/run-four-pacc-requests.py NAME \
  --runtime /srv/lanxin/qwen35-a3b/runtime-xm-request-batch-v18 \
  --case /srv/lanxin/qwen35-a3b/bin/cross-long.json \
  --reference /srv/lanxin/qwen35-a3b/logs/cross-long-cpu.json \
  --ubatch 8 --timeout 21600
```

This is a diagnostic command, not a claim that ubatch=8 currently passes. Use
cross-fast.json/cross-fast-cpu.json and ubatch=2 for the short correctness gate.
The isolation wrapper verifies boot ID, owned DDR base and device exclusivity.
Do not bypass it or expand the payload windows. It does not reboot/reload firmware.

## Remaining work

The final RVV-enabled short model run is qualified. Next fix oversized RoPE staging and
reproduce larger-batch RMSNorm diagnostics before accepting sustained throughput.
The profile points to quantization packing/scaling and fine-grained staging, not
evidence of an interrupt-dominated bottleneck. Resident packed weights, activation
reuse and fewer checked submissions need their own ownership/cache-coherency ABI
and numerical tests. A 500 TPS claim requires completed, correct long runs with
independent requests; operator speedup alone is not sufficient.
