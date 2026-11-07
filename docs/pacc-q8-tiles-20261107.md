# Native Q8 tiled transfer and XM execution, 2026-11-07

## Delivered

The Q8_0 x Q8_1 matrix that previously required a 23,549,120-byte generic
ELF/argument submission now runs as native bounded row tiles in the existing
2,097,152-byte payload. No allocation enlargement, guessed address, software
matrix fallback, success-on-error or host reboot was used.

The runtime validates shape, strides, allocation extents and completion
policy. It copies activations once per operation, packs each weight tile,
submits the native MMVF job, waits, reads and validates output, and then
reuses the buffers under the existing shared-DDR stage lock. Rows are capped
at 512 by default and further bounded by actual payload capacity. Offsets
are aligned and checked against the payload end. This implementation is
serial submit/wait/readback, not an unverified asynchronous copy overlap.

The named CUDA interception accepts the non-fused Q8_0 variant in addition
to Q6_K. It rejects ids, fusion, unsupported dimensions and inconsistent
buffer extents instead of invoking incomplete generic warp execution.
The new public symbol is `hetgpu_sifive_submit_q8_0_q8_1_tiled`.

The device ABI uses x_type=4, reserved0=6, byte input strides and element
output strides. Supported shapes are 1<=N<=8, K divisible by 32 and <=32768.
Q8_0/BF16's existing ABI is not changed. PACC performs K=32 dots on native XM
using exact integer BF16 operands, then applies Q8 block scales outside XM.
A dot has magnitude <=524288, so integer results fit exactly in FP32.
Nonfinite, fractional and out-of-bound dots, invalid scales and leaf failures
are rejected. Q8 currently uses one worker; Q6 retains four persistent workers.

## Validation

Host-only layout tests use a scalar test double, which is NOT linked into
firmware. Actual PACC tests use the real XM leaf and compare against GGML
Q8_0 dequantization plus a double-precision reference. Strides include padding,
outputs have tail guards, and input quants include -128 and 127.

| M x N x K | Tiles | Runtime call ms | Max absolute error |
| --- | ---: | ---: | ---: |
| 33 x 1 x 32 | 1 | 1.931 | 0.0000007153 |
| 1031 x 8 x 2048 | 3 | 40.661 | 0.0002843421 |
| 3 x 3 x 32768 | 1 | 4.093 | 0.0067304354 |
| 8192 x 2 x 2048 | 16 | 174.070 | 0.0003369395 |

All values satisfy the per-element criterion abs_error <= 0.0005 +
0.0001*abs(reference). Invalid stride and K checks intentionally emit errors
in the successful test log. Fault tests reject NaN/infinity/fractional and
out-of-range leaf results, leaf error, nonfinite scales and invalid shapes;
zero and the maximum integer dot are accepted.

Hardware telemetry: 21/21 completed submissions, marker 0x20032, no index
gaps or incomplete phase sets. Aggregate device input copy=25.294ms,
compute=157.438ms, output copy=2.213ms. Completion publication p99=7us;
this is not end-to-end completion latency. Sampling uses double-read
stability, not atomic source-side snapshots.

The original Q6 hardware suite passed 493/493 submissions on the new
firmware. That first suite used the original runtime; the subsequent Qwen
regression exercises the rebuilt runtime's shared Q6/Q8 transport code.
That regression passed all 64 expected tokens (4 requests x 16), 8245/8245
completed submissions, with no index gaps. Cohort wall time was 60.887697s,
or 1.051115 TPS including prefill and excluding startup. This is a correctness
regression, not a claim of a statistically established throughput gain.

## Qwen integration gate

ngl=2 now completes the formerly failing 8192x2x2048 Q8 operation in
175.619ms, with 16 observed successful tiles. The next operation is
`rms_norm_f32<256,true,false>`, grid=(16,2,1). The default minimum hidden
size is 1024, and host fallback is disabled, so the server aborts there.

A second bounded run set HETGPU_SIFIVE_RMSNORM_OFFLOAD_MIN_HIDDEN=1 and both
RMSNORM_STRICT_STRIDES and RMSNORM_STRICT_WEIGHT_SHAPE to 1 (with the
HETGPU_SIFIVE_ prefix). Q8 again completed, in 176.346ms. RMSNorm was then
rejected by the named-offload parser/layout checks. The log does not isolate
which strict check failed. Neither run reached readiness or generated valid
tokens. The Q8 payload blocker is fixed; ngl=2 inference/TPS is not.

## Deployment and provenance

Firmware: v50_q8_tiles, marker 0x20032, SHA256
`bfc733006cf66d4d7f01e46d9634186bb4f4ff345f91e5d705382d99ea0a5a01`.
Compiler profile remains `-O3 -march=rv64gcv_zvl1024b -mrvv-vector-bits=1024`.
The 128-bit host only runs separately compiled VLA layout tests; candidate
fixed-1024 code is executed on the PACC guest.

Candidate `/srv/lanxin/qwen35-a3b/runtime-q8/libnvcuda.so` SHA256:
`c172ec346cf1732ea962354f21fa7877c677b1c9c151d824418040082565653d`.
The original runtime remains installed. The candidate was built from
`/home/ubuntu/Concordia` with its pre-existing changes preserved, using
rustup Cargo, LLVM_ZLUDA_PREBUILT=/usr/lib/llvm-20 and the sifive,embed_cudart
features. Build completed with warnings. An initial build without the
prebuilt prefix began rebuilding LLVM; only that owned build tree was stopped
before retrying with the verified prefix. This is not a clean release-branch
build or a claim that its earlier JIT-O3 changes were deployed.

PACC0 was reloaded only through the checked vendor driver loader. Host boot
ID remains 89a6c374-059a-40da-8c4f-c00179fd48d4, owned DDR remains
0x2075d000000+0x400000, and the canonical vendor firmware was restored.
PACC1-3 and the unrelated CPU server were not changed.

The evidence manifest hashes raw captures, source and binaries. Large raw
profiles/build logs remain under `/srv/lanxin/qwen35-a3b/logs`; firmware and
runtime binaries are not committed. Failure captures remain explicitly
invalid as whole-run performance evidence, despite their completed Q8 tiles.
