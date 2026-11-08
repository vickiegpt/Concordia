# Qwen3.5-35B-A3B full-layer PACC bring-up

## Scope and current gate

Initial target: Qwen3.5-35B-A3B Q4_K_M, `ngl=99`, one PACC0, batch=256,
ubatch=1, flash attention disabled. This is full-layer compute offload with
host-backed weights staged through the owned 2 MiB payload aperture, not a
claim that all model weights reside in PACC RAM. Host-side quantization and
existing pointwise shims remain. GDN, softmax and top-k run on the PACC core;
quantized integer matrix dots use XM. No host matrix fallback is enabled.

**Current end-to-end gate: two-token correctness smoke PASSED.** The corrected
gather runtime with v53 firmware generates `[17, 220]` (`2 `), matching the CPU
reference for raw prompt `1+1=`. Prefill took 1109742.836 ms, one decode step took
269051.840 ms, and the measured request window was 1378.804553 s. This yields
0.0014505319 output tokens/s including prefill, excluding model load. The single
decode interval is 0.00371676 tokens/s, not sustained decode throughput. No host
matrix fallback was enabled. Four-card qualification is recorded separately below.

The previous clean-release two-token run completed but produced `[17, 1046]` (`2 gr`), versus
the CPU reference `[17, 220]` (`2 `). All logits were finite. Prefill took
1187607.727 ms and the following decode took 265996.446 ms; these are failed-run
diagnostic timings, not a valid throughput result. `aggregate_output_tps` is null.
The phase recorder observed 8720 completed preloaded submissions and no incomplete
ones, but those records do not cover all ELF kernel execution. Do not infer an
end-to-end bottleneck percentage from the preloaded phase sum.

The first token matching does not validate recurrent state or sustained decode.
No host reboot or PACC1/2/3 programming was used for this result.

## First numerical defect found

The four-layer diagnostic prefix preserves 76 original tensors byte-for-byte.
Its CPU layer-0 and layer-3 outputs for the first input token also exactly match
the corresponding full-model CPU outputs. It is not a replacement benchmark.

With the old runtime, the first token's `state_predelta-1` should be all zero
but contains 2048 nonzero trailing floats (max absolute value 0.084170967).
The host data-copy shim treats `k_get_rows_float_vec`'s `ne00v` as a scalar
float count, although GGML passes a count of 16-byte vectors. It copies only
one quarter of an FP32 recurrent state, leaving the rest stale. This defect
affects both initial scratch-buffer reuse and state restoration on later tokens.

The fix multiplies the vector width by four while preserving scalar-element
strides, shares that width calculation with staged buffer metadata, and rejects
unsupported types, negative indices, and out-of-range source rows. No-op gather
can no longer return success. This is a host data-copy repair, not a host matmul
fallback or an XM speedup.

`test-getrows-vector-abi.py` reproduces the old failure, then passes scalar,
vector, 524288-float state, guard preservation, negative-index and short-buffer
checks with the repaired library. The full-model post-fix two-token run passes;
longer generation and varied prompts remain unqualified. The single-token
layer-31 residual comparison also amplifies small
upstream differences; that threshold crossing alone is not proof of another bug.

Post-fix hardware traces show all-zero initial states in layers 0/1/2 and five
byte-identical state-carry pairs across consecutive input tokens. The copy-only
checker is `check-recurrent-trace.py`; its success is not a full-model TPS claim.

## Repairs

- Correct concat argument counts and sizes for the current GGML CUDA ABI.
- Add device-side gated delta, top-k and masked attention softmax paths.
- Preserve exact lazily registered CUDA function identity before compilation.
- Tile FP32 MMVF within actual shared-DDR capacity and preserve FP32 inputs.
- Handle multihead attention strides and broadcast without host matmul.
- Add Q4_K, Q5_K and Q8_0 XM integer-dot tiles, expert selection, gate fusion,
  bias application, and small-K row grouping.
- Preserve untouched KV rows and size scatter output from actual indices.
- Keep preloaded and ELF completion ownership separate; retain completed ELF
  results so a cleared status record cannot cause duplicate execution.
- Replace unbounded retained ELF mappings with a 64-entry, 32 MiB total-budget
  cache. Lookup checks symbol and all ELF bytes, not only a hash. Restore writable
  PT_LOAD data and BSS from the relocated initial image before reuse. Exhaustion
  fails explicitly rather than growing memory or selecting CPU fallback.
- Add persistent phase, mapping-count and fault diagnostics in the reserved,
  disabled-telemetry area. Fault PC/address are meaningful only if signal != 0.

## Loader failure evidence

v52, using the old retained-mapping policy plus diagnostics, failed the repeated
cell probe at zero-based iteration 4085. The last stage was 0x5104, before ELF
execution, with 4086 retained mappings totaling 0x3fd8000 bytes. A later tiny
kernel could not complete either. This isolates a loader/resource-lifetime
failure; it is not evidence of a new XM opcode mismatch.

v53 passed 6000 hardware submissions, including a mutable static counter that
must restart at one for every fresh invocation. With the identity probe included,
only two ELF mappings (32 KiB) remained. Host cache tests additionally cover
10000 hits, writable/BSS restoration, different ELF bytes, and bounded exhaustion.

Firmware: `v53_bounded_elf`, marker `0x20035`.
SHA256: `b6e1b133a71c34820c55250a6cab341a0de9ec4c784980791606061e4adf9129`.
No host reboot. Only PACC0 was rebound through the vendor driver. The canonical
vendor image was restored and its SHA checked after every load.

## Single-device reproduction

These paths and the loader's boot/allocation checks are specific to lanxin2.
Deploy the tools into `/srv/lanxin/qwen35-a3b/bin` and run with PACC0 idle:

```sh
CONCORDIA_RELEASE=/srv/lanxin/qwen35-a3b/runtime-ngl99-getrows \
  bash /srv/lanxin/qwen35-a3b/bin/run-ngl99-direct.sh UNIQUE_PREFIX 0x20036
```

The direct runner verifies firmware identity and numerical results, then makes
the full-offload request against the CPU reference. Failure stops the run.
The current image is v54, marker `0x20036`; the recorded fixed single-device run
used v53, marker `0x20035`. This command does not load firmware or reboot anything.
Run it only while all other PACC clients are idle: this older single-device runner
uses the whole 2 MiB payload, not the four-worker partition wrapper. Tests using
actual PTX require the existing llama.cpp PTX cache.

Release builds use the LLVM submodule at the recorded gitlink, LLVM 20, and the
evaluation environment's Cargo.lock. The lock is not normally tracked by this
repository; preserve its hash with the evidence when reproducing a build.

For direct generation without the server's two-token sequence-removal probe,
use `run-ngl99-direct.sh UNIQUE_PREFIX 0x20035`. It still computes all prompt and
decode steps, checks every output logit is finite, and compares generated text
and token IDs with an actual CPU run. Its default is two output tokens.

## Initial four-device readiness audit

Read-only inspection found all four `/dev/paccN` and helper nodes present and
bound. However, all four helpers report the same base, `0x2075d000000`, with one
4 MiB allocation. The current PACC1/2/3 beacon fields are zero; this does not
establish their firmware identity, readiness, or numerical correctness.

Do not enable four simultaneous clients with the current single-slot settings:

- Only PACC0 has the qualified v53 image and numerical/completion evidence.
- The current builder and recovery loader are explicitly PACC0-only.
- Four equal payload slots would be 512 KiB each, below some current Q4/Q5
  launches. Tile bounds must use actual per-device capacity, not a fixed 2 MiB.
- The enabled early-devmem startup marker writes completion records for all
  four slots. A new card boot must not overwrite another card's live completion.
- Diagnostic rings use slots 8..11, while RMS debug records start in slot 9.
  That layout must be separated before trusting four-card profiling.
- A single numeric `HETGPU_SIFIVE_VISIBLE_DEVICES` value means a device count;
  explicit physical-device lists require the list syntax. Verify physical
  affinity rather than assuming `1` selects PACC1.

For aggregate throughput, qualify independent requests on four isolated device
queues first, then evaluate expert or tensor parallelism. No four-card TPS or
linear speedup has been measured. This audit did not reset or program PACC1/2/3.

### Ownership-qualified v54 deployment

v54 `owned_slots` / build `0x20036` now requires an explicit valid device id
before startup writes. Both startup marker paths write only that id's completion
record. Per-device diagnostic rings remain in slots 8..11; RMS debug moves to
slot 12. A host mmap/canary test verifies all four ownership cases and rejects
missing/invalid ids. All four id-specific firmware images were built and loaded
through the audited vendor driver. Each physical device then passed four Q8 XM
shapes, with matching build `0x20036`, complete diagnostic submissions and no
observed event gaps. This qualifies the individual matrix paths, not inference.

The partition-aware runtime computes Q4/Q5/Q8 and softmax row tiles from actual
ELF slot capacity. `run-pacc-isolated.sh` gives physical device N the payload
range `[0x200000 + N*0x80000, 0x280000 + N*0x80000)`, within the single owned
4 MiB allocation. A per-worker logical upper bound confines both ELF and
preloaded GEMM staging. Four independent processes avoid process-local kernel
mutex serialization. Geometry, oversize rejection and summary-gating tests pass.

The vendor driver audit shows `pacc_probe` computes firmware fallback slots as
`PaccBase + (id << 27)`, length 128 MiB, and reset passes the selected id to SBI.
The loader checks boot id, reserved memory, exact firmware SHA, driver SHA and
srcversion (for nonzero devices), and an idle target before driver unbind/bind.
It serializes canonical-firmware changes, restores the vendor file in `finally`,
and never writes raw reset registers. Each target still needs a hardware identity
and numerical check after loading. Do not treat a successful bind as that check.

Initially nvtop PID 1375297 held all four `/dev/paccN` handles and the loader
refused the busy device. The user approved ending that monitor; it had already
exited when checked, so no process was killed. All four targets were idle for
the subsequent rebinds. The host boot id is unchanged and the canonical vendor
firmware was restored after every load. No unrelated inference service was stopped.

At 512 KiB per worker, the raw single-tile Q4 API correctly rejects the oversized
512x2048 fused test. The CUDA ABI path instead splits the same shape and passes
both unfused and fused expert-indexed numerical checks with output guards intact.

`run-four-pacc-smoke.py` requires four per-device numerical/profile qualification
JSON files before launching. It leaves TPS null unless all four actual generated
token sequences match a CPU reference and each profile has the requested firmware
build and device id. Reported windows use the shared host monotonic clock and
include cross-worker skew; the short smoke is not sustained large-batch evidence.

Four-worker reproduction with the already qualified, boot-specific artifacts:

```sh
cd /srv/lanxin/qwen35-a3b
python3 bin/run-four-pacc-smoke.py UNIQUE_PREFIX \
  --runtime runtime-ngl99-four \
  --reference logs/ngl99-direct-cpu-reference.json \
  --qualification logs/four-v54-qualification --timeout 5400
python3 bin/audit-four-pacc-staging.py logs/UNIQUE_PREFIX \
  --output logs/UNIQUE_PREFIX/staging-audit.json
```

Run the staging audit after inference has exited so its log hashes describe
immutable evidence. It validates logged ELF staging spans and physical affinity,
not every memory transaction performed by the device. The helper's allocation
must remain pinned and its boot-specific base must not be reused after reboot.

### Four-device full-model result

`four-ngl99-v54-v1` passed on all four physical devices. Each request loaded
41/41 layers, used the full original 35B Q4_K_M model, and generated `[17, 220]`
(`2 `) for `1+1=`, exactly matching the CPU token/text reference. All logits were
finite. Each recorder reports matching firmware build, valid capture, and zero
observed event gaps, with 17408 completed preloaded submissions per device.
Those phase durations exclude ELF execution and are not a full end-to-end
compute/copy/wait breakdown. All four processes exited with status 0.

| Physical device | Prefill (s) | One decode step (s) | Token sequence |
| --- | ---: | ---: | --- |
| PACC0 | 1151.052523 | 288.993212 | 17, 220 |
| PACC1 | 1199.717347 | 305.046173 | 17, 220 |
| PACC2 | 1269.484211 | 315.773141 | 17, 220 |
| PACC3 | 1184.042147 | 297.867910 | 17, 220 |

- Eight output tokens / 1585.268998 s shared request window:
  **0.0050464622 aggregate output TPS**, including prefill, excluding model load.
- Four post-prefill tokens / 434.208998 s shared decode window:
  **0.0092121536 aggregate decode-window TPS**, including worker start skew.
- Outer elapsed time including model load and capture finalization: 1647.469875 s.
- Per device: 19750 logged ELF submissions, 68200 staged buffer bindings, no
  wrong-device or out-of-partition spans. This is host-log audit evidence, not
  an independent hardware bus trace or a security-isolation proof.

This is a four-request, two-output-token correctness smoke with `ubatch=1`.
It is not sustained decode, a large-batch benchmark, or peak throughput. It is
far below 500 TPS. The single-device and four-device runs differ in firmware
version and payload capacity, so do not present their ratio as controlled scaling.
Host weight storage, data movement, quantization and existing pointwise shims
remain; integer matrix dots execute on XM, with no host matrix fallback enabled.

Raw logs and profiles remain at
`/srv/lanxin/qwen35-a3b/logs/four-ngl99-v54-v1/`. The committed summary, per-device
results, artifact hashes, compact profiles and staging audit are under
`tools/pacc-qwen35/evidence-ngl99/four-v54/`. Compact profiles retain the absolute
raw-file paths and SHA256 values. The default installed runtime was not replaced;
the tested runtime is `/srv/lanxin/qwen35-a3b/runtime-ngl99-four`.

## Remaining boundaries

- General CUDA warp reductions are not universally implemented by these named
  kernel specializations. Unsupported combinations return errors.
- Four independent ubatch=1 workers do not qualify multi-token MoE CUDA kernels,
  longer output sequences, other prompts, or sustained large-batch throughput.
- Board VLEN=1024 code is not executed on the host's VLEN=128 CPU.
- Do not reuse the owned shared-DDR address after a host reboot.
- The default installed runtime is unchanged; candidate libraries are separate.
