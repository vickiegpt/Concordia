# Four-PACC recovery and 2048-request attempt

## Outcome

Host lanxin2/wr5219 was NOT rebooted. The unrelated CPU llama-server PID 4807
was preserved. Four PACC devices were recovered using vendor driver unbind/bind,
and all four passed numerical tests. The canonical boot firmware was restored
after every temporary image load; no persistent boot configuration was changed.

The recovered full-model short test passes with aggregate decode **0.06262262 TPS**.
The actual **2048 independent requests (512 per PACC)** attempt was stopped by the
host-memory guard during initialization, before any XM submissions or generated
tokens. Its TPS is **null**, not zero and not a successful throughput result.
500 TPS and sustained large-batch inference have NOT been achieved.

## Recovery and ownership

- Boot ID: `4ca2fa21-0d48-4090-93a5-51206ac1de29`.
- Verified the vendor firmware reservation separately from shared-DDR ownership.
  Firmware slots occupy EFI-reserved `[0x20100000000, 0x20120000000)`.
- Stopped all four PACC before normally unloading the helper. No force unload or
  raw reset-register writes were used.
- Replaced the unqualified fixed 16 MiB mapping with a fresh, kernel-owned 4 MiB
  allocation: base `0x20ef7000000`, override `0`, DMA synchronization enabled.
- A transient `pacc-owned-ddr-pin-20261108.service` keeps a helper FD/module
  reference. It refuses normal termination while a PACC is bound. Do not kill
  this holder or free the allocation underneath running jobd instances.
- Payload windows are disjoint: PACC0 `[0x200000,0x280000)`, PACC1
  `[0x280000,0x300000)`, PACC2 `[0x300000,0x380000)`, PACC3 `[0x380000,0x400000)`.
  These offsets are inside the owned allocation; each window is 512 KiB.
- Reused the previously qualified jobd executable, build `0x20036`, SHA256
  `3a0507b2c7be070d07aebd21ee287543869dd69bdd54c50e9a81f184d70288ce`.
  Packaging reproduced the previous qualified image byte-for-byte as a control,
  then generated four images with this boot's address and physical IDs. The
  old-address control image was never loaded.
- Canonical vendor firmware SHA256 remains
  `19cdacf4ddfa8f6861477daa2be79f3e1e52d1a11780c7a209a9419e4a2a83a8`.

The recovery scripts are deliberately boot-specific, not a universal installer.
`prepare` refuses a second allocation when its state exists. A later boot needs
fresh layout/ownership checks and freshly packaged images. The old isolation
wrapper remains unchanged and refuses this boot; use `run-pacc-recovered.py`.

## Validation

| Gate | Result |
|---|---|
| Host-view owned-DDR mmap/pread/pwrite | Passed; not itself PACC evidence |
| Q8 numerical qualification | Four shapes per card; 42 complete submissions/card; all passed |
| Completion/build capture | All four valid, build 0x20036, no incomplete jobs |
| Concurrent XM/RMSNorm stress | 64 alternating rounds/card, deterministic XM outputs and RMS numerical/guard checks passed |
| Full Qwen3.5-35B-A3B-Q4_K_M, ngl=99 | Four cards x two requests x two tokens; all 16 tokens match CPU |
| 512-sequence CPU reference | 512 independent states, distinct numeric prompts, all 1024 reference tokens generated |
| Four-card 2048-request inference | Memory-guard failure before XM; no application TPS |

The short run used the new 512-sequence library and v4 harness; a live process
mapping confirmed `build-seq512/bin/libllama.so.0.3.0` was loaded. Each card reports
12,000 verified multi-column XM tiles, maximum N=2. Total decode is 8 tokens /
127.749366 s = 0.06262262 TPS. Total output including prefill is 16 /
250.448181 s = 0.06388547 TPS. Model-load time is excluded from these two metrics.
This is a two-token correctness smoke, not sustained throughput or an optimization
over the previous boot. Existing host packing, quantization and pointwise
compatibility paths remain; ngl=99 does not establish an entirely device-resident
graph. No CPU matrix fallback was added.

Initial Q8 device-side completion-publish p99 was 5, 7, 6, 6 us respectively.
This excludes host observation latency and is not an end-to-end completion p99.
The streaming profiler covers PRELOADED jobs, not all ELF/XM work.

## Actual 2048-Request Attempt

Configuration: four independent processes, each 512 sequence states, ngl=99,
microbatch 8, two output tokens per request, CPU-reference checks, 600 s timeout.
The same 512 distinct prompts were submitted independently on each card, with
separate KV/recurrent states. This is not merely llama.cpp's `-b 2048`.

All four basic hardware qualifications passed. During model/recurrent-state
initialization, minimum sampled MemAvailable reached **30,429,664 KiB**
(29.02 GiB), below the configured 32 GiB reserve. The runner terminated only its
four workers; return codes were `[1,1,1,1]`. Elapsed including load was 68.348 s.
All XM batch counters were zero. No complete inference result was produced;
both aggregate TPS fields are null. Available memory recovered after cleanup.
A second four-card 64-round XM/RMSNorm test passed after the guarded exit;
all devices remain bound and the owned-DDR holder remains active.
The CPU capacity-only test from the earlier commit does not establish that four
ngl=99 workers fit: this path additionally stores separate CUDA-shim model buffers.

The previous microbatch-8 RoPE staging failure (592,512 bytes required vs 524,288
owned) remains unresolved, but this 2048-request attempt did not reach it. Memory
protection was not disabled to force progress. The next step is to measure and
reduce replicated model/state allocations before retrying 2048, then fix/tile
RoPE staging before claiming sustained large-batch throughput.

## Scripts and Evidence

All working paths below are under `/srv/lanxin/qwen35-a3b`:

- `bin/recover-pacc-ddr.py`: validated stop, dynamic allocation and lifetime holder.
- `bin/package-recovery-jobd.py`: reproducible control and four new-address images.
- `bin/load-recovered-pacc.py`: one stopped device at a time; restore canonical image.
- `bin/run-pacc-recovered.py`: boot/ownership checks, per-card lease and strict policy.
- `bin/run-four-pacc-requests.py`: selectable wrapper/harness/library, memory floor,
  strict CPU-reference/phase/XM gates and shared-clock aggregation.
- `logs/recovery-four-fast-20261108/summary.json`: valid short-run results.
- `logs/recovery-four-512-20261108/summary.json`: failed 2048-request result.
- `logs/recovery-512-cpu-reference.json`: 512-state CPU reference only.
- `tools/pacc-qwen35/evidence-recovery-20261108`: bounded logs, JSON and SHA256 manifest.

The existing aggregation regression passes. A 4096 GiB floor test confirms refusal
before launching hardware workers. Python syntax checks pass. Firmware binaries,
model weights and full large logs are not committed; their provenance is recorded.
