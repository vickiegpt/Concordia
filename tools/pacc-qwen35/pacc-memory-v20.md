# Qwen four-PACC memory reduction

## Results

Model: Qwen3.5-35B-A3B-Q4_K_M. Host: lanxin2/wr5219.
No reboot, driver reload, firmware rebind, CPU-server stop, or reduction of the
32 GiB memory reserve was performed. The original runtime and libllama build
remain available. These are explicit opt-in candidate paths.

The memory initialization barrier is now passed with **2048 independent sequence
states, four PACC workers x 512, ngl=99**. Sharing immutable weights alone passes,
even when all state buffers use the original eager initialization. This result
does not rely on lazy zero pages hiding the state allocation.

| Measurement | Workers / sequences | Aggregate PSS | Private dirty | Minimum MemAvailable |
|---|---:|---:|---:|---:|
| Original allocation policy | 1 / 512 | 57.159 GiB | 56.646 GiB | 190.259 GiB |
| Shared weights; eager private state | 4 / 2048 | 167.125 GiB | 146.620 GiB | 99.843 GiB |
| Shared weights; demand-zero private state | 4 / 2048 | 22.912 GiB | 2.407 GiB | 244.455 GiB |

Measurements were taken while every worker held its context at a common barrier.
PSS, not the sum of RSS, accounts for shared file pages. The first row is a
one-worker baseline, not a successful four-worker measurement. The earlier
four-worker baseline exited at 29.02 GiB available under the 32 GiB guard.

Immediately after model loading, per-worker anonymous PSS falls from 20.0734 GiB
to 0.0828 GiB. The weights are backed by the same read-only GGUF pages, not four
private CUDA-shim copies. Live smaps show `r--s` model mappings and zero model
Private_Dirty on all four workers. The shared-weight/eager-state test retains
**32,160 MiB RS per worker**, plus KV and
compute/output buffers; no independent request state is merged or compressed.

The 22.912 GiB figure is **initialization residency only**. Anonymous pages are
materialized as requests write their independent states. It is not a claim of
22.912 GiB peak memory during sustained inference. The eager-state measurement
above demonstrates that the full allocation also fits after weight sharing.

## Implementation

- The isolated libllama build reuses its existing model mmap lifetime. A dedicated
  borrowed buffer keeps the CUDA buffer type for scheduling, without allocating
  or copying another model-sized buffer. All compute dispatch remains unchanged.
- This path requires `HETGPU_SIFIVE_MMAP_WEIGHTS=1` and the versioned runtime
  capability `hetgpu_sifive_host_backing_mode_v1() == 1`. Native-device and
  shared-DDR heap policies are rejected. There is no automatic activation for
  an ordinary NVIDIA device.
- Model buffers are immutable. Write/clear callbacks fail; buffer destruction
  releases only metadata. The model owns the file mapping. Tensors requiring
  backend padding are rejected rather than reading adjacent file bytes as padding.
- The existing pointwise broadcast compatibility path accepts readable mapped
  inputs under this explicit mode. Its output still requires an owned CUDA
  allocation. No CPU matrix fallback was added.
- With `HETGPU_SIFIVE_LAZY_ZERO=1`, host-backed CUDA allocations of at least 1 MiB
  use dedicated private anonymous mmap regions. D8/D32 zero operations validate
  ownership and bounds, discard only fully covered pages, and explicitly clear
  partial edge pages. Nonzero writes and failed madvise operations retain normal
  memset behavior. Free unmaps the owned region.
- Driver BOs, shared-DDR payloads, IPC mappings and small host allocations do not
  use this discard path. CUDA logical allocation counters are not physical RSS;
  OS PSS/Private_Dirty measurements are used in this report.

## Correctness and Application Tests

- Exported CUDA allocator/memset test: a 32 MiB + 123 byte allocation initially
  has zero resident pages. After a nonzero fill it has 8193 resident pages. An
  unaligned interior zero leaves only two boundary pages resident, preserving
  prefix/suffix bytes. D32, range rejection, fork isolation and unmap tests pass.
- Borrowed-buffer test passes reads, lifetime retention, backend-padding refusal,
  native-device refusal and intentional rejection of weight writes. Its child
  SIGABRT/backtrace is an expected negative test, not a failed inference.
- Full-model short run: four cards x two requests x two tokens, ngl=99. All
  **16 tokens match the CPU reference**; all four phase captures are valid.
  Each card completes 12,000 verified multi-column XM tiles with maximum N=2.
- Aggregate short-run decode is **0.0612601 TPS**: 8 tokens / 130.590704 seconds.
  Output including prefill is 0.0632734 TPS: 16 / 252.871046 seconds. This does
  not establish a speedup over the preceding 0.0626226 TPS short run. The primary
  demonstrated improvement is memory capacity, not compute throughput.
- Runtime build, isolated libllama build, benchmark aggregation regression and
  four-card 64-round XM/RMSNorm tests after the failed large run pass.

## Remaining 2048-Request Blocker

The new four-worker / 512-sequence-per-worker inference attempt gets past memory
initialization, with minimum MemAvailable **255,808,824 KiB** and no memory-guard
trigger. It fails after 9.664 s including load, before any XM batch is completed:

```text
quantize_mmq_q8_1<mmq_q8_1_ds_layout0, false>
SIFIVE kernel image must be a non-empty ELF64 payload
```

This is the next observed large-batch quantization/launch-path blocker. The older
microbatch-8 RoPE staging issue remains unresolved but was not reached in this
attempt. There are no valid 2048-request output tokens or TPS, and no 500 TPS
claim. Initial v19 also exposed a mapped-input rejection in bin_bcast; v20 fixes
that read-only input compatibility check. Both failed runs are retained.

## Using the Candidate

Root directory: `/srv/lanxin/qwen35-a3b`.

```bash
env HETGPU_SIFIVE_MMAP_WEIGHTS=1 HETGPU_SIFIVE_LAZY_ZERO=1 \
  HETGPU_SIFIVE_QK_VECTOR_REDUCE=1 \
  python3 bin/run-four-pacc-requests.py NEW_RUN_NAME \
  --runtime /srv/lanxin/qwen35-a3b/runtime-memory-v20 \
  --wrapper /srv/lanxin/qwen35-a3b/bin/run-pacc-recovered.py \
  --harness /srv/lanxin/qwen35-a3b/build-seq512-memory/bin/qwen-multiseq-v4 \
  --llama-library-dir /srv/lanxin/qwen35-a3b/build-seq512-memory/bin \
  --case /srv/lanxin/qwen35-a3b/bin/cross-fast.json \
  --reference /srv/lanxin/qwen35-a3b/logs/seq512-fast-cpu-reference-20261108.json \
  --ubatch 2 --timeout 600 --min-available-gib 32
```

This is the qualified short case, not a working 2048-request serving command.
Set LAZY_ZERO=0 to retain eager state allocation while sharing weights.
`run-memory-capacity.py` reproduces the barrier measurements; it reports null TPS.
`build-memory-candidate.sh` builds separate candidate directories from the saved
seq512 source and the tracked patch/helpers without replacing installed binaries.
Source/runtime hashes and bounded logs are in `evidence-memory-v20`; full raw logs
remain under `logs/memory-*` on lanxin2.
