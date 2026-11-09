# Qwen3.5-35B-A3B NGL sweep on lanxin2, 2026-11-08

## Decision

The fastest fully validated configuration in this experiment is **ngl=0**:
**31.5464 aggregate decode tokens/s**, the median of two 32-token runs.
This is CPU throughput, not PACC throughput. Including prefill, the median
aggregate output rate is **25.9503 tokens/s**. No default service configuration
was changed.

The fastest XM-qualified two-token screening point is ngl=2 (1.1546 decode
tokens/s), but it fails the longer CPU-token comparison in both repetitions.
Do not use that screening rate as a correct sustained PACC result. The other
positive NGL points have only the short screening evidence, not a 32-token pass.

## Conditions and accounting

- Host: lanxin2 / wr5219, riscv64, 32 online CPUs, Linux 6.18.3+.
- Model: `/srv/lanxin/qwen35-a3b/models/Qwen3.5-35B-A3B-Q4_K_M.gguf`.
- Runtime: `runtime-memory-v20`, private `build-seq512-memory` libllama,
  `qwen-multiseq-v5`, original `build-pacc-cuda-gcc14` GGML libraries.
- Four workers, two independent sequences each: **8 requests, not 2048**.
- Positive NGL workers are individually leased to physical PACC0/1/2/3.
  The CUDA/GPU wording in llama logs names the compatibility backend, not an RTX GPU.
- Eight CPU threads per worker, ubatch=2, greedy sampling, finite-logit checks,
  independent recurrent state, batched prefill, exact CPU token reference.
- Mapped immutable model weights and lazy-zero backing remain enabled.
- All workers load and initialize before a common start barrier is released.
- Decode TPS = sum of valid generated tokens after each request's first token /
  (latest worker finish - earliest worker decode start). Individual TPS are NOT summed.
- Output TPS includes prefill but excludes model loading. Qualification and
  telemetry initialization are outside the inference clock window.
- Positive NGL uses streaming jobd telemetry. CPU baseline does not; these are
  instrumented application-path comparisons, not isolated chip peak measurements.
- The unrelated CPU llama-server (PID 4807) and other host services were retained.

## Two-token screening

Prompts are `1` and `2`; each worker emits four tokens. Each row covers 16 output
tokens total, including 8 decode tokens. This is a short numerical/placement
screen, not a natural-language quality or sustained-throughput benchmark.

| Requested NGL | Observed offloaded layers | Aggregate decode TPS | Qualification |
| --- | --- | --- | --- |
| 0 | 0 | 22.2167 | CPU-only; all tokens match |
| 1 | 1/41 | not XM-qualified | All tokens match, but no cross-request XM-column record |
| 2 | 2/41 | 1.1546 | Tokens and four-device XM gates pass |
| 3 | 3/41 | 0.8301 | Tokens and four-device XM gates pass |
| 4 | 4/41 | 0.6120 | Tokens and four-device XM gates pass |
| 8 | 8/41 | 0.3145 | Tokens and four-device XM gates pass |
| 16 | 16/41 | 0.1595 | Tokens and four-device XM gates pass |
| 32 | 32/41 | 0.08114 | Tokens and four-device XM gates pass |
| 99 | 41/41 | 0.06250 | Tokens and four-device XM gates pass |

## 32-token validation

Prompts ask for detailed explanations of virtual memory and matrix multiplication.
Each passing round emits 256 tokens, of which 248 are decode tokens. A fresh
CPU-only reference completed both 32-token sequences. Test order was 0,1,2 and
then 2,1,0, without overlapping benchmark points.

| NGL | First round | Second round | Valid median decode TPS |
| --- | --- | --- | --- |
| 0 | 31.1116 TPS, all 256 tokens match | 31.9811 TPS, all 256 tokens match | **31.5464** |
| 1 | CPU-token mismatch | Telemetry capture rejected; workers terminated | null |
| 2 | CPU-token mismatch | Same CPU-token mismatch | null |

The mismatch is sequence index 1 (matrix multiplication), generation step 7
(the eighth generated token): expected token **4530**, actual **2876**.
It was captured on all four workers for ngl=1 round 0 and on three saved worker
results in each ngl=2 round; the fail-fast runner terminated peers before all
could write results. XM multi-column submissions were observed for ngl=2, but
they do not establish full-model numerical correctness.

In ngl=1 round 1 the PACC1 phase collector reported
`RuntimeError: duplicate phase or reused completed sequence`; the runner stopped
the group before result JSONs were produced. This is a separate capture failure,
not a second measured token mismatch. No throughput is assigned to failed runs.
Logs also contain PTX parser warnings; no causal attribution is established here.

Exact greedy-token divergence alone does not distinguish harmless rounding near
an argmax tie from an incorrect kernel/state transition. Logit margins and
layer-level error are not captured by this harness. The conservative acceptance
gate remains failed rather than silently relaxing correctness.

## Profile and memory observations

Across four workers, the PRELOADED compute sum grows from 16.936 s at ngl=2 to
124.213 s at ngl=32 in screening. Input-copy sums stay near 5 s. These sums include
startup and exclude other ELF kernel work, so they are NOT decode wall-time
percentages and do not prove that interrupts or completion observation dominate.

Minimum observed host MemAvailable across all points was **245.370 GiB**;
the 32 GiB guard did not fire. This is a host-wide availability measurement, not
per-model PSS, VRAM use, or a 2048-request capacity validation. The 2048-request
MMQ ELF-generation blocker from the previous experiment remains unresolved.

All four cards passed the post-sweep test: 64 alternating XM/RMS submissions per
card, deterministic XM outputs, numerical RMS checks and unchanged guards.
Boot ID stayed `4ca2fa21-0d48-4090-93a5-51206ac1de29`; PID 4807 remained alive and
`pacc-owned-ddr-pin-20261108.service` remained active. No reboot, firmware reload,
driver reset, or default runtime replacement was performed.

## Reproduction and artifacts

`build-ngl-sweep.sh` builds the barrier-enabled harness against the existing
seq512-memory source and libraries. `run-four-pacc-requests.py` adds `--ngl` and
`--start-barrier`, CPU-only isolation, memory/timeout cleanup, and separate token
and XM acceptance fields. `test-four-request-summary.py` covers matching NGL,
missing hardware proof, CPU separation, and invalid per-worker timing.

The deployment under `/srv/lanxin/qwen35-a3b/bin` includes the helper scripts and
both CPU references. With that deployment:

```sh
ROOT=/srv/lanxin/qwen35-a3b
python3 "$ROOT/bin/run-ngl-sweep.py" --phase coarse --ngls 0,1,2,3,4,8,16,32,99
python3 "$ROOT/bin/run-ngl-sweep.py" --phase long --ngls 0,1,2 --repeats 2
```

The sweep reuses completed run directories after configuration/artifact checks;
it does not overwrite raw results or silently rerun them. For a fresh point,
invoke `run-four-pacc-requests.py` with a new prefix and the same recorded options.
The sweep progress `elapsed_s` is orchestration time (including near-zero cache
reuse), NOT inference time. Use per-run monotonic timestamps and TPS fields.

`evidence-ngl-sweep-v20/` contains all 15 run summaries, per-worker results,
settings, artifact hashes, qualification/phase summaries, full gzip-compressed
logs, CPU references, input cases and post-sweep checks. `results.json` separates
token errors from profile errors. The top-level progress recheck corrects the
distinction between observed XM-column log records and a fully qualified capture;
original per-run summaries/logs are preserved unchanged. A few older short-run
scope strings say 'no CPU matrix fallback'; they must not be interpreted as
all-model device execution: non-offloaded layers intentionally execute on CPU.

Next correctness work should compare forced-prefix logits around sequence 1,
step 7 for ngl=0/1/2, and separately diagnose the phase-capture duplicate. Only
after passing those gates should larger cross-request batches be ranked by TPS.
