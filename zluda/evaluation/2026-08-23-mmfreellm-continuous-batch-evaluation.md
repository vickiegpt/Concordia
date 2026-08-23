# MatMulFreeLM Continuous-Batch Throughput Evaluation

## Result

The MatMulFreeLM 2.7B CUDA path passed the 200 aggregate generated-token/s
qualification. Two deterministic measured runs used four windowed batches of
16 requests, generated exactly eight tokens per request, completed all 512
generated tokens, and sustained 363.62 and 366.62 aggregate tok/s.

This result is GPU-native BitLinear throughput. The TernIP/CXL adapter was
disabled, `fpga_tps_reported=false`, and no throughput in this evaluation is
attributed to the Agilex FPGA. The FPGA `MAX_LANES = 4` policy was not changed.

## Evaluated system

- HetGPU evaluated Git SHA: `3a9c4436080bf2d13de316054d687fed95c77ab8`
- HetGPU branch: `codex/mmfreellm-continuous-batch-20260823`
- HetGPU evaluation-relevant worktree status: clean (`git-status.txt` is empty)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- GPU UUID: `GPU-f07ea2df-1b6f-9a02-b534-5090abf3c174`
- Driver: 595.84
- GPU memory: 97887 MiB
- Python: 3.13.9
- PyTorch: 2.9.1+cu128
- CUDA reported by PyTorch: 12.8
- Transformers: 5.5.4
- Model: `/root/.cache/huggingface/hub/models--ridger--MMfreeLM-2.7B/snapshots/77deff0c1c9ac79aa51eb3ab7dd34fc375bf9324`
- Parameter count: 2,702,357,504
- Dtype: FP16
- BitLinear backend: `default`
- TernIP adapter: `disabled`

## External MatMulFreeLM source provenance

The imported `/root/matmulfreellm` worktree was intentionally evaluated as it
existed rather than cleaned or rewritten:

- Git SHA: `4ff8b0b8ef94ccf5b2eeb4450bea6c42ecd75c2b`
- Branch: `codex/16core-kimi-proof-20260817`
- Worktree status: dirty and preserved in
  `environment/mmfreellm-git-status-before.txt`
- Imported Python source files captured: 39

The package had four modified tracked Python files and five untracked Python
files under `mmfreelm/`. The proof contains the complete imported source tree,
staged and unstaged patches, inventories, and runtime hashes. Source inventory,
Git status, and all 39 source hashes were identical before and after benchmark
execution. This binds the result to the exact dirty source that ran without
discarding unrelated user work.

## Batch-1 control

The existing single-sequence benchmark ran three measured iterations after one
warmup. Mean throughput was 21.7490 tok/s and median throughput was 21.6969
tok/s. Its non-empty decoded output retained the prompt:

```text
The quick brown fox and in in the to in the in
```

It appended exactly eight integer token IDs:
`[304, 297, 297, 272, 298, 297, 272, 297]`.

## Batch-16 qualification

All latency columns are milliseconds converted from the canonical JSON. Peak
memory is the maximum CUDA allocation observed in each run. Aggregate elapsed
time is first enqueue through last completion; per-request latency begins at
each request's own enqueue, after bounded-queue backpressure where applicable.

| Run | Aggregate tok/s | Aggregate elapsed (s) | Queue p50 / p95 / max (ms) | Service p50 / p95 / max (ms) | Request E2E p50 / p95 / max (ms) | Observed batches | Peak CUDA memory | Completed | Generated tokens |
|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|
| 1 | 363.6208 | 1.408060 | 346.793 / 371.295 / 371.346 | 347.479 / 371.841 / 371.841 | 692.959 / 714.355 / 714.406 | 16, 16, 16, 16 | 5,784,931,840 B (5.39 GiB) | 64 | 512 |
| 2 | 366.6200 | 1.396541 | 346.527 / 359.386 / 359.441 | 347.231 / 359.589 / 359.589 | 697.152 / 709.384 / 709.439 | 16, 16, 16, 16 | 5,784,931,840 B (5.39 GiB) | 64 | 512 |

Both runs exceeded the 200 tok/s threshold independently. Request IDs were
complete and unique, no request failed, every decoded output retained the
prompt, and each request appended exactly eight token IDs. Generated token IDs
were identical request-by-request across both qualified runs. The representative
qualified output was:

```text
The quick brown fox in in in in the in the in
```

The batch-1 and batch-16 token IDs differed, so
`cross_batch_token_ids_equal=false`. This is recorded but is not an acceptance
failure: the hard determinism gate compares repeated runs at the same qualified
batch shape.

## Proof artifacts

Immutable proof directory:

`/home/victoryang00/hetGPU/.proof/mmfreellm-continuous-batch-20260823T235900Z`

Key artifacts:

- `manifest.json`: configuration, both repository identities, exit codes, and status
- `batch1/result.json`: batch-1 canonical record
- `batch16/result.json`: both complete batch-16 records and every request
- `batch1/stdout.log`, `batch1/stderr.log`: raw control output
- `batch16/stdout.log`, `batch16/stderr.log`: raw qualification output
- `environment/`: GPU, packages, and external-repository before/after status
- `source/inventory.txt`, `source/files/`: HetGPU and imported MatMulFreeLM source
- `source/mmfreellm/`: external inventories and staged/unstaged patches
- `hashes/mmfreellm-captured.sha256`: captured-copy hashes equal to runtime-before hashes
- `hashes/mmfreellm-runtime-before.sha256` and `-after.sha256`: identical runtime source hashes
- `hashes/source.sha256`: 49 verified captured source files
- `hashes/artifacts.sha256`: 75 verified proof artifacts

Selected artifact SHA-256 values:

```text
760bef4da3bdec180617400738314d73d4e94a64e5fadf704b3d088aac6bbdbc  batch1/result.json
9a7a1597e2ae0478fcac3f8ee3682620f60659398ba9fb30dfa6e3125f887ab7  batch16/result.json
484b2e88b6061b2d8200e678fd5286ffafae402c898927da798e846743f5b058  manifest.json
```

Both hash manifests were independently checked with `sha256sum --check` after
the live run. The overall `qualification-status.txt` was written as `pass` only
after manifest generation, artifact hashing, and hash verification succeeded.
