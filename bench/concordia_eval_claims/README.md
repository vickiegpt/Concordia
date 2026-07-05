# Concordia 05_eval Claim Runner

This harness maps the measurable claims in
`69b5f215f5c67f0702bd8f65/05_eval.tex` to repo-local evidence commands.
It does not rewrite paper numbers or fabricate hardware measurements. Claims
that require unavailable GPUs, Kimi model shards, NCCL ranks, or heterogeneous
targets are emitted as `blocked` or `partial` rows with the exact blocker.

## Run

```bash
python3 bench/concordia_eval_claims/run_eval_claims.py \
  --repo-root . \
  --paper 69b5f215f5c67f0702bd8f65/05_eval.tex \
  --claims bench/concordia_eval_claims/claims.json \
  --work-dir /tmp/concordia-eval-claims/work \
  --csv /tmp/concordia-eval-claims/eval_claims.csv \
  --jsonl /tmp/concordia-eval-claims/eval_claims.jsonl \
  --markdown /tmp/concordia-eval-claims/eval_claims.md
```

Use `--static-only` for a compile/static/fixture-only pass:

```bash
bash bench/concordia_eval_claims/test_eval_claims.sh
```

If `cargo-oxide` is not installed but the prebuilt cuda-oxide benchmark binary
exists, the runner uses:

```text
bench/concordia_delta_checkpoint/target/release/concordia_delta_checkpoint_bench
```

Override that path with:

```bash
CONCORDIA_DELTA_BENCH_BINARY=/path/to/concordia_delta_checkpoint_bench
```

`CONCORDIA_FORCE_GPU_COUNT=N` is intended for tests and constrained
launchers only; normal runs probe `nvidia-smi -L`.

## Output Schema

```text
claim_id,status,source,expected,observed,unit,runner,artifact,message
```

`status` is:

- `pass`: every runnable evidence command passed.
- `partial`: at least one evidence command passed, but live hardware/model
  evidence is blocked.
- `blocked`: no live evidence could run because a prerequisite is missing.
- `fail`: a required evidence command failed or the claim text was not found
  in `05_eval.tex`.
