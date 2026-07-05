#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${root}/../.." && pwd)"
tmp="$(mktemp -d /tmp/concordia-eval-claims.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

python3 "${root}/run_eval_claims.py" \
  --repo-root "${repo_root}" \
  --paper "${repo_root}/69b5f215f5c67f0702bd8f65/05_eval.tex" \
  --claims "${root}/claims.json" \
  --work-dir "${tmp}/work" \
  --csv "${tmp}/eval_claims.csv" \
  --jsonl "${tmp}/eval_claims.jsonl" \
  --markdown "${tmp}/eval_claims.md" \
  --static-only

test -s "${tmp}/eval_claims.csv"
test -s "${tmp}/eval_claims.jsonl"
test -s "${tmp}/eval_claims.md"

head -n 1 "${tmp}/eval_claims.csv" | grep -Fxq "claim_id,status,source,expected,observed,unit,runner,artifact,message"
grep -q '^pk-dispatch-latency,' "${tmp}/eval_claims.csv"
grep -q '^delta-checkpoint-speedup,' "${tmp}/eval_claims.csv"
grep -q '^qwen-inference-tps,' "${tmp}/eval_claims.csv"
grep -q '^two-gpu-nccl-boundary,' "${tmp}/eval_claims.csv"
grep -q '^cross-arch-migration,' "${tmp}/eval_claims.csv"
grep -q '^realworld-llm-throughput,' "${tmp}/eval_claims.csv"
grep -q '"claim_id": "lora-sft-delta"' "${tmp}/eval_claims.jsonl"
grep -q '| pk-dispatch-latency |' "${tmp}/eval_claims.md"
grep -q '05_eval.tex' "${tmp}/eval_claims.md"

row_count="$(awk -F, 'NR > 1 { count++ } END { print count + 0 }' "${tmp}/eval_claims.csv")"
if [[ "${row_count}" -lt 12 ]]; then
  echo "expected at least 12 eval claims, saw ${row_count}" >&2
  exit 1
fi

fake_delta="${tmp}/fake_delta_bench.sh"
cat >"${fake_delta}" <<'SH'
#!/usr/bin/env bash
if [[ "${1:-}" == "--self-test" ]]; then
  echo "# device=fake RTX PRO 6000 ordinal=0 device_count=1 cc=12.0 rank=none"
  echo "self-test passed: dirty_pages=2 cpu_copy_ms=0.1 gpu_diff_ms=0.01"
  exit 0
fi
cat <<'OUT'
# table5_sparse_delta_checkpoint
region_mb,requested_dirty_pages,observed_dirty_pages,cpu_dtoh_ms,cpu_diff_ms,gpu_diff_ms,gpu_append_ms,cpu_total_ms,gpu_total_ms,speedup
16,1,1,1.0,4.0,0.1,0.1,5.0,0.2,25.0
# table6_dirty_scaling_256mb
region_mb,requested_dirty_pages,observed_dirty_pages,cpu_dtoh_ms,cpu_diff_ms,gpu_diff_ms,gpu_append_ms,cpu_total_ms,gpu_total_ms,speedup
256,32,32,10.0,100.0,0.5,0.2,110.0,0.7,157.1
# launch_overhead_us
mode,p50_us
sync,7.6
batch_per_launch,2.5
OUT
SH
chmod +x "${fake_delta}"

CONCORDIA_FORCE_GPU_COUNT=1 \
CONCORDIA_DELTA_BENCH_BINARY="${fake_delta}" \
python3 "${root}/run_eval_claims.py" \
  --repo-root "${repo_root}" \
  --paper "${repo_root}/69b5f215f5c67f0702bd8f65/05_eval.tex" \
  --claims "${root}/claims.json" \
  --work-dir "${tmp}/work-real" \
  --csv "${tmp}/eval_claims_real.csv" \
  --jsonl "${tmp}/eval_claims_real.jsonl" \
  --markdown "${tmp}/eval_claims_real.md"

grep -q '^delta-checkpoint-speedup,pass,' "${tmp}/eval_claims_real.csv"
grep -q '^delta-dirty-scaling,pass,' "${tmp}/eval_claims_real.csv"
grep -q 'delta_real:exit_0' "${tmp}/eval_claims_real.csv"

echo "eval claim runner tests passed"
