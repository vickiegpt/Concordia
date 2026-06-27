#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d /tmp/kimi-k26-tps-fake.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

runner="${tmp}/llama-cli"
model_dir="${tmp}/moonshotai_Kimi-K2.6-IQ1_M"
model_prefix="$(basename "${model_dir}")"
mkdir -p "${model_dir}"
for shard_index in 1 2 3 4 5 6; do
  touch "$(printf '%s/%s-%05d-of-00006.gguf' "${model_dir}" "${model_prefix}" "${shard_index}")"
done

cat >"${runner}" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf 'Kimi fake output\n'
cat >&2 <<'EOF'
llama_perf_context_print:        eval time =     500.00 ms /    25 tokens (   20.00 ms per token,    50.00 tokens per second)
llama_perf_context_print:       total time =     750.00 ms /    40 tokens
EOF
SH
chmod +x "${runner}"

BITNET_LLAMA_CLI="${runner}" \
MODEL_DIR="${model_dir}" \
MODEL_PREFIX="${model_prefix}" \
KIMI_TPS_WORKDIR="${tmp}/work" \
KIMI_TPS_CASES="baseline" \
KIMI_TPS_BUILD_ZLUDA=0 \
KIMI_TPS_BASELINE_WITH_SHIM=0 \
KIMI_TPS_REQUIRE_RUN=1 \
bash "${root}/run_kimi_k26_tps.sh"

csv="${tmp}/work/kimi_k26_tps.csv"
jsonl="${tmp}/work/kimi_k26_tps.jsonl"
grep -q '^baseline,pass,50,25,750,0,' "${csv}"
grep -q '"status": "pass"' "${jsonl}"
grep -q '"tps": 50' "${jsonl}"
grep -q '"tokens": 25' "${jsonl}"

echo "runner fake-run test passed"
