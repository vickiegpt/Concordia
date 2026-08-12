#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d /tmp/kimi-k26-tps-runner.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

BITNET_LLAMA_CLI="/does/not/exist/llama-cli" \
MODEL_DIR="/does/not/exist/model" \
KIMI_TPS_WORKDIR="${tmp}" \
KIMI_TPS_CASES="baseline" \
KIMI_TPS_REQUIRE_RUN=0 \
bash "${root}/run_kimi_k26_tps.sh"

csv="${tmp}/kimi_k26_tps.csv"
jsonl="${tmp}/kimi_k26_tps.jsonl"

test -s "${csv}"
test -s "${jsonl}"
grep -q '^baseline,skipped_missing_runner,0,0,0,0,/does/not/exist/llama-cli,' "${csv}"
grep -q '"status": "skipped_missing_runner"' "${jsonl}"
grep -q 'KIMI_TPS_ZLUDA_FEATURES' "${root}/run_kimi_k26_tps.sh"
grep -q 'KIMI_TMATMUL_COCOTB' "${root}/run_kimi_k26_tps.sh"
grep -q 'KIMI_TMATMUL_FPGA' "${root}/README.md"
grep -q 'KIMI_TMATMUL_FPGA' "${root}/../../tools/run_kimi_k26_iq1m_bitnet.sh"

echo "runner static test passed"
