#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d /tmp/kimi-k26-tps-parser.XXXXXX)"
trap 'rm -rf "${tmp}"' EXIT

aof="${tmp}/kimi.aof"
truncate -s 4096 "${aof}"

python3 "${root}/parse_kimi_tps.py" \
  --case concordia \
  --stdout "${root}/fixtures/kimi.stdout" \
  --stderr "${root}/fixtures/llama_timings.stderr" \
  --exit-code 0 \
  --total-ms 1300 \
  --aof "${aof}" \
  --runner /opt/bitnet/llama-cli \
  --model /models/kimi.gguf \
  --gpu "NVIDIA RTX PRO 6000 Blackwell" \
  --commit deadbeef \
  --csv "${tmp}/evidence.csv" \
  --jsonl "${tmp}/evidence.jsonl"

expected_header="case,status,tps,tokens,total_ms,aof_bytes,runner,model,gpu,commit,checkpoint_markers,message"
actual_header="$(sed -n '1p' "${tmp}/evidence.csv")"
if [[ "${actual_header}" != "${expected_header}" ]]; then
  echo "unexpected CSV header: ${actual_header}" >&2
  exit 1
fi

grep -q '^concordia,pass,81.12,64,1245.67,4096,/opt/bitnet/llama-cli,/models/kimi.gguf,NVIDIA RTX PRO 6000 Blackwell,deadbeef,2,ok$' "${tmp}/evidence.csv"

python3 - "${tmp}/evidence.jsonl" <<'PY'
import json
import sys
row = json.loads(open(sys.argv[1], encoding="utf-8").readline())
assert row["case"] == "concordia"
assert row["status"] == "pass"
assert row["tps"] == 81.12
assert row["tokens"] == 64
assert row["total_ms"] == 1245.67
assert row["aof_bytes"] == 4096
assert row["checkpoint_markers"] == 2
PY

python3 "${root}/parse_kimi_tps.py" \
  --case baseline \
  --stdout "${root}/fixtures/kimi.stdout" \
  --stderr "${root}/fixtures/no_timings.stderr" \
  --exit-code 0 \
  --total-ms 2222 \
  --runner /opt/bitnet/llama-cli \
  --model /models/kimi.gguf \
  --gpu "NVIDIA RTX PRO 6000 Blackwell" \
  --commit deadbeef \
  --csv "${tmp}/missing.csv" \
  --jsonl "${tmp}/missing.jsonl"

grep -q '^baseline,missing_timing,0,0,2222,0,/opt/bitnet/llama-cli,/models/kimi.gguf,NVIDIA RTX PRO 6000 Blackwell,deadbeef,0,no_generation_timing$' "${tmp}/missing.csv"

python3 "${root}/parse_kimi_tps.py" \
  --case failed \
  --stdout "${root}/fixtures/kimi.stdout" \
  --stderr "${root}/fixtures/no_timings.stderr" \
  --exit-code 42 \
  --total-ms 333 \
  --runner /opt/bitnet/llama-cli \
  --model /models/kimi.gguf \
  --gpu "NVIDIA RTX PRO 6000 Blackwell" \
  --commit deadbeef \
  --csv "${tmp}/failed.csv" \
  --jsonl "${tmp}/failed.jsonl"

grep -q '^failed,run_failed,0,0,333,0,/opt/bitnet/llama-cli,/models/kimi.gguf,NVIDIA RTX PRO 6000 Blackwell,deadbeef,0,runner_exit_42$' "${tmp}/failed.csv"

echo "parser tests passed"
