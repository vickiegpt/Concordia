#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(git -C "${script_dir}" rev-parse --show-toplevel)
harness=${script_dir}/run_v3_batch_evaluation.sh
fake_bin=${script_dir}/eval-fixtures/bin
test_root=$(mktemp -d /tmp/hetgpu-v3-eval-static.XXXXXX)
trap 'rm -rf -- "${test_root}"' EXIT

marker_baseline=${test_root}/baseline-executed
marker_enabled=${test_root}/enabled-executed
record_only_result=${test_root}/record-only
PATH="${fake_bin}:${PATH}" \
HETGPU_RUN_LIVE_V3_BATCH=0 \
HETGPU_RUN_LIVE_WORKLOAD_EVAL=0 \
HETGPU_EVAL_BASELINE_CMD="touch '${marker_baseline}'" \
HETGPU_EVAL_ENABLED_CMD="touch '${marker_enabled}'" \
"${harness}" "${record_only_result}"
test ! -e "${marker_baseline}"
test ! -e "${marker_enabled}"
python3 -c '
import json, pathlib, sys
summary = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["workload_commands_selected"] is True
assert summary["workloads_executed"] is False
assert summary["live_fixture_executed"] is False
' "${record_only_result}/summary.json"
test -f "${record_only_result}/patches/evaluation-relevant.staged.patch"
test -f "${record_only_result}/patches/evaluation-relevant.unstaged.patch"
test -f "${record_only_result}/source/evaluation-relevant-untracked.txt"
test -s "${record_only_result}/hashes/source.sha256"
test -s "${record_only_result}/hashes/source.final.sha256"
cmp -s "${record_only_result}/hashes/source.sha256" \
    "${record_only_result}/hashes/source.final.sha256"
test "$(cat "${record_only_result}/status/source_tree_stability.status")" -eq 0
test -s "${record_only_result}/hashes/final-artifacts.sha256"
test "$(cat "${record_only_result}/evaluated-git-head.txt")" = "$(git -C "${repo_root}" rev-parse HEAD)"
required_inventory=(
    Cargo.lock
    ext/nvidia_runtime-sys/src/lib.rs
    zluda/Cargo.toml
    zluda/build.rs
    zluda/src/cublas_shim.c
    zluda/src/cudart_dax_pool.c
    zluda/src/cudart_dax_pool.h
    zluda/src/cudart_shim.c
    zluda/src/impl/driver.rs
    zluda/src/impl/function.rs
    zluda/src/impl/iq1s_tmatmul.rs
    zluda/src/impl/mod.rs
    zluda/src/lib.rs
    zluda/tests/batch_scheduler_integration_test.rs
    zluda/tests/run_v3_batch_evaluation.sh
)
for expected_path in "${required_inventory[@]}"; do
    grep -Fxq "${expected_path}" \
        "${record_only_result}/source/evaluation-source-inventory.txt"
    grep -Fq "  ${expected_path}" "${record_only_result}/hashes/source.sha256"
    test -f "${record_only_result}/source/files/${expected_path}"
done
while IFS= read -r dirty_path; do
    [[ -n ${dirty_path} ]] || continue
    grep -Fxq "${dirty_path}" \
        "${record_only_result}/source/evaluation-source-inventory.txt"
done <"${record_only_result}/source/dirty-build-inputs.txt"

zero_result=${test_root}/zero-tests
set +e
PATH="${fake_bin}:${PATH}" \
HETGPU_FAKE_CARGO_ZERO_GATE=planner \
"${harness}" "${zero_result}" >"${test_root}/zero.stdout" 2>"${test_root}/zero.stderr"
zero_status=$?
set -e
test "${zero_status}" -ne 0
test ! -e "${zero_result}/summary.json"
grep -q 'planner.*zero tests\|planner.*expected at least' \
    "${zero_result}/logs/planner_test_count.stderr.log"

unpaired_result=${test_root}/unpaired
set +e
HETGPU_EVAL_BASELINE_CMD='true' "${harness}" "${unpaired_result}" \
    >"${test_root}/unpaired.stdout" 2>"${test_root}/unpaired.stderr"
unpaired_status=$?
set -e
test "${unpaired_status}" -eq 2
test ! -e "${unpaired_result}"

mutation_file=${test_root}/source-mutation-fixture.txt
printf '%s\n' stable >"${mutation_file}"
mutation_result=${test_root}/source-mutation
set +e
PATH="${fake_bin}:${PATH}" \
HETGPU_EVAL_STATIC_TEST_MODE=1 \
HETGPU_EVAL_EXTRA_SOURCE_FILE="${mutation_file}" \
HETGPU_FAKE_CARGO_MUTATE_FILE="${mutation_file}" \
"${harness}" "${mutation_result}" \
    >"${test_root}/mutation.stdout" 2>"${test_root}/mutation.stderr"
mutation_status=$?
set -e
test "${mutation_status}" -ne 0
test ! -e "${mutation_result}/summary.json"
test ! -e "${mutation_result}/STATUS"
test -s "${mutation_result}/hashes/source.sha256"
test -s "${mutation_result}/hashes/source.final.sha256"
test "$(cat "${mutation_result}/status/source_tree_final_hash.status")" -eq 0
test "$(cat "${mutation_result}/status/source_tree_stability.status")" -ne 0

echo "static v3 evaluation harness tests passed"
