#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
fixture_root=$(mktemp -d /tmp/mmfreellm-continuous-eval-test.XXXXXX)
trap 'rm -rf -- "${fixture_root}"' EXIT

batch1_fixture=${fixture_root}/batch1-pass
batch1_bad_fixture=${fixture_root}/batch1-bad-output
batch16_pass_fixture=${fixture_root}/batch16-pass
batch16_fail_fixture=${fixture_root}/batch16-fail

cat >"${batch1_fixture}" <<'PY'
#!/usr/bin/env python3
import json

record = {
    "schema": "matmulfreellm-generation-benchmark-v1",
    "validated": True,
    "device": "cuda",
    "dtype": "half",
    "max_new_tokens": 8,
    "prompt": "The quick brown fox",
    "output": "The quick brown fox jumps",
    "generated_token_ids": list(range(10, 18)),
    "total_runs": 3,
    "total_generated_tokens": 24,
    "runs": [{"generated_tokens": 8} for _ in range(3)],
    "mean_tokens_per_second": 20.0,
    "median_tokens_per_second": 20.0,
}
print("MMFREELM_BENCHMARK_JSON=" + json.dumps(record, sort_keys=True))
PY

cat >"${batch1_bad_fixture}" <<'PY'
#!/usr/bin/env python3
import json

record = {
    "schema": "matmulfreellm-generation-benchmark-v1",
    "validated": True,
    "device": "cuda",
    "dtype": "half",
    "max_new_tokens": 8,
    "prompt": "The quick brown fox",
    "output": "semantic mismatch",
    "generated_token_ids": list(range(10, 18)),
    "total_runs": 3,
    "total_generated_tokens": 24,
    "runs": [{"generated_tokens": 8} for _ in range(3)],
    "mean_tokens_per_second": 20.0,
    "median_tokens_per_second": 20.0,
}
print("MMFREELM_BENCHMARK_JSON=" + json.dumps(record, sort_keys=True))
PY

cat >"${batch16_pass_fixture}" <<'PY'
#!/usr/bin/env python3
import json

requests = [
    {
        "request_id": request_id,
        "output": "The quick brown fox jumps",
        "generated_token_ids": list(range(10, 18)),
    }
    for request_id in range(64)
]
runs = [
    {
        "run_index": run_index,
        "requested_requests": 64,
        "completed_requests": 64,
        "failed_requests": 0,
        "total_generated_tokens": 512,
        "aggregate_tokens_per_second": 250.0 + run_index,
        "observed_batch_sizes": [16, 16, 16, 16],
        "requests": requests,
    }
    for run_index in (1, 2)
]
record = {
    "schema": "matmulfreellm-continuous-batch-benchmark-v1",
    "validated": True,
    "qualification_passed": True,
    "failure_reasons": [],
    "deterministic_generated_token_ids": True,
    "device": "cuda",
    "dtype": "half",
    "request_count": 64,
    "max_batch_size": 16,
    "max_new_tokens": 8,
    "qualification_runs": 2,
    "min_aggregate_tps": 200.0,
    "bitlinear_backend": "default",
    "ternip_adapter": "disabled",
    "fpga_tps_reported": False,
    "runs": runs,
}
print("MMFREELM_CONTINUOUS_BATCH_JSON=" + json.dumps(record, sort_keys=True))
PY

cat >"${batch16_fail_fixture}" <<'PY'
#!/usr/bin/env python3
import json

requests = [
    {
        "request_id": request_id,
        "output": "The quick brown fox jumps",
        "generated_token_ids": list(range(10, 18)),
    }
    for request_id in range(64)
]
runs = [
    {
        "run_index": run_index,
        "requested_requests": 64,
        "completed_requests": 64,
        "failed_requests": 0,
        "total_generated_tokens": 512,
        "aggregate_tokens_per_second": 199.0 if run_index == 1 else 250.0,
        "observed_batch_sizes": [16, 16, 16, 16],
        "requests": requests,
    }
    for run_index in (1, 2)
]
record = {
    "schema": "matmulfreellm-continuous-batch-benchmark-v1",
    "validated": True,
    "qualification_passed": False,
    "failure_reasons": ["run 1 below target"],
    "deterministic_generated_token_ids": True,
    "device": "cuda",
    "dtype": "half",
    "request_count": 64,
    "max_batch_size": 16,
    "max_new_tokens": 8,
    "qualification_runs": 2,
    "min_aggregate_tps": 200.0,
    "bitlinear_backend": "default",
    "ternip_adapter": "disabled",
    "fpga_tps_reported": False,
    "runs": runs,
}
print("MMFREELM_CONTINUOUS_BATCH_JSON=" + json.dumps(record, sort_keys=True))
raise SystemExit(1)
PY
chmod 700 \
    "${batch1_fixture}" "${batch1_bad_fixture}" \
    "${batch16_pass_fixture}" "${batch16_fail_fixture}"

pass_dir=${fixture_root}/proof-pass
HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
HETGPU_MMFREELM_BATCH1_CMD="${batch1_fixture}" \
HETGPU_MMFREELM_BATCH16_CMD="${batch16_pass_fixture}" \
    "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" "${pass_dir}"

test "$(cat "${pass_dir}/qualification-status.txt")" = pass
test -s "${pass_dir}/batch1/result.json"
test -s "${pass_dir}/batch16/result.json"
test -d "${pass_dir}/source/files"
python3 - "${pass_dir}/manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert manifest["qualification_passed"] is True
assert manifest["cross_batch_token_ids_equal"] is True
assert manifest["configuration"]["fpga_tps_reported"] is False
external = manifest["external_model_repository"]
assert external["path"] == "/root/matmulfreellm"
assert len(external["git_sha"]) == 40
assert external["source_file_count"] > 0
PY
test -f "${pass_dir}/environment/mmfreellm-git-status-before.txt"
test -s "${pass_dir}/hashes/mmfreellm-runtime-before.sha256"
cmp \
    "${pass_dir}/hashes/mmfreellm-runtime-before.sha256" \
    "${pass_dir}/hashes/mmfreellm-runtime-after.sha256"
(cd "${pass_dir}" && sha256sum --check hashes/source.sha256 >/dev/null)
(cd "${pass_dir}" && sha256sum --check hashes/artifacts.sha256 >/dev/null)

fail_dir=${fixture_root}/proof-fail
if HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
   HETGPU_MMFREELM_BATCH1_CMD="${batch1_fixture}" \
   HETGPU_MMFREELM_BATCH16_CMD="${batch16_fail_fixture}" \
       "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" "${fail_dir}"; then
    echo "failed qualification unexpectedly passed" >&2
    exit 1
fi
test "$(cat "${fail_dir}/qualification-status.txt")" = failed
if [[ ! -s ${fail_dir}/batch16/result.json ]]; then
    echo "failed proof did not preserve batch-16 JSON" >&2
    find "${fail_dir}" -maxdepth 3 -type f -printf '%P %s bytes\n' >&2
    sed -n '1,160p' "${fail_dir}/manifest.json" >&2
    tail -3 "${fail_dir}/batch16/stdout.log" >&2
    if [[ -s ${fail_dir}/batch16/stderr.log ]]; then
        sed -n '1,120p' "${fail_dir}/batch16/stderr.log" >&2
    fi
    exit 1
fi

bad_control_dir=${fixture_root}/proof-bad-control
if HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
   HETGPU_MMFREELM_BATCH1_CMD="${batch1_bad_fixture}" \
   HETGPU_MMFREELM_BATCH16_CMD="${batch16_pass_fixture}" \
       "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" \
       "${bad_control_dir}"; then
    echo "semantically invalid batch-1 control unexpectedly passed" >&2
    exit 1
fi
test "$(cat "${bad_control_dir}/qualification-status.txt")" = failed

finalize_fail_dir=${fixture_root}/proof-finalize-fail
if HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
   HETGPU_MMFREELM_INJECT_FINALIZE_FAILURE=1 \
   HETGPU_MMFREELM_BATCH1_CMD="${batch1_fixture}" \
   HETGPU_MMFREELM_BATCH16_CMD="${batch16_pass_fixture}" \
       "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" \
       "${finalize_fail_dir}"; then
    echo "injected finalization failure unexpectedly passed" >&2
    exit 1
fi
test "$(cat "${finalize_fail_dir}/qualification-status.txt")" = failed

capture_fail_dir=${fixture_root}/proof-capture-fail
if HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
   HETGPU_MMFREELM_INJECT_CAPTURE_MISMATCH=1 \
   HETGPU_MMFREELM_BATCH1_CMD="${batch1_fixture}" \
   HETGPU_MMFREELM_BATCH16_CMD="${batch16_pass_fixture}" \
       "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" \
       "${capture_fail_dir}"; then
    echo "external-source capture mismatch unexpectedly passed" >&2
    exit 1
fi
test "$(cat "${capture_fail_dir}/qualification-status.txt")" = failed

if HETGPU_MMFREELM_STATIC_TEST_MODE=1 \
   HETGPU_MMFREELM_BATCH1_CMD="${batch1_fixture}" \
   HETGPU_MMFREELM_BATCH16_CMD="${batch16_pass_fixture}" \
       "${script_dir}/run_mmfreellm_continuous_batch_evaluation.sh" "${pass_dir}"; then
    echo "existing proof unexpectedly overwritten" >&2
    exit 1
fi

echo "MatMulFreeLM continuous-batch evaluation static tests passed"
