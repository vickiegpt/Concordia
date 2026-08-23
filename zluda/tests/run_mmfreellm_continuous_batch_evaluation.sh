#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

usage() {
    echo "usage: $0 NEW_RESULT_DIRECTORY" >&2
}

fail() {
    echo "run_mmfreellm_continuous_batch_evaluation: $*" >&2
    exit 2
}

[[ $# -eq 1 ]] || {
    usage
    exit 2
}

result_argument=$1
[[ -n ${result_argument} ]] || fail "result directory must not be empty"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(git -C "${script_dir}" rev-parse --show-toplevel)
result_parent=$(dirname -- "${result_argument}")
result_name=$(basename -- "${result_argument}")
[[ ${result_name} != . && ${result_name} != .. && -n ${result_name} ]] \
    || fail "unsafe result directory name"
[[ -d ${result_parent} ]] \
    || fail "result directory parent does not exist: ${result_parent}"
result_parent=$(cd -- "${result_parent}" && pwd -P)
result_dir=${result_parent}/${result_name}
case ${result_dir} in
    /|"${repo_root}"|"${script_dir}")
        fail "refusing unsafe result directory: ${result_dir}"
        ;;
esac
[[ ! -e ${result_dir} && ! -L ${result_dir} ]] \
    || fail "result directory already exists; refusing to overwrite: ${result_dir}"

static_test_mode=${HETGPU_MMFREELM_STATIC_TEST_MODE:-0}
inject_finalize_failure=${HETGPU_MMFREELM_INJECT_FINALIZE_FAILURE:-0}
inject_capture_mismatch=${HETGPU_MMFREELM_INJECT_CAPTURE_MISMATCH:-0}
batch1_override=${HETGPU_MMFREELM_BATCH1_CMD:-}
batch16_override=${HETGPU_MMFREELM_BATCH16_CMD:-}
[[ ${static_test_mode} == 0 || ${static_test_mode} == 1 ]] \
    || fail "HETGPU_MMFREELM_STATIC_TEST_MODE must be 0 or 1"
[[ ${inject_finalize_failure} == 0 || ${inject_finalize_failure} == 1 ]] \
    || fail "HETGPU_MMFREELM_INJECT_FINALIZE_FAILURE must be 0 or 1"
[[ ${inject_capture_mismatch} == 0 || ${inject_capture_mismatch} == 1 ]] \
    || fail "HETGPU_MMFREELM_INJECT_CAPTURE_MISMATCH must be 0 or 1"
if [[ ${inject_finalize_failure} == 1 && ${static_test_mode} != 1 ]]; then
    fail "finalization failure injection is restricted to static mode"
fi
if [[ ${inject_capture_mismatch} == 1 && ${static_test_mode} != 1 ]]; then
    fail "source capture mismatch injection is restricted to static mode"
fi
if [[ ${static_test_mode} == 1 ]]; then
    [[ -n ${batch1_override} && -n ${batch16_override} ]] \
        || fail "static mode requires both benchmark command overrides"
else
    [[ -z ${batch1_override} && -z ${batch16_override} ]] \
        || fail "benchmark command overrides are restricted to static mode"
fi

bitlinear_backend=${MMFREELM_BITLINEAR_BACKEND:-default}
[[ ${bitlinear_backend} == default ]] \
    || fail "MMFREELM_BITLINEAR_BACKEND must be default"
ternip_adapter=${MMFREELM_TERNIP_ADAPTER:-disabled}
case ${ternip_adapter,,} in
    ""|0|false|disabled) ;;
    *) fail "MMFREELM_TERNIP_ADAPTER must be disabled" ;;
esac
export MMFREELM_BITLINEAR_BACKEND=default
export MMFREELM_TERNIP_ADAPTER=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

model_repo=${HETGPU_MMFREELM_REPO_ROOT:-/root/matmulfreellm}
if [[ -n ${HETGPU_MMFREELM_REPO_ROOT:-} && ${static_test_mode} != 1 ]]; then
    fail "HETGPU_MMFREELM_REPO_ROOT override is restricted to static mode"
fi
model_repo=$(realpath -e -- "${model_repo}") \
    || fail "MatMulFreeLM repository does not exist"
git -C "${model_repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || fail "MatMulFreeLM repository is not a Git worktree: ${model_repo}"
[[ -d ${model_repo}/mmfreelm ]] \
    || fail "MatMulFreeLM Python package is missing: ${model_repo}/mmfreelm"
model_repo_sha=$(git -C "${model_repo}" rev-parse HEAD)
model_repo_branch=$(git -C "${model_repo}" branch --show-current)

started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_sha=$(git -C "${repo_root}" rev-parse HEAD)
git_branch=$(git -C "${repo_root}" branch --show-current)
batch1_exit=-1
batch16_exit=-1
cross_batch_token_ids_equal=false
failure_reason=
finalizing=0
model_source_file_count=0

mkdir -m 700 -- "${result_dir}"
mkdir -p -m 700 -- \
    "${result_dir}/batch1" \
    "${result_dir}/batch16" \
    "${result_dir}/commands" \
    "${result_dir}/environment" \
    "${result_dir}/hashes" \
    "${result_dir}/logs" \
    "${result_dir}/source/files" \
    "${result_dir}/source/mmfreellm"

finalize() {
    local qualification=$1
    local exit_code=$2
    local effective_qualification=${qualification}
    local finalization_failed=0
    local status=failed
    finalizing=1
    trap - ERR
    set +e
    if [[ ${inject_finalize_failure} == 1 ]]; then
        finalization_failed=1
        effective_qualification=false
        failure_reason="injected finalization failure"
    fi
    ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if ! python3 - \
        "${result_dir}/manifest.json" \
        "${started_utc}" "${ended_utc}" "${repo_root}" \
        "${git_sha}" "${git_branch}" "${static_test_mode}" \
        "${batch1_exit}" "${batch16_exit}" "${effective_qualification}" \
        "${cross_batch_token_ids_equal}" "${failure_reason}" \
        "${model_repo}" "${model_repo_sha}" "${model_repo_branch}" \
        "${model_source_file_count}" <<'PY'
import json
import pathlib
import sys

(
    output,
    started,
    ended,
    repo,
    sha,
    branch,
    static_mode,
    batch1_exit,
    batch16_exit,
    qualified,
    cross_batch_equal,
    failure_reason,
    model_repo,
    model_repo_sha,
    model_repo_branch,
    model_source_file_count,
) = sys.argv[1:]
manifest = {
    "schema": "matmulfreellm-continuous-batch-evaluation-v1",
    "started_utc": started,
    "ended_utc": ended,
    "repo_root": repo,
    "git_sha": sha,
    "git_branch": branch,
    "git_status_artifact": "git-status.txt",
    "qualification_passed": qualified == "true",
    "failure_reason": failure_reason or None,
    "cross_batch_token_ids_equal": cross_batch_equal == "true",
    "external_model_repository": {
        "path": model_repo,
        "git_sha": model_repo_sha,
        "git_branch": model_repo_branch,
        "source_file_count": int(model_source_file_count),
        "git_status_before": "environment/mmfreellm-git-status-before.txt",
        "git_status_after": "environment/mmfreellm-git-status-after.txt",
        "runtime_hashes_before": "hashes/mmfreellm-runtime-before.sha256",
        "runtime_hashes_after": "hashes/mmfreellm-runtime-after.sha256",
        "captured_source_hashes": "hashes/mmfreellm-captured.sha256",
    },
    "configuration": {
        "static_test_mode": static_mode == "1",
        "request_count": 64,
        "max_batch_size": 16,
        "max_new_tokens": 8,
        "qualification_runs": 2,
        "minimum_aggregate_tps": 200.0,
        "bitlinear_backend": "default",
        "ternip_adapter": "disabled",
        "fpga_tps_reported": False,
    },
    "command_exit_codes": {
        "batch1": int(batch1_exit),
        "batch16": int(batch16_exit),
    },
    "artifacts": {
        "batch1_json": "batch1/result.json",
        "batch16_json": "batch16/result.json",
        "source_hashes": "hashes/source.sha256",
        "artifact_hashes": "hashes/artifacts.sha256",
        "external_source_inventory": "source/mmfreellm/inventory-before.txt",
    },
}
with pathlib.Path(output).open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
    then
        finalization_failed=1
        effective_qualification=false
    fi
    if ! (
        cd -- "${result_dir}"
        find . -type f \
            ! -path './hashes/artifacts.sha256' \
            ! -path './hashes/artifacts.sha256.pending' \
            ! -path './qualification-status.txt' \
            -print0 \
            | sort -z \
            | xargs -0 sha256sum >hashes/artifacts.sha256.pending \
            && mv -- hashes/artifacts.sha256.pending hashes/artifacts.sha256 \
            && sha256sum --check hashes/artifacts.sha256 >/dev/null
    ); then
        finalization_failed=1
        effective_qualification=false
    fi
    if [[ ${finalization_failed} == 0 \
          && ${effective_qualification} == true \
          && ${exit_code} == 0 ]]; then
        status=pass
    fi
    if ! printf '%s\n' "${status}" >"${result_dir}/qualification-status.txt"; then
        exit 1
    fi
    if [[ ${finalization_failed} != 0 ]]; then
        exit 1
    fi
    exit "${exit_code}"
}

unexpected_failure() {
    local exit_code=$?
    if [[ ${finalizing} == 0 ]]; then
        failure_reason="unexpected harness command failure (exit ${exit_code})"
        finalize false "${exit_code}"
    fi
    exit "${exit_code}"
}
trap unexpected_failure ERR

evaluation_fail() {
    failure_reason=$1
    echo "run_mmfreellm_continuous_batch_evaluation: ${failure_reason}" >&2
    finalize false 1
}

git -C "${repo_root}" status --short >"${result_dir}/git-status.txt"
printf '%s\n' "${git_sha}" >"${result_dir}/evaluated-git-head.txt"
printf '%s\n' \
    "MMFREELM_BITLINEAR_BACKEND=default" \
    "MMFREELM_TERNIP_ADAPTER=disabled" \
    "HF_HUB_OFFLINE=1" \
    "TRANSFORMERS_OFFLINE=1" \
    "TOKENIZERS_PARALLELISM=false" \
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" \
    >"${result_dir}/environment/allowlist.txt"
nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader \
    >"${result_dir}/environment/nvidia-smi.csv" 2>"${result_dir}/environment/nvidia-smi.stderr" \
    || true
python3 - <<'PY' >"${result_dir}/environment/python-packages.txt" 2>&1
import sys
import torch
import transformers

print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("transformers", transformers.__version__)
PY

source_files=(
    docs/superpowers/specs/2026-08-21-mmfreellm-continuous-batch-throughput-design.md
    docs/superpowers/plans/2026-08-23-mmfreellm-continuous-batch-throughput.md
    zluda/tests/mmfreellm_continuous_batch.py
    zluda/tests/test_mmfreellm_continuous_batch.py
    zluda/tests/run_mmfreellm_2p7b_benchmark.py
    zluda/tests/test_run_mmfreellm_2p7b_benchmark.py
    zluda/tests/run_mmfreellm_continuous_batch_benchmark.py
    zluda/tests/test_run_mmfreellm_continuous_batch_benchmark.py
    zluda/tests/run_mmfreellm_continuous_batch_evaluation.sh
    zluda/tests/test_run_mmfreellm_continuous_batch_evaluation_static.sh
)
printf '%s\n' "${source_files[@]}" >"${result_dir}/source/inventory.txt"
for relative_path in "${source_files[@]}"; do
    [[ -f ${repo_root}/${relative_path} ]] \
        || evaluation_fail "evaluation source is missing: ${relative_path}"
    mkdir -p -- "${result_dir}/source/files/$(dirname -- "${relative_path}")"
    cp -- "${repo_root}/${relative_path}" \
        "${result_dir}/source/files/${relative_path}"
done

mapfile -d '' model_source_files < <(
    cd -- "${model_repo}"
    find mmfreelm -type f -name '*.py' -print0 | sort -z
)
model_source_file_count=${#model_source_files[@]}
[[ ${model_source_file_count} -gt 0 ]] \
    || evaluation_fail "MatMulFreeLM source inventory is empty"
printf '%s\n' "${model_source_files[@]}" \
    >"${result_dir}/source/mmfreellm/inventory-before.txt"
for relative_path in "${model_source_files[@]}"; do
    destination=${result_dir}/source/files/external/matmulfreellm/${relative_path}
    mkdir -p -- "$(dirname -- "${destination}")"
    cp -- "${model_repo}/${relative_path}" "${destination}"
    printf 'external/matmulfreellm/%s\n' "${relative_path}" \
        >>"${result_dir}/source/inventory.txt"
done
git -C "${model_repo}" status --short \
    >"${result_dir}/environment/mmfreellm-git-status-before.txt"
git -C "${model_repo}" diff --binary -- mmfreelm \
    >"${result_dir}/source/mmfreellm/unstaged.patch"
git -C "${model_repo}" diff --cached --binary -- mmfreelm \
    >"${result_dir}/source/mmfreellm/staged.patch"
(
    cd -- "${model_repo}"
    sha256sum "${model_source_files[@]}"
) >"${result_dir}/hashes/mmfreellm-runtime-before.sha256"
if [[ ${inject_capture_mismatch} == 1 ]]; then
    printf '\n# injected capture mismatch\n' \
        >>"${result_dir}/source/files/external/matmulfreellm/${model_source_files[0]}"
fi
(
    cd -- "${result_dir}/source/files/external/matmulfreellm"
    sha256sum "${model_source_files[@]}"
) >"${result_dir}/hashes/mmfreellm-captured.sha256"
cmp \
    "${result_dir}/hashes/mmfreellm-runtime-before.sha256" \
    "${result_dir}/hashes/mmfreellm-captured.sha256" \
    || evaluation_fail "captured MatMulFreeLM source differs from runtime source"
(
    cd -- "${result_dir}"
    find source/files -type f -print0 | sort -z | xargs -0 sha256sum \
        >hashes/source.sha256
)

batch1_command=(
    python3 "${script_dir}/run_mmfreellm_2p7b_benchmark.py"
    --repo-root "${model_repo}"
    --device cuda --dtype half --max-new-tokens 8 --warmup-runs 1 --runs 3
)
batch16_command=(
    python3 "${script_dir}/run_mmfreellm_continuous_batch_benchmark.py"
    --repo-root "${model_repo}"
    --device cuda --dtype half --request-count 64 --max-batch-size 16
    --max-new-tokens 8 --queue-timeout-ms 2 --interarrival-ms 0
    --warmup-runs 1 --runs 2 --min-aggregate-tps 200
    --result-json "${result_dir}/batch16/result.json"
)
if [[ ${static_test_mode} == 1 ]]; then
    printf '%s\n' "${batch1_override}" >"${result_dir}/commands/batch1.txt"
    printf '%s\n' "${batch16_override}" >"${result_dir}/commands/batch16.txt"
else
    printf '%q ' "${batch1_command[@]}" >"${result_dir}/commands/batch1.txt"
    printf '\n' >>"${result_dir}/commands/batch1.txt"
    printf '%q ' "${batch16_command[@]}" >"${result_dir}/commands/batch16.txt"
    printf '\n' >>"${result_dir}/commands/batch16.txt"
fi

extract_record() {
    local prefix=$1
    local stdout_path=$2
    local json_path=$3
    python3 - "${prefix}" "${stdout_path}" "${json_path}" <<'PY'
import json
import pathlib
import sys

prefix, stdout_name, output_name = sys.argv[1:]
lines = pathlib.Path(stdout_name).read_text(encoding="utf-8").splitlines()
matches = [line[len(prefix):] for line in lines if line.startswith(prefix)]
if not matches:
    raise SystemExit(f"missing canonical JSON prefix: {prefix}")
record = json.loads(matches[-1])
output = pathlib.Path(output_name)
if output.exists():
    existing = json.loads(output.read_text(encoding="utf-8"))
    if existing != record:
        raise SystemExit(f"stdout JSON does not match result file: {output}")
else:
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
PY
}

trap - ERR
set +e
if [[ ${static_test_mode} == 1 ]]; then
    bash -c "${batch1_override}" \
        >"${result_dir}/batch1/stdout.log" \
        2>"${result_dir}/batch1/stderr.log"
else
    "${batch1_command[@]}" \
        >"${result_dir}/batch1/stdout.log" \
        2>"${result_dir}/batch1/stderr.log"
fi
batch1_exit=$?
set -e
trap unexpected_failure ERR
extract_record "MMFREELM_BENCHMARK_JSON=" \
    "${result_dir}/batch1/stdout.log" "${result_dir}/batch1/result.json"
[[ ${batch1_exit} == 0 ]] || evaluation_fail "batch-1 control exited ${batch1_exit}"

python3 - "${result_dir}/batch1/result.json" <<'PY'
import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert record.get("validated") is True, "batch-1 result is not validated"
assert record.get("device") == "cuda", "batch-1 result is not CUDA"
assert record.get("max_new_tokens") == 8, "batch-1 token count is not eight"
prompt = str(record.get("prompt", ""))
output = str(record.get("output", ""))
assert prompt.strip(), "batch-1 prompt is empty"
assert output.startswith(prompt), "batch-1 output does not retain prompt"
token_ids = record.get("generated_token_ids", [])
assert len(token_ids) == 8, "batch-1 IDs are not eight"
assert all(isinstance(token_id, int) and not isinstance(token_id, bool)
           for token_id in token_ids), "batch-1 IDs are not integers"
runs = record.get("runs", [])
assert len(runs) == record.get("total_runs"), "batch-1 run accounting mismatch"
assert all(run.get("generated_tokens") == 8 for run in runs), (
    "batch-1 measured run token count mismatch"
)
assert record.get("total_generated_tokens") == 8 * len(runs), (
    "batch-1 total generated-token accounting mismatch"
)
PY

trap - ERR
set +e
if [[ ${static_test_mode} == 1 ]]; then
    bash -c "${batch16_override}" \
        >"${result_dir}/batch16/stdout.log" \
        2>"${result_dir}/batch16/stderr.log"
else
    "${batch16_command[@]}" \
        >"${result_dir}/batch16/stdout.log" \
        2>"${result_dir}/batch16/stderr.log"
fi
batch16_exit=$?
set -e
trap unexpected_failure ERR
extract_record "MMFREELM_CONTINUOUS_BATCH_JSON=" \
    "${result_dir}/batch16/stdout.log" "${result_dir}/batch16/result.json"
[[ ${batch16_exit} == 0 ]] \
    || evaluation_fail "batch-16 qualification exited ${batch16_exit}"

cross_batch_token_ids_equal=$(python3 - \
    "${result_dir}/batch1/result.json" \
    "${result_dir}/batch16/result.json" <<'PY'
import json
import pathlib
import sys

batch1 = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
record = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
assert record.get("schema") == "matmulfreellm-continuous-batch-benchmark-v1"
assert record.get("validated") is True
assert record.get("qualification_passed") is True
assert record.get("fpga_tps_reported") is False
assert record.get("device") == "cuda"
assert record.get("dtype") == "half"
assert record.get("bitlinear_backend") == "default"
assert record.get("ternip_adapter") == "disabled"
assert record.get("request_count") == 64
assert record.get("max_batch_size") == 16
assert record.get("max_new_tokens") == 8
assert record.get("qualification_runs") == 2
assert record.get("min_aggregate_tps") == 200.0
assert record.get("deterministic_generated_token_ids") is True
runs = record.get("runs", [])
assert len(runs) == 2
token_maps = []
for run in runs:
    assert run.get("requested_requests") == 64
    assert run.get("completed_requests") == 64
    assert run.get("failed_requests") == 0
    assert run.get("total_generated_tokens") == 512
    assert float(run.get("aggregate_tokens_per_second", 0)) >= 200.0
    sizes = run.get("observed_batch_sizes", [])
    assert sizes and max(sizes) <= 16
    requests = run.get("requests", [])
    assert len(requests) == 64
    ids = [request.get("request_id") for request in requests]
    assert sorted(ids) == list(range(64))
    assert all(str(request.get("output", "")).strip() for request in requests)
    assert all(len(request.get("generated_token_ids", [])) == 8 for request in requests)
    token_maps.append({
        request["request_id"]: tuple(request["generated_token_ids"])
        for request in requests
    })
assert token_maps[0] == token_maps[1]
print(str(tuple(batch1["generated_token_ids"]) == token_maps[0][0]).lower())
PY
)

mapfile -d '' model_source_files_after < <(
    cd -- "${model_repo}"
    find mmfreelm -type f -name '*.py' -print0 | sort -z
)
printf '%s\n' "${model_source_files_after[@]}" \
    >"${result_dir}/source/mmfreellm/inventory-after.txt"
cmp \
    "${result_dir}/source/mmfreellm/inventory-before.txt" \
    "${result_dir}/source/mmfreellm/inventory-after.txt" \
    || evaluation_fail "MatMulFreeLM source inventory changed during evaluation"
(
    cd -- "${model_repo}"
    sha256sum "${model_source_files_after[@]}"
) >"${result_dir}/hashes/mmfreellm-runtime-after.sha256"
cmp \
    "${result_dir}/hashes/mmfreellm-runtime-before.sha256" \
    "${result_dir}/hashes/mmfreellm-runtime-after.sha256" \
    || evaluation_fail "MatMulFreeLM source hashes changed during evaluation"
git -C "${model_repo}" status --short \
    >"${result_dir}/environment/mmfreellm-git-status-after.txt"
cmp \
    "${result_dir}/environment/mmfreellm-git-status-before.txt" \
    "${result_dir}/environment/mmfreellm-git-status-after.txt" \
    || evaluation_fail "MatMulFreeLM Git status changed during evaluation"

finalize true 0
