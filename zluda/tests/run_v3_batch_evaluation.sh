#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

usage() {
    echo "usage: $0 NEW_RESULT_DIRECTORY" >&2
}

fail() {
    echo "run_v3_batch_evaluation: $*" >&2
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
[[ ${result_name} != "." && ${result_name} != ".." && -n ${result_name} ]] \
    || fail "unsafe result directory name"
[[ -d ${result_parent} ]] || fail "result directory parent does not exist: ${result_parent}"
result_parent=$(cd -- "${result_parent}" && pwd -P)
result_dir=${result_parent}/${result_name}

case ${result_dir} in
    /|"${repo_root}"|"${script_dir}") fail "refusing unsafe result directory: ${result_dir}" ;;
esac
[[ ! -e ${result_dir} && ! -L ${result_dir} ]] \
    || fail "result directory already exists; refusing to overwrite: ${result_dir}"

run_live=${HETGPU_RUN_LIVE_V3_BATCH:-0}
run_live_workload=${HETGPU_RUN_LIVE_WORKLOAD_EVAL:-0}
static_test_mode=${HETGPU_EVAL_STATIC_TEST_MODE:-0}
[[ ${run_live} == 0 || ${run_live} == 1 ]] \
    || fail "HETGPU_RUN_LIVE_V3_BATCH must be 0 or 1"
[[ ${run_live_workload} == 0 || ${run_live_workload} == 1 ]] \
    || fail "HETGPU_RUN_LIVE_WORKLOAD_EVAL must be 0 or 1"
[[ ${static_test_mode} == 0 || ${static_test_mode} == 1 ]] \
    || fail "HETGPU_EVAL_STATIC_TEST_MODE must be 0 or 1"
extra_source_file=${HETGPU_EVAL_EXTRA_SOURCE_FILE:-}
if [[ -n ${extra_source_file} ]]; then
    [[ ${static_test_mode} == 1 ]] \
        || fail "HETGPU_EVAL_EXTRA_SOURCE_FILE is restricted to static harness tests"
    extra_source_file=$(realpath -e -- "${extra_source_file}") \
        || fail "static extra source file does not exist"
    case ${extra_source_file} in
        /tmp/*) ;;
        *) fail "static extra source file must be below /tmp" ;;
    esac
    [[ -f ${extra_source_file} && ! -L ${extra_source_file} ]] \
        || fail "static extra source input must be a regular non-symlink file"
fi

baseline_command=${HETGPU_EVAL_BASELINE_CMD:-}
enabled_command=${HETGPU_EVAL_ENABLED_CMD:-}
if [[ -n ${baseline_command} && -z ${enabled_command} ]] \
    || [[ -z ${baseline_command} && -n ${enabled_command} ]]; then
    fail "HETGPU_EVAL_BASELINE_CMD and HETGPU_EVAL_ENABLED_CMD must be supplied together"
fi
workloads_selected=0
if [[ -n ${baseline_command} ]]; then
    workloads_selected=1
fi
workload_timing_kind=${HETGPU_EVAL_WORKLOAD_KIND:-auto}
case ${workload_timing_kind} in
    auto|kimi|matmulfreellm) ;;
    *) fail "HETGPU_EVAL_WORKLOAD_KIND must be auto, kimi, or matmulfreellm" ;;
esac
fixture_batch_limit=${HETGPU_V3_FIXTURE_BATCH_LIMIT:-2}
[[ ${fixture_batch_limit} == 2 ]] \
    || fail "HETGPU_V3_FIXTURE_BATCH_LIMIT must be 2 to prove two batch-2 slices"
if [[ ${run_live_workload} == 1 ]]; then
    [[ ${run_live} == 1 ]] \
        || fail "live workload evaluation requires HETGPU_RUN_LIVE_V3_BATCH=1"
    [[ ${workloads_selected} == 1 ]] \
        || fail "live workload evaluation requires both baseline and enabled commands"
fi

control_device=${HETGPU_CXL_TMATMUL_DEVICE:-/dev/cxl_tmatmul3b001}
dax_device=${HETGPU_CXL_TMATMUL_DAX:-/dev/dax6.0}
devdax_validator=${HETGPU_CXL_DEVDAX_VALIDATOR:-/root/.config/superpowers/worktrees/ternary_matmul/kernel7-real-cxl-mem-20260825/synth/intel_ia780i/sw/validate_cxl_devdax_binding.py}
base_dpa=${HETGPU_CXL_TMATMUL_V3_BASE_DPA:-0x01000000}
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_sha=$(git -C "${repo_root}" rev-parse HEAD)
git_branch=$(git -C "${repo_root}" branch --show-current)

mkdir -m 700 -- "${result_dir}"
mkdir -m 700 -- "${result_dir}/commands" "${result_dir}/logs" \
    "${result_dir}/status" "${result_dir}/timing" "${result_dir}/test-counts" \
    "${result_dir}/patches" "${result_dir}/source" "${result_dir}/hashes"
mkdir -m 700 -- "${result_dir}/source/files"
[[ -f ${devdax_validator} && ! -L ${devdax_validator} ]] \
    || fail "CXL devdax validator is missing or not a regular file: ${devdax_validator}"
devdax_validator=$(realpath -e -- "${devdax_validator}")
git -C "${repo_root}" status --short >"${result_dir}/git-status.txt"
printf '%s\n' "${git_sha}" >"${result_dir}/evaluated-git-head.txt"

evaluation_files=(
    Cargo.lock
    docs/superpowers/plans/2026-08-20-fpga-v3-batch-scheduler-implementation.md
    docs/superpowers/specs/2026-08-20-fpga-v3-batch-scheduler-design.md
    ext/nvidia_runtime-sys/src/lib.rs
    zluda/Cargo.toml
    zluda/build.rs
    zluda/src/cublas_shim.c
    zluda/src/cudart_dax_pool.c
    zluda/src/cudart_dax_pool.h
    zluda/src/cudart_shim.c
    zluda/src/lib.rs
    zluda/src/impl/mod.rs
    zluda/src/impl/batch_scheduler.rs
    zluda/src/impl/cxl_tmatmul_v3.rs
    zluda/src/impl/driver.rs
    zluda/src/impl/iq1s_tmatmul.rs
    zluda/src/impl/function.rs
    zluda/test_batch_scheduler.rs
    zluda/tests/batch_scheduler_integration_test.rs
    zluda/tests/cudart_dax_pool_test.c
    zluda/tests/cudart_dax_preload_test.c
    zluda/tests/run_v3_batch_evaluation.sh
    zluda/tests/parse_v3_workload_timing.py
    zluda/tests/test_parse_v3_workload_timing.py
    zluda/tests/run_mmfreellm_2p7b_benchmark.py
    zluda/tests/test_run_mmfreellm_2p7b_benchmark.py
    zluda/tests/validate_v3_workload_completion.py
    zluda/tests/test_validate_v3_workload_completion.py
    zluda/tests/test_run_v3_batch_evaluation_static.sh
    zluda/tests/eval-fixtures/bin/cargo
)
declare -A evaluation_seen=()
for relative_path in "${evaluation_files[@]}"; do
    evaluation_seen["${relative_path}"]=1
done
build_input_pathspecs=(
    Cargo.toml
    Cargo.lock
    ext/nvidia_runtime-sys
    zluda/Cargo.toml
    zluda/build.rs
    zluda/src
)
dirty_build_inputs=()
while IFS= read -r -d '' relative_path; do
    dirty_build_inputs+=("${relative_path}")
    if [[ -z ${evaluation_seen["${relative_path}"]+present} ]]; then
        evaluation_files+=("${relative_path}")
        evaluation_seen["${relative_path}"]=1
    fi
done < <(
    {
        git -C "${repo_root}" diff --name-only -z -- "${build_input_pathspecs[@]}"
        git -C "${repo_root}" diff --cached --name-only -z -- "${build_input_pathspecs[@]}"
        git -C "${repo_root}" ls-files --others --exclude-standard -z -- \
            "${build_input_pathspecs[@]}"
    } | sort -zu
)
printf '%s\n' "${dirty_build_inputs[@]}" \
    >"${result_dir}/source/dirty-build-inputs.txt"
git -C "${repo_root}" diff --cached -- "${evaluation_files[@]}" \
    >"${result_dir}/patches/evaluation-relevant.staged.patch"
git -C "${repo_root}" diff -- "${evaluation_files[@]}" \
    >"${result_dir}/patches/evaluation-relevant.unstaged.patch"
git -C "${repo_root}" status --short --ignored -- "${evaluation_files[@]}" \
    >"${result_dir}/source/evaluation-relevant-git-status.txt"
for relative_path in "${evaluation_files[@]}"; do
    if ! git -C "${repo_root}" ls-files --error-unmatch -- "${relative_path}" >/dev/null 2>&1; then
        classification=untracked
        if git -C "${repo_root}" check-ignore -q -- "${relative_path}"; then
            classification=ignored-untracked
        fi
        printf '%s\t%s\n' "${classification}" "${relative_path}"
    fi
done >"${result_dir}/source/evaluation-relevant-untracked.txt"
for relative_path in "${evaluation_files[@]}"; do
    if [[ -f ${repo_root}/${relative_path} ]]; then
        mkdir -p -- "${result_dir}/source/files/$(dirname -- "${relative_path}")"
        cp -- "${repo_root}/${relative_path}" "${result_dir}/source/files/${relative_path}"
    fi
done
mkdir -p -- "${result_dir}/source/files/external"
cp -- "${devdax_validator}" \
    "${result_dir}/source/files/external/validate_cxl_devdax_binding.py"
source_hash_files=("${evaluation_files[@]}" "${devdax_validator}")
if [[ -n ${extra_source_file} ]]; then
    source_hash_files+=("${extra_source_file}")
fi
printf '%s\n' "${source_hash_files[@]}" \
    >"${result_dir}/source/evaluation-source-inventory.txt"
source_hash_python='
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2])
rows = []
for listed_path in sys.argv[3:]:
    listed = pathlib.Path(listed_path)
    source = listed if listed.is_absolute() else root / listed
    if not source.is_file():
        raise SystemExit(f"evaluation source is missing or not a regular file: {listed_path}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    rows.append(f"{digest.hexdigest()}  {listed_path}\n")
with output.open("x", encoding="utf-8") as stream:
    stream.writelines(rows)
'
python3 -c "${source_hash_python}" "${repo_root}" \
    "${result_dir}/hashes/source.sha256" "${source_hash_files[@]}"

python3 -c '
import json, sys
path, started, repo, sha, branch, status_path, live, live_workload, commands_selected, control, dax, validator, base, batch_limit, fixture_limit, timing_kind, baseline, enabled = sys.argv[1:]
manifest = {
    "schema": "hetgpu-v3-batch-evaluation-v1",
    "started_utc": started,
    "repo_root": repo,
    "git_sha": sha,
    "git_branch": branch,
    "git_status_artifact": status_path,
    "configuration": {
        "run_live_v3_batch": live == "1",
        "run_live_workload_evaluation": live_workload == "1",
        "workload_commands_selected": commands_selected == "1",
        "control_device": control,
        "dax_device": dax,
        "devdax_validator": validator,
        "base_dpa": base,
        "configured_batch_limit": batch_limit or None,
        "live_fixture_batch_limit": int(fixture_limit),
        "workload_timing_kind": timing_kind,
        "baseline_command": baseline or None,
        "enabled_command": enabled or None,
    },
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
' "${result_dir}/manifest.json" "${started_utc}" "${repo_root}" "${git_sha}" \
    "${git_branch}" "git-status.txt" "${run_live}" "${run_live_workload}" \
    "${workloads_selected}" "${control_device}" "${dax_device}" \
    "${devdax_validator}" "${base_dpa}" \
    "${HETGPU_FPGA_BATCH_LIMIT:-}" "${fixture_batch_limit}" "${workload_timing_kind}" \
    "${baseline_command}" "${enabled_command}"

run_gate() {
    local gate=$1
    shift
    local command_file=${result_dir}/commands/${gate}.txt
    local stdout_file=${result_dir}/logs/${gate}.stdout.log
    local stderr_file=${result_dir}/logs/${gate}.stderr.log
    local status_file=${result_dir}/status/${gate}.status
    local timing_file=${result_dir}/timing/${gate}.json
    local started_ns ended_ns exit_status

    {
        printf '%q ' "$@"
        printf '\n'
    } >"${command_file}"
    echo "[v3-eval] ${gate}" >&2
    started_ns=$(date +%s%N)
    set +e
    "$@" >"${stdout_file}" 2>"${stderr_file}"
    exit_status=$?
    set -e
    ended_ns=$(date +%s%N)
    printf '%s\n' "${exit_status}" >"${status_file}"
    python3 -c '
import json, sys
path, gate, start, end, status = sys.argv[1:]
start, end, status = int(start), int(end), int(status)
with open(path, "x", encoding="utf-8") as stream:
    json.dump({
        "gate": gate,
        "started_unix_ns": start,
        "ended_unix_ns": end,
        "elapsed_seconds": (end - start) / 1_000_000_000,
        "exit_status": status,
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
' "${timing_file}" "${gate}" "${started_ns}" "${ended_ns}" "${exit_status}"
    if [[ ${exit_status} -ne 0 ]]; then
        echo "[v3-eval] ${gate} failed with status ${exit_status}; see ${stderr_file}" >&2
        return "${exit_status}"
    fi
}

assert_rust_test_count_python='
import json, pathlib, re, sys
log_path, output_path, gate, minimum = sys.argv[1:]
minimum = int(minimum)
text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
counts = [
    int(match.group(1))
    for match in re.finditer(r"test result:\s+ok\.\s+([0-9]+) passed;", text)
]
passed = sum(counts)
if not counts:
    raise SystemExit(f"{gate} produced no parseable Rust test result")
if passed < minimum:
    raise SystemExit(f"{gate} passed {passed} tests, expected at least {minimum}; zero tests are not accepted")
with open(output_path, "x", encoding="utf-8") as stream:
    json.dump({"gate": gate, "passed": passed, "minimum": minimum, "validated": True}, stream, indent=2, sort_keys=True)
    stream.write("\n")
'

run_rust_test_gate() {
    local gate=$1
    local minimum=$2
    shift 2
    run_gate "${gate}" "$@"
    run_gate "${gate}_test_count" python3 -c "${assert_rust_test_count_python}" \
        "${result_dir}/logs/${gate}.stdout.log" \
        "${result_dir}/test-counts/${gate}.json" "${gate}" "${minimum}"
}

cd -- "${repo_root}"

run_rust_test_gate integration_contract 3 env CARGO_INCREMENTAL=0 \
    cargo test -p zluda --no-default-features --features nvidia,evaluation \
    --test batch_scheduler_integration_test -- --nocapture
run_rust_test_gate planner 13 env CARGO_INCREMENTAL=0 \
    cargo test -p zluda --no-default-features --features nvidia \
    batch_scheduler -- --nocapture
run_rust_test_gate iq1s 25 env -u HETGPU_RUN_LIVE_V3_BATCH CARGO_INCREMENTAL=0 \
    cargo test -p zluda --no-default-features --features nvidia \
    iq1s_tmatmul -- --nocapture
run_rust_test_gate cxl_v3 31 env CARGO_INCREMENTAL=0 \
    cargo test -p zluda --no-default-features --features nvidia \
    cxl_tmatmul_v3 -- --nocapture
run_rust_test_gate nvidia_completion 4 env CARGO_INCREMENTAL=0 \
    cargo test -p zluda --no-default-features --features nvidia \
    nvidia_iq1s_completed -- --nocapture
run_gate workload_timing_parser_tests python3 \
    zluda/tests/test_parse_v3_workload_timing.py
run_gate mmfreellm_benchmark_tests python3 \
    zluda/tests/test_run_mmfreellm_2p7b_benchmark.py
run_gate workload_completion_validator_tests python3 \
    zluda/tests/test_validate_v3_workload_completion.py
run_gate cargo_check env CARGO_INCREMENTAL=0 \
    cargo check -p zluda --no-default-features --features nvidia,evaluation

if [[ ${run_live} == 1 ]]; then
    run_gate live_device_nodes bash -c "
        [[ -c \$1 ]] || { echo \"not a character device: \$1\" >&2; exit 3; }
        [[ -c \$2 ]] || { echo \"not a character device: \$2\" >&2; exit 3; }
    " _ "${control_device}" "${dax_device}"

    run_gate cxl_devdax_binding python3 "${devdax_validator}" \
        --control "${control_device}" --dax "${dax_device}" \
        --negative-fd --output "${result_dir}/cxl-devdax-binding.json"

    query_caps_python='
import datetime
import fcntl
import json
import os
import struct
import sys

control_device, dax_device, output_path = sys.argv[1:]
query_caps_v3 = 0xC080CE10
bind_dax_v3 = 0xC048CE16
unbind_dax_v3 = 0x4040CE17
caps_layout = struct.Struct("<IIQ8I3Q7Q")
bind_layout = struct.Struct("<IIiI3Q4Q")
unbind_layout = struct.Struct("<IIQ6Q")
expected_hpa = 0x0C1000000000
expected_bytes = 0x800000000

def query(fd):
    buffer = bytearray(caps_layout.pack(caps_layout.size, *([0] * 20)))
    fcntl.ioctl(fd, query_caps_v3, buffer, True)
    return caps_layout.unpack(buffer)

def validate(values, phase):
    expected_caps = 0xFA if phase == "pre_bind" else 0xFB
    expected_dax = 0 if phase == "pre_bind" else expected_bytes
    if values[0] != caps_layout.size or values[1] != 3:
        raise SystemExit(f"{phase} QUERY_CAPS_V3 returned invalid size/version")
    if values[2] != expected_caps:
        raise SystemExit(f"{phase} capabilities {values[2]:#x} != {expected_caps:#x}")
    if values[3] != 4 or values[12] != 0x0F:
        raise SystemExit(f"{phase} does not expose exactly lanes 0 through 3")
    if values[11] != expected_dax:
        raise SystemExit(f"{phase} DAX bytes {values[11]:#x} != {expected_dax:#x}")
    if values[5] < 2:
        raise SystemExit(f"{phase} max_batch is incompatible with batch-2 slices")
    return {
        "phase": phase,
        "size": values[0],
        "version": values[1],
        "capabilities": values[2],
        "num_instances": values[3],
        "dim_d": values[4],
        "max_batch": values[5],
        "max_descriptors": values[6],
        "max_inflight_submissions": values[7],
        "max_timeout_ms": values[8],
        "ddr_data_width_bits": values[9],
        "dax_alignment_bytes": values[10],
        "dax_bytes": values[11],
        "per_lane_counter_mask": values[12],
        "accelerator_clock_hz": values[13],
        "reserved": list(values[14:]),
    }

control_fd = os.open(control_device, os.O_RDWR | os.O_CLOEXEC)
dax_fd = -1
generation = 0
try:
    pre = validate(query(control_fd), "pre_bind")
    dax_fd = os.open(dax_device, os.O_RDWR | os.O_SYNC | os.O_CLOEXEC)
    bind = bytearray(
        bind_layout.pack(bind_layout.size, 0, dax_fd, 0, *([0] * 7))
    )
    fcntl.ioctl(control_fd, bind_dax_v3, bind, True)
    bound = bind_layout.unpack(bind)
    hpa, length, generation = bound[4], bound[5], bound[6]
    if hpa != expected_hpa or length != expected_bytes or generation == 0:
        raise SystemExit(
            f"BIND_DAX returned hpa={hpa:#x} bytes={length:#x} generation={generation}"
        )
    post = validate(query(control_fd), "post_bind")
finally:
    if generation:
        unbind = bytearray(
            unbind_layout.pack(unbind_layout.size, 0, generation, *([0] * 6))
        )
        fcntl.ioctl(control_fd, unbind_dax_v3, unbind, True)
    if dax_fd >= 0:
        os.close(dax_fd)
    os.close(control_fd)

caps = {
    "schema": "hetgpu-v3-bound-caps/v1",
    "validated": True,
    "control_device": control_device,
    "dax_device": dax_device,
    "query_caps_v3_ioctl": query_caps_v3,
    "query_caps_v3_ioctl_hex": hex(query_caps_v3),
    "bind_dax_v3_ioctl": bind_dax_v3,
    "unbind_dax_v3_ioctl": unbind_dax_v3,
    "bound_hpa": hpa,
    "bound_bytes": length,
    "generation": generation,
    "pre_bind": pre,
    "post_bind": post,
    "queried_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open(output_path, "x", encoding="utf-8") as stream:
    json.dump(caps, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(caps, sort_keys=True))
'
    run_gate query_caps_v3 python3 -c "${query_caps_python}" \
        "${control_device}" "${dax_device}" "${result_dir}/capabilities-v3.json"
    run_rust_test_gate live_batch_fixture 1 env CARGO_INCREMENTAL=0 \
        HETGPU_RUN_LIVE_V3_BATCH=1 \
        HETGPU_FPGA_BATCH_LIMIT="${fixture_batch_limit}" \
        HETGPU_CXL_TMATMUL_DEVICE="${control_device}" \
        HETGPU_CXL_TMATMUL_DAX="${dax_device}" \
        HETGPU_CXL_TMATMUL_V3_BASE_DPA="${base_dpa}" \
        cargo test -p zluda --no-default-features --features nvidia \
        live_v3_host_captured_batch_four_fixture -- --nocapture

    extract_live_completion_python='
import json, pathlib, sys
output, *logs = sys.argv[1:]
records = []
for log in logs:
    for raw in pathlib.Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("event") == "hetgpu_v3_live_batch_fixture_completed":
            records.append(record)
if len(records) != 1 or records[0].get("validated") is not True:
    raise SystemExit(f"expected one validated live fixture completion record, found {len(records)}")
with open(output, "x", encoding="utf-8") as stream:
    stream.write(json.dumps(records[0], sort_keys=True) + "\n")
'
    run_gate extract_live_completion python3 -c "${extract_live_completion_python}" \
        "${result_dir}/live-completion-evidence.jsonl" \
        "${result_dir}/logs/live_batch_fixture.stdout.log" \
        "${result_dir}/logs/live_batch_fixture.stderr.log"
fi

workloads_executed=0
if [[ ${run_live_workload} == 1 ]]; then
    run_gate workload_baseline bash -lc "${baseline_command}"
    run_gate workload_enabled bash -lc "${enabled_command}"
    run_gate parse_workload_baseline_timing python3 \
        zluda/tests/parse_v3_workload_timing.py \
        --label baseline --kind "${workload_timing_kind}" \
        --stdout "${result_dir}/logs/workload_baseline.stdout.log" \
        --stderr "${result_dir}/logs/workload_baseline.stderr.log" \
        --output "${result_dir}/timing/workload-baseline.json"
    run_gate parse_workload_enabled_timing python3 \
        zluda/tests/parse_v3_workload_timing.py \
        --label enabled --kind "${workload_timing_kind}" \
        --stdout "${result_dir}/logs/workload_enabled.stdout.log" \
        --stderr "${result_dir}/logs/workload_enabled.stderr.log" \
        --output "${result_dir}/timing/workload-enabled.json"

    run_gate extract_workload_completion python3 \
        zluda/tests/validate_v3_workload_completion.py \
        --output "${result_dir}/workload-completion-evidence.jsonl" \
        "${result_dir}/logs/workload_enabled.stdout.log" \
        "${result_dir}/logs/workload_enabled.stderr.log"
    workloads_executed=1
fi

built_artifacts=()
while IFS= read -r -d '' built_path; do
    built_artifacts+=("${built_path}")
done < <(
    find target/debug -maxdepth 2 -type f \
        \( -name 'libnvcuda.so' -o -name 'libnvcuda-*.rlib' -o \
           -name 'libnvcuda-*.so' -o -name 'batch_scheduler_integration_test-*' -o \
           -name 'nvcuda-*' \) -print0 | sort -z
)
if [[ ${#built_artifacts[@]} -gt 0 ]]; then
    sha256sum -- "${built_artifacts[@]}" >"${result_dir}/hashes/built-artifacts.sha256"
else
    : >"${result_dir}/hashes/built-artifacts.sha256"
fi

run_gate source_tree_final_hash python3 -c "${source_hash_python}" \
    "${repo_root}" "${result_dir}/hashes/source.final.sha256" \
    "${source_hash_files[@]}"
run_gate source_tree_stability cmp -- \
    "${result_dir}/hashes/source.sha256" \
    "${result_dir}/hashes/source.final.sha256"

completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 -c '
import json, pathlib, sys
path, completed, live_fixture, commands_selected, workload_requested, workloads_executed = sys.argv[1:]
root = pathlib.Path(path).parent
statuses = {}
for status_path in sorted((root / "status").glob("*.status")):
    statuses[status_path.stem] = int(status_path.read_text(encoding="utf-8").strip())
if any(status != 0 for status in statuses.values()):
    raise SystemExit("refusing success summary because a selected gate failed")
summary = {
    "schema": "hetgpu-v3-batch-evaluation-summary-v1",
    "status": "completed",
    "completed_utc": completed,
    "software_gates_passed": True,
    "live_fixture_executed": live_fixture == "1",
    "workload_commands_selected": commands_selected == "1",
    "live_workload_evaluation_requested": workload_requested == "1",
    "workloads_executed": workloads_executed == "1",
    "gate_statuses": statuses,
    "tps_claimed": False,
}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(summary, stream, indent=2, sort_keys=True)
    stream.write("\n")
' "${result_dir}/summary.json" "${completed_utc}" "${run_live}" \
    "${workloads_selected}" "${run_live_workload}" "${workloads_executed}"
printf 'completed\n' >"${result_dir}/STATUS"
(
    cd -- "${result_dir}"
    find . -type f ! -path './hashes/final-artifacts.sha256' \
        ! -path './source/artifact-inventory.txt' -print \
        | sort
) >"${result_dir}/source/artifact-inventory.txt"
(
    cd -- "${result_dir}"
    find . -type f ! -path './hashes/final-artifacts.sha256' -print0 \
        | sort -z | xargs -0 sha256sum
) >"${result_dir}/hashes/final-artifacts.sha256"
echo "[v3-eval] completed: ${result_dir}" >&2
