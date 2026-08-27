#!/usr/bin/env bash
set -euo pipefail

oracle=${1:?usage: run_au250_xrt_tq1.sh <upstream-oracle> <proof-dir>}
proof_dir=${2:?usage: run_au250_xrt_tq1.sh <upstream-oracle> <proof-dir>}
xclbin=${HETGPU_XRT_XCLBIN:-/au250_xrt/example/MaxCores_370M.xclbin}
libnvcuda=${HETGPU_TQ1_EVALUATION_LIBRARY:-/qwen-build/hetgpu-target/release/libnvcuda.so}

for required in "${oracle}" "${xclbin}" "${libnvcuda}"; do
    [[ -f "${required}" ]] || { echo "missing TQ1 live-gate input: ${required}" >&2; exit 1; }
done
install -d "${proof_dir}"

xclbin_info="$(xclbinutil --info --input "${xclbin}" 2>&1)"
printf '%s\n' "${xclbin_info}" > "${proof_dir}/xclbin-info.txt"
require_instance_bank() {
    local instance=$1
    local bank=$2
    awk -v instance="${instance}" -v bank="${bank}" '
        $1 == "Instance:" { active = ($2 == instance) }
        active && $1 == "Memory:" && $2 == bank { found = 1 }
        END { exit !found }
    ' <<<"${xclbin_info}" || {
        echo "TQ1 xclbin topology missing ${instance}->${bank}" >&2
        exit 1
    }
}
require_instance_bank ternip_big_1 bank0
require_instance_bank ternip_big_2 bank3
require_instance_bank ternip_big_3 bank2
require_instance_bank ternip_small_1 bank1

export HETGPU_XRT_XCLBIN="${xclbin}"
export HETGPU_XRT_NUM_VECTOR_REGISTERS=4
export HETGPU_XRT_TIMEOUT_MS=10000
export HETGPU_XRT_CLOCK_HZ=300000000
export HETGPU_XRT_CU_CONFIG='{"version":1,"cus":[{"ip_name":"ternip_big:ternip_big_1","memory_group":0,"lanes":9},{"ip_name":"ternip_big:ternip_big_2","memory_group":3,"lanes":9},{"ip_name":"ternip_big:ternip_big_3","memory_group":2,"lanes":9},{"ip_name":"ternip_small:ternip_small_1","memory_group":1,"lanes":6}]}'
export HETGPU_XRT_EXECUTION_LOG="${proof_dir}/xrt.jsonl"
rm -f -- "${proof_dir}/xrt.jsonl"

xbutil examine -d 0000:64:00.1 -r dynamic-regions -r error -r firewall -r thermal \
    > "${proof_dir}/xbutil-before.txt" 2>&1
grep -Fq 'Level 0 : 0x0 (GOOD)' "${proof_dir}/xbutil-before.txt" || {
    echo "AU250 firewall is not GOOD before TQ1 live gate" >&2
    exit 1
}
if grep -Eiq '(^|[^[:alpha:]])fatal([^[:alpha:]]|$)' "${proof_dir}/xbutil-before.txt"; then
    echo "AU250 reported a fatal error before TQ1 live gate" >&2
    exit 1
fi

ORACLE="${oracle}" PROOF_DIR="${proof_dir}" LIBNVCUDA="${libnvcuda}" python3 - <<'PY'
import ctypes
import json
import math
import os
import pathlib
import struct
import subprocess

proof = pathlib.Path(os.environ["PROOF_DIR"])
oracle = os.environ["ORACLE"]
library_path = os.environ["LIBNVCUDA"]
INPUT_HEADER = struct.Struct("<8sIIIIIQQ")
OUTPUT_HEADER = struct.Struct("<8sIIIIIQQ")
Q8_HEADER = struct.Struct("<8sQ")
CAPACITIES = [9, 9, 9, 6]
MATRIX_BYTES = 1024 * 1024 // 4
PROGRAM_BYTES = 6 * 16


class XorShift64:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFFFFFFFFFF
        value ^= value >> 7
        value ^= (value << 17) & 0xFFFFFFFFFFFFFFFF
        self.state = value & 0xFFFFFFFFFFFFFFFF
        return self.state


def make_fixture(path, k, rows, tokens, experts, seed):
    rng = XorShift64(seed)
    blocks = bytearray()
    for _ in range(experts * rows * (k // 256)):
        blocks.extend((rng.next() & 0xFF) for _ in range(52))
        scale = 0.015625 * (1 + rng.next() % 31)
        blocks.extend(struct.pack("<e", scale))
    activations = []
    for _ in range(tokens * experts * k):
        activations.append(((rng.next() % 8191) - 4095) / 2048.0)
    with path.open("wb") as stream:
        stream.write(INPUT_HEADER.pack(
            b"TQ1FX1\0\0", 1, k, rows, tokens, experts, len(blocks), len(activations)
        ))
        stream.write(blocks)
        stream.write(struct.pack(f"<{len(activations)}f", *activations))
    return bytes(blocks), activations


def read_reference(path, expected):
    payload = path.read_bytes()
    if len(payload) < OUTPUT_HEADER.size:
        raise RuntimeError("upstream reference is truncated")
    magic, version, k, rows, tokens, experts, output_count, q8_count = OUTPUT_HEADER.unpack_from(payload)
    if magic != b"TQ1RF1\0\0" or version != 1 or (k, rows, tokens, experts) != expected:
        raise RuntimeError("upstream reference header mismatch")
    cursor = OUTPUT_HEADER.size
    output_bytes = output_count * 4
    if len(payload) != cursor + output_bytes + q8_count * 260:
        raise RuntimeError("upstream reference extent mismatch")
    outputs = list(struct.unpack_from(f"<{output_count}f", payload, cursor))
    return outputs, payload[cursor + output_bytes:], q8_count


def expected_jobs(tile_count):
    cursor = 0
    per_cu = [0, 0, 0, 0]
    jobs = []
    for _ in range(tile_count):
        remaining = 8
        while remaining:
            cu = cursor % len(CAPACITIES)
            cursor += 1
            assigned = min(CAPACITIES[cu], remaining)
            remaining -= assigned
            per_cu[cu] += 1
            jobs.append(cu)
    return jobs, per_cu


library = ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
evaluate = library.hetgpu_tq1_evaluate_raw_v1
evaluate.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
]
evaluate.restype = ctypes.c_int

cases = [
    ("single_tile", 1024, 1024, 1, 1, 0xA2500001),
    ("tiled", 2048, 2048, 4, 10, 0xA2500002),
]
summary = {"schema_version": 1, "status": "pass", "cases": {}}
for name, k, rows, tokens, experts, seed in cases:
    fixture_path = proof / f"{name}.fixture.bin"
    reference_path = proof / f"{name}.reference.bin"
    output_path = proof / f"{name}.xrt-output.bin"
    q8_path = proof / f"{name}.rust-q8.bin"
    evidence_path = proof / f"{name}.evidence.jsonl"
    blocks, activations = make_fixture(fixture_path, k, rows, tokens, experts, seed)
    subprocess.run([oracle, str(fixture_path), str(reference_path)], check=True)
    reference, reference_q8, q8_count = read_reference(
        reference_path, (k, rows, tokens, experts)
    )
    block_array = (ctypes.c_uint8 * len(blocks)).from_buffer_copy(blocks)
    activation_array = (ctypes.c_float * len(activations))(*activations)
    output_array = (ctypes.c_float * len(reference))()
    os.environ["HETGPU_TQ1_EVIDENCE_LOG"] = str(evidence_path)
    os.environ["HETGPU_TQ1_Q8_DUMP"] = str(q8_path)
    evidence_path.unlink(missing_ok=True)
    q8_path.unlink(missing_ok=True)
    result = evaluate(
        block_array, len(blocks), activation_array, len(activations),
        k, rows, tokens, experts, output_array, len(output_array),
    )
    if result != 1:
        raise RuntimeError(f"{name}: Rust TQ1 evaluation returned {result}")
    actual = list(output_array)
    output_path.write_bytes(struct.pack(f"<{len(actual)}f", *actual))
    max_error = 0.0
    for index, (got, expected_value) in enumerate(zip(actual, reference)):
        error = abs(got - expected_value)
        tolerance = 1e-4 + 1e-5 * abs(expected_value)
        if not math.isfinite(got) or error > tolerance:
            raise RuntimeError(
                f"{name}: output {index} got {got}, expected {expected_value}, "
                f"error {error}, tolerance {tolerance}"
            )
        max_error = max(max_error, error)
    q8_payload = q8_path.read_bytes()
    magic, rust_q8_count = Q8_HEADER.unpack_from(q8_payload)
    if magic != b"TQ1Q8K1\0" or rust_q8_count != q8_count:
        raise RuntimeError(f"{name}: Rust Q8_K header mismatch")
    if q8_payload[Q8_HEADER.size:] != reference_q8:
        raise RuntimeError(f"{name}: Rust and upstream Q8_K scales/quants differ")
    records = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    if len(records) != 1 or records[0].get("route") != "handled":
        raise RuntimeError(f"{name}: expected one handled evidence record")
    evidence = records[0]["evidence"]
    tile_count = tokens * experts * math.ceil(rows / 1024) * (k // 1024)
    jobs, per_cu = expected_jobs(tile_count)
    expected_matrix = len(jobs) * MATRIX_BYTES
    expected_io = sum(CAPACITIES[cu] * 1024 * 2 for cu in jobs)
    expected_program = len(jobs) * PROGRAM_BYTES
    required = {
        "backend": "xrt-tq1-v1",
        "eligible_operations": 1,
        "handled_operations": 1,
        "submission_count": len(jobs),
        "completion_count": len(jobs),
        "per_cu_submissions": per_cu,
        "per_cu_completions": per_cu,
        "matrix_bytes": expected_matrix,
        "input_bytes": expected_io,
        "output_bytes": expected_io,
        "program_bytes": expected_program,
        "clock_hz": 300000000,
    }
    for key, expected_value in required.items():
        if evidence.get(key) != expected_value:
            raise RuntimeError(
                f"{name}: evidence {key}={evidence.get(key)!r}, expected {expected_value!r}"
            )
    if evidence.get("stall_codes") != [1] * len(jobs):
        raise RuntimeError(f"{name}: non-terminal STALL evidence")
    if not (-16384 <= evidence.get("raw_min", -16385) <= evidence.get("raw_max", 16385) <= 16384):
        raise RuntimeError(f"{name}: raw bound evidence is invalid")
    for timing in ("dispatch_to_stall_ns", "derived_accelerator_cycles", "decode_ns", "pack_ns", "xrt_ns", "reconstruct_ns"):
        if not isinstance(evidence.get(timing), int) or evidence[timing] <= 0:
            raise RuntimeError(f"{name}: missing positive {timing}")
    if name == "tiled" and not all(count > 0 for count in per_cu):
        raise RuntimeError("tiled: all four CUs were not active")
    summary["cases"][name] = {
        "status": "pass",
        "dimensions": {"k": k, "rows": rows, "tokens": tokens, "experts": experts},
        "outputs": len(actual),
        "q8_blocks": q8_count,
        "max_absolute_error": max_error,
        "per_cu_completions": per_cu,
        "evidence": evidence,
    }

temporary = proof / "summary.json.partial"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
temporary.replace(proof / "summary.json")
print(json.dumps(summary, sort_keys=True))
PY

xbutil examine -d 0000:64:00.1 -r dynamic-regions -r error -r firewall -r thermal \
    > "${proof_dir}/xbutil-after.txt" 2>&1
grep -Fq 'Level 0 : 0x0 (GOOD)' "${proof_dir}/xbutil-after.txt" || {
    echo "AU250 firewall is not GOOD after TQ1 live gate" >&2
    exit 1
}
if grep -Eiq '(^|[^[:alpha:]])fatal([^[:alpha:]]|$)' "${proof_dir}/xbutil-after.txt"; then
    echo "AU250 reported a fatal error after TQ1 live gate" >&2
    exit 1
fi
for cu in ternip_big_1 ternip_big_2 ternip_big_3 ternip_small_1; do
    grep -Eq "${cu}.*\(DONE\)" "${proof_dir}/xbutil-after.txt" || {
        echo "AU250 CU ${cu} is not DONE after TQ1 live gate" >&2
        exit 1
    }
done
echo "PASS: upstream-linked single-tile and tiled TQ1 AU250 qualification"
