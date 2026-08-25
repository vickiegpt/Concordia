#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


REQUIRED_SUMMARY_KEYS = {
    "model_sha256",
    "xclbin_sha256",
    "libnvcuda_sha256",
    "runner_sha256",
    "exit_code",
    "generated_token_ids",
    "prompt_tokens_per_second",
    "generation_tokens_per_second",
    "gpu_name",
    "fpga_bdf",
    "firewall_status",
    "fatal_errors",
    "fpga_temperature_c",
}
HASH_KEYS = {"model_sha256", "xclbin_sha256", "libnvcuda_sha256", "runner_sha256"}
APPROVED_CUS = (
    "ternip_big:ternip_big_1",
    "ternip_big:ternip_big_2",
    "ternip_big:ternip_big_3",
    "ternip_small:ternip_small_1",
)
ATTENTION_MARKERS = ("attention", "attn", "flash", "softmax", "qkv", "query", "key", "value", "kq", "qk")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProofError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ProofError(message)


def load_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProofError(f"unable to read JSON {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def load_jsonl(path):
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ProofError(f"unable to read JSONL {path}: {error}") from error
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProofError(f"invalid JSONL {path}:{line_number}: {error}") from error
        require(isinstance(record, dict), f"{path}:{line_number} must be a JSON object")
        records.append(record)
    return records


def is_iq1s_matmul(kernel):
    name = str(kernel).lower()
    return "ggml_type19" in name and ("mul_mat_q" in name or "mul_mat_vec_q" in name)


def is_attention(kernel):
    name = str(kernel).lower()
    return any(marker in name for marker in ATTENTION_MARKERS)


def validate(proof_dir):
    proof_dir = Path(proof_dir)
    require(proof_dir.is_dir(), f"proof directory does not exist: {proof_dir}")
    summary = load_json(proof_dir / "summary.json")
    missing = sorted(REQUIRED_SUMMARY_KEYS - summary.keys())
    require(not missing, f"summary.json is missing keys: {', '.join(missing)}")
    for key in HASH_KEYS:
        require(isinstance(summary[key], str) and SHA256_RE.fullmatch(summary[key]), f"{key} is not a SHA-256")
    require(summary["exit_code"] == 0, f"runner exit code is {summary['exit_code']}")
    tokens = summary["generated_token_ids"]
    require(isinstance(tokens, list) and tokens, "generated_token_ids must be nonempty")
    require(all(isinstance(token, int) and not isinstance(token, bool) for token in tokens), "generated_token_ids must contain only integers")
    for key in ("prompt_tokens_per_second", "generation_tokens_per_second"):
        value = summary[key]
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0, f"{key} must be positive")
    require(isinstance(summary["gpu_name"], str) and summary["gpu_name"].strip(), "gpu_name is empty")
    require(re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", str(summary["fpga_bdf"])), "fpga_bdf is invalid")
    require(summary["firewall_status"] == "GOOD", "FPGA firewall is not GOOD")
    require(summary["fatal_errors"] == [], "proof contains fatal errors")
    temperature = summary["fpga_temperature_c"]
    require(isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and temperature < 85, "FPGA temperature is not below 85 C")

    routes = load_jsonl(proof_dir / "routes.jsonl")
    selected_iq1s = [record for record in routes if is_iq1s_matmul(record.get("kernel"))]
    require(selected_iq1s, "no qualified IQ1_S BitLinear route was observed")
    for record in selected_iq1s:
        require(record.get("route") == "cxl_tmatmul", f"IQ1_S route fell through: {record}")
        require(record.get("backend") == "xrt", f"IQ1_S route did not select XRT: {record}")
        require(record.get("xrt_enabled") is True, f"IQ1_S XRT route was not enabled: {record}")
        require(record.get("hardware_matmul_enabled") is True, f"IQ1_S route lacks physical hardware evidence: {record}")
    gpu_attention = [record for record in routes if is_attention(record.get("kernel")) and record.get("route") == "gpu"]
    require(gpu_attention, "no native GPU attention route was observed")

    xrt_records = load_jsonl(proof_dir / "xrt.jsonl")
    require(len(xrt_records) == len(selected_iq1s), "XRT completion count does not match selected IQ1_S routes")
    physical_submissions = 0
    aggregate_per_cu = [0] * len(APPROVED_CUS)
    for record in xrt_records:
        require(record.get("event") == "au250_xrt_iq1s_completed", f"unexpected XRT event: {record}")
        evidence = record.get("evidence")
        require(isinstance(evidence, dict), "XRT completion has no evidence object")
        require(evidence.get("backend") == "xrt" and evidence.get("comparison_status") == "pass", f"XRT completion did not pass: {evidence}")
        submissions = evidence.get("submission_count")
        per_cu = evidence.get("per_cu_submissions")
        stalls = evidence.get("stall_codes")
        require(isinstance(submissions, int) and submissions > 0, "XRT submission_count must be positive")
        require(isinstance(per_cu, list) and len(per_cu) == len(APPROVED_CUS), "XRT per-CU table must match the approved four-CU ABI")
        require(all(isinstance(value, int) and value >= 0 for value in per_cu), "XRT per-CU counts are invalid")
        require(sum(per_cu) == submissions, "XRT per-CU counts do not account for every submission")
        require(isinstance(stalls, list) and len(stalls) == submissions, "XRT STALL evidence does not account for every submission")
        require(all(isinstance(stall, int) and stall != 0 for stall in stalls), "XRT completion contains a zero STALL code")
        require(-4096 <= evidence.get("raw_min", -4097) <= evidence.get("raw_max", 4097) <= 4096, "XRT raw result is outside signed i16 proof bounds")
        require(isinstance(evidence.get("reference_checked_components"), int) and evidence["reference_checked_components"] > 0, "XRT completion lacks reference-checked components")
        physical_submissions += submissions
        aggregate_per_cu = [left + right for left, right in zip(aggregate_per_cu, per_cu)]

    result = {
        "status": "pass",
        "xrt_completions": len(xrt_records),
        "xrt_routes": len(selected_iq1s),
        "gpu_attention_routes": len(gpu_attention),
        "physical_submissions": physical_submissions,
        "per_cu_submissions": dict(zip(APPROVED_CUS, aggregate_per_cu)),
        "generated_token_ids": tokens,
        "prompt_tokens_per_second": summary["prompt_tokens_per_second"],
        "generation_tokens_per_second": summary["generation_tokens_per_second"],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate strict Kimi GPU-attention/AU250-BitLinear proof")
    parser.add_argument("proof_dir", type=Path)
    args = parser.parse_args()
    try:
        result = validate(args.proof_dir)
    except ProofError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
