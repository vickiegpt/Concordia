import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_au250_hybrid_proof as validator


SHA = "a" * 64


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def make_valid_proof(tmp_path):
    proof = tmp_path / "proof"
    proof.mkdir()
    summary = {
        "model_sha256": SHA,
        "xclbin_sha256": "b" * 64,
        "libnvcuda_sha256": "c" * 64,
        "runner_sha256": "d" * 64,
        "exit_code": 0,
        "generated_token_ids": [1234],
        "prompt_tokens_per_second": 42.0,
        "generation_tokens_per_second": 3.5,
        "gpu_name": "NVIDIA RTX PRO 6000 Blackwell",
        "fpga_bdf": "0000:64:00.1",
        "firewall_status": "GOOD",
        "fatal_errors": [],
        "fpga_temperature_c": 28.0,
    }
    (proof / "summary.json").write_text(json.dumps(summary))
    write_jsonl(
        proof / "routes.jsonl",
        [
            {
                "kernel": "_Z9mul_mat_qIL9ggml_type19E",
                "route": "cxl_tmatmul",
                "backend": "xrt",
                "xrt_enabled": True,
                "hardware_matmul_enabled": True,
            },
            {
                "kernel": "flash_attn_fwd",
                "route": "gpu",
                "backend": "xrt",
                "xrt_enabled": True,
                "hardware_matmul_enabled": False,
            },
        ],
    )
    write_jsonl(
        proof / "xrt.jsonl",
        [
            {
                "event": "au250_xrt_iq1s_completed",
                "evidence": {
                    "backend": "xrt",
                    "comparison_status": "pass",
                    "submission_count": 4,
                    "per_cu_submissions": [1, 1, 1, 1],
                    "stall_codes": [1, 1, 1, 1],
                    "raw_min": -32,
                    "raw_max": 64,
                    "reference_checked_components": 8192,
                },
            }
        ],
    )
    (proof / "stderr.log").write_text("llama_perf_context_print: eval time = 100 ms / 1 tokens (10 tokens per second)\n")
    (proof / "xbutil.txt").write_text("Firewall Level 0 : 0x0 (GOOD)\nFPGA : 28 C\n")
    return proof


def test_accepts_gpu_attention_and_physical_xrt_bitlinear(tmp_path):
    result = validator.validate(make_valid_proof(tmp_path))
    assert result["status"] == "pass"
    assert result["xrt_completions"] == 1
    assert result["gpu_attention_routes"] == 1
    assert result["physical_submissions"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_xrt_completion",
        "bitlinear_gpu_fallback",
        "missing_attention_gpu_route",
        "invalid_token",
        "nonzero_exit",
        "firewall_bad",
        "fatal_error",
        "hash_missing",
        "zero_stall",
    ],
)
def test_rejects_incomplete_or_ambiguous_proof(tmp_path, mutation):
    proof = make_valid_proof(tmp_path)
    summary_path = proof / "summary.json"
    summary = json.loads(summary_path.read_text())
    routes = [json.loads(line) for line in (proof / "routes.jsonl").read_text().splitlines()]
    xrt = [json.loads(line) for line in (proof / "xrt.jsonl").read_text().splitlines()]

    if mutation == "missing_xrt_completion":
        xrt = []
    elif mutation == "bitlinear_gpu_fallback":
        routes[0].update(route="gpu", backend="xrt", hardware_matmul_enabled=False)
    elif mutation == "missing_attention_gpu_route":
        routes = routes[:1]
    elif mutation == "invalid_token":
        summary["generated_token_ids"] = ["not-an-integer"]
    elif mutation == "nonzero_exit":
        summary["exit_code"] = 1
    elif mutation == "firewall_bad":
        summary["firewall_status"] = "TRIPPED"
    elif mutation == "fatal_error":
        summary["fatal_errors"] = ["fatal DMA error"]
    elif mutation == "hash_missing":
        summary["libnvcuda_sha256"] = ""
    elif mutation == "zero_stall":
        xrt[0]["evidence"]["stall_codes"][2] = 0

    summary_path.write_text(json.dumps(summary))
    write_jsonl(proof / "routes.jsonl", routes)
    write_jsonl(proof / "xrt.jsonl", xrt)
    with pytest.raises(validator.ProofError):
        validator.validate(proof)
