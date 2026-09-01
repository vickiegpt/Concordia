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
        "cudart_shim_sha256": "e" * 64,
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
        "gpu_layers": 24,
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


def test_accepts_explicit_nonfinite_q8_native_capability_fallback(tmp_path):
    proof = make_valid_proof(tmp_path)
    routes = [json.loads(line) for line in (proof / "routes.jsonl").read_text().splitlines()]
    routes.insert(
        1,
        {
            "kernel": "_Z9mul_mat_qIL9ggml_type19E",
            "route": "gpu",
            "backend": "xrt",
            "xrt_enabled": True,
            "hardware_matmul_enabled": False,
            "reason": "nonfinite_q8_not_representable_by_au250_abi",
        },
    )
    write_jsonl(proof / "routes.jsonl", routes)

    result = validator.validate(proof)
    assert result["xrt_completions"] == 1
    assert result["iq1s_native_capability_fallbacks"] == 1


def test_accepts_explicit_gpu_single_token_iq1s_vector_route(tmp_path):
    proof = make_valid_proof(tmp_path)
    routes = [json.loads(line) for line in (proof / "routes.jsonl").read_text().splitlines()]
    routes.insert(
        1,
        {
            "kernel": "_Z13mul_mat_vec_qIL9ggml_type19ELi1E",
            "route": "gpu",
            "source": "explicit_gpu_env",
            "backend": "xrt",
            "xrt_enabled": True,
            "hardware_matmul_enabled": False,
        },
    )
    write_jsonl(proof / "routes.jsonl", routes)

    result = validator.validate(proof)
    assert result["xrt_completions"] == 1
    assert result["iq1s_gpu_vector_routes"] == 1


def test_parses_current_llama_perf_lines_without_mixing_prompt_and_eval():
    stderr = """\
llama_perf_context_print: prompt eval time = 4710.77 ms / 33 tokens (142.75 ms per token, 7.01 tokens per second)
llama_perf_context_print:        eval time = 250.00 ms / 1 runs (250.00 ms per token, 4.00 tokens per second)
"""
    assert validator.parse_perf_rate(stderr, "prompt eval") == 7.01
    assert validator.parse_perf_rate(stderr, "eval") == 4.0


def test_recognizes_current_cuda_soft_max_symbol_as_attention():
    assert validator.is_attention("_Z12soft_max_f32ILb1ELi64ELi64EfEvPKf")


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
        "cudart_hash_missing",
        "zero_stall",
        "zero_gpu_layers",
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
    elif mutation == "cudart_hash_missing":
        summary["cudart_shim_sha256"] = ""
    elif mutation == "zero_stall":
        xrt[0]["evidence"]["stall_codes"][2] = 0
    elif mutation == "zero_gpu_layers":
        summary["gpu_layers"] = 0

    summary_path.write_text(json.dumps(summary))
    write_jsonl(proof / "routes.jsonl", routes)
    write_jsonl(proof / "xrt.jsonl", xrt)
    with pytest.raises(validator.ProofError):
        validator.validate(proof)
