#!/usr/bin/env python3
"""Mutation tests for the strict Qwen mixed-quant IQ1_S proof validator."""

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("validate_qwen35_iq1s_au250_proof.py")
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
MODEL_SIZE = 94_155_830_880
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"
EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}


def load_validator():
    spec = importlib.util.spec_from_file_location("qwen_iq1s_proof_validator", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def measurement(value):
    return {
        "model_load_ms": 1000.0 + value,
        "prompt_tokens_per_second": 20.0 + value,
        "ttft_ms": 50.0 + value,
        "generation_tokens_per_second": 5.0 + value,
        "end_to_end_ms": 7000.0 + value,
    }


def xrt_evidence(hybrid):
    return {
        "submission_count": 4 if hybrid else 0,
        "completion_count": 4 if hybrid else 0,
        "per_cu_submissions": [1, 1, 1, 1] if hybrid else [0, 0, 0, 0],
        "per_cu_completions": [1, 1, 1, 1] if hybrid else [0, 0, 0, 0],
        "request_ids": [(1 << 32) + index for index in range(4)] if hybrid else [],
        "stall_codes": [1, 2, 3, 4] if hybrid else [],
        "raw_min": -123 if hybrid else 0,
        "raw_max": 456 if hybrid else 0,
        "reference_checked_components": 64 if hybrid else 0,
        "operation_count": 1 if hybrid else 0,
    }


def route_counts(hybrid):
    return {
        "eligible": 8 if hybrid else 0,
        "handled": 8 if hybrid else 0,
        "fallback": 0,
        "error": 0,
    }


def mode_record(mode):
    hybrid = mode == "hybrid"
    return {
        "schema_version": 2,
        "evidence_kind": "iq1s",
        "mode": mode,
        "model_size": MODEL_SIZE,
        "model_sha256": MODEL_SHA256,
        "model_audit_sha256": "AUTO",
        "llama_revision": LLAMA_REVISION,
        "binary_sha256": "b" * 64,
        "placement": {"all_layers_on_gpu": True, "cpu_layers": 0},
        "prompt_tokens": 256,
        "prompt_text": "fixed prompt",
        "prompt_token_ids": list(range(256)),
        "generated_token_ids": list(range(1000, 1032)),
        "semantic": {"text": "OK", "token_ids": [777]},
        "hardware_probe": {"token_ids": [777, 778], "n_predict": 2},
        "semantic_hardware_gate": (
            {
                "routes": {"eligible": 1, "handled": 1, "fallback": 0, "error": 0},
                "xrt": xrt_evidence(True),
                "gpu_attention_routes": 1,
            }
            if hybrid
            else None
        ),
        "routes": route_counts(hybrid),
        "xrt": xrt_evidence(hybrid),
        "gpu_attention_routes": 4 if hybrid else 0,
        "measurements": [measurement(index) for index in range(5)],
        "process": {"exit_code": 0, "server_termination_code": -15},
        "device_health": {
            "before": {"firewall": "GOOD", "fatal_errors": []},
            "after": {"firewall": "GOOD", "fatal_errors": []},
        },
        "request_contract": {
            "prompt": list(range(256)),
            "n_predict": 32,
            "temperature": 0.0,
            "seed": 42,
            "cache_prompt": False,
            "return_tokens": True,
            "stream": True,
            "timings_per_token": True,
            "return_progress": True,
        },
        "warmup_count": 1,
    }


def valid_proof():
    return {
        "audit": {
            "schema_version": 1,
            "status": "pass",
            "model_sha256": MODEL_SHA256,
            "architecture": "qwen35moe",
            "routed_expert_count": 180,
            "routed_expert_types": copy.deepcopy(EXPERT_TYPES),
            "tq1_0_total": 0,
            "non_expert_iq1s": [],
        },
        "cuda": mode_record("cuda"),
        "hybrid": mode_record("hybrid"),
        "numerical": {
            "schema_version": 1,
            "status": "pass",
            "cases": {
                "single_tile": {"status": "pass", "max_absolute_error": 1e-6},
                "tiled": {"status": "pass", "max_absolute_error": 2e-6},
            },
        },
    }


def write_proof(root, proof):
    audit_raw = json.dumps(proof["audit"], sort_keys=True).encode("utf-8") + b"\n"
    (root / "model-tensor-audit.json").write_bytes(audit_raw)
    audit_hash = hashlib.sha256(audit_raw).hexdigest()
    for mode in ("cuda", "hybrid"):
        if proof[mode]["model_audit_sha256"] == "AUTO":
            proof[mode]["model_audit_sha256"] = audit_hash
        (root / f"{mode}.json").write_text(json.dumps(proof[mode]), encoding="utf-8")
    numerical = root / "numerical"
    numerical.mkdir()
    (numerical / "summary.json").write_text(
        json.dumps(proof["numerical"]), encoding="utf-8"
    )


def mutate_audit_hash(proof):
    proof["hybrid"]["model_audit_sha256"] = "0" * 64


def mutate_audit_distribution(proof):
    proof["audit"]["routed_expert_types"]["IQ1_S"] = 140


def mutate_nonexpert_iq1s(proof):
    proof["audit"]["non_expert_iq1s"] = ["token_embd.weight"]


def mutate_tq1_tensor(proof):
    proof["audit"]["tq1_0_total"] = 1


def mutate_model_size(proof):
    proof["cuda"]["model_size"] -= 1


def mutate_model_hash(proof):
    proof["hybrid"]["model_sha256"] = "0" * 64


def mutate_revision(proof):
    proof["cuda"]["llama_revision"] = "0" * 40


def mutate_binary(proof):
    proof["hybrid"]["binary_sha256"] = "c" * 64


def mutate_cpu_layer(proof):
    proof["hybrid"]["placement"]["cpu_layers"] = 1


def mutate_prompt(proof):
    proof["hybrid"]["prompt_token_ids"][0] = 999


def mutate_generated_tokens(proof):
    proof["hybrid"]["generated_token_ids"][0] = 999


def mutate_semantic_text(proof):
    proof["cuda"]["semantic"]["text"] = "Okay"


def mutate_semantic_route(proof):
    proof["hybrid"]["semantic_hardware_gate"]["routes"]["eligible"] = 0
    proof["hybrid"]["semantic_hardware_gate"]["routes"]["handled"] = 0


def mutate_fallback(proof):
    proof["hybrid"]["routes"]["fallback"] = 1


def mutate_route_error(proof):
    proof["hybrid"]["routes"]["error"] = 1


def mutate_duplicate_request(proof):
    proof["hybrid"]["xrt"]["request_ids"][3] = proof["hybrid"]["xrt"]["request_ids"][2]


def mutate_completion_count(proof):
    proof["hybrid"]["xrt"]["completion_count"] = 3


def mutate_inactive_cu(proof):
    xrt = proof["hybrid"]["xrt"]
    xrt.update(
        {
            "submission_count": 3,
            "completion_count": 3,
            "per_cu_submissions": [1, 1, 1, 0],
            "per_cu_completions": [1, 1, 1, 0],
            "request_ids": xrt["request_ids"][:3],
            "stall_codes": xrt["stall_codes"][:3],
        }
    )


def mutate_zero_stall(proof):
    proof["hybrid"]["xrt"]["stall_codes"][0] = 0


def mutate_raw_overflow(proof):
    proof["hybrid"]["xrt"]["raw_max"] = 4097


def mutate_numerical(proof):
    proof["numerical"]["cases"]["tiled"]["status"] = "fail"


def mutate_nonfinite_timing(proof):
    proof["cuda"]["measurements"][0]["ttft_ms"] = math.nan


def mutate_negative_timing(proof):
    proof["hybrid"]["measurements"][0]["generation_tokens_per_second"] = -1.0


def mutate_measurement_count(proof):
    proof["cuda"]["measurements"].pop()


def mutate_process_failure(proof):
    proof["hybrid"]["process"]["exit_code"] = 1


def mutate_firewall(proof):
    proof["cuda"]["device_health"]["after"]["firewall"] = "BAD"


def mutate_fatal_text(proof):
    proof["hybrid"]["device_health"]["before"]["fatal_errors"] = ["FATAL poison"]


def mutate_request_contract(proof):
    proof["hybrid"]["request_contract"]["seed"] = 41


def mutate_warmup_count(proof):
    proof["cuda"]["warmup_count"] = 0


def mutate_attention_route(proof):
    proof["hybrid"]["gpu_attention_routes"] = 0


def mutate_semantic_xrt(proof):
    proof["hybrid"]["semantic_hardware_gate"]["xrt"]["submission_count"] = 0


MUTATIONS = [
    mutate_audit_hash,
    mutate_audit_distribution,
    mutate_nonexpert_iq1s,
    mutate_tq1_tensor,
    mutate_model_size,
    mutate_model_hash,
    mutate_revision,
    mutate_binary,
    mutate_cpu_layer,
    mutate_prompt,
    mutate_generated_tokens,
    mutate_semantic_text,
    mutate_semantic_route,
    mutate_fallback,
    mutate_route_error,
    mutate_duplicate_request,
    mutate_completion_count,
    mutate_inactive_cu,
    mutate_zero_stall,
    mutate_raw_overflow,
    mutate_numerical,
    mutate_nonfinite_timing,
    mutate_negative_timing,
    mutate_measurement_count,
    mutate_process_failure,
    mutate_firewall,
    mutate_fatal_text,
    mutate_request_contract,
    mutate_warmup_count,
    mutate_attention_route,
    mutate_semantic_xrt,
]


def test_accepts_complete_mixed_quant_proof(tmp_path):
    validator = load_validator()
    write_proof(tmp_path, valid_proof())

    normalized = validator.validate_proof(tmp_path)

    assert normalized["schema_version"] == 2
    assert normalized["status"] == "pass"
    assert normalized["token_ids_match"] is True
    assert normalized["eligible_route_coverage"] == 1.0
    assert normalized["tensor_eligibility_coverage"] == pytest.approx(141 / 180)
    assert normalized["all_cus_active"] is True
    assert normalized["numerical"]["cases"]["tiled"]["max_absolute_error"] == 2e-6


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda function: function.__name__)
def test_rejects_one_fault_at_a_time(tmp_path, mutation):
    validator = load_validator()
    proof = copy.deepcopy(valid_proof())
    mutation(proof)
    write_proof(tmp_path, proof)

    with pytest.raises(validator.ProofInvalid):
        validator.validate_proof(tmp_path)


def test_cli_fails_closed_without_printing_throughput(tmp_path):
    proof = valid_proof()
    mutate_generated_tokens(proof)
    write_proof(tmp_path, proof)

    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(tmp_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("QWEN_IQ1S_PROOF_INVALID:")
    assert "median" not in result.stderr


def test_rejects_missing_file_and_unknown_schema(tmp_path):
    validator = load_validator()
    proof = valid_proof()
    proof["cuda"]["schema_version"] = 3
    write_proof(tmp_path, proof)
    with pytest.raises(validator.ProofInvalid):
        validator.validate_proof(tmp_path)

    (tmp_path / "model-tensor-audit.json").unlink()
    with pytest.raises(validator.ProofInvalid):
        validator.validate_proof(tmp_path)
