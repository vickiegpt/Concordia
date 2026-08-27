#!/usr/bin/env python3
import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("validate_qwen35_tq1_au250_proof.py")
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
MODEL_SIZE = 94155830880
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"


def load_validator():
    spec = importlib.util.spec_from_file_location("qwen_proof_validator", MODULE_PATH)
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


def mode_record(mode):
    hybrid = mode == "hybrid"
    return {
        "schema_version": 1,
        "mode": mode,
        "model_size": MODEL_SIZE,
        "model_sha256": MODEL_SHA256,
        "llama_revision": LLAMA_REVISION,
        "binary_sha256": "b" * 64,
        "placement": {"all_layers_on_gpu": True, "cpu_layers": 0},
        "prompt_tokens": 256,
        "prompt_token_ids": list(range(256)),
        "generated_token_ids": [42, 17, 99],
        "semantic": {"text": "OK", "token_ids": [777]},
        "routes": {
            "eligible": 8 if hybrid else 0,
            "handled": 8 if hybrid else 0,
            "fallback": 0,
            "error": 0,
        },
        "xrt": {
            "submission_count": 4 if hybrid else 0,
            "completion_count": 4 if hybrid else 0,
            "per_cu_submissions": [1, 1, 1, 1] if hybrid else [0, 0, 0, 0],
            "per_cu_completions": [1, 1, 1, 1] if hybrid else [0, 0, 0, 0],
            "request_ids": [1, 2, 3, 4] if hybrid else [],
            "stall_codes": [1, 1, 1, 1] if hybrid else [],
            "raw_min": -123 if hybrid else 0,
            "raw_max": 456 if hybrid else 0,
        },
        "measurements": [measurement(index) for index in range(5)],
        "process": {"exit_code": 0},
        "device_health": {
            "before": {"firewall": "GOOD", "fatal_errors": []},
            "after": {"firewall": "GOOD", "fatal_errors": []},
        },
    }


def valid_proof():
    return {
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
    (root / "cuda.json").write_text(json.dumps(proof["cuda"]), encoding="utf-8")
    (root / "hybrid.json").write_text(json.dumps(proof["hybrid"]), encoding="utf-8")
    numerical = root / "numerical"
    numerical.mkdir()
    (numerical / "summary.json").write_text(
        json.dumps(proof["numerical"]), encoding="utf-8"
    )


def mutate_model_size(proof):
    proof["cuda"]["model_size"] -= 1


def mutate_model_hash(proof):
    proof["hybrid"]["model_sha256"] = "0" * 64


def mutate_llama_revision(proof):
    proof["cuda"]["llama_revision"] = "0" * 40


def mutate_binary(proof):
    proof["hybrid"]["binary_sha256"] = "c" * 64


def mutate_cpu_layer(proof):
    proof["hybrid"]["placement"]["cpu_layers"] = 1


def mutate_prompt_count(proof):
    proof["cuda"]["prompt_tokens"] = 255


def mutate_generated_tokens(proof):
    proof["hybrid"]["generated_token_ids"] = [42, 18, 99]


def mutate_semantic(proof):
    proof["cuda"]["semantic"]["text"] = "Okay"


def mutate_eligible_zero(proof):
    proof["hybrid"]["routes"]["eligible"] = 0
    proof["hybrid"]["routes"]["handled"] = 0


def mutate_handled(proof):
    proof["hybrid"]["routes"]["handled"] = 7


def mutate_fallback(proof):
    proof["hybrid"]["routes"]["fallback"] = 1


def mutate_route_error(proof):
    proof["hybrid"]["routes"]["error"] = 1


def mutate_cu_zero(proof):
    proof["hybrid"]["xrt"]["per_cu_completions"][3] = 0


def mutate_missing_request(proof):
    proof["hybrid"]["xrt"]["request_ids"].pop()


def mutate_duplicate_request(proof):
    proof["hybrid"]["xrt"]["request_ids"][3] = 3


def mutate_stall(proof):
    proof["hybrid"]["xrt"]["stall_codes"][0] = 0


def mutate_raw_bound(proof):
    proof["hybrid"]["xrt"]["raw_max"] = 16385


def mutate_numerical(proof):
    proof["numerical"]["cases"]["tiled"]["status"] = "fail"


def mutate_exit(proof):
    proof["cuda"]["process"]["exit_code"] = 1


def mutate_health(proof):
    proof["hybrid"]["device_health"]["after"]["firewall"] = "BAD"


def mutate_measurement_count(proof):
    proof["cuda"]["measurements"].pop()


def mutate_missing_timing(proof):
    del proof["hybrid"]["measurements"][0]["ttft_ms"]


def mutate_nonfinite_timing(proof):
    proof["cuda"]["measurements"][0]["end_to_end_ms"] = math.nan


def mutate_negative_timing(proof):
    proof["hybrid"]["measurements"][0]["generation_tokens_per_second"] = -1.0


MUTATIONS = [
    mutate_model_size,
    mutate_model_hash,
    mutate_llama_revision,
    mutate_binary,
    mutate_cpu_layer,
    mutate_prompt_count,
    mutate_generated_tokens,
    mutate_semantic,
    mutate_eligible_zero,
    mutate_handled,
    mutate_fallback,
    mutate_route_error,
    mutate_cu_zero,
    mutate_missing_request,
    mutate_duplicate_request,
    mutate_stall,
    mutate_raw_bound,
    mutate_numerical,
    mutate_exit,
    mutate_health,
    mutate_measurement_count,
    mutate_missing_timing,
    mutate_nonfinite_timing,
    mutate_negative_timing,
]


def test_accepts_valid_proof_and_summarizes_metrics(tmp_path):
    validator = load_validator()
    write_proof(tmp_path, valid_proof())
    summary = validator.validate_proof(tmp_path)
    assert summary["status"] == "pass"
    assert summary["token_ids_match"] is True
    assert summary["eligible_route_coverage"] == 1.0
    assert summary["all_cus_active"] is True
    assert summary["modes"]["cuda"]["measurements"] == 5
    assert summary["modes"]["hybrid"]["metrics"]["ttft_ms"]["median"] == 52.0


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda function: function.__name__)
def test_rejects_one_fault_at_a_time(tmp_path, mutation):
    validator = load_validator()
    proof = copy.deepcopy(valid_proof())
    mutation(proof)
    write_proof(tmp_path, proof)
    with pytest.raises(validator.ProofInvalid):
        validator.validate_proof(tmp_path)


def test_cli_fails_closed_without_printing_metrics(tmp_path):
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
    assert result.stderr.startswith("QWEN_TQ1_PROOF_INVALID:")
    assert "median" not in result.stderr


def test_rejects_unknown_schema_and_missing_keys(tmp_path):
    validator = load_validator()
    for patch in ("schema", "missing"):
        proof = valid_proof()
        if patch == "schema":
            proof["cuda"]["schema_version"] = 2
        else:
            del proof["hybrid"]["routes"]
        case = tmp_path / patch
        case.mkdir()
        write_proof(case, proof)
        with pytest.raises(validator.ProofInvalid):
            validator.validate_proof(case)
