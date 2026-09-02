#!/usr/bin/env python3
"""Mutation tests for the strict three-mode Qwen IQ1_S proof validator."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("validate_qwen35_iq1s_au250_proof.py")
MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
MODEL_SIZE = 94_155_830_880
LLAMA_REVISION = "925e1179947ea0c0ebfb0032df18af3a729822be"
EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}
TRACE_ASSEMBLY = (
    "ldv v0, PARAM_INPUT\n"
    "tmatmul_import v0\n"
    "tmatmul_go PARAM_MATRIX\n"
    "tmatmul_export v0\n"
    "sv v0, PARAM_OUTPUT\n"
    "stall\n"
)
TRACE_INSTRUCTIONS = [
    ["ldv", "v0", "PARAM_INPUT"],
    ["tmatmul_import", "v0"],
    ["tmatmul_go", "PARAM_MATRIX"],
    ["tmatmul_export", "v0"],
    ["sv", "v0", "PARAM_OUTPUT"],
    ["stall"],
]
IQ1S_KERNEL = "_Z9mul_mat_qIL9ggml_type19ELi32ELi8ELb0EEvPKcS2_PfS3_iiiiiii"


def load_validator():
    spec = importlib.util.spec_from_file_location("qwen_iq1s_proof_validator", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def semantic_sha256(instructions=TRACE_INSTRUCTIONS):
    digest = hashlib.sha256(b"hetgpu-tmatmul-semantic-trace-v1\0")
    for instruction in instructions:
        for token in instruction:
            digest.update(token.encode())
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def physical_completion(mode, index):
    encoded = bytes(range(16))
    program_sha = hashlib.sha256(encoded).hexdigest()
    cache_hit = index >= 2
    return {
        "request_id": (1 << 32) + index + 1,
        "cu_index": index,
        "stall_code": index + 1,
        "dispatch_to_stall_ns": 1000 + index,
        "matrix_key_sha256": f"{index + 1:064x}",
        "matrix_content_sha256": f"{index + 5:064x}",
        "matrix_address": 0x1000 + index * 0x1000,
        "matrix_cache_hit": cache_hit,
        "matrix_bytes_transferred": 0 if cache_hit else 262_144,
        "trace_mode": mode,
        "model_context_limit": 262_144,
        "trace_semantic_sha256": semantic_sha256(),
        "trace_assembly_sha256": hashlib.sha256(TRACE_ASSEMBLY.encode()).hexdigest(),
        "replay_safe_program_sha256": program_sha,
        "trace_assembly": TRACE_ASSEMBLY,
        "trace_instructions": copy.deepcopy(TRACE_INSTRUCTIONS),
        "encoded_program_sha256": program_sha,
        "encoded_program_hex": encoded.hex(),
        "program_address": 0x10000 + index * 0x1000,
        "program_bytes": len(encoded),
        "program_cache_hit": cache_hit,
    }


def xrt_evidence(mode):
    base = {
        "submission_count": 0,
        "completion_count": 0,
        "per_cu_submissions": [0, 0, 0, 0],
        "per_cu_completions": [0, 0, 0, 0],
        "request_ids": [],
        "stall_codes": [],
        "raw_min": 0,
        "raw_max": 0,
        "reference_checked_components": 0,
        "operation_count": 0,
        "resident_matrix_hits": 0,
        "resident_matrix_misses": 0,
        "resident_matrix_bytes_transferred": 0,
        "program_cache_hits": 0,
        "program_cache_misses": 0,
        "host_pack_hits": 0,
        "host_pack_misses": 0,
        "host_pack_bytes_built": 0,
        "physical_completions": [],
    }
    if mode == "cuda":
        return base
    base.update({
        "submission_count": 4,
        "completion_count": 4,
        "per_cu_submissions": [1, 1, 1, 1],
        "per_cu_completions": [1, 1, 1, 1],
        "request_ids": [(1 << 32) + index + 1 for index in range(4)],
        "stall_codes": [1, 2, 3, 4],
        "raw_min": -123,
        "raw_max": 456,
        "reference_checked_components": 64,
        "operation_count": 1,
        "resident_matrix_hits": 2,
        "resident_matrix_misses": 2,
        "resident_matrix_bytes_transferred": 2 * 262_144,
        "program_cache_hits": 2,
        "program_cache_misses": 2,
        "host_pack_hits": 2,
        "host_pack_misses": 2,
        "host_pack_bytes_built": 2 * 262_144,
        "physical_completions": [physical_completion(mode, index) for index in range(4)],
    })
    return base


def routes(mode):
    if mode == "cuda":
        return {"eligible": 0, "handled": 0, "fallback": 0, "error": 0, "eligible_kernels": []}
    return {
        "eligible": 8,
        "handled": 8,
        "fallback": 0,
        "error": 0,
        "eligible_kernels": [IQ1S_KERNEL] * 8,
    }


def measurement(index):
    wall = 100.0 + index
    return {
        "model_load_ms": 1000.0,
        "prompt_tokens_per_second": 20.0 + index,
        "ttft_ms": 50.0 + index,
        "generation_tokens_per_second": 2048 / wall,
        "single_request_generation_tokens_per_second": 5.0,
        "end_to_end_ms": 7000.0 + index,
        "queue_ms": 20.0,
        "service_ms": 6500.0,
        "measured_wall_seconds": wall,
        "request_count": 64,
        "max_active": 16,
        "generated_tokens": 2048,
    }


def sampled_comparison(mode):
    if mode == "cuda":
        return None
    return {
        "status": "pass",
        "reference_backend": "scalar_iq1s",
        "checked_elements": 3,
        "atol": 1.0e-4,
        "rtol": 1.0e-3,
        "max_absolute_error": 1.0e-5,
        "max_relative_error": 1.0e-5,
        "reference_outputs": [1.0, -2.0, 0.0],
        "actual_outputs": [1.00001, -2.0, 0.0],
        "phase": "pre_timed",
        "kernel": IQ1S_KERNEL,
    }


def mode_record(mode):
    hybrid = mode != "cuda"
    xrt = xrt_evidence(mode)
    generated = list(range(1000, 1032))
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
        "generated_token_ids": generated,
        "generated_token_ids_by_request": [generated[:] for _ in range(64)],
        "token_equivalence_measurement": 0,
        "slot_ids_by_request": [index % 16 for index in range(64)],
        "slot_erase_evidence": [[0] * 16 for _ in range(4)],
        "wave_slot_erase_evidence": [[([287] * 16) for _ in range(3)] for _ in range(4)],
        "semantic": {"text": "OK", "token_ids": [777]},
        "hardware_probe": {"token_ids": [777, 778], "n_predict": 2},
        "sampled_ffn_comparison": sampled_comparison(mode),
        "semantic_hardware_gate": (
            {"routes": {"eligible": 1, "handled": 1, "fallback": 0, "error": 0, "eligible_kernels": [IQ1S_KERNEL]},
             "xrt": copy.deepcopy(xrt), "gpu_attention_routes": 1}
            if hybrid else None
        ),
        "routes": routes(mode),
        "xrt": xrt,
        "gpu_attention_routes": 4 if hybrid else 0,
        "measurements": [measurement(index) for index in range(3)],
        "process": {"exit_code": 0, "server_termination_code": -15},
        "device_health": {
            "before": {"firewall": "GOOD", "fatal_errors": []},
            "after": {"firewall": "GOOD", "fatal_errors": []},
        },
        "request_contract": {
            "prompt": list(range(256)), "n_predict": 32, "temperature": 0.0,
            "seed": 42, "ignore_eos": True, "cache_prompt": False, "return_tokens": True,
            "stream": True, "timings_per_token": True, "return_progress": True,
        },
        "warmup_count": 1,
        "request_count": 64,
        "max_active_requests": 16,
        "generated_tokens_per_request": 32,
        "context_tokens_per_request": 512,
        "server_context_tokens": 8192,
    }


def valid_proof():
    return {
        "audit": {
            "schema_version": 1, "status": "pass", "model_sha256": MODEL_SHA256,
            "architecture": "qwen35moe", "routed_expert_count": 180,
            "routed_expert_types": copy.deepcopy(EXPERT_TYPES), "tq1_0_total": 0,
            "non_expert_iq1s": [],
        },
        "cuda": mode_record("cuda"),
        "handwritten": mode_record("handwritten"),
        "compiler": mode_record("compiler"),
        "artifact_hashes": {
            "model_sha256": MODEL_SHA256, "llama_server_sha256": "b" * 64,
            "libggml_sha256": "c" * 64, "libnvcuda_sha256": "d" * 64,
            "cuda13_launch_shim_sha256": "e" * 64, "xclbin_sha256": "f" * 64,
            "llama_revision": LLAMA_REVISION, "build_threads": "32", "threads": "64",
        },
        "repository_diff_sha256": "1" * 64,
        "pcie": "LnkCap: Speed 16GT/s, Width x16\nLnkSta: Speed 8GT/s (downgraded), Width x4 (downgraded)\n",
    }


def write_proof(root, proof):
    audit_raw = json.dumps(proof["audit"], sort_keys=True).encode() + b"\n"
    (root / "model-tensor-audit.json").write_bytes(audit_raw)
    audit_hash = hashlib.sha256(audit_raw).hexdigest()
    for mode in ("cuda", "handwritten", "compiler"):
        if proof[mode]["model_audit_sha256"] == "AUTO":
            proof[mode]["model_audit_sha256"] = audit_hash
        (root / f"{mode}.json").write_text(json.dumps(proof[mode]), encoding="utf-8")
    (root / "model-verification.json").write_text(json.dumps({
        "path": "/models/qwen/Qwen3.5-397B-A17B-UD-TQ1_0.gguf", "size": MODEL_SIZE,
        "device": 1, "inode": 2, "mtime_ns": 3, "ctime_ns": 4, "sha256": MODEL_SHA256,
    }), encoding="utf-8")
    (root / "qwen-build-preflight.json").write_text(json.dumps({
        "schema_version": 1, "build_root": "/qwen-build",
        "libggml_path": "/qwen-build/llama-build/bin/libggml.so.0.22.0",
        "libggml_sha256": proof["artifact_hashes"]["libggml_sha256"],
        "llama_revision": LLAMA_REVISION,
        "required_symbols": ["dequantize_row_iq1_s", "ggml_init", "ggml_free"],
        "status": "pass",
    }), encoding="utf-8")
    (root / "artifact-hashes.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in proof["artifact_hashes"].items()), encoding="utf-8")
    (root / "repository-diff.sha256").write_text(proof["repository_diff_sha256"] + "\n", encoding="utf-8")
    (root / "pcie-link.txt").write_text(proof["pcie"], encoding="utf-8")
    (root / "xclbin-info.txt").write_text("\n".join(
        f"Instance:        {name}" for name in ("ternip_big_1", "ternip_big_2", "ternip_big_3", "ternip_small_1")) + "\n", encoding="utf-8")


def mutate_audit_distribution(p): p["audit"]["routed_expert_types"]["IQ1_S"] = 140
def mutate_nonexpert_iq1s(p): p["audit"]["non_expert_iq1s"] = ["token_embd.weight"]
def mutate_model_hash(p): p["compiler"]["model_sha256"] = "0" * 64
def mutate_build_hash(p): p["artifact_hashes"]["libggml_sha256"] = "0" * 64
def mutate_source_hash(p): p["repository_diff_sha256"] = "bad"
def mutate_binary(p): p["compiler"]["binary_sha256"] = "a" * 64
def mutate_build_threads(p): p["artifact_hashes"]["build_threads"] = "33"
def mutate_workload_count(p): p["handwritten"]["request_count"] = 63
def mutate_concurrency(p): p["compiler"]["max_active_requests"] = 17
def mutate_token_count(p): p["cuda"]["generated_tokens_per_request"] = 31
def mutate_per_request_context(p): p["cuda"]["context_tokens_per_request"] = 256
def mutate_server_context(p): p["compiler"]["server_context_tokens"] = 512
def mutate_slot_assignment(p): p["handwritten"]["slot_ids_by_request"][17] = 2
def mutate_slot_erase(p): p["compiler"]["slot_erase_evidence"][2].pop()
def mutate_wave_slot_erase(p): p["compiler"]["wave_slot_erase_evidence"][1][2].pop()
def mutate_token_equivalence_batch(p): p["cuda"]["token_equivalence_measurement"] = 1
def mutate_request_ids(p): p["handwritten"]["generated_token_ids_by_request"].pop()
def mutate_tokens(p): p["compiler"]["generated_token_ids_by_request"][3][0] = 999
def mutate_ffn_value(p): p["handwritten"]["sampled_ffn_comparison"]["actual_outputs"][0] = 2.0
def mutate_ffn_tolerance(p): p["compiler"]["sampled_ffn_comparison"]["rtol"] = 1.0
def mutate_fallback(p): p["handwritten"]["routes"]["fallback"] = 1
def mutate_route_set(p): p["compiler"]["routes"]["eligible_kernels"][0] += "_different"
def mutate_trace_mode(p): p["compiler"]["xrt"]["physical_completions"][0]["trace_mode"] = "handwritten"
def mutate_context_limit(p): p["compiler"]["xrt"]["physical_completions"][0]["model_context_limit"] = 262_145
def mutate_semantic_hash(p): p["compiler"]["xrt"]["physical_completions"][0]["trace_semantic_sha256"] = "0" * 64
def mutate_assembly_body(p): p["handwritten"]["xrt"]["physical_completions"][0]["trace_assembly"] += "stall\n"
def mutate_assembly_hash(p): p["compiler"]["xrt"]["physical_completions"][0]["trace_assembly_sha256"] = "0" * 64
def mutate_program_hash(p): p["compiler"]["xrt"]["physical_completions"][0]["encoded_program_sha256"] = "0" * 64
def mutate_replay_hash(p): p["handwritten"]["xrt"]["physical_completions"][0]["replay_safe_program_sha256"] = "0" * 64
def mutate_program_address(p): p["compiler"]["xrt"]["physical_completions"][0]["program_address"] = 0
def mutate_program_bytes(p): p["compiler"]["xrt"]["physical_completions"][0]["program_bytes"] = 32
def mutate_cache_transfer(p): p["handwritten"]["xrt"]["physical_completions"][0]["matrix_bytes_transferred"] = 0
def mutate_cache_hits(p): p["compiler"]["xrt"]["resident_matrix_hits"] = 0
def mutate_duplicate_completion(p): p["handwritten"]["xrt"]["physical_completions"][3]["request_id"] = p["handwritten"]["xrt"]["physical_completions"][2]["request_id"]
def mutate_inactive_cu(p): p["compiler"]["xrt"]["physical_completions"][3]["cu_index"] = 2
def mutate_zero_stall(p): p["handwritten"]["xrt"]["physical_completions"][0]["stall_code"] = 0
def mutate_firewall(p): p["compiler"]["device_health"]["after"]["firewall"] = "BAD"
def mutate_pcie(p): p["pcie"] = "LnkSta: Speed 8GT/s, Width x0\n"
def mutate_throughput(p):
    m = p["compiler"]["measurements"][1]
    m["measured_wall_seconds"] = 200.0
    m["generation_tokens_per_second"] = 2048 / 200.0
def mutate_throughput_math(p): p["handwritten"]["measurements"][0]["generation_tokens_per_second"] += 1.0
def mutate_measurement_count(p): p["compiler"]["measurements"].pop()
def mutate_process(p): p["handwritten"]["process"]["exit_code"] = 1


MUTATIONS = [
    mutate_audit_distribution, mutate_nonexpert_iq1s, mutate_model_hash, mutate_build_hash,
    mutate_source_hash, mutate_binary, mutate_build_threads, mutate_workload_count, mutate_concurrency,
    mutate_token_count, mutate_per_request_context, mutate_server_context, mutate_slot_assignment,
    mutate_slot_erase, mutate_wave_slot_erase,
    mutate_token_equivalence_batch,
    mutate_request_ids, mutate_tokens, mutate_ffn_value,
    mutate_ffn_tolerance, mutate_fallback, mutate_route_set, mutate_trace_mode,
    mutate_context_limit, mutate_semantic_hash, mutate_assembly_body, mutate_assembly_hash,
    mutate_program_hash, mutate_replay_hash, mutate_program_address, mutate_program_bytes,
    mutate_cache_transfer, mutate_cache_hits, mutate_duplicate_completion, mutate_inactive_cu,
    mutate_zero_stall, mutate_firewall, mutate_pcie, mutate_throughput,
    mutate_throughput_math, mutate_measurement_count, mutate_process,
]


def test_accepts_complete_three_mode_proof(tmp_path):
    validator = load_validator()
    write_proof(tmp_path, valid_proof())
    normalized = validator.validate_proof(tmp_path)
    assert normalized["schema_version"] == 3
    assert normalized["status"] == "pass"
    assert normalized["token_ids_match"] is True
    assert normalized["all_cus_active"] is True
    assert normalized["tensor_eligibility_coverage"] == pytest.approx(141 / 180)
    assert normalized["pcie_link"] == {"observed": "Gen3 x4", "downgrade_risk": True}
    assert set(normalized["modes"]) == {"cuda", "handwritten", "compiler"}
    assert all(item >= 15.0 for mode in ("handwritten", "compiler")
               for item in normalized["modes"][mode]["measured_throughput"])


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
    mutate_tokens(proof)
    write_proof(tmp_path, proof)
    result = subprocess.run([sys.executable, str(MODULE_PATH), str(tmp_path)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("QWEN_IQ1S_PROOF_INVALID:")
    assert "tok/s" not in result.stderr


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
