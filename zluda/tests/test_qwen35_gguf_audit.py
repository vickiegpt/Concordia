#!/usr/bin/env python3
"""Behavior tests for the fixed Qwen3.5 mixed-quant GGUF audit."""

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest


AUDITOR = Path(__file__).parents[2] / "tools" / "qwen35_gguf_audit.py"
EXPECTED_EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}
TYPE_IDS = {"IQ1_S": 19, "IQ2_XXS": 16, "IQ3_S": 21, "MXFP4": 39}


def load_auditor():
    spec = importlib.util.spec_from_file_location("qwen35_gguf_audit", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def gguf_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def expert_tensors(distribution=EXPECTED_EXPERT_TYPES):
    names = [
        f"blk.{layer}.ffn_{projection}_exps.weight"
        for layer in range(60)
        for projection in ("gate", "up", "down")
    ]
    type_ids = []
    for type_name, count in distribution.items():
        type_ids.extend([TYPE_IDS[type_name]] * count)
    assert len(names) == len(type_ids) == 180
    return [(name, type_id) for name, type_id in zip(names, type_ids)]


def encode_tensor(name, type_id):
    return b"".join(
        [
            gguf_string(name),
            struct.pack("<I", 2),
            struct.pack("<QQ", 1024, 1024),
            struct.pack("<I", type_id),
            struct.pack("<Q", 0),
        ]
    )


def write_gguf(path, architecture="qwen35moe", tensors=None):
    tensors = expert_tensors() if tensors is None else tensors
    metadata = gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string(architecture)
    body = b"".join(encode_tensor(name, type_id) for name, type_id in tensors)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + metadata + body)


def test_fixed_contract_accepts_only_expected_mixed_experts(tmp_path):
    model = tmp_path / "model.gguf"
    write_gguf(model)

    audit = load_auditor().audit_model(
        model,
        expected_size=model.stat().st_size,
        expected_sha256="a" * 64,
        verify_hash=False,
    )

    assert audit["status"] == "pass"
    assert audit["architecture"] == "qwen35moe"
    assert audit["routed_expert_count"] == 180
    assert audit["routed_expert_types"] == EXPECTED_EXPERT_TYPES
    assert audit["all_tensor_types"] == EXPECTED_EXPERT_TYPES
    assert audit["tq1_0_total"] == 0
    assert audit["non_expert_iq1s"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_architecture", "architecture"),
        ("wrong_expert_count", "180"),
        ("wrong_expert_distribution", "distribution"),
        ("add_tq1_tensor", "TQ1_0"),
        ("add_nonexpert_iq1s", "non-expert IQ1_S"),
        ("duplicate_tensor", "duplicate tensor"),
    ],
)
def test_fixed_contract_rejects_model_drift(tmp_path, mutation, message):
    model = tmp_path / f"{mutation}.gguf"
    architecture = "qwen35moe"
    tensors = expert_tensors()
    if mutation == "wrong_architecture":
        architecture = "qwen3"
    elif mutation == "wrong_expert_count":
        tensors = tensors[:-1]
    elif mutation == "wrong_expert_distribution":
        tensors[0] = (tensors[0][0], TYPE_IDS["IQ2_XXS"])
    elif mutation == "add_tq1_tensor":
        tensors.append(("output.weight", 34))
    elif mutation == "add_nonexpert_iq1s":
        tensors.append(("token_embd.weight", TYPE_IDS["IQ1_S"]))
    elif mutation == "duplicate_tensor":
        tensors.append(tensors[0])
    write_gguf(model, architecture=architecture, tensors=tensors)

    auditor = load_auditor()
    with pytest.raises(auditor.AuditError, match=message):
        auditor.audit_model(
            model,
            expected_size=model.stat().st_size,
            expected_sha256="a" * 64,
            verify_hash=False,
        )


@pytest.mark.parametrize("malformation", ["string_length", "metadata_type", "descriptor_truncation"])
def test_bounded_parser_rejects_malformed_headers(tmp_path, malformation):
    model = tmp_path / f"{malformation}.gguf"
    if malformation == "string_length":
        model.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 1, 100))
    elif malformation == "metadata_type":
        model.write_bytes(
            b"GGUF"
            + struct.pack("<IQQ", 3, 0, 1)
            + gguf_string("general.architecture")
            + struct.pack("<I", 13)
        )
    else:
        model.write_bytes(
            b"GGUF"
            + struct.pack("<IQQ", 3, 1, 1)
            + gguf_string("general.architecture")
            + struct.pack("<I", 8)
            + gguf_string("qwen35moe")
            + gguf_string("blk.0.ffn_gate_exps.weight")
            + struct.pack("<I", 2)
            + struct.pack("<Q", 1024)
        )

    auditor = load_auditor()
    with pytest.raises(auditor.AuditError):
        auditor.audit_model(
            model,
            expected_size=model.stat().st_size,
            expected_sha256="a" * 64,
            verify_hash=False,
        )


def test_cli_binds_to_independent_file_identity_and_writes_audit(tmp_path):
    model = tmp_path / "model.gguf"
    output = tmp_path / "audit.json"
    verification = tmp_path / "verification.json"
    write_gguf(model)
    auditor = load_auditor()
    stat = model.stat()
    verification.write_text(
        json.dumps(
            {
                "path": str(model),
                "size": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "sha256": auditor.sha256(model),
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            str(model),
            "--model-verification",
            str(verification),
            "--output",
            str(output),
            "--expected-size",
            str(stat.st_size),
            "--expected-sha256",
            auditor.sha256(model),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["schema_version"] == 1
    assert value["status"] == "pass"
    assert value["model_sha256"] == auditor.sha256(model)
    assert not output.with_suffix(".json.partial").exists()


def test_cli_rejects_identity_change_after_hash_record(tmp_path):
    model = tmp_path / "model.gguf"
    output = tmp_path / "audit.json"
    verification = tmp_path / "verification.json"
    write_gguf(model)
    stat = model.stat()
    verification.write_text(
        json.dumps(
            {
                "path": str(model),
                "size": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns + 1,
                "sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            str(model),
            "--model-verification",
            str(verification),
            "--output",
            str(output),
            "--expected-size",
            str(stat.st_size),
            "--expected-sha256",
            "a" * 64,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "changed after" in result.stderr
    assert not output.exists()
