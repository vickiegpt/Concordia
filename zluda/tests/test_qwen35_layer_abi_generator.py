import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "tools" / "qwen35-iq1s-layer-abi.json"
GENERATOR = ROOT / "tools" / "generate_qwen35_iq1s_layer_abi.py"


def run_generator(
    tmp_path: Path, *extra: str, schema: Path = SCHEMA
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    c_out = tmp_path / "iq1s_layer_generated.h"
    rust_out = tmp_path / "iq1s_layer_abi.rs"
    sv_out = tmp_path / "iq1s_layer_abi_pkg.sv"
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--schema",
            str(schema),
            "--c-out",
            str(c_out),
            "--rust-out",
            str(rust_out),
            "--sv-out",
            str(sv_out),
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, c_out, rust_out, sv_out


def test_canonical_schema_layout() -> None:
    schema = json.loads(SCHEMA.read_text())

    def field(record: str, name: str) -> tuple[int, str]:
        for field_name, field_type, offset in schema[record]["fields"]:
            if field_name == name:
                return offset, field_type
        raise AssertionError(f"missing {record}.{name}")

    assert schema["abi_version"] == 2
    assert schema["endian"] == "little"
    assert schema["command"]["size"] == 128
    assert schema["completion"]["size"] == 128
    assert field("command", "transaction_id") == (24, "u64")
    assert field("command", "arena_offset") == (64, "u64")
    assert field("completion", "status") == (8, "u32")


def test_generator_emits_deterministic_c_rust_and_sv(tmp_path: Path) -> None:
    result, c_out, rust_out, sv_out = run_generator(tmp_path)
    assert result.returncode == 0, result.stderr

    c_header = c_out.read_text()
    rust_source = rust_out.read_text()
    sv_source = sv_out.read_text()
    schema_hash = hashlib.sha256(SCHEMA.read_bytes()).hexdigest()

    assert "#define HETGPU_IQ1S_COMMAND_BYTES 128u" in c_header
    assert "pub const IQ1S_COMMAND_BYTES: usize = 128;" in rust_source
    assert "localparam int IQ1S_COMMAND_BYTES = 128;" in sv_source
    assert "command CRC32 is IEEE CRC32 with bytes 8..11 zeroed" in c_header
    assert schema_hash in c_header
    assert schema_hash in rust_source
    assert schema_hash in sv_source

    first = (c_header, rust_source, sv_source)
    result, c_out, rust_out, sv_out = run_generator(tmp_path)
    assert result.returncode == 0, result.stderr
    assert first == (c_out.read_text(), rust_out.read_text(), sv_out.read_text())


def test_check_mode_detects_generated_output_drift(tmp_path: Path) -> None:
    result, c_out, rust_out, sv_out = run_generator(tmp_path)
    assert result.returncode == 0, result.stderr
    c_out.write_text(c_out.read_text().replace("TRANSACTION_ID_OFFSET 24u", "TRANSACTION_ID_OFFSET 32u"))

    result, _, _, _ = run_generator(tmp_path, "--check")
    assert result.returncode != 0
    assert "ABI output differs from canonical schema" in result.stderr


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (("command", 6, 1, "u128"), "unknown integer type"),
        (("command", 6, 2, 25), "not naturally aligned"),
        (("command", 8, 2, 32), "overlaps the previous field"),
        (("command", "size", None, 129), "expected exact record size"),
    ],
)
def test_generator_rejects_invalid_record_layouts(
    tmp_path: Path, mutation: tuple[object, ...], error: str
) -> None:
    schema = json.loads(SCHEMA.read_text())
    record_name = mutation[0]
    if mutation[1] == "size":
        schema[record_name]["size"] = mutation[3]
    else:
        field_index = mutation[1]
        element_index = mutation[2]
        schema[record_name]["fields"][field_index][element_index] = mutation[3]
    mutated_schema = tmp_path / "mutated-schema.json"
    mutated_schema.write_text(json.dumps(schema))

    result, _, _, _ = run_generator(tmp_path, schema=mutated_schema)
    assert result.returncode == 2
    assert error in result.stderr
