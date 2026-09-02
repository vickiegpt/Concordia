#!/usr/bin/env python3
"""Generate the Qwen IQ1_S persistent ABI from one canonical JSON schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


TYPE_INFO = {
    "u16": (2, "uint16_t", "u16"),
    "u32": (4, "uint32_t", "u32"),
    "u64": (8, "uint64_t", "u64"),
}
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SchemaError(ValueError):
    pass


def upper(name: str) -> str:
    return name.upper()


def require_identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise SchemaError(f"{context} must be an identifier")
    return value


def parse_constant(value: Any, context: str) -> int:
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value, 0)
        except ValueError as error:
            raise SchemaError(f"{context} is not an integer") from error
    else:
        raise SchemaError(f"{context} is not an integer")
    if not 0 <= parsed <= 0xFFFFFFFF:
        raise SchemaError(f"{context} must fit u32")
    return parsed


def validate_record(schema: dict[str, Any], record_name: str) -> None:
    record = schema.get(record_name)
    if not isinstance(record, dict):
        raise SchemaError(f"missing {record_name} record")
    size = record.get("size")
    fields = record.get("fields")
    if not isinstance(size, int) or size <= 0:
        raise SchemaError(f"{record_name}.size must be positive")
    if not isinstance(fields, list) or not fields:
        raise SchemaError(f"{record_name}.fields must be nonempty")

    names: set[str] = set()
    cursor = 0
    for index, field in enumerate(fields):
        if not isinstance(field, list) or len(field) != 3:
            raise SchemaError(f"{record_name}.fields[{index}] must have name, type, offset")
        name = require_identifier(field[0], f"{record_name}.fields[{index}].name")
        type_name = field[1]
        offset = field[2]
        if name in names:
            raise SchemaError(f"duplicate field {record_name}.{name}")
        names.add(name)
        if type_name not in TYPE_INFO:
            raise SchemaError(f"unknown integer type {record_name}.{name}: {type_name}")
        if not isinstance(offset, int) or offset < 0:
            raise SchemaError(f"{record_name}.{name} offset must be nonnegative")
        width = TYPE_INFO[type_name][0]
        if offset % width:
            raise SchemaError(f"{record_name}.{name} is not naturally aligned")
        if offset < cursor:
            raise SchemaError(f"{record_name}.{name} overlaps the previous field")
        if offset != cursor:
            raise SchemaError(
                f"{record_name} has an unnamed byte gap at {cursor}..{offset - 1}; "
                "represent every gap with a reserved field"
            )
        cursor = offset + width
        if cursor > size:
            raise SchemaError(f"{record_name}.{name} exceeds record size")
    if cursor != size:
        raise SchemaError(
            f"{record_name} fields end at {cursor}, expected exact record size {size}; "
            "represent trailing bytes with a reserved field"
        )


def validate_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise SchemaError("schema root must be an object")
    if schema.get("abi_version") != 2:
        raise SchemaError("abi_version must be 2")
    if schema.get("endian") != "little":
        raise SchemaError("endian must be little")

    constants = schema.get("constants")
    if not isinstance(constants, dict) or not constants:
        raise SchemaError("constants must be a nonempty object")
    for name, value in constants.items():
        require_identifier(name, "constant name")
        parse_constant(value, f"constants.{name}")

    enums = schema.get("enums")
    if not isinstance(enums, dict) or not enums:
        raise SchemaError("enums must be a nonempty object")
    for enum_name, members in enums.items():
        require_identifier(enum_name, "enum name")
        if not isinstance(members, dict) or not members:
            raise SchemaError(f"enums.{enum_name} must be nonempty")
        seen_values: set[int] = set()
        for member_name, value in members.items():
            require_identifier(member_name, f"enums.{enum_name} member")
            if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                raise SchemaError(f"enums.{enum_name}.{member_name} must fit u32")
            if value in seen_values:
                raise SchemaError(f"enums.{enum_name} has duplicate value {value}")
            seen_values.add(value)

    registers = schema.get("registers")
    if not isinstance(registers, dict) or not registers:
        raise SchemaError("registers must be a nonempty object")
    seen_offsets: set[int] = set()
    for name, offset in registers.items():
        require_identifier(name, "register name")
        if not isinstance(offset, int) or offset < 0 or offset % 4:
            raise SchemaError(f"registers.{name} must be a nonnegative 4-byte offset")
        if offset in seen_offsets:
            raise SchemaError(f"duplicate register offset {offset}")
        seen_offsets.add(offset)

    validate_record(schema, "command")
    validate_record(schema, "completion")
    return schema


def banner(schema_hash: str, prefix: str) -> list[str]:
    return [
        f"{prefix} Generated from tools/qwen35-iq1s-layer-abi.json; do not edit.",
        f"{prefix} Canonical schema SHA-256: {schema_hash}",
    ]


def generate_c(schema: dict[str, Any], schema_hash: str) -> str:
    lines = banner(schema_hash, "//")
    lines += [
        "#ifndef HETGPU_QWEN35_IQ1S_LAYER_GENERATED_H",
        "#define HETGPU_QWEN35_IQ1S_LAYER_GENERATED_H",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "",
        f'#define HETGPU_IQ1S_SCHEMA_SHA256 "{schema_hash}"',
        f"#define HETGPU_IQ1S_ABI_VERSION {schema['abi_version']}u",
    ]
    for name, value in schema["constants"].items():
        lines.append(f"#define HETGPU_IQ1S_{upper(name)} UINT32_C(0x{parse_constant(value, name):08x})")
    for enum_name, members in schema["enums"].items():
        for member_name, value in members.items():
            lines.append(f"#define HETGPU_IQ1S_{upper(enum_name)}_{upper(member_name)} {value}u")
    lines.append("")
    for name, offset in schema["registers"].items():
        lines.append(f"#define HETGPU_IQ1S_REG_{upper(name)}_OFFSET {offset}u")
    lines.append("")

    for record_name in ("command", "completion"):
        record = schema[record_name]
        lines.append(f"#define HETGPU_IQ1S_{upper(record_name)}_BYTES {record['size']}u")
        for field_name, _, offset in record["fields"]:
            lines.append(
                f"#define HETGPU_IQ1S_{upper(record_name)}_{upper(field_name)}_OFFSET {offset}u"
            )
        lines += ["", "#pragma pack(push, 1)"]
        lines.append(f"typedef struct hetgpu_iq1s_{record_name}_v2 {{")
        for field_name, type_name, _ in record["fields"]:
            lines.append(f"    {TYPE_INFO[type_name][1]} {field_name};")
        lines.append(f"}} hetgpu_iq1s_{record_name}_v2;")
        lines += ["#pragma pack(pop)", ""]
        lines.append(
            f"_Static_assert(sizeof(hetgpu_iq1s_{record_name}_v2) == "
            f"HETGPU_IQ1S_{upper(record_name)}_BYTES, \"{record_name} ABI size\");"
        )
        for field_name, _, _ in record["fields"]:
            lines.append(
                f"_Static_assert(offsetof(hetgpu_iq1s_{record_name}_v2, {field_name}) == "
                f"HETGPU_IQ1S_{upper(record_name)}_{upper(field_name)}_OFFSET, "
                f"\"{record_name}.{field_name} ABI offset\");"
            )
        lines.append("")

    lines += [
        "// command CRC32 is IEEE CRC32 with bytes 8..11 zeroed.",
        "static inline uint32_t hetgpu_iq1s_command_crc32(const void *record) {",
        "    const uint8_t *bytes = (const uint8_t *)record;",
        "    uint32_t crc = UINT32_C(0xffffffff);",
        "    for (size_t i = 0; i < HETGPU_IQ1S_COMMAND_BYTES; ++i) {",
        "        uint32_t byte = (i >= 8u && i <= 11u) ? 0u : bytes[i];",
        "        crc ^= byte;",
        "        for (unsigned bit = 0; bit < 8u; ++bit)",
        "            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (0u - (crc & 1u)));",
        "    }",
        "    return ~crc;",
        "}",
        "",
        "#endif",
    ]
    return "\n".join(lines) + "\n"


def generate_rust(schema: dict[str, Any], schema_hash: str) -> str:
    lines = banner(schema_hash, "//")
    lines += [
        "",
        f'pub const IQ1S_SCHEMA_SHA256: &str = "{schema_hash}";',
        f"pub const IQ1S_ABI_VERSION: u32 = {schema['abi_version']};",
    ]
    for name, value in schema["constants"].items():
        lines.append(f"pub const IQ1S_{upper(name)}: u32 = 0x{parse_constant(value, name):08x};")
    for enum_name, members in schema["enums"].items():
        for member_name, value in members.items():
            lines.append(f"pub const IQ1S_{upper(enum_name)}_{upper(member_name)}: u32 = {value};")
    lines.append("")
    for name, offset in schema["registers"].items():
        lines.append(f"pub const IQ1S_REG_{upper(name)}_OFFSET: usize = {offset};")
    lines.append("")

    for record_name in ("command", "completion"):
        record = schema[record_name]
        rust_type = "Iq1s" + record_name.title()
        lines.append(f"pub const IQ1S_{upper(record_name)}_BYTES: usize = {record['size']};")
        for field_name, _, offset in record["fields"]:
            lines.append(
                f"pub const IQ1S_{upper(record_name)}_{upper(field_name)}_OFFSET: usize = {offset};"
            )
        lines += ["", "#[repr(C)]", "#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]"]
        lines.append(f"pub struct {rust_type} {{")
        for field_name, type_name, _ in record["fields"]:
            lines.append(f"    pub {field_name}: {TYPE_INFO[type_name][2]},")
        lines += ["}", ""]
        lines.append(
            f"const _: [(); IQ1S_{upper(record_name)}_BYTES] = "
            f"[(); core::mem::size_of::<{rust_type}>()];"
        )
        for field_name, _, _ in record["fields"]:
            lines.append(
                f"const _: [(); IQ1S_{upper(record_name)}_{upper(field_name)}_OFFSET] = "
                f"[(); core::mem::offset_of!({rust_type}, {field_name})];"
            )
        lines.append("")

    lines += [
        "/// command CRC32 is IEEE CRC32 with bytes 8..11 zeroed.",
        "pub fn iq1s_command_crc32(record: &[u8; IQ1S_COMMAND_BYTES]) -> u32 {",
        "    let mut crc = 0xffff_ffffu32;",
        "    for (index, value) in record.iter().copied().enumerate() {",
        "        let byte = if (8..=11).contains(&index) { 0 } else { value };",
        "        crc ^= u32::from(byte);",
        "        for _ in 0..8 {",
        "            crc = (crc >> 1) ^ (0xedb8_8320 & 0u32.wrapping_sub(crc & 1));",
        "        }",
        "    }",
        "    !crc",
        "}",
    ]
    return "\n".join(lines) + "\n"


def generate_sv(schema: dict[str, Any], schema_hash: str) -> str:
    lines = banner(schema_hash, "//")
    lines += [
        "package iq1s_layer_abi_pkg;",
        "",
        f'localparam string IQ1S_SCHEMA_SHA256 = "{schema_hash}";',
        f"localparam int IQ1S_ABI_VERSION = {schema['abi_version']};",
    ]
    for name, value in schema["constants"].items():
        lines.append(
            f"localparam logic [31:0] IQ1S_{upper(name)} = 32'h{parse_constant(value, name):08x};"
        )
    for enum_name, members in schema["enums"].items():
        for member_name, value in members.items():
            lines.append(f"localparam int IQ1S_{upper(enum_name)}_{upper(member_name)} = {value};")
    lines.append("")
    for name, offset in schema["registers"].items():
        lines.append(f"localparam int IQ1S_REG_{upper(name)}_OFFSET = {offset};")
    lines.append("")
    for record_name in ("command", "completion"):
        record = schema[record_name]
        lines.append(f"localparam int IQ1S_{upper(record_name)}_BYTES = {record['size']};")
        for field_name, _, offset in record["fields"]:
            lines.append(
                f"localparam int IQ1S_{upper(record_name)}_{upper(field_name)}_OFFSET = {offset};"
            )
        lines.append("")
    lines += [
        "// command CRC32 is IEEE CRC32 with bytes 8..11 zeroed.",
        "localparam logic [31:0] IQ1S_CRC32_REFLECTED_POLYNOMIAL = 32'hedb88320;",
        "",
        "endpackage",
    ]
    return "\n".join(lines) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        try:
            return path.read_text() == content
        except FileNotFoundError:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--c-out", required=True, type=Path)
    parser.add_argument("--rust-out", required=True, type=Path)
    parser.add_argument("--sv-out", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        schema_bytes = args.schema.read_bytes()
        schema = validate_schema(json.loads(schema_bytes))
    except (OSError, json.JSONDecodeError, SchemaError) as error:
        print(f"invalid ABI schema: {error}", file=sys.stderr)
        return 2

    schema_hash = hashlib.sha256(schema_bytes).hexdigest()
    outputs = (
        (args.c_out, generate_c(schema, schema_hash)),
        (args.rust_out, generate_rust(schema, schema_hash)),
        (args.sv_out, generate_sv(schema, schema_hash)),
    )
    mismatches = [str(path) for path, content in outputs if not write_or_check(path, content, args.check)]
    if mismatches:
        print(
            "ABI output differs from canonical schema: " + ", ".join(mismatches),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
