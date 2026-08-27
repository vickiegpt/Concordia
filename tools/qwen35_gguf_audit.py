#!/usr/bin/env python3
"""Audit the fixed Qwen3.5 mixed-quant GGUF tensor contract."""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path


EXPECTED_MODEL_SIZE = 94_155_830_880
EXPECTED_MODEL_SHA256 = "0a32c2702fbb61934960cfeef34524b81ec6d9267158f246d45fc86f5aaa7568"
EXPECTED_EXPERT_TYPES = {"IQ1_S": 141, "IQ2_XXS": 24, "IQ3_S": 4, "MXFP4": 11}
GGML_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}
EXPERT_NAME = re.compile(
    r"^blk\.(?:0|[1-9][0-9]*)\.ffn_(?:gate|up|down|gate_up)_exps\.weight$"
)
MAX_STRING_BYTES = 16 * 1024 * 1024
MAX_CONTAINER_ITEMS = 10_000_000
METADATA_FIXED_WIDTHS = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


class AuditError(ValueError):
    pass


class Tensor:
    __slots__ = ("name", "dimensions", "type_id", "type_name", "offset")

    def __init__(self, name, dimensions, type_id, offset):
        self.name = name
        self.dimensions = dimensions
        self.type_id = type_id
        self.type_name = GGML_TYPES.get(type_id, f"GGML_TYPE_{type_id}")
        self.offset = offset


class BoundedReader:
    def __init__(self, stream, size):
        self.stream = stream
        self.size = size
        self.offset = 0

    @property
    def remaining(self):
        return self.size - self.offset

    def read_exact(self, size, context):
        if size < 0 or size > self.remaining:
            raise AuditError(
                f"truncated GGUF while reading {context}: need {size} bytes, "
                f"have {self.remaining}"
            )
        value = self.stream.read(size)
        if len(value) != size:
            raise AuditError(f"short read while reading {context}")
        self.offset += size
        return value

    def skip(self, size, context):
        if size < 0 or size > self.remaining:
            raise AuditError(
                f"truncated GGUF while skipping {context}: need {size} bytes, "
                f"have {self.remaining}"
            )
        self.stream.seek(size, os.SEEK_CUR)
        self.offset += size

    def unpack(self, format_text, context):
        size = struct.calcsize(format_text)
        return struct.unpack(format_text, self.read_exact(size, context))

    def u32(self, context):
        return self.unpack("<I", context)[0]

    def u64(self, context):
        return self.unpack("<Q", context)[0]

    def string(self, context, decode=True):
        size = self.u64(f"{context} length")
        if size > MAX_STRING_BYTES:
            raise AuditError(f"{context} is too large: {size} bytes")
        if decode:
            raw = self.read_exact(size, context)
            try:
                return raw.decode("utf-8", errors="strict")
            except UnicodeError as error:
                raise AuditError(f"{context} is not valid UTF-8: {error}") from error
        self.skip(size, context)
        return None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_count(reader, count, context):
    if count > MAX_CONTAINER_ITEMS:
        raise AuditError(f"{context} count {count} exceeds {MAX_CONTAINER_ITEMS}")
    if count > reader.remaining:
        raise AuditError(f"{context} count {count} cannot fit in {reader.remaining} bytes")


def _skip_metadata_value(reader, type_id, context, depth=0):
    if depth > 16:
        raise AuditError(f"{context} metadata nesting exceeds 16")
    fixed = METADATA_FIXED_WIDTHS.get(type_id)
    if fixed is not None:
        reader.skip(fixed, context)
        return
    if type_id == 8:
        reader.string(context, decode=False)
        return
    if type_id == 9:
        element_type = reader.u32(f"{context} array element type")
        count = reader.u64(f"{context} array count")
        _checked_count(reader, count, f"{context} array")
        for index in range(count):
            _skip_metadata_value(reader, element_type, f"{context}[{index}]", depth + 1)
        return
    raise AuditError(f"unsupported GGUF metadata value type {type_id} for {context}")


def parse_header(path):
    path = Path(path)
    size = path.stat().st_size
    try:
        with path.open("rb") as stream:
            reader = BoundedReader(stream, size)
            if reader.read_exact(4, "magic") != b"GGUF":
                raise AuditError("model does not begin with GGUF magic")
            version = reader.u32("GGUF version")
            if version != 3:
                raise AuditError(f"unsupported GGUF version {version}, expected 3")
            tensor_count = reader.u64("tensor count")
            metadata_count = reader.u64("metadata count")
            _checked_count(reader, tensor_count, "tensor")
            _checked_count(reader, metadata_count, "metadata")

            architecture = None
            for index in range(metadata_count):
                key = reader.string(f"metadata key {index}")
                type_id = reader.u32(f"metadata {key!r} type")
                if key == "general.architecture":
                    if architecture is not None:
                        raise AuditError("duplicate general.architecture metadata")
                    if type_id != 8:
                        raise AuditError("general.architecture metadata is not a string")
                    architecture = reader.string("general.architecture")
                else:
                    _skip_metadata_value(reader, type_id, f"metadata {key!r}")

            tensors = []
            for index in range(tensor_count):
                name = reader.string(f"tensor {index} name")
                rank = reader.u32(f"tensor {name!r} rank")
                if rank == 0 or rank > 4:
                    raise AuditError(f"tensor {name!r} has invalid rank {rank}")
                dimensions = tuple(
                    reader.u64(f"tensor {name!r} dimension {dimension}")
                    for dimension in range(rank)
                )
                if any(value == 0 for value in dimensions):
                    raise AuditError(f"tensor {name!r} contains a zero dimension")
                type_id = reader.u32(f"tensor {name!r} type")
                offset = reader.u64(f"tensor {name!r} offset")
                tensors.append(Tensor(name, dimensions, type_id, offset))
    except OSError as error:
        raise AuditError(f"cannot read model {path}: {error}") from error

    if architecture is None:
        raise AuditError("GGUF omitted general.architecture")
    return architecture, tensors


def validate_contract(architecture, tensors):
    if architecture != "qwen35moe":
        raise AuditError(f"architecture is {architecture!r}, expected 'qwen35moe'")
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        raise AuditError("GGUF contains duplicate tensor names")
    experts = [tensor for tensor in tensors if EXPERT_NAME.fullmatch(tensor.name)]
    distribution = Counter(tensor.type_name for tensor in experts)
    non_expert_iq1s = sorted(
        tensor.name
        for tensor in tensors
        if tensor.type_name == "IQ1_S" and not EXPERT_NAME.fullmatch(tensor.name)
    )
    tq1_total = sum(tensor.type_name == "TQ1_0" for tensor in tensors)
    if len(experts) != 180:
        raise AuditError(f"routed expert tensor count is {len(experts)}, expected 180")
    if dict(distribution) != EXPECTED_EXPERT_TYPES:
        raise AuditError(f"routed expert distribution is {dict(distribution)!r}")
    if tq1_total != 0:
        raise AuditError(f"model contains {tq1_total} unexpected TQ1_0 tensors")
    if non_expert_iq1s:
        raise AuditError(f"non-expert IQ1_S tensors: {non_expert_iq1s}")
    return experts, distribution, non_expert_iq1s, tq1_total


def audit_model(path, expected_size, expected_sha256, verify_hash=True):
    path = Path(path)
    try:
        stat = path.stat()
    except OSError as error:
        raise AuditError(f"cannot stat model {path}: {error}") from error
    if stat.st_size != expected_size:
        raise AuditError(
            f"model byte count is {stat.st_size}, expected {expected_size}"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise AuditError("expected model SHA-256 is not 64 lowercase hexadecimal digits")
    if verify_hash and sha256(path) != expected_sha256:
        raise AuditError("model SHA-256 mismatch")

    architecture, tensors = parse_header(path)
    experts, distribution, non_expert_iq1s, tq1_total = validate_contract(
        architecture, tensors
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "model_path": str(path),
        "model_size": stat.st_size,
        "model_sha256": expected_sha256,
        "architecture": architecture,
        "tensor_count": len(tensors),
        "all_tensor_types": dict(Counter(tensor.type_name for tensor in tensors)),
        "routed_expert_count": len(experts),
        "routed_expert_types": dict(distribution),
        "tq1_0_total": tq1_total,
        "non_expert_iq1s": non_expert_iq1s,
    }


def verify_independent_record(path, record_path, expected_size, expected_sha256):
    path = Path(path)
    try:
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read model verification record: {error}") from error
    stat = path.stat()
    required = {
        "path": str(path),
        "size": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "sha256": expected_sha256,
    }
    if record != required or stat.st_size != expected_size:
        raise AuditError("model changed after the independently hashed preflight")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except OSError as error:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise AuditError(f"cannot write audit {path}: {error}") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--model-verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-size", type=int, default=EXPECTED_MODEL_SIZE)
    parser.add_argument("--expected-sha256", default=EXPECTED_MODEL_SHA256)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        verify_independent_record(
            args.model,
            args.model_verification,
            args.expected_size,
            args.expected_sha256,
        )
        audit = audit_model(
            args.model,
            expected_size=args.expected_size,
            expected_sha256=args.expected_sha256,
            verify_hash=False,
        )
        atomic_json(args.output, audit)
    except (AuditError, OSError) as error:
        print(f"QWEN_GGUF_AUDIT_INVALID: {error}", file=sys.stderr)
        return 1
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
