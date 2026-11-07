#!/usr/bin/env python3
"""Normalize existing vendor XM aliases using libfdt; never add capabilities."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import struct

ALIASES = {b"xlxmmbase": b"xsfmmbase", b"xlxmm32a8i": b"xsfmm32a8i",
           b"xlxmm32a16f": b"xsfmm32a16f", b"xlxmm32a32f": b"xsfmm32a32f",
           b"xlxmm64t": b"xsfmm64t"}


def normalize(raw, separator):
    if not raw.endswith(b"\0"):
        raise ValueError("unterminated ISA property")
    parts = raw[:-1].split(separator)
    if any(parts.count(old) != 1 for old in ALIASES):
        raise ValueError("expected exactly one of each existing XM alias")
    if any(new in parts for new in ALIASES.values()):
        raise ValueError("mixed aliases")
    result = separator.join(ALIASES.get(p, p) for p in parts) + b"\0"
    if len(result) != len(raw):
        raise ValueError("property length changed")
    return result


def patch(data, offset):
    magic, size = struct.unpack_from(">II", data, offset)
    if magic != 0xd00dfeed or size < 40 or offset + size > len(data):
        raise ValueError("invalid FDT bounds")
    lib = C.CDLL("libfdt.so.1")
    lib.fdt_check_header.argtypes = [C.c_void_p]
    lib.fdt_next_node.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int)]
    lib.fdt_getprop.argtypes = [C.c_void_p, C.c_int, C.c_char_p, C.POINTER(C.c_int)]
    lib.fdt_getprop.restype = C.c_void_p
    lib.fdt_get_path.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    lib.fdt_setprop_inplace.argtypes = [C.c_void_p, C.c_int, C.c_char_p, C.c_void_p, C.c_int]
    blob = C.create_string_buffer(data[offset:offset+size], size)
    if lib.fdt_check_header(blob):
        raise ValueError("libfdt rejected header")
    depth = C.c_int(0)
    node, cpus, changes = -1, [], []
    while True:
        node = lib.fdt_next_node(blob, node, C.byref(depth))
        if node == -1:
            break
        if node < 0:
            raise ValueError(f"FDT iteration error {node}")
        length = C.c_int()
        ptr = lib.fdt_getprop(blob, node, b"device_type", C.byref(length))
        if not ptr or C.string_at(ptr, length.value) != b"cpu\0":
            continue
        path = C.create_string_buffer(512)
        if lib.fdt_get_path(blob, node, path, len(path)):
            raise ValueError("CPU path error")
        cpus.append(path.value.decode())
        for prop, sep in ((b"riscv,isa", b"_"), (b"riscv,isa-extensions", b"\0")):
            ptr = lib.fdt_getprop(blob, node, prop, C.byref(length))
            if not ptr or length.value <= 0:
                raise ValueError(f"missing {prop!r}")
            original = C.string_at(ptr, length.value)
            updated = normalize(original, sep)
            if lib.fdt_setprop_inplace(blob, node, prop, updated, len(updated)):
                raise ValueError("inplace FDT edit failed")
            changes.append({"node": path.value.decode(), "property": prop.decode(),
                            "before": original.decode(), "after": updated.decode()})
    if sorted(cpus) != [f"/cpus/cpu@{i}" for i in range(4)]:
        raise ValueError(f"unexpected CPU set {cpus}")
    result = data[:offset] + blob.raw + data[offset+size:]
    changed = [i for i, (a, b) in enumerate(zip(data, result)) if a != b]
    if len(changed) != 80 or any(not offset <= i < offset+size for i in changed):
        raise ValueError(f"unexpected byte changes: {len(changed)}")
    return result, {"fdt_offset": offset, "fdt_size": size,
                    "changed_offsets": changed, "properties": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--offset", type=lambda s: int(s, 0), default=0x1f000)
    args = parser.parse_args()
    data = args.source.read_bytes()
    before = hashlib.sha256(data).hexdigest()
    if before != args.sha256:
        raise SystemExit("source hash mismatch")
    result, report = patch(data, args.offset)
    report.update(source_sha256=before, output_sha256=hashlib.sha256(result).hexdigest())
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or manifest.exists():
        raise SystemExit("refusing to overwrite output or manifest")
    with args.output.open("xb") as f:
        f.write(result)
    with manifest.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "properties"}))


if __name__ == "__main__":
    main()
