#!/usr/bin/env python3
"""
Minimal cuobjdump replacement for hetGPU workflows.

Currently supported:
  cuobjdump --extract-ptx all <input>

Behavior:
  - Accepts an ELF shared object / object file or a raw CUDA fatbin.
  - Extracts PTX blobs from .nv_fatbin-like sections and writes them as
    `module_XXX.ptx` files into the current working directory.
  - Falls back to searching for raw PTX markers when structured parsing does
    not find any embedded PTX entries.

This is intentionally narrow: it exists to unblock the cudart shim's
`.so -> PTX -> ELF` path when a system `cuobjdump` is unavailable.
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import sys
import zlib


FATBIN_MAGIC = 0xBA55ED50
FATBINC_MAGIC = 0x466243B1
INTERESTING_SECTIONS = {".nv_fatbin", ".nv.fatbin", ".nvFatBinSegment", ".nv.module.ptx", ".ptx"}


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cuobjdump", add_help=True)
    parser.add_argument("--extract-ptx", dest="extract_ptx", metavar="WHAT")
    parser.add_argument("input", nargs="?")
    ns, extra = parser.parse_known_args()
    if extra:
        parser.error(f"unsupported arguments: {' '.join(extra)}")
    if ns.extract_ptx != "all" or not ns.input:
        parser.error("only `--extract-ptx all <input>` is currently supported")
    return ns


def read_u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0]


def read_u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def read_u64(data: bytes, off: int) -> int:
    return struct.unpack_from("<Q", data, off)[0]


def is_probably_ptx(data: bytes) -> bool:
    stripped = data.lstrip(b"\x00\r\n\t ")
    return stripped.startswith(b".version ") or (
        b".version " in stripped[:256] and b".target " in stripped[:1024]
    )


def sanitize_ptx_blob(data: bytes) -> bytes | None:
    stripped = data.strip(b"\x00\r\n\t ")
    if not stripped:
        return None
    if not is_probably_ptx(stripped):
        return None
    try:
        text = stripped.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if ".version " not in text or ".target " not in text:
        return None
    return (text.rstrip() + "\n").encode("utf-8")


def try_decompress_zlib(data: bytes) -> bytes | None:
    try:
        return zlib.decompress(data)
    except zlib.error:
        return None


def extract_textish_ptx_blobs(data: bytes) -> list[bytes]:
    blobs: list[bytes] = []
    start = 0
    version = b".version "
    while True:
        pos = data.find(version, start)
        if pos < 0:
            break
        end = pos
        while end < len(data):
            b = data[end]
            if b == 0:
                break
            if b in (9, 10, 13) or 32 <= b <= 126:
                end += 1
                continue
            break
        blob = sanitize_ptx_blob(data[pos:end])
        if blob is not None:
            blobs.append(blob)
        start = pos + len(version)
    return dedupe_blobs(blobs)


def dedupe_blobs(blobs: list[bytes]) -> list[bytes]:
    seen: set[bytes] = set()
    out: list[bytes] = []
    for blob in blobs:
        if blob in seen:
            continue
        seen.add(blob)
        out.append(blob)
    return out


def extract_from_fatbin(data: bytes) -> list[bytes]:
    blobs: list[bytes] = []
    if len(data) < 16 or read_u32(data, 0) != FATBIN_MAGIC:
        return blobs

    header_size = read_u16(data, 6)
    if header_size <= 0 or header_size >= len(data):
        return blobs

    off = header_size
    while off + 24 <= len(data):
        kind = read_u16(data, off)
        entry_header_size = read_u16(data, off + 4)
        payload_size = read_u64(data, off + 8)
        uncompressed_size = read_u64(data, off + 16)
        if entry_header_size <= 0:
            break
        payload_start = off + entry_header_size
        payload_end = payload_start + payload_size
        if payload_end > len(data):
            break

        payload = data[payload_start:payload_end]
        if kind == 0x01:
            blob = sanitize_ptx_blob(payload)
            if blob is not None:
                blobs.append(blob)
            elif uncompressed_size > payload_size and payload[:1] == b"\x78":
                inflated = try_decompress_zlib(payload)
                if inflated is not None:
                    blob = sanitize_ptx_blob(inflated)
                    if blob is not None:
                        blobs.append(blob)

        entry_total = entry_header_size + payload_size
        aligned = (entry_total + 7) & ~7
        if aligned <= 0:
            break
        off += aligned

    if not blobs:
        blobs.extend(extract_textish_ptx_blobs(data))
    return dedupe_blobs(blobs)


def extract_from_elf(binary: bytes) -> list[bytes]:
    blobs: list[bytes] = []
    if len(binary) < 64 or binary[:4] != b"\x7fELF":
        return blobs
    if binary[4] != 2 or binary[5] != 1:
        return blobs

    shoff = read_u64(binary, 40)
    shentsize = read_u16(binary, 58)
    shnum = read_u16(binary, 60)
    shstrndx = read_u16(binary, 62)
    if not shoff or not shentsize or not shnum:
        return blobs
    if shoff + shnum * shentsize > len(binary):
        return blobs
    if shstrndx >= shnum:
        return blobs

    shstr_off = shoff + shstrndx * shentsize
    strtab_offset = read_u64(binary, shstr_off + 24)
    strtab_size = read_u64(binary, shstr_off + 32)
    if strtab_offset + strtab_size > len(binary):
        return blobs
    strtab = binary[strtab_offset : strtab_offset + strtab_size]

    sections: list[tuple[str, int, int, int]] = []

    for i in range(shnum):
        sec = shoff + i * shentsize
        name_off = read_u32(binary, sec)
        sec_addr = read_u64(binary, sec + 16)
        sec_offset = read_u64(binary, sec + 24)
        sec_size = read_u64(binary, sec + 32)
        if sec_offset + sec_size > len(binary):
            continue
        end = strtab.find(b"\x00", name_off)
        if end < 0:
            continue
        name = strtab[name_off:end].decode("utf-8", errors="ignore")
        sections.append((name, sec_addr, sec_offset, sec_size))
        if name not in INTERESTING_SECTIONS and "fatbin" not in name and "ptx" not in name:
            continue
        sec_data = binary[sec_offset : sec_offset + sec_size]
        section_blobs: list[bytes] = []
        if name == ".nvFatBinSegment" and len(sec_data) >= 24 and read_u32(sec_data, 0) == FATBINC_MAGIC:
            wrapped_addr = read_u64(sec_data, 8)
            wrapped_off = None
            for sec_name, cand_addr, cand_off, cand_size in sections:
                if cand_size == 0:
                    continue
                if cand_addr <= wrapped_addr < cand_addr + cand_size:
                    wrapped_off = cand_off + (wrapped_addr - cand_addr)
                    break
            if wrapped_off is not None and wrapped_off < len(binary):
                section_blobs = extract_from_fatbin(binary[wrapped_off:])

        if not section_blobs:
            section_blobs = extract_from_fatbin(sec_data)
        if not section_blobs:
            section_blobs = extract_textish_ptx_blobs(sec_data)
        blobs.extend(section_blobs)

    if not blobs:
        blobs.extend(extract_textish_ptx_blobs(binary))
    return dedupe_blobs(blobs)


def extract_ptx(input_path: pathlib.Path) -> list[bytes]:
    data = input_path.read_bytes()
    blobs = extract_from_elf(data)
    if blobs:
        return blobs
    blobs = extract_from_fatbin(data)
    if blobs:
        return blobs
    return extract_textish_ptx_blobs(data)


def write_blobs(blobs: list[bytes], cwd: pathlib.Path) -> int:
    written = 0
    for idx, blob in enumerate(blobs):
        out = cwd / f"module_{idx:03d}.ptx"
        out.write_bytes(blob)
        print(out.name)
        written += 1
    return written


def main() -> int:
    ns = parse_args()
    input_path = pathlib.Path(ns.input).resolve()
    if not input_path.exists():
        eprint(f"cuobjdump: input not found: {input_path}")
        return 1

    blobs = extract_ptx(input_path)
    if not blobs:
        eprint(f"cuobjdump: no PTX found in {input_path}")
        return 1

    count = write_blobs(blobs, pathlib.Path.cwd())
    eprint(f"cuobjdump: extracted {count} PTX file(s) from {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
