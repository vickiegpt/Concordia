#!/usr/bin/env python3
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def host_target_triple() -> str:
    override = os.environ.get("HETGPU_SOURCE_TO_PTX_HOST_TARGET")
    if override:
        return override

    machine = platform.machine()
    if machine == "riscv64":
        return "riscv64-linux-gnu"
    if machine in ("aarch64", "arm64"):
        return "aarch64-linux-gnu"
    if machine in ("x86_64", "amd64"):
        return "x86_64-linux-gnu"
    return f"{machine}-linux-gnu"


def command_from_entry(entry):
    args = entry.get("arguments")
    if args:
        return list(args)
    command = entry.get("command")
    if command:
        return shlex.split(command)
    return []


def cxx_include_flags(host_target: str) -> list[str]:
    override = os.environ.get("HETGPU_SOURCE_TO_PTX_CXX_INCLUDE_DIRS")
    if override:
        flags = []
        for path in override.split(os.pathsep):
            if path:
                flags.extend(["-isystem", path])
        return flags

    root = Path("/usr/include/c++")
    if not root.is_dir():
        return []

    versions = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not versions:
        return []

    version = versions[-1].name
    candidates = [
        root / version,
        Path("/usr/include") / host_target / "c++" / version,
        root / version / host_target,
    ]
    flags = []
    for path in candidates:
        if path.is_dir():
            flags.extend(["-isystem", str(path)])
    return flags


def write_cuda_prelude(out_dir: str) -> str:
    path = os.path.join(out_dir, "__hetgpu_source_to_ptx_prelude.hpp")
    text = """#pragma once
// fake_cuda defines __noinline__ before Clang's -include file is parsed.  That
// collides with libstdc++ headers that use __attribute__((__noinline__, ...)).
// Pre-include the C++ headers with the CUDA macro hidden, then restore it for
// any CUDA/device code that still expects the spelling.
#pragma push_macro("__noinline__")
#undef __noinline__
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#pragma pop_macro("__noinline__")
"""
    with open(path, "w") as f:
        f.write(text)
    return path


def source_entries(entries):
    for entry in entries:
        src = entry.get("file", "")
        if "/ggml/src/ggml-cuda/" not in src or not src.endswith(".cu"):
            continue
        cmd = command_from_entry(entry)
        if cmd:
            yield src, cmd


def sanitize_ptx_for_parser(path: str) -> None:
    with open(path, "r") as f:
        text = f.read()

    # Our PTX parser only accepts the base target directive.  Clang can emit
    # ".target sm_80, debug" when the original compile command carried debug
    # flags; the target option is not needed by the PACC lowering path.
    text = re.sub(r"(?m)^(\s*\.target\s+[^,\n]+),[^\n]*$", r"\1", text)

    with open(path, "w") as f:
        f.write(text)


def compile_one(src: str, cmd: list[str], out_dir: str, clang: str, host_target: str, prelude: str):
    out = os.path.join(out_dir, os.path.basename(src) + ".ptx")
    if os.path.exists(out) and os.path.getsize(out) > 50:
        return src, 0, ""

    new = [
        clang,
        f"--target={host_target}",
        "--sysroot=/",
        "--gcc-toolchain=/usr",
        "-Wno-unknown-cuda-version",
    ]
    new.extend(cxx_include_flags(host_target))
    new.extend(["-include", prelude])
    skip = False
    skip_next_source = False

    for a in cmd[1:]:
        if skip:
            skip = False
            continue
        if skip_next_source:
            skip_next_source = False
            continue
        if a == "-o":
            skip = True
            continue
        if a == "-c":
            skip_next_source = True
            continue
        if a == "-Xcompiler":
            skip = True
            continue
        if a in ("-target", "--target", "--gcc-toolchain", "--sysroot"):
            skip = True
            continue
        if (
            a.startswith("--target=")
            or a.startswith("-target=")
            or a.startswith("--gcc-toolchain=")
            or a.startswith("--sysroot=")
            or a in ("-nostdinc++", "-nostdinc")
        ):
            continue
        if a in ("-extended-lambda", "--expt-extended-lambda", "-use_fast_math"):
            continue
        if (
            a in ("-g", "-G", "--device-debug", "--generate-line-info", "-lineinfo")
            or a.startswith("-gline-")
            or a.startswith("-gdwarf")
            or a.startswith("-ggdb")
            or a.startswith("-gmodules")
            or a.startswith("-fdebug-")
        ):
            continue
        if a.startswith("-compress-mode=") or a in ("--ptx", "--cubin", "--fatbin"):
            continue
        new.append(a)

    new.append("-DTHRUST_IGNORE_CUB_VERSION_CHECK")
    new.extend(["--cuda-device-only", "-S", src, "-o", out])
    result = subprocess.run(new, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        sanitize_ptx_for_parser(out)
    return src, result.returncode, result.stderr[-4000:]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: source_to_ptx.py <compile_commands.json> <out_dir>", file=sys.stderr)
        return 2

    cc_path, out_dir = sys.argv[1], sys.argv[2]
    entries = json.load(open(cc_path, "r"))
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    host_target = host_target_triple()
    clang = os.environ.get("HETGPU_SOURCE_TO_PTX_CLANG", "/usr/bin/clang++-20")
    jobs = int(os.environ.get("HETGPU_SOURCE_TO_PTX_JOBS", str(min(4, os.cpu_count() or 1))))
    jobs = max(1, jobs)
    work = list(source_entries(entries))
    only = os.environ.get("HETGPU_SOURCE_TO_PTX_ONLY")
    if only:
        needles = [item for item in only.split(os.pathsep) if item]
        work = [(src, cmd) for src, cmd in work if any(needle in src for needle in needles)]
    exclude = os.environ.get("HETGPU_SOURCE_TO_PTX_EXCLUDE")
    if exclude:
        needles = [item for item in exclude.split(os.pathsep) if item]
        work = [(src, cmd) for src, cmd in work if not any(needle in src for needle in needles)]
    prelude = write_cuda_prelude(out_dir)

    if jobs == 1 or len(work) <= 1:
        results = [compile_one(src, cmd, out_dir, clang, host_target, prelude) for src, cmd in work]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(compile_one, src, cmd, out_dir, clang, host_target, prelude)
                for src, cmd in work
            ]
            results = [future.result() for future in futures]

    for src, returncode, stderr_tail in results:
        if returncode != 0:
            sys.stderr.write("[cudart_shim] source->PTX compile failed for %s\n" % src)
            sys.stderr.write(stderr_tail)
            continue
        count += 1

    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
