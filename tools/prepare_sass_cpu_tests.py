#!/usr/bin/env python3
"""Prepare CPU-only SASS tests using real source modules and the real PTX parser.

This is an explicit subset, not a replacement for cargo test --workspace.
No compiler, test, CUDA tool, or network process is started by this script.
"""
import argparse
import json
import pathlib
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=pathlib.Path, required=True,
                        help="New isolated harness/build directory; must not exist")
    parser.add_argument("--lockfile", type=pathlib.Path,
                        help="Optional existing dependency lock seed; never modified")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    modules = ["instruction", "cubin_parser", "disassembler", "lifter",
               "fuzz", "translation_pipeline"]
    integrations = ["sass_lifter_fuzz", "sass_translation_pipeline"]
    inputs = [source / "ptx/src/sass" / f"{name}.rs" for name in modules]
    inputs += [source / "ptx/tests" / f"{name}.rs" for name in integrations]
    inputs += [source / "ptx_parser/Cargo.toml"]
    if args.lockfile:
        inputs.append(args.lockfile)
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing required real source/input: {path}")
    output.mkdir(parents=True, exist_ok=False)
    if args.lockfile:
        shutil.copy2(args.lockfile, output / "Cargo.lock")
    manifest = '''[package]
name = "concordia_sass_cpu_contract"
version = "0.1.0"
edition = "2021"
publish = false
[workspace]
[lib]
name = "ptx"
path = "lib.rs"
[dependencies]
serde = { version = "1.0", features = ["derive"] }
object = { version = "0.32", default-features = false, features = ["read", "elf"] }
tempfile = "3"
'''
    manifest += "ptx_parser = { path = " + json.dumps(str(source / "ptx_parser")) + " }\n"
    for name in integrations:
        manifest += ("\n[[test]]\nname = " + json.dumps(name)
                     + "\npath = " + json.dumps(str(source / "ptx/tests" / f"{name}.rs")) + "\n")
    (output / "Cargo.toml").write_text(manifest)
    rust = "pub mod sass {\n"
    for name in modules:
        rust += ("#[path = " + json.dumps(str(source / "ptx/src/sass" / f"{name}.rs"))
                 + f"]\npub mod {name};\npub use {name}::*;\n")
    rust += "}\npub use sass::*;\n"
    (output / "lib.rs").write_text(rust)
    (output / "SOURCE.json").write_text(json.dumps({
        "source": str(source), "modules": modules, "integration_tests": integrations,
        "scope": "CPU SASS subset; no parser stubs, no workspace/GPU claim",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
