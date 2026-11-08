#!/usr/bin/env python3
"""Copy original GGUF tensors for a diagnostic layer prefix, not an LLM benchmark."""
import argparse
import hashlib
import json
import pathlib
import re
import gguf

p = argparse.ArgumentParser()
p.add_argument("source", type=pathlib.Path)
p.add_argument("output", type=pathlib.Path)
p.add_argument("--layers", type=int, default=4)
a = p.parse_args()
if a.output.exists() or not a.output.name.startswith("DIAGNOSTIC-"):
    p.error("output must be new and explicitly named DIAGNOSTIC-*")
r = gguf.GGUFReader(a.source)
arch = r.fields["general.architecture"].contents()
if arch != "qwen35moe" or not 1 <= a.layers < r.fields[f"{arch}.block_count"].contents():
    p.error("unsupported architecture or layer count")
w = gguf.GGUFWriter(a.output, arch)
for name, field in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    value = field.contents()
    if name == f"{arch}.block_count":
        value = a.layers
    if name == "general.name":
        value = f"DIAGNOSTIC ONLY Qwen35 prefix {a.layers} layers"
    w.add_key_value(name, value, field.types[0], field.types[-1] if len(field.types) > 1 else None)
records = []
for tensor in r.tensors:
    match = re.match(r"blk\.(\d+)\.", tensor.name)
    if match and int(match[1]) >= a.layers:
        continue
    w.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    records.append({"name": tensor.name, "bytes": tensor.n_bytes,
                    "sha256": hashlib.sha256(tensor.data).hexdigest()})
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
check = gguf.GGUFReader(a.output)
assert len(check.tensors) == len(records)
for tensor, record in zip(check.tensors, records):
    assert tensor.name == record["name"]
    assert hashlib.sha256(tensor.data).hexdigest() == record["sha256"]
a.output.with_suffix(".provenance.json").write_text(json.dumps({
    "diagnostic_only": True, "source": str(a.source.resolve()), "layers": a.layers,
    "tensors_byte_identical": True, "tensors": records}, indent=2) + "\n")
print(f"DIAGNOSTIC ONLY: {len(records)} byte-verified tensors, {a.layers} layers")
