#!/usr/bin/env python3
"""Verify single-token recurrent state carry in consecutive tensor snapshots."""
import argparse
import array
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
p.add_argument("--minimum-pairs", type=int, default=1)
a = p.parse_args()
initial, pairs = [], []
for path in sorted(a.directory.glob("state_predelta-*.f32")):
    match = re.fullmatch(r"state_predelta-(\d+)\.(\d+)\.f32", path.name)
    if not match:
        continue
    layer, token = map(int, match.groups())
    data = path.read_bytes()
    if token == 0:
        floats = array.array("f")
        floats.frombytes(data)
        assert all(x == 0 for x in floats), (layer, "initial state not zero")
        initial.append(layer)
    else:
        previous = a.directory / f"new_state-{layer}.{token-1}.f32"
        assert previous.read_bytes() == data, (layer, token, "state carry not byte-identical")
        pairs.append([layer, token])
assert initial and len(pairs) >= a.minimum_pairs, "insufficient state evidence"
print(json.dumps({"initial_zero_layers": initial, "byte_identical_carry_pairs": pairs,
                  "all_correct": True, "scope": "state copying only; no inference or TPS qualification"}, indent=2))
