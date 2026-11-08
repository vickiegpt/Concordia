#!/usr/bin/env python3
"""Audit recorded ELF staging spans; this is not a DMA hardware trace."""
import argparse
import hashlib
import json
from pathlib import Path
import re

BASE = 0x2075d000000
LAYOUT = re.compile(r'helper slot layout dev=(\d+).*slot_bytes=(\d+) slot_off=(0x[0-9a-f]+) required=(\d+)')
BINDING = re.compile(r"staged kernel='[^']+' arg=\d+ host=0x[0-9a-f]+ shared=(0x[0-9a-f]+) size=(\d+)")
SUBMIT = re.compile(r'helper-submit .* shared=(0x[0-9a-f]+) submit=(\d+) bytes on sifive(\d+) slot(\d+)')

def audit(lines, device):
    start = 0x200000 + device * 0x80000
    counts = dict(layouts=0, bindings=0, submissions=0)
    violations = []
    violation_count = 0
    def check(condition, number, reason):
        nonlocal violation_count
        if not condition:
            violation_count += 1
            if len(violations) < 20:
                violations.append(dict(line=number, reason=reason))
    for number, line in enumerate(lines, 1):
        if match := LAYOUT.search(line):
            dev, size, off, required = (int(x, 0) for x in match.groups())
            counts['layouts'] += 1
            check(dev == device and size == 0x80000 and off == start and 0 < required <= size,
                  number, 'wrong device, slot, or image size')
        if match := BINDING.search(line):
            address, size = (int(x, 0) for x in match.groups())
            counts['bindings'] += 1
            check(size > 0 and BASE + start <= address and address + size <= BASE + start + 0x80000,
                  number, 'binding outside worker payload')
        if match := SUBMIT.search(line):
            address, size, dev, slot = (int(x, 0) for x in match.groups())
            counts['submissions'] += 1
            check(address == BASE + start and 0 < size <= 0x80000 and dev == device and slot == 0,
                  number, 'submission outside worker payload or device')
    return dict(device=device, counts=counts, violations=violations, violation_count=violation_count,
                passed=all(counts.values()) and not violation_count)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for device in range(4):
        path = args.directory / f'pacc{device}-phases.log'
        with path.open(errors='replace') as stream:
            result = audit(stream, device)
        with path.open('rb') as stream:
            result['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        result['log'] = str(path)
        results.append(result)
    report = dict(all_passed=all(r['passed'] for r in results), devices=results,
                  scope='Logged ELF host staging bounds and physical affinity only; not all device memory accesses')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['all_passed'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
