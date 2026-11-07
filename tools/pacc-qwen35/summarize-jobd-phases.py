#!/usr/bin/env python3
import json
from pathlib import Path
import re
import argparse

p = argparse.ArgumentParser()
p.add_argument('captures', nargs='+')
p.add_argument('--host-log', type=Path)
args = p.parse_args()
if args.host_log and len(args.captures) != 1:
    p.error('--host-log requires exactly one capture')
for filename in args.captures:
    path = Path(filename)
    data = json.loads(path.read_text())
    rows = [value for key, value in sorted(data['submissions'].items(), key=lambda item: int(item[0]))]
    cursor = 0
    host = None
    calls = []
    for line in (args.host_log or path.with_suffix('.log')).read_text().splitlines():
        match = re.search(r'profile pack_us=(\d+) write_us=(\d+) submit_wait_us=(\d+) readback_us=(\d+)', line)
        if match:
            host = dict(zip(('pack', 'write', 'submit_wait', 'readback'), map(int, match.groups())))
        match = re.search(r'dev=(\d+) m=(\d+) n=(\d+) k=(\d+) tiles=(\d+) elapsed_ms=([\d.]+)', line)
        if not match:
            continue
        dev, m, n, k, tiles = map(int, match.groups()[:5])
        group = rows[cursor:cursor+tiles]
        cursor += tiles
        if len(group) != tiles or any(len(row) != 7 for row in group):
            raise SystemExit('incomplete phase records: refuse summary')
        device = {phase: sum(row[phase] for row in group) for phase in group[0]}
        calls.append(dict(m=m, n=n, k=k, tiles=tiles,
                          elapsed_ms=float(match[6]), host_us=host, device_us=device,
                          wait_residual_us=None if host is None else host['submit_wait']-
                          device['device_before_publish']-device['completion_publish']))
        host = None
    if cursor != len(rows):
        raise SystemExit('phase count does not match logged submissions')
    result = dict(source=str(path), calls=calls, capabilities=data['capabilities'],
                  limitations='Residual includes dispatch, host polling and diagnostics; not isolated IRQ latency.')
    destination = path.with_suffix('.summary.json')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
