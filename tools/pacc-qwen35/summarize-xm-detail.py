#!/usr/bin/env python3
"""Require a complete per-job XM breakdown before comparing profiles."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('captures', nargs='+', type=Path)
a = p.parse_args()
for path in a.captures:
    data = json.loads(path.read_text())
    if not data['capture_valid'] or data['observed_index_gaps']:
        raise SystemExit(f'invalid or lossy capture: {path}')
    jobs = {}
    names = ('activation_pack_us', 'weight_unpack_us', 'xm_leaf_us', 'reduce_us', 'xm_calls', 'workers')
    for record in data['raw_records']:
        status = record['status']
        if 0x5500 <= status <= 0x5505:
            row = jobs.setdefault(str(record['seq']), {})
            name = names[status-0x5500]
            if name in row:
                raise SystemExit('duplicate XM phase')
            row[name] = record['aux']
    # Older single-worker images did not emit a worker count.
    for row in jobs.values():
        row.setdefault('workers', 1)
    if set(jobs) != set(data['submissions']) or any(len(row) != 6 for row in jobs.values()):
        raise SystemExit('incomplete XM breakdown')
    sums = {name: sum(row[name] for row in jobs.values()) for name in names[:5]}
    sums['workers_range'] = [min(row['workers'] for row in jobs.values()), max(row['workers'] for row in jobs.values())]
    sums['detail_timing_semantics'] = 'Sum of worker elapsed durations; overlaps when workers > 1. compute_us is serial dispatch wall time.'
    sums.update(source=str(path), jobs=len(jobs), build_ids=sorted(set(data['submission_builds'].values())),
                input_copy_us=data['phases_us']['input_copy']['sum_us'],
                compute_us=data['phases_us']['compute']['sum_us'])
    for seq, row in jobs.items():
        if not 1 <= row['workers'] <= 4 or sum(row[n] for n in names[:4]) > data['submissions'][seq]['compute']*row['workers']:
            raise SystemExit(f'inconsistent timing for {seq}')
    path.with_suffix('.xm.json').write_text(json.dumps(sums, indent=2)+'\n')
    print(json.dumps(sums, indent=2))
