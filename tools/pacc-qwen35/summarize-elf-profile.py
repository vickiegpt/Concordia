#!/usr/bin/env python3
"""Separate host phases from nested device durations, never mix clock epochs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

def summarize(path):
    host, device, batches = defaultdict(lambda: defaultdict(float)), defaultdict(float), defaultdict(float)
    paired = defaultdict(float)
    pending = None
    def fields(line):
        return dict(part.split('=', 1) for part in line.split()[1:] if '=' in part)
    with path.open(errors='replace') as stream:
        for line in stream:
            if line.startswith('PACC_ELF_PROFILE '):
                row = fields(line); values = {k:float(v) for k,v in row.items() if k.endswith('_us')}
                target = host[row['kernel']]; target['calls'] += 1
                for key, value in values.items(): target[key] += value
                pending = (row, values)
            elif line.startswith('PACC_QK_PROFILE '):
                row = fields(line); device['calls'] += 1
                for key,value in row.items():
                    if key.endswith('_us') or key == 'xm_calls': device[key] += float(value)
                if not pending or pending[0]['kernel'] != 'pacc_q4k_tile_v1' or pending[0]['dev'] != row['dev']:
                    raise ValueError('device record has no adjacent host submission')
                hv = pending[1]
                residual = hv['publish_us']+hv['wait_us']-float(row['device_us'])
                paired['calls'] += 1
                paired['publish_wait_minus_device_us'] += residual
                paired['negative_residuals'] += residual < -100
                pending = None
            elif line.startswith('PACC_QK_BATCH '):
                row = fields(line); batches['calls'] += 1
                for key,value in row.items(): batches[key] += float(value)
    return dict(log=str(path.resolve()), host_by_kernel=dict(host), device_qk=dict(device),
                qk_host_batch=dict(batches), paired_qk=dict(paired),
                limitations=['Host wait contains device execution: do not add the two durations.',
                    'Publish+wait minus kernel duration includes image publishing, guest setup and cleanup, scheduling, polling and completion visibility; it is not pure IRQ latency.',
                    'Device syscall timing adds overhead; compare throughput with detailed timing disabled.',
                    'Only Q4/Q5/Q8 ELF kernels have internal device timings here; other kernels expose host phases only.'])

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('log',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();report=summarize(args.log)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
