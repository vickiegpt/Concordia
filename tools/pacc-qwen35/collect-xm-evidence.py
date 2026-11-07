#!/usr/bin/env python3
"""Export bounded summaries plus hashes; retain large raw traces on the host."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=Path('/srv/lanxin/qwen35-a3b'))
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
manifest = []


def record(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    manifest.append(dict(path=str(path), size=path.stat().st_size, sha256=h.hexdigest()))


for name in ('profile-v35-xm-config', 'profile-v36-isa-alias', 'profile-v37-xm-batch',
             'profile-v38-q6k-xm', 'profile-v38-xm-large', 'phases-v38-xm-count16-p4'):
    path = a.root/'logs'/(name+'.json')
    data = json.loads(path.read_text())
    record(path)
    summary = {k: v for k, v in data.items() if k not in ('raw_records', 'submissions', 'submission_builds')}
    summary['build_values'] = sorted(set(data.get('submission_builds', {}).values()))
    summary['capability_events'] = [r for r in data['raw_records'] if 0x5320 <= r['status'] < 0x5330 or 0x5400 <= r['status'] < 0x5448]
    summary['source_sha256'] = manifest[-1]['sha256']
    (a.output/(name+'.summary.json')).write_text(json.dumps(summary, indent=2)+'\n')
    if name != 'phases-v38-xm-count16-p4':
        log = path.with_suffix('.log')
        record(log)
        shutil.copyfile(log, a.output/log.name)
for name in ('concurrent-count16-v38-xm-p4.json', 'concurrent-count16-v34-pacc-p4.json',
             'concurrent-count16-cpu-p4.json', 'profile-v38-xm-large.summary.json'):
    path = a.root/'logs'/name
    record(path)
    destination = 'profile-v38-xm-large.calls.json' if name == 'profile-v38-xm-large.summary.json' else name
    shutil.copyfile(path, a.output/destination)
for name in ('concurrent-count16-v38-xm-p4.server.log', 'build-v38-q6k-xm.log'):
    record(a.root/'logs'/name)
for name in ('v35_xm_config', 'v36_isa_alias', 'v37_xm_batch', 'v38_q6k_xm'):
    record(a.root/'firmware'/f'lx500_pacc_qwen35_pacc0_{name}.bin')
    path = a.root/'firmware'/f'lx500_pacc_qwen35_pacc0_{name}.bin.json'
    if path.exists():
        shutil.copyfile(path, a.output/path.name)
        record(path)
for name in ('hetgpu_pacc_jobd.c', 'q6k-batch-xm.c', 'xsfmm_native_bf16.c',
             'xm-user-probe.c', 'xm-config-probe.c', 'build-xm-batch-jobd.sh'):
    record(a.root/'bin'/name)
record(a.root/'firmware'/'hetgpu_pacc_jobd_v38_q6k_xm')
(a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Wrote {len(manifest)} provenance entries to {a.output}')
