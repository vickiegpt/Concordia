#!/usr/bin/env python3
"""Export selected measurements with raw hashes, not firmware or model binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=Path('/srv/lanxin/qwen35-a3b'))
p.add_argument('--output', type=Path, required=True)
p.add_argument('--profiles', nargs='+', required=True)
p.add_argument('--cohorts', nargs='+', required=True)
p.add_argument('--images', nargs='+', required=True)
p.add_argument('--incomplete-profiles', nargs='*', default=[])
p.add_argument('--attachments', nargs='*', default=[])
p.add_argument('--inputs', nargs='*', type=Path, default=[])
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
manifest = []


def record(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    manifest.append(dict(path=str(path), size=path.stat().st_size, sha256=h.hexdigest()))
    return h.hexdigest()


for name in a.profiles + a.incomplete_profiles:
    path = a.root/'logs'/(name+'.json')
    data = json.loads(path.read_text())
    sha = record(path)
    complete = data.get('capture_valid') and not data.get('observed_index_gaps')
    if not complete and name not in a.incomplete_profiles:
        raise SystemExit(f'Incomplete evidence: {path}')
    summary = {k: v for k, v in data.items()
               if k not in ('raw_records', 'submissions', 'submission_builds')}
    summary['build_values'] = sorted(set(data['submission_builds'].values()))
    summary['capability_events'] = [r for r in data['raw_records']
                                   if 0x5320 <= r['status'] < 0x5330 or 0x5400 <= r['status'] < 0x5448]
    summary['source_sha256'] = sha
    summary['diagnostic_coverage'] = 'complete' if complete else 'incomplete; do not use as a complete timing breakdown'
    (a.output/(name+'.summary.json')).write_text(json.dumps(summary, indent=2)+'\n')
    for suffix in ('.log', '.summary.json', '.xm.json'):
        detail = path.with_suffix(suffix)
        if detail.exists():
            record(detail)
            destination = name+'.calls.json' if suffix == '.summary.json' else detail.name
            shutil.copyfile(detail, a.output/destination)
for name in a.cohorts:
    path = a.root/'logs'/(name+'.json')
    data = json.loads(path.read_text())
    if not data['all_correct'] or not data['aggregate_output_tps']:
        raise SystemExit(f'Invalid token cohort: {path}')
    record(path)
    shutil.copyfile(path, a.output/path.name)
    record(path.with_suffix('.server.log'))
for version in a.images:
    image = a.root/'firmware'/f'lx500_pacc_qwen35_pacc0_{version}.bin'
    record(image)
    sidecar = image.with_suffix('.bin.json')
    record(sidecar)
    shutil.copyfile(sidecar, a.output/sidecar.name)
# These are the current v46 source files, not reconstructed intermediate versions.
for name in ('hetgpu_pacc_jobd.c', 'q6k-batch-xm.c', 'q6k-xm-workers.c',
             'q6k-xm-profile.h', 'payload-copy-rvv.c', 'xsfmm_native_bf16.c',
             'build-xm-batch-jobd.sh', 'bench-qwen-requests.py', 'run-xm-qwen-benchmark.sh',
             'profile-jobd-phases.py', 'run-xm-validation.sh', 'summarize-xm-detail.py',
             'summarize-jobd-phases.py', 'collect-xm-performance-evidence.py',
             'test-payload-copy.c', 'test-xm-reject.c', 'test_jobd_q6k.c'):
    record(a.root/'bin'/name)
record(a.root/'firmware/hetgpu_pacc_jobd_v46_xm_reduce')
for relative in a.attachments:
    path = a.root/relative
    record(path)
    shutil.copyfile(path, a.output/path.name)
for path in a.inputs:
    record(path)
(a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Exported {len(manifest)} provenance entries to {a.output}')
