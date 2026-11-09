#!/usr/bin/env python3
"""Preserve fixed-workload results and full compressed logs, without binaries."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path('/srv/lanxin/qwen35-a3b')
repo=root/'Concordia-xm-release'
out=repo/'tools/pacc-qwen35/evidence-ngl-sweep-v20'
out.mkdir(exist_ok=False)
files={}
def record(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    files[str(path)]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
def copy(path,dest):
    record(path)
    if path.suffix=='.log':
        with path.open('rb') as src,gzip.open(str(dest)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
    else:shutil.copy2(path,dest)

rows=[]
for phase in ('coarse','long'):
    progress=root/f'logs/ngl-sweep-{phase}-progress.json'
    copy(progress,out/progress.name)
    for run in json.loads(progress.read_text())['records']:
        summary=Path(run['summary']);dest=out/summary.parent.name;dest.mkdir()
        for path in sorted(summary.parent.iterdir()):
            if path.is_file() and path.suffix in ('.json','.log'):copy(path,dest/path.name)
        r=json.loads(summary.read_text())
        row=dict(run,phase=phase)
        row['errors']=[dict(device=i,error=q.get('error'),first_mismatch=q.get('first_mismatch'))
                       for i,q in enumerate(r.get('requests',[])) if not q.get('all_correct')]
        row['profile_errors']=[dict(device=p.get('device'),error=p.get('error'),returncode=p.get('returncode'))
                              for p in r.get('phase_profiles',[]) if p.get('error')]
        row['device_phase_sum_us']={key:sum(p.get('phases_us',{}).get(key,{}).get('sum_us',0)
            for p in r.get('phase_profiles',[])) for key in
            ('input_copy','compute','output_copy','completion_publish','device_before_publish')}
        rows.append(row)
for name in ('seq512-fast-cpu-reference-20261108.json','ngl-sweep-long-cpu.json','ngl-sweep-long-cpu.log'):
    copy(root/'logs'/name,out/name)
for name in ('cross-fast.json','ngl-sweep-long.json'):
    copy(root/'bin'/name,out/name)
for device in range(4):
    path=root/f'logs/ngl-sweep-postguard-pacc{device}.log'
    assert path.read_text().splitlines()[-1].startswith(f'PASS XM/RMS alternating device={device} repeats=64')
    copy(path,out/path.name)
for name in ('libnvcuda.so','libhetgpu_cuda_shim.so'):record(root/'runtime-memory-v20'/name)
for name in ('libllama.so.0.3.0','qwen-multiseq-v5'):record(root/'build-seq512-memory/bin'/name)
for name in ('libggml.so','libggml-base.so','libggml-cpu.so','libggml-cuda.so'):
    record(root/'llama.cpp/build-pacc-cuda-gcc14/bin'/name)
for name in ('sweep-qwen-multiseq.cpp','run-ngl-sweep.py','run-four-pacc-requests.py',
             'test-four-request-summary.py','build-ngl-sweep.sh','collect-ngl-sweep-evidence.py'):
    record(repo/'tools/pacc-qwen35'/name)
state=json.loads((root/'logs/recovery-20261108/state.json').read_text())
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot==state['boot_id']
assert Path('/proc/4807/comm').read_text().strip()=='llama-server'
assert subprocess.run(['systemctl','is-active','--quiet','pacc-owned-ddr-pin-20261108.service']).returncode==0
model=root/'models/Qwen3.5-35B-A3B-Q4_K_M.gguf'
manifest=dict(boot_id=boot,cpu_server_pid=4807,state=state,files=files,
    parent_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    model=dict(path=str(model),bytes=model.stat().st_size,mtime_ns=model.stat().st_mtime_ns),
    limitations=['Four workers x two sequences, not 2048 concurrency.',
                 'Device phase sums cover PRELOADED jobs including startup, not all ELF work or pure decode.',
                 'Positive ngl uses streaming telemetry; CPU baseline does not.',
                 'CPU server and unrelated host services were not stopped.'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print(out)
