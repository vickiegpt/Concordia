#!/usr/bin/env python3
"""Archive failures and successes separately; keep full trace hashes on record."""
import argparse
import array
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/srv/lanxin/qwen35-a3b')
REPO=ROOT/'Concordia-xm-release'
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
p.add_argument('--runs',nargs='+',required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False)
files={}
def record(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    files[str(path)]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
def copy(path,dest):
    record(path)
    if path.suffix in ('.log','.f32') or (path.suffix in ('.json','.jsonl') and path.stat().st_size>65536):
        with path.open('rb') as src,gzip.open(str(dest)+'.gz','wb') as dst:
            shutil.copyfileobj(src,dst)
    else:shutil.copy2(path,dest)
rows=[]
host_breakdown=[]
for name in a.runs:
    folder=ROOT/'logs'/name
    summary=json.loads((folder/'summary.json').read_text())
    dest=a.output/name;dest.mkdir()
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix in ('.log','.json'):copy(path,dest/path.name)
    rows.append(dict(run=name,all_correct=summary['all_correct'],
        decode_tps=summary.get('aggregate_decode_tps'),output_tps=summary.get('aggregate_output_tps'),
        errors=[r.get('first_mismatch',r.get('error')) for r in summary.get('requests',[]) if not r.get('all_correct')]))
    if summary['all_correct']:
        head_us=decode_us=0.0
        for device in range(4):
            result=json.loads((folder/f'pacc{device}-result.json').read_text())
            calls=[]
            for line in (folder/f'pacc{device}-phases.log').read_text().splitlines():
                if line.startswith('PACC Q6_KxQ8_1 '):
                    fields=dict(item.split('=',1) for item in line.split()[2:])
                    if fields.get('m')=='248320' and fields.get('completion')=='verified':calls.append(fields)
            steps=result['steps']
            assert result['ubatch']==result['sequences'] and len(calls)>=len(steps)>0
            selected=calls[-len(steps):]
            assert all(int(c['n'])==result['sequences'] for c in selected)
            head_us+=sum(float(c['elapsed_ms'])*1000 for c in selected)
            decode_us+=sum(s['elapsed_us'] for s in steps)
        host_breakdown.append(dict(run=name,q6_head_host_elapsed_us=head_us,
            decode_call_host_elapsed_us=decode_us,fraction=head_us/decode_us,
            scope='Sum of per-worker decode-call times; head includes staging, submission, wait and readback, NOT device compute time'))
names=['head-oracle-pacc1','head-oracle-rne','cpu-q81-long-v21','cpu-q81-layer39-long-v23',
       'exact-v23-single-ngl2','compare-layer39-rne','compare-q81-v23-final','compare-q81-noauto-final',
       'build-rne-exact-v23','build-numerical-tools-v23-check']
names += [f'test-rms-exact-v23-pacc{i}' for i in range(4)]
names += [f'{prefix}-pacc{i}' for prefix in ('test-rms-exact-v23-post','test-q6-v23-post') for i in range(4)]
names += [name+'-cpu' for name in a.runs if (ROOT/f'logs/{name}-cpu.json').exists()]
for name in names:
    for suffix in ('.json','.log'):
        path=ROOT/f'logs/{name}{suffix}'
        if path.exists():copy(path,a.output/path.name)
for name in ('logit-cpu-v1','logit-pacc1-v1','logit-pacc2-v1',
             'trace-q81-cpu-long-v23','trace-q81-pacc1-long-v23',
             'trace-q81-cpu-layers-v23','trace-q81-pacc1-auto-layers-v23','trace-q81-pacc1-noauto-v23'):
    copy(ROOT/f'logs/{name}.json',a.output/f'{name}.json')
    source=ROOT/'logs'/name;dest=a.output/name;dest.mkdir()
    copy(source/'tensors.jsonl',dest/'tensors.jsonl')
    for path in sorted(source.glob('*.f32')):
        record(path)
        # First divergent layer boundary, before the later greedy-token mismatch.
        if path.name.startswith('19-') and ('l_out-18-' in path.name or 'l_out-19-' in path.name):
            copy(path,dest/path.name)
        # Full failing logits for the original native-CPU comparison and same-math diagnosis.
        if path.name=='logits-1-7.f32' and name in ('logit-cpu-v1','logit-pacc1-v1'):
            copy(path,dest/path.name)
        if path.name=='logits-1-20.f32' and name in ('trace-q81-cpu-long-v23','trace-q81-pacc1-long-v23','trace-q81-pacc1-noauto-v23'):
            copy(path,dest/path.name)
for file in ('runtime-rne-exact-v23/libnvcuda.so','runtime-rne-exact-v23/libhetgpu_cuda_shim.so',
             'build-seq512-memory/bin/qwen-q81-validation-v3','build-seq512-memory/bin/trace-multiseq-v3'):
    record(ROOT/file)
for path in REPO.joinpath('tools/pacc-qwen35').glob('*'):
    if path.is_file() and path.name in ('cpu-q81-head.h','qwen-q81-validation.cpp','trace-multiseq.cpp',
        'run-four-pacc-requests.py','test-rms-exact.py','run-numerical-eval.py'):record(path)
state=json.loads((ROOT/'logs/recovery-20261108/state.json').read_text())
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot==state['boot_id']
assert Path('/proc/4807/comm').read_text().strip()=='llama-server'
assert subprocess.run(['systemctl','is-active','--quiet','pacc-owned-ddr-pin-20261108.service']).returncode==0
manifest=dict(boot_id=boot,cpu_server_pid=4807,allocation=state,files=files,
    parent_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    limitations=['CUDA Q8_1 reference differs from native CPU Q8_K activations.',
        'Trace-only teacher-forcing runs never contribute TPS.',
        'Phase sampler failures are retained; qualified runs do not claim device phase accounting.',
        'NGL1 is output-head offload; NGL2 is layer39 plus output-head offload.',
        'No 2048-request or 500 TPS claim.'])
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
(a.output/'host-head-timing.json').write_text(json.dumps(host_breakdown,indent=2)+'\n')
cpu=array.array('f');cpu.frombytes((ROOT/'logs/trace-q81-cpu-long-v23/logits-1-20.f32').read_bytes())
comparisons={}
for name in ('trace-q81-pacc1-long-v23','trace-q81-pacc1-noauto-v23'):
    actual=array.array('f');actual.frombytes((ROOT/f'logs/{name}/logits-1-20.f32').read_bytes())
    assert len(actual)==len(cpu) and all(math.isfinite(v) for v in actual)
    delta=[float(x)-y for x,y in zip(actual,cpu)]
    comparisons[name]=dict(max_abs=max(map(abs,delta)),
        relative_rms=math.sqrt(sum(v*v for v in delta)/sum(v*v for v in cpu)),
        expected_argmax=max(range(len(cpu)),key=cpu.__getitem__),
        actual_argmax=max(range(len(actual)),key=actual.__getitem__))
(a.output/'same-math-logit-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print(json.dumps(rows,indent=2))
