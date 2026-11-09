#!/usr/bin/env python3
"""Sequential fixed-workload sweep; persist failed points without inventing TPS."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT=Path('/srv/lanxin/qwen35-a3b')
p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['coarse','long'],required=True)
p.add_argument('--ngls',required=True)
p.add_argument('--repeats',type=int,default=1)
a=p.parse_args();ngls=[int(s) for s in a.ngls.split(',')]
lock=(ROOT/'logs/ngl-sweep.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert ngls and len(set(ngls))==len(ngls) and all(0<=n<=99 for n in ngls) and 1<=a.repeats<=5
assert a.phase!='coarse' or a.repeats==1, 'coarse names are single-pass; use long for repeats'
case=ROOT/'bin'/('cross-fast.json' if a.phase=='coarse' else 'ngl-sweep-long.json')
reference=ROOT/'logs'/('seq512-fast-cpu-reference-20261108.json' if a.phase=='coarse' else 'ngl-sweep-long-cpu.json')
ref=json.loads(reference.read_text());assert ref['reference_complete'] and ref['config']==json.loads(case.read_text())
env=os.environ.copy();env.update(HETGPU_SIFIVE_MMAP_WEIGHTS='1',HETGPU_SIFIVE_LAZY_ZERO='1',HETGPU_SIFIVE_QK_VECTOR_REDUCE='1')
records=[];summary=ROOT/'logs'/f'ngl-sweep-{a.phase}-progress.json'
for repeat in range(a.repeats):
    # Reverse the second pass so one configuration is not always measured first.
    for ngl in ngls if repeat%2==0 else list(reversed(ngls)):
        suffix=f'-r{repeat}' if a.phase=='long' else ''
        name=f'ngl-sweep-{a.phase}-{ngl}{suffix}'
        path=ROOT/'logs'/name/'summary.json'
        before=time.monotonic()
        if path.exists():
            settings=json.loads(path.with_name('settings.json').read_text())
            assert settings['ngl']==ngl and settings['start_barrier']
            assert settings['ubatch']==2 and settings['runtime']==str(ROOT/'runtime-memory-v20')
            assert settings['harness']==str(ROOT/'build-seq512-memory/bin/qwen-multiseq-v5')
            hashes=json.loads(path.with_name('artifacts.json').read_text())
            for artifact in (case,reference,Path(settings['harness'])):
                assert hashes[str(artifact.resolve())]==hashlib.sha256(artifact.read_bytes()).hexdigest(), f'changed artifact: {artifact}'
        else:
            with (ROOT/'logs'/f'{name}-driver.log').open('x') as log:
                command=['python3',str(ROOT/'bin/run-four-pacc-requests.py'),name,
                    '--runtime',str(ROOT/'runtime-memory-v20'),'--wrapper',str(ROOT/'bin/run-pacc-recovered.py'),
                    '--harness',str(ROOT/'build-seq512-memory/bin/qwen-multiseq-v5'),
                    '--llama-library-dir',str(ROOT/'build-seq512-memory/bin'),
                    '--case',str(case),'--reference',str(reference),'--ubatch','2','--ngl',str(ngl),
                    '--start-barrier','--timeout','900','--min-available-gib','32']
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
        if not path.exists():raise RuntimeError(f'runner did not produce {path}')
        r=json.loads(path.read_text());offloaded=[]
        for device in range(4):
            file=path.parent/f'pacc{device}-phases.log'
            if ngl==0:file=path.parent/f'pacc{device}-driver.log'
            if not file.exists():continue
            for line in file.read_text(errors='replace').splitlines():
                if 'offloaded ' in line and '/41 layers' in line:offloaded.append(line.strip())
        record=dict(ngl=ngl,repeat=repeat,passed=r.get('all_correct',False),
            tokens_correct=r.get('tokens_correct',len(r.get('requests',[]))==4 and all(x.get('all_correct') for x in r.get('requests',[]))),
            xm_batch_verified=r.get('xm_batch_verified',ngl>0 and r.get('all_correct',False)),
            xm_columns_observed=ngl>0 and len(r.get('xm_batch_proof',[]))==4 and all(x.get('multi_tiles',0)>0 and x.get('max_n',0)>1 for x in r.get('xm_batch_proof',[])),
            decode_tps=r.get('aggregate_decode_tps'),output_tps=r.get('aggregate_output_tps'),
            token_matched_decode_tps=r.get('token_matched_decode_tps'),
            decode_tokens=r.get('decode_tokens'),decode_window_s=r.get('decode_window_s'),
            min_mem_available_kib=r.get('min_mem_available_kib'),elapsed_s=time.monotonic()-before,
            summary=str(path),offloaded=offloaded,scope=r.get('scope'))
        records.append(record)
        grouped={}
        for n in ngls:
            rows=[row for row in records if row['ngl']==n]
            valid=[row['decode_tps'] for row in rows if row['passed'] and row['decode_tps'] is not None]
            grouped[str(n)]=dict(completed=len(rows),valid=len(valid),median_decode_tps=statistics.median(valid) if valid and len(valid)==len(rows) else None)
        summary.write_text(json.dumps(dict(phase=a.phase,case=str(case),records=records,grouped=grouped),indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if r.get('runner_timed_out') or any(p.get('timed_out') for p in r.get('phase_profiles',[])) or r.get('stage')=='hardware/profile qualification':
            raise RuntimeError('stop sweep after timeout or failed hardware qualification')
