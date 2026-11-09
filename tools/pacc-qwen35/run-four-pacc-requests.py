#!/usr/bin/env python3
"""Four devices, independent multi-sequence workers, fail-closed aggregate TPS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT=Path('/srv/lanxin/qwen35-a3b')

def available_kib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])
    raise RuntimeError('MemAvailable is missing')

def aggregate(reference,results,profiles,proofs,codes):
    config=reference['config'];count=config['max_tokens'];batch=len(config['prompts'])
    valid=(len(results)==len(profiles)==len(proofs)==len(codes)==4 and not any(codes) and
           reference.get('ngl')==0 and reference.get('reference_complete') is True and
           not config.get('diagnostic',False) and
           all(r.get('all_correct') is True and r.get('ngl')==99 and not r.get('diagnostic',False) and
               r.get('sequences')==batch and r.get('model')==reference.get('model') and
               r.get('config')==config and r.get('tokens')==reference['tokens'] and
               r.get('input_tokens')==reference['input_tokens'] and
               r.get('output_tokens')==batch*count and r.get('decode_tokens')==batch*(count-1) and
               all(len(v)==count for v in r.get('tokens',[])) for r in results) and
           {p.get('device') for p in profiles}=={0,1,2,3} and
           all(p.get('capture_valid') is True and p.get('build_matches') is True for p in profiles) and
           {p.get('device') for p in proofs}=={0,1,2,3} and
           all(p.get('multi_tiles',0)>0 and p.get('max_n',0)>1 for p in proofs))
    out=dict(all_correct=valid,aggregate_output_tps=None,aggregate_decode_tps=None,
             scope='four hardware workers, real cross-request XM columns; no CPU matrix fallback')
    if valid:
        start=min(r['started_monotonic_us'] for r in results)
        dec=min(r['decode_started_monotonic_us'] for r in results)
        end=max(r['finished_monotonic_us'] for r in results)
        if not start<dec<end:raise ValueError('invalid shared-clock interval')
        out.update(output_tokens=4*batch*count,decode_tokens=4*batch*(count-1),
                   request_window_s=(end-start)/1e6,decode_window_s=(end-dec)/1e6,
                   aggregate_output_tps=4*batch*count*1e6/(end-start),
                   aggregate_decode_tps=4*batch*(count-1)*1e6/(end-dec))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('prefix');p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=21600);p.add_argument('--ubatch',type=int,default=8)
    p.add_argument('--wrapper',type=Path,default=ROOT/'bin/run-pacc-isolated.sh')
    p.add_argument('--harness',type=Path,default=ROOT/'bin/qwen-multiseq-v3')
    p.add_argument('--llama-library-dir',type=Path)
    p.add_argument('--min-available-gib',type=int,default=16)
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.prefix):p.error('invalid prefix')
    config=json.loads(a.case.read_text());reference=json.loads(a.reference.read_text())
    if (reference.get('config')!=config or reference.get('ngl')!=0 or
            not reference.get('reference_complete') or config.get('diagnostic',False) or
            len(config['prompts'])<2 or not 2<=a.ubatch<=8 or a.min_available_gib<1):p.error('invalid CPU reference or batch')
    floor_kib=a.min_available_gib*1024*1024
    if available_kib()<floor_kib:p.error('insufficient available memory before launch')
    out=ROOT/'logs'/a.prefix;out.mkdir(exist_ok=False)
    env=os.environ.copy();env['CONCORDIA_RELEASE']=str(a.runtime.resolve())
    wrapper=[('python3' if a.wrapper.suffix=='.py' else 'bash'),str(a.wrapper.resolve())]
    if a.llama_library_dir:
        env['LLAMA_RUNTIME_LIBRARY_DIR']=str(a.llama_library_dir.resolve())
    for key in ('HETGPU_SIFIVE_QK_REQUEST_BATCH','HETGPU_SIFIVE_QK_LOCAL_BATCH',
                'HETGPU_SIFIVE_QK_VECTOR_PACK','HETGPU_SIFIVE_QK_EXPERT_BATCH'):env[key]='1'
    for key in ('HETGPU_SIFIVE_QK_PROFILE','HETGPU_SIFIVE_ELF_PROFILE',
                'HETGPU_SIFIVE_LOG_KERNEL_LAUNCHES'):env[key]='0'
    env['Q8_TEST_LIBRARY']=str(a.runtime.resolve()/'libnvcuda.so')
    settings={key:env.get(key,'0') for key in (
        'HETGPU_SIFIVE_QK_REQUEST_BATCH','HETGPU_SIFIVE_QK_LOCAL_BATCH',
        'HETGPU_SIFIVE_QK_VECTOR_PACK','HETGPU_SIFIVE_QK_EXPERT_BATCH',
        'HETGPU_SIFIVE_QK_VECTOR_REDUCE','HETGPU_SIFIVE_QK_PROFILE','HETGPU_SIFIVE_ELF_PROFILE',
        'HETGPU_SIFIVE_MMAP_WEIGHTS','HETGPU_SIFIVE_LAZY_ZERO')}
    settings.update(ubatch=a.ubatch,timeout_s=a.timeout,runtime=str(a.runtime.resolve()))
    settings.update(wrapper=str(a.wrapper.resolve()),harness=str(a.harness.resolve()),
                    llama_library_dir=env.get('LLAMA_RUNTIME_LIBRARY_DIR'),
                    min_available_gib=a.min_available_gib)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    artifacts={}
    files=[a.runtime/'libnvcuda.so',a.runtime/'libhetgpu_cuda_shim.so',a.harness,a.wrapper,a.case,a.reference]
    if a.llama_library_dir:files.append(a.llama_library_dir/'libllama.so')
    for file in files:
        artifacts[str(file.resolve())]=hashlib.sha256(file.read_bytes()).hexdigest()
    (out/'artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    for device in range(4):
        with (out/f'pacc{device}-qualification-driver.log').open('x') as log:
            qualification=subprocess.run([*wrapper,str(device),'--','python3',
                str(ROOT/'bin/profile-jobd-stream.py'),'--device',str(device),'--require-build','0x20036',
                '--timeout','180','--output',str(out/f'pacc{device}-qualification.json'),'--',
                str(ROOT/'bin/test_jobd_q8'),'--hardware',str(device)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if qualification.returncode:
            report=dict(all_correct=False,aggregate_output_tps=None,aggregate_decode_tps=None,
                        stage='hardware/profile qualification',failed_device=device,
                        returncode=qualification.returncode)
            (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
            return 1
    processes=[];logs=[];started=time.monotonic()
    min_available=available_kib();memory_guard_triggered=False
    def interrupt(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        for device in range(4):
            log=(out/f'pacc{device}-driver.log').open('x');logs.append(log)
            cmd=[*wrapper,str(device),'--','python3',
                 str(ROOT/'bin/profile-jobd-stream.py'),'--device',str(device),'--require-build','0x20036',
                 '--timeout',str(a.timeout),'--output',str(out/f'pacc{device}-phases.json'),'--',
                 str(a.harness.resolve()),str(ROOT/'models/Qwen3.5-35B-A3B-Q4_K_M.gguf'),
                 '99',str(a.case.resolve()),str(out/f'pacc{device}-result.json'),str(a.reference.resolve()),str(a.ubatch)]
            processes.append(subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
        while any(x.poll() is None for x in processes):
            min_available=min(min_available,available_kib())
            memory_guard_triggered=min_available<floor_kib
            if memory_guard_triggered or any(x.poll() not in (None,0) for x in processes):
                for x in processes:
                    if x.poll() is None:x.terminate()
                break
            time.sleep(1)
        codes=[x.wait(timeout=30) for x in processes]
    finally:
        for x in processes:
            if x.poll() is None:
                x.terminate()
                try:x.wait(20)
                except subprocess.TimeoutExpired:x.kill();x.wait()
        for log in logs:log.close()
    def load(suffix):
        return [json.loads(file.read_text()) for i in range(4) if (file:=out/f'pacc{i}-{suffix}.json').exists()]
    proofs=[]
    for i in range(4):
        proof=dict(device=i,calls=0,multi_tiles=0,max_n=0)
        file=out/f'pacc{i}-phases.log'
        if file.exists():
            with file.open(errors='replace') as stream:
                for line in stream:
                    if not line.startswith('PACC_XM_REQUEST_BATCH '):continue
                    fields=dict(part.split('=',1) for part in line.split()[1:])
                    if fields.get('completion')!='verified':continue
                    proof['calls']+=1;proof['multi_tiles']+=int(fields['multi_tiles'])
                    proof['max_n']=max(proof['max_n'],int(fields['max_n']))
        proofs.append(proof)
    results,profiles=load('result'),load('phases')
    report=aggregate(reference,results,profiles,proofs,codes)
    if memory_guard_triggered:
        report.update(all_correct=False,aggregate_output_tps=None,aggregate_decode_tps=None,
                      failure='available memory crossed the configured floor')
    report.update(returncodes=codes,requests=results,phase_profiles=profiles,xm_batch_proof=proofs,
                  elapsed_including_load_s=time.monotonic()-started,
                  min_mem_available_kib=min_available,memory_guard_triggered=memory_guard_triggered)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('requests','phase_profiles')},indent=2))
    return 0 if report['all_correct'] else 1

if __name__=='__main__':raise SystemExit(main())
