#!/usr/bin/env python3
"""CPU-independent CUDA-quantization oracle, then strict four-device validation."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path('/srv/lanxin/qwen35-a3b')
p=argparse.ArgumentParser()
p.add_argument('prefix')
p.add_argument('--ngl',type=int,choices=(1,2),required=True)
p.add_argument('--case',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--harness',type=Path,required=True)
p.add_argument('--ubatch',type=int,choices=(2,4,8),required=True)
p.add_argument('--op-offload',choices=('0','1'),default='0')
p.add_argument('--timeout',type=int,default=1800)
a=p.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_-]+',a.prefix)
reference=ROOT/f'logs/{a.prefix}-cpu.json'
assert not reference.exists()
env=os.environ.copy()
for key in ('LD_PRELOAD','QWEN_BARRIER_DIR','QWEN_WORKER_ID',
            'QWEN_CPU_Q81_HEAD_REFERENCE','QWEN_CPU_Q81_LAYER39_REFERENCE'):
    env.pop(key,None)
env.update(HETGPU_SIFIVE_RMS_EXACT='1',HETGPU_SIFIVE_MMAP_WEIGHTS='1',
           HETGPU_SIFIVE_LAZY_ZERO='1',HETGPU_SIFIVE_QK_VECTOR_REDUCE='1',
           QWEN_OP_OFFLOAD=a.op_offload)
cpu=env.copy()
cpu.update(QWEN_CPU_Q81_HEAD_REFERENCE='1',
    LD_LIBRARY_PATH=f'{ROOT}/build-seq512-memory/bin:{ROOT}/llama.cpp/build-pacc-cuda-gcc14/bin')
if a.ngl==2:cpu['QWEN_CPU_Q81_LAYER39_REFERENCE']='1'
with (ROOT/f'logs/{a.prefix}-cpu.log').open('x') as log:
    subprocess.run([str(a.harness),str(ROOT/'models/Qwen3.5-35B-A3B-Q4_K_M.gguf'),
        '0',str(a.case),str(reference),'-',str(a.ubatch)],env=cpu,stdout=log,
        stderr=subprocess.STDOUT,timeout=a.timeout,check=True)
assert json.loads(reference.read_text()).get('reference_complete') is True
with (ROOT/f'logs/{a.prefix}-driver.log').open('x') as log:
    result=subprocess.run(['python3',str(ROOT/'bin/run-four-pacc-requests.py'),a.prefix,
        '--runtime',str(a.runtime),'--harness',str(a.harness),
        '--wrapper',str(ROOT/'bin/run-pacc-recovered.py'),
        '--llama-library-dir',str(ROOT/'build-seq512-memory/bin'),
        '--case',str(a.case),'--reference',str(reference),'--ubatch',str(a.ubatch),
        '--ngl',str(a.ngl),'--start-barrier','--phase-profile','qualification',
        '--timeout',str(a.timeout),'--min-available-gib','32'],env=env,
        stdout=log,stderr=subprocess.STDOUT)
print(ROOT/f'logs/{a.prefix}/summary.json',flush=True)
raise SystemExit(result.returncode)
