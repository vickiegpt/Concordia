#!/usr/bin/env python3
"""Archive bounded evidence and hashes; full logs remain on the evaluation host."""
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/srv/lanxin/qwen35-a3b')
repo=ROOT/'Concordia-xm-release'
out=repo/'tools/pacc-qwen35/evidence-xm-requests'
out.mkdir(exist_ok=False)
manifest={}
def record(file):
    h=hashlib.sha256()
    with file.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    manifest[str(file)]=dict(bytes=file.stat().st_size,sha256=h.hexdigest())
def copy(file,name=None):
    if file.exists():record(file);shutil.copy2(file,out/(name or file.name))
def excerpt(file,name):
    record(file);first=[];last=deque(maxlen=24);tokens=[];errors=[];counts={}
    with file.open(errors='replace') as stream:
        for line in stream:
            if line.startswith('MULTISEQ_'):tokens.append(line)
            if any(v in line for v in ('nonfinite','ELF submit failed','PACC request batch rejected','PACC RoPE unsupported')):
                errors.append(line[:2000])
            if line.startswith(('PACC_XM_REQUEST_BATCH ','PACC_QK_PROFILE ','PACC GDN ','PACC RoPE ')):
                key=line.split()[0];counts[key]=counts.get(key,0)+1
                if len(first)<24:first.append(line)
                last.append(line)
    (out/name).write_text(''.join(first+['\nLAST EVENTS\n']+list(last)+['\nTOKENS\n']+tokens+['\nERRORS\n']+errors))
    return counts

files=['cross-long-cpu.json','cross-eight-cpu.json','cross-fast-cpu.json','cross-diagnostic-cpu.json',
       'cross-diagnostic-pacc-v10.json','gdn-local-baseline-v10.json','gdn-local-candidate-v11.json',
       'test-cross-argsort-v8.log','test-topk-v8.log','test-request-q4-v8.log','test-request-q5-v10.log',
       'test-request-q8-v12.log','test-cross-layout-v8-retry.log','test-cross-reduce-v10.log',
       'test-cross-mmf-v12.log','test-rms-batch-pacc3-v11.log','stream-qualification-pacc1.json',
       'cross-regressions-v16.log','cross-regressions-v18.log','test-elf-error-v16.log',
       'test-completion-contract-v16.log','test-runtime-v18.log',
       'bench-xm-columns-q4-v15-isolated.json','bench-xm-columns-q4-v15-isolated.log',
       'bench-xm-columns-q4-v15-isolated-phases.json','four-xm-rms-v14-driver.log',
       'four-xm-rms-v18-driver.log']
for name in files:copy(ROOT/'logs'/name)
for prefix in sys.argv[1:]:
    directory=ROOT/'logs'/prefix
    for file in directory.glob('*.json'):copy(file,prefix+'-'+file.name)
    for device in range(4):
        file=directory/f'pacc{device}-phases.log'
        if file.exists():excerpt(file,f'{prefix}-pacc{device}-excerpt.log')
for name in ('cross-eight-pacc3-v12.log','cross-fast-pacc-v8.log'):
    file=ROOT/'logs'/name
    if file.exists():excerpt(file,name.replace('.log','-excerpt.log'))
for pattern in ('four-xm-rms-v14-pacc*.log','four-xm-rms-v18-pacc*.log',
                'bench-xm-vector-*.json','bench-xm-vector-*.log'):
    for file in (ROOT/'logs').glob(pattern):copy(file)
for runtime in ('runtime-xm-request-batch-v12','runtime-xm-request-batch-v16','runtime-xm-request-batch-v18'):
    for name in ('libnvcuda.so','libhetgpu_cuda_shim.so'):record(ROOT/runtime/name)
for file in (ROOT/'bin').glob('qwen-multiseq-v3'):record(file)
sources=subprocess.check_output(['git','-C',str(repo),'diff','HEAD','--name-only'],text=True).splitlines()
sources+=['ext/sifive_runtime-sys/src/concat_tile.c','ext/sifive_runtime-sys/src/concat_tile.rs',
          'zluda/src/impl/sifive_concat.rs','zluda/src/impl/sifive_set_rows.rs']
for path in sources:record(repo/path)
for file in (repo/'tools/pacc-qwen35').iterdir():
    if file.is_file() and file.name!='test-cross-rope.py' and file.suffix in ('.py','.cpp','.json','.sh'):record(file)
copy(ROOT/'logs/build-cross-v16.log')
copy(ROOT/'logs/build-cross-v18.log')
record(Path('/lib/firmware/lanxin/lx500_pacc.bin'))
(out/'manifest.json').write_text(json.dumps(dict(files=manifest,
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    parent_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
print(out)
