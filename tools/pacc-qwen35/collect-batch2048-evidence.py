#!/usr/bin/env python3
"""Archive CPU capacity evidence separately from unperformed accelerator runs."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path('/srv/lanxin/qwen35-a3b')
repo=root/'Concordia-xm-release';out=repo/'tools/pacc-qwen35/evidence-batch2048-capacity'
out.mkdir(exist_ok=False);manifest={}
def record(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
    manifest[str(path)]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
def copy(path,name=None):
    record(path);shutil.copy2(path,out/(name or path.name))
for name in ('batch2048-capacity-20261108.json','batch2048-capacity-seq512-20261108.json',
             'batch2048-cpu-allocation-20261108.json','batch2048-cpu-allocation-20261108.log',
             'seq512-fast-cpu-reference-20261108.json','seq512-fast-cpu-reference-20261108.log',
             'build-seq512-20261108.log'):
    copy(root/'logs'/name)
for file in (root/'logs/batch2048-four-cpu-capacity-20261108').iterdir():
    if file.suffix in ('.json','.log','.ready'):copy(file,'four-'+file.name)
original=json.loads((root/'logs/cross-fast-cpu.json').read_text())
candidate=json.loads((root/'logs/seq512-fast-cpu-reference-20261108.json').read_text())
assert candidate['ngl']==0 and candidate['reference_complete']
assert all(candidate[k]==original[k] for k in ('tokens','input_tokens','config','model'))
(out/'cpu-reference-comparison.json').write_text(json.dumps(dict(passed=True,
    tokens=candidate['tokens'],scope='two-request CPU regression only; no PACC TPS'),indent=2)+'\n')
for directory in ('src','include'):
    for file in sorted((root/'seq512-source'/directory).rglob('*')):
        if file.is_file():record(file)
for name in ('libllama.so.0.3.0','qwen-capacity-probe','qwen-multiseq-v4'):
    record(root/'build-seq512/bin'/name)
for name in ('libggml.so','libggml-base.so','libggml-cpu.so','libggml-cuda.so'):
    record(root/'llama.cpp/build-pacc-cuda-gcc14/bin'/name)
record(root/'llama.cpp/src/llama-cparams.h')
record(root/'runtime-xm-request-batch-v18/libnvcuda.so')
record(root/'build-seq512/CMakeCache.txt')
for name in ('check-request-capacity.py','seq512-CMakeLists.txt','seq512-llama-max-seq.patch',
             'seq512-qwen-multiseq.cpp','qwen-capacity-probe.cpp','probe-four-context-capacity.py'):
    record(repo/'tools/pacc-qwen35'/name)
(out/'manifest.json').write_text(json.dumps(dict(files=manifest,
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    parent_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
    accelerator_execution_performed=False),indent=2)+'\n')
print(out)
