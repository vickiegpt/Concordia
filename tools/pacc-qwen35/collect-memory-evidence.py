#!/usr/bin/env python3
"""Archive PSS-based memory tests separately from inference correctness/TPS."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path('/srv/lanxin/qwen35-a3b')
repo = root / 'Concordia-xm-release'
out = repo / 'tools/pacc-qwen35/evidence-memory-v20'
out.mkdir(exist_ok=False)
files = {}
def record(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''): h.update(data)
    files[str(path)] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
def copy(path, dest):
    record(path); shutil.copy2(path, dest)
def tail(path, dest, count=80):
    record(path)
    dest.write_text('\n'.join(path.read_text(errors='replace').splitlines()[-count:]) + '\n')

for run in ('memory-four-fast-v19', 'memory-four-fast-v20', 'memory-four-512-v20',
            'memory-baseline-one-v20', 'memory-mmap-four-v20', 'memory-both-four-v20'):
    dest = out / run; dest.mkdir()
    for file in sorted((root / 'logs' / run).iterdir()):
        if file.suffix in ('.json', '.ready'): copy(file, dest / file.name)
        elif file.suffix == '.log': tail(file, dest / (file.stem + '-tail.log'), 160)
for name in ('test-lazy-host-memory.json', 'test-lazy-host-memory-v20.json',
             'test-pacc-mmap-weights.log', 'memory-fast-v20-live-smaps.json'):
    copy(root / 'logs' / name, out / name)
for name in ('build-memory-v19.log', 'build-memory-v20.log', 'build-seq512-memory.log'):
    tail(root / 'logs' / name, out / name)
for device in range(4):
    path = root / f'logs/memory-postguard-v20-pacc{device}.log'
    assert path.read_text().splitlines()[-1].startswith(f'PASS XM/RMS alternating device={device} repeats=64')
    tail(path, out / path.name, 8)
for name in ('libnvcuda.so', 'libhetgpu_cuda_shim.so'):
    record(root / 'runtime-memory-v20' / name)
for name in ('libllama.so.0.3.0', 'qwen-multiseq-v4', 'qwen-memory-probe'):
    record(root / 'build-seq512-memory/bin' / name)
for name in ('libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'libggml-cuda.so'):
    record(root / 'llama.cpp/build-pacc-cuda-gcc14/bin' / name)
for name in ('llama-model.cpp', 'pacc-mmap-weights.cpp', 'pacc-mmap-weights.h', 'llama-cparams.h'):
    record(root / 'seq512-memory-source/src' / name)
for name in ('memory.rs', 'function.rs'): record(repo / 'zluda/src/impl' / name)
for name in ('pacc-mmap-weights.cpp', 'pacc-mmap-weights.h', 'seq512-mmap-weights.patch',
             'seq512-CMakeLists.txt', 'run-memory-capacity.py', 'qwen-memory-probe.cpp',
             'run-four-pacc-requests.py', 'test-lazy-host-memory.py', 'test-pacc-mmap-weights.cpp',
             'build-memory-candidate.sh', 'collect-memory-evidence.py'):
    record(repo / 'tools/pacc-qwen35' / name)
state = json.loads((root / 'logs/recovery-20261108/state.json').read_text())
boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot == state['boot_id']
assert Path('/proc/4807/comm').read_text().strip() == 'llama-server'
assert subprocess.run(['systemctl', 'is-active', '--quiet', 'pacc-owned-ddr-pin-20261108.service']).returncode == 0
record(Path('/lib/firmware/lanxin/lx500_pacc.bin'))
assert files['/lib/firmware/lanxin/lx500_pacc.bin']['sha256'] == state['vendor_sha256']
(out / 'manifest.json').write_text(json.dumps(dict(boot_id=boot, cpu_server_pid=4807,
    parent_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    state=state, files=files), indent=2) + '\n')
print(out)
