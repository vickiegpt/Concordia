#!/usr/bin/env python3
"""Archive bounded recovery evidence; never equate capacity with throughput."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path('/srv/lanxin/qwen35-a3b')
repo = root / 'Concordia-xm-release'
out = repo / 'tools/pacc-qwen35/evidence-recovery-20261108'
out.mkdir(exist_ok=False)
manifest = {}

def record(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    manifest[str(path)] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())

def copy(path, name=None):
    record(path)
    shutil.copy2(path, out / (name or path.name))

copy(root / 'logs/recovery-20261108/state.json')
copy(root / 'logs/recovery-20261108/preflight-after.json', 'preflight-after.txt')
copy(root / 'firmware/recovery-20261108/manifest.json', 'firmware-manifest.json')
copy(root / 'logs/package-recovery-20261108.log')
for device in range(4):
    for suffix in ('json', 'log'):
        copy(root / f'logs/recovery-pacc{device}-qualification.{suffix}')
    file = root / f'logs/recovery-mixed-20261108-pacc{device}.log'
    record(file)
    # Full logs remain on the host, with an integrity hash in the manifest.
    lines = file.read_text().splitlines()
    (out / f'mixed-pacc{device}-tail.log').write_text('\n'.join(lines[-8:]) + '\n')
    file = root / f'logs/recovery-postguard-20261108-pacc{device}.log'
    record(file)
    lines = file.read_text().splitlines()
    assert lines[-1].startswith(f'PASS XM/RMS alternating device={device} repeats=64')
    (out / f'postguard-pacc{device}-tail.log').write_text('\n'.join(lines[-8:]) + '\n')
    record(root / f'firmware/recovery-20261108/pacc{device}-owned.bin')

for name in ('recovery-four-fast-20261108', 'recovery-four-512-20261108'):
    source = root / 'logs' / name
    if not source.exists():
        continue
    dest = out / name
    dest.mkdir()
    for file in sorted(source.iterdir()):
        if not file.is_file():
            continue
        record(file)
        if file.suffix == '.json':
            shutil.copy2(file, dest / file.name)
        elif file.suffix == '.log':
            lines = file.read_text(errors='replace').splitlines()
            (dest / (file.stem + '-tail.log')).write_text('\n'.join(lines[-50:]) + '\n')

for name in ('cross-512-recovery.json',):
    copy(root / 'bin' / name)
for name in ('recovery-512-cpu-reference.json', 'recovery-512-cpu-reference.log'):
    if (root / 'logs' / name).exists():
        copy(root / 'logs' / name)

for name in ('recover-pacc-ddr.py', 'package-recovery-jobd.py', 'load-recovered-pacc.py',
             'run-pacc-recovered.py', 'run-four-pacc-requests.py', 'test-four-xm-rms.py',
             'make-recovery-512-case.py', 'run-recovery-512-reference.py', 'collect-recovery-evidence.py'):
    record(root / 'bin' / name)
record(root / 'runtime-xm-request-batch-v18/libnvcuda.so')
record(root / 'build-seq512/bin/libllama.so.0.3.0')
record(root / 'build-seq512/bin/qwen-multiseq-v4')
record(Path('/lib/firmware/lanxin/lx500_pacc.bin'))
status = dict(
    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
    cpu_server_pid4807_comm=Path('/proc/4807/comm').read_text().strip(),
    pin_service_active=subprocess.run(['systemctl', 'is-active', '--quiet',
                                       'pacc-owned-ddr-pin-20261108.service']).returncode == 0,
    bound_devices=[i for i in range(4) if Path(f'/sys/bus/platform/drivers/pacc/PACC0001:0{i}').is_symlink()],
    helper_parameters={name: Path('/sys/module/hetgpu_sifive_mbox_ddr_coh/parameters', name).read_text().strip()
                       for name in ('shared_ddr_base_override', 'shared_ddr_size', 'shared_ddr_dma_sync')},
    parent_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    files=manifest)
state = json.loads((root / 'logs/recovery-20261108/state.json').read_text())
assert status['boot_id'] == state['boot_id']
assert status['cpu_server_pid4807_comm'] == 'llama-server'
assert status['pin_service_active'] and status['bound_devices'] == list(range(4))
assert manifest['/lib/firmware/lanxin/lx500_pacc.bin']['sha256'] == state['vendor_sha256']
(out / 'manifest.json').write_text(json.dumps(status, indent=2) + '\n')
print(out)
