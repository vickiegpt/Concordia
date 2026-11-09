#!/usr/bin/env python3
"""Measure ngl=99 state allocation with a four-worker barrier and 32 GiB reserve."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path('/srv/lanxin/qwen35-a3b')
p = argparse.ArgumentParser()
p.add_argument('prefix'); p.add_argument('--variant', choices=['baseline', 'mmap', 'both'], required=True)
p.add_argument('--workers', type=int, choices=[1, 4], required=True)
p.add_argument('--runtime', type=Path, required=True)
a = p.parse_args()
assert a.prefix.replace('-', '').isalnum()
out = ROOT / 'logs' / a.prefix
out.mkdir(exist_ok=False)
def available():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
def snapshot(pid):
    report = {}
    for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == 'kB': report[parts[0][:-1]] = int(parts[1])
    model = dict(Rss=0, Pss=0, Shared_Clean=0, Private_Clean=0, Private_Dirty=0, Shared_Dirty=0)
    active = False; perms = set()
    for line in Path(f'/proc/{pid}/smaps').read_text().splitlines():
        parts = line.split()
        if parts and '-' in parts[0] and ':' not in parts[0]:
            active = str(ROOT / 'models/Qwen3.5-35B-A3B-Q4_K_M.gguf') in line
            if active: perms.add(parts[1])
        elif active and len(parts) == 3 and parts[2] == 'kB' and parts[0][:-1] in model:
            model[parts[0][:-1]] += int(parts[1])
    return dict(pid=pid, rollup_kib=report, model_kib=model, model_permissions=sorted(perms))
env = os.environ.copy()
env.update(CONCORDIA_RELEASE=str(a.runtime.resolve()),
           LLAMA_RUNTIME_LIBRARY_DIR=str(ROOT / 'build-seq512-memory/bin'),
           HETGPU_SIFIVE_MMAP_WEIGHTS='0' if a.variant == 'baseline' else '1',
           HETGPU_SIFIVE_LAZY_ZERO='1' if a.variant == 'both' else '0')
initial = minimum = available(); children = []; logs = []; reports = []; samples = []; error = None
if initial < 100 * 1024**2: raise SystemExit('insufficient initial memory')
def interrupt(signum, frame): raise RuntimeError(f'signal {signum}')
signal.signal(signal.SIGTERM, interrupt)
try:
    for i in range(a.workers):
        log = (out / f'worker{i}.log').open('x'); logs.append(log)
        children.append(subprocess.Popen(['python3', str(ROOT / 'bin/run-pacc-recovered.py'), str(i), '--',
            str(ROOT / 'build-seq512-memory/bin/qwen-memory-probe'),
            str(ROOT / 'models/Qwen3.5-35B-A3B-Q4_K_M.gguf'), '512', str(out / f'worker{i}.json'), str(out)],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True))
    start = time.monotonic()
    while True:
        current = available(); minimum = min(minimum, current)
        if current < 32 * 1024**2: raise RuntimeError('32 GiB memory guard')
        if any(c.poll() is not None for c in children): raise RuntimeError('worker exited before barrier')
        ready = [out / f'worker{i}.json.ready' for i in range(a.workers)]
        if all(f.exists() for f in ready):
            try: reports = [json.loads(f.read_text()) for f in ready]
            except json.JSONDecodeError: time.sleep(.1); continue
            assert all(r.get('allocation_passed') and r.get('sequences') == 512 for r in reports)
            samples = [snapshot(c.pid) for c in children]
            (out / 'release').touch(exist_ok=False)
            break
        if time.monotonic() - start > 240: raise RuntimeError('capacity timeout')
        time.sleep(.1)
    codes = [c.wait(timeout=60) for c in children]
    assert codes == [0] * a.workers
except BaseException as exc: error = f'{type(exc).__name__}: {exc}'
finally:
    for c in children:
        if c.poll() is None:
            os.killpg(c.pid, signal.SIGTERM)
            try: c.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(c.pid, signal.SIGKILL); c.wait()
    for log in logs: log.close()
report = dict(capacity_passed=error is None, variant=a.variant, workers=reports, barrier_samples=samples,
              ngl=99, simultaneous_sequences=512 * a.workers if samples else 0,
              initial_mem_available_kib=initial, min_mem_available_kib=minimum,
              sum_pss_kib=sum(s['rollup_kib']['Pss'] for s in samples),
              sum_private_dirty_kib=sum(s['rollup_kib']['Private_Dirty'] for s in samples),
              aggregate_decode_tps=None, error=error, returncodes=[c.returncode for c in children],
              limitation='Initialization residency, not peak active-state memory or inference throughput')
(out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('workers', 'barrier_samples')}, indent=2))
raise SystemExit(0 if error is None else 1)
