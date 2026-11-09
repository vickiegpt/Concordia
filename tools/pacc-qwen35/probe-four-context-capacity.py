#!/usr/bin/env python3
"""Hold four CPU-only contexts simultaneously; never report inference throughput."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--probe',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(exist_ok=False)
def available_kib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])
    raise RuntimeError('MemAvailable missing')
initial=available_kib();minimum=initial
if initial<220*1024**2:raise SystemExit('need at least 220 GiB available before this probe')
children=[];logs=[];ready=[];error=None;held=False;codes=[]
def interrupt(signum,frame):raise RuntimeError(f'signal {signum}')
signal.signal(signal.SIGTERM,interrupt)
try:
    for i in range(4):
        log=(a.output/f'worker{i}.log').open('x');logs.append(log)
        children.append(subprocess.Popen([str(a.probe.resolve()),str(a.model.resolve()),'512',
            str((a.output/f'worker{i}.json').resolve()),str(a.output.resolve())],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
    started=time.monotonic()
    while True:
        now_available=available_kib();minimum=min(minimum,now_available)
        if now_available<20*1024**2:raise RuntimeError('less than 20 GiB available')
        if any(c.poll() is not None for c in children):raise RuntimeError('worker exited before barrier')
        files=[a.output/f'worker{i}.json.ready' for i in range(4)]
        if all(f.exists() for f in files):
            try:ready=[json.loads(f.read_text()) for f in files]
            except json.JSONDecodeError:time.sleep(.1);continue
            if not all(r.get('allocation_passed') and r.get('sequences')==512 for r in ready):
                raise RuntimeError('invalid ready report')
            held=True;break
        if time.monotonic()-started>240:raise RuntimeError('allocation timeout')
        time.sleep(.1)
    (a.output/'release').touch(exist_ok=False)
    codes=[c.wait(timeout=60) for c in children]
except BaseException as exc:error=f'{type(exc).__name__}: {exc}'
finally:
    for c in children:
        if c.poll() is None:
            os.killpg(c.pid,signal.SIGTERM)
            try:c.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    for log in logs:log.close()
report=dict(stage='CPU context capacity only; no PACC execution or generated tokens',
            simultaneously_held_sequences=2048 if held else 0,
            capacity_passed=held and codes==[0]*4 and error is None,
            initial_available_kib=initial,min_available_kib=minimum,
            aggregate_decode_tps=None,workers=ready,returncodes=[c.returncode for c in children],error=error)
(a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['capacity_passed'] else 1)
