#!/usr/bin/env python3
"""Long-run device phase capture without retaining multi-GB Python record lists."""
import argparse
from collections import deque
import gc
import json
import mmap
import os
from pathlib import Path
import runpy
import signal
import subprocess
import time
from jobd_phase_accumulator import PhaseAccumulator

base = runpy.run_path(str(Path(__file__).with_name('profile-jobd-phases.py')))
EVENT, MAGIC, RING_SIZE = base['EVENT'],base['MAGIC'],base['RING_SIZE']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',type=int,choices=range(4),required=True)
    p.add_argument('--require-build',type=lambda s:int(s,0),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=21600)
    p.add_argument('--sample-ms',type=float,default=1)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or not .5<=a.sample_ms<=100:p.error('invalid command/sample interval')
    if a.output.exists() or a.output.with_suffix('.log').exists():p.error('output exists')
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    fd=os.open(f'/dev/hetgpu_sifive_mbox_ddr_coh{a.device}',os.O_RDONLY)
    if base['read_shared_base'](fd)!=int(os.environ['HETGPU_SIFIVE_SHARED_DDR_BASE'],0):
        raise RuntimeError('owned allocation changed')
    offset=base['RING_OFFSET']+a.device*0x2000
    region=mmap.mmap(fd,offset+RING_SIZE,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ)
    def sample():
        first=region[offset:offset+RING_SIZE];second=region[offset:offset+RING_SIZE]
        for i in range(0,RING_SIZE,EVENT.size):
            raw=first[i:i+EVENT.size]
            if raw!=second[i:i+EVENT.size]:continue
            magic,index,status,job,aux,seq=EVENT.unpack(raw)
            if magic==MAGIC:yield(index,seq,status,job,aux)
    seen=set(sample());recent=deque(seen)
    accumulator=PhaseAccumulator(max((r[0] for r in seen),default=0),a.require_build)
    started=time.monotonic();last=started;maximum=0;timed_out=False;finished=None;error=None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        gc.disable()
        try:
            while True:
                now=time.monotonic();maximum=max(maximum,now-last);last=now
                for event in sorted(sample(),key=lambda event:event[0]):
                    if event in seen:continue
                    seen.add(event);recent.append(event)
                    if len(recent)>768:seen.remove(recent.popleft())
                    accumulator.observe(dict(zip(('index','seq','status','job','aux'),event)))
                if child.poll() is not None:
                    if finished is None:finished=now
                    if now-finished>1:break
                elif now-started>a.timeout:
                    timed_out=True
                    break
                time.sleep(a.sample_ms/1000)
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            region.close();os.close(fd);gc.enable()
    report=accumulator.report(child.returncode,timed_out)
    report.update(device=a.device,command=command,sample_ms=a.sample_ms,max_sample_gap_ms=maximum*1000,
                  collection='streaming histogram; no full raw ring retained',error=error,
                  limitations=['Double-read stability is not source-side atomicity.',
                    'These phases cover PRELOADED jobs, not all ELF kernels.',
                    'Completion publish excludes host observation latency.'])
    if error:report['capture_valid']=False
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['capture_valid'] else 1

if __name__=='__main__':raise SystemExit(main())
