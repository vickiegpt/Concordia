#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import subprocess
import sys
import time

root=Path('/srv/lanxin/qwen35-a3b')
parser=argparse.ArgumentParser();parser.add_argument('prefix')
parser.add_argument('--wrapper',type=Path,default=root/'bin/run-pacc-isolated.sh')
args=parser.parse_args();prefix=args.prefix
wrapper=[('python3' if args.wrapper.suffix=='.py' else 'bash'),str(args.wrapper.resolve())]
assert prefix.replace('-','').isalnum()
processes=[];logs=[];codes=[]
try:
    for device in range(4):
        log=(root/'logs'/f'{prefix}-pacc{device}.log').open('x');logs.append(log)
        processes.append(subprocess.Popen([*wrapper,str(device),'--',
            'python3',str(root/'bin/test-xm-rms-alternation.py'),str(device)],cwd=root,
            stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy()))
        time.sleep(2)
    codes=[p.wait(timeout=300) for p in processes]
finally:
    for p in processes:
        if p.poll() is None:
            p.terminate()
            try:p.wait(10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
    for log in logs:log.close()
print('four-device mixed XM/RMS returncodes:',codes)
assert codes==[0,0,0,0]
