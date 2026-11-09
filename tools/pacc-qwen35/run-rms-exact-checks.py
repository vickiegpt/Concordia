#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import argparse
root=Path('/srv/lanxin/qwen35-a3b')
parser=argparse.ArgumentParser()
parser.add_argument('--runtime',required=True)
parser.add_argument('--prefix',required=True)
parser.add_argument('--test',choices=('test-rms-exact.py','test-q6-batch-reference.py'),default='test-rms-exact.py')
args=parser.parse_args()
env=os.environ.copy();env['CONCORDIA_RELEASE']=args.runtime
for device in range(4):
    with (root/f'logs/{args.prefix}-pacc{device}.log').open('x') as log:
        subprocess.run(['python3',str(root/'bin/run-pacc-recovered.py'),str(device),'--','python3',
            str(root/'bin'/args.test),str(device),str(root/f'logs/{args.prefix}-pacc{device}.json')],
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    print('PASS',device,flush=True)
