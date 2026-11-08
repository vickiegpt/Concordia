#!/usr/bin/env python3
"""Same rows/weights, distinct request columns; native XM, not application TPS."""
import ctypes as c
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

ref=runpy.run_path(str(Path(__file__).with_name('test-q4k-xm.py')))
fn=ref['fn'];qtype=ref['qtype'];m=128 if qtype==8 else 256;nb=8
device=int(os.environ.get('HETGPU_SIFIVE_FORCE_DEVICE','0'));assert device in range(4)
data=[]
for n in (1,2,4,8):
    x,y=ref['generate'](m,n,nb,0);expected=ref['reference'](x,y,m,n,nb,0)
    xb=c.create_string_buffer(x);yb=c.create_string_buffer(y)
    out=(c.c_float*(m*n+8))(*([math.nan]*(m*n)+[12345.]*8))
    times=[]
    for repeat in range(9):
        started=time.monotonic_ns()
        rc=fn(device,xb,yb,out,m,n,nb,0)
        times.append((time.monotonic_ns()-started)/1000)
        assert rc==0
        assert all(math.isfinite(out[i]) and abs(out[i]-v)<5e-5+5e-5*abs(v) for i,v in enumerate(expected))
        assert list(out)[m*n:]==[12345.]*8
    mean=sum(times[1:])/8
    data.append(dict(device=device,qtype=qtype,m=m,n=n,k=nb*256,latency_us=times,mean_us=mean,
                     requests_per_s=n*1e6/mean,gflop_per_s=2*m*n*nb*256/(mean*1000)))
with open(sys.argv[1],'x') as stream:json.dump(dict(runtime=os.environ['CONCORDIA_RELEASE'],cases=data,
    scope='hardware operator throughput; NOT LLM tokens/sec'),stream,indent=2)
print(json.dumps(data,indent=2))
