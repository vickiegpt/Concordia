#!/usr/bin/env python3
"""A/B device scalar versus RVV scaling around real XM matrix products."""
import ctypes as c
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

ref=runpy.run_path(str(Path(__file__).with_name('test-q4k-xm.py')))
device=int(os.environ['HETGPU_SIFIVE_FORCE_DEVICE'])
cases=[]
base_m=128 if ref['qtype']==8 else 256
for m,n,nb,gate in [(base_m,1,8,0),(base_m,2,8,0),(base_m,4,8,0),(base_m,8,8,0),
                    (7,3,3,1),(31,8,3,1),(3,2,128,1)]:
    x,y=ref['generate'](m,n,nb,gate)
    expected=ref['reference'](x,y,m,n,nb,gate)
    xb=c.create_string_buffer(x);yb=c.create_string_buffer(y)
    results={};baseline=None
    for mode in ('0','1'):
        os.environ['HETGPU_SIFIVE_QK_VECTOR_REDUCE']=mode
        times=[]
        for repeat in range(9):
            out=(c.c_float*(m*n+8))(*([math.nan]*(m*n)+[12345.]*8))
            start=time.monotonic_ns()
            rc=ref['fn'](device,xb,yb,out,m,n,nb,gate)
            times.append((time.monotonic_ns()-start)/1000)
            assert rc==0,(m,n,nb,gate,mode,rc)
            assert all(math.isfinite(out[i]) and abs(out[i]-v)<5e-5+5e-5*abs(v)
                       for i,v in enumerate(expected))
            assert list(out)[m*n:]==[12345.]*8
            raw=c.string_at(c.addressof(out),m*n*4)
            if baseline is None:baseline=raw
            assert raw==baseline,('scalar/vector bit mismatch',m,n,nb,gate,mode,repeat)
        results[mode]=dict(latency_us=times,mean_us=sum(times[1:])/8,
                           output_sha256=hashlib.sha256(raw).hexdigest())
    cases.append(dict(m=m,n=n,k=nb*256,gate=gate,results=results,
                      speedup=results['0']['mean_us']/results['1']['mean_us']))
    print('PASS',ref['qtype'],m,n,nb*256,gate,'bitwise_equal',cases[-1]['speedup'],flush=True)
report=dict(runtime=os.environ['CONCORDIA_RELEASE'],device=device,qtype=ref['qtype'],cases=cases,
            scope='native XM operator; scalar/RVV device postprocess comparison, not LLM TPS')
with open(sys.argv[1],'x') as stream:json.dump(report,stream,indent=2)
