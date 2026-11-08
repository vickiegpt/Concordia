#!/usr/bin/env python3
import ctypes as c
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time

base=runpy.run_path(str(Path(__file__).with_name('test-gdn-hardware.py')))
submit,reference=base['submit'],base['reference']
result=[]
for s,n,kda,keep,slots in [(128,1,0,0,1),(128,3,0,0,1),(32,5,1,1,3)]:
    ins=3*n*s+n*(s if kda else 1)+n+s*s;outs=n*s+slots*s*s
    values=(c.c_float*ins)(*[((i*13)%47-23)*.003 for i in range(ins)])
    output=(c.c_float*(outs+8))(*([math.nan]*outs+[12345.]*8))
    scale=c.c_float(1/math.sqrt(s)).value;times=[];digests=[]
    expected=reference(list(values),s,n,kda,keep,slots,scale)
    for repeat in range(8):
        before=time.monotonic_ns()
        assert submit(0,values,output,s,n,kda,keep,slots,scale)==0
        times.append((time.monotonic_ns()-before)/1000)
        assert list(output)[outs:]==[12345.]*8
        assert all(math.isfinite(v) and abs(v-ref)<=3e-5+3e-5*abs(ref) for v,ref in zip(output,expected))
        digests.append(hashlib.sha256(bytes(output)[:outs*4]).hexdigest())
    assert len(set(digests))==1
    result.append(dict(width=s,tokens=n,kda=kda,keep=keep,slots=slots,latency_us=times,
                       mean_us=sum(times[1:])/7,output_sha256=digests[0]))
with open(sys.argv[1],'x') as stream:json.dump(dict(runtime=os.environ['CONCORDIA_RELEASE'],cases=result),stream,indent=2)
print(json.dumps(result,indent=2))
