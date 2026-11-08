#!/usr/bin/env python3
import ctypes as c
import hashlib
import math
import os
from pathlib import Path
import runpy
import sys

dev=int(sys.argv[1]);assert dev in range(4)
ref=runpy.run_path(str(Path(__file__).with_name('test-q4k-xm.py')));lib=ref['lib'];gemm=ref['fn']
rms=lib.hetgpu_sifive_submit_rmsnorm_on
rms.argtypes=[c.c_int]+[c.c_void_p]*3+[c.c_uint64]*2+[c.c_float,c.c_int];rms.restype=c.c_int
m,n,nb=176,3,16;x,y=ref['generate'](m,n,nb,0)
xb=c.create_string_buffer(x);yb=c.create_string_buffer(y);out=(c.c_float*(m*n))()
digest=None
for repeat in range(64):
    assert gemm(dev,xb,yb,out,m,n,nb,0)==0
    h=hashlib.sha256(bytes(out)).hexdigest()
    if digest is None:digest=h
    assert h==digest,(dev,repeat,'XM output changed')
    rows=(3,8,1)[repeat%3];hidden=2048
    a=(c.c_float*(rows*hidden))(*[((i*7+repeat*3+dev)%31-15)/8 for i in range(rows*hidden)])
    w=(c.c_float*hidden)(*[.5+(i%5)/8 for i in range(hidden)])
    b=(c.c_float*(rows*hidden+8))(*([math.nan]*(rows*hidden)+[12345.]*8))
    assert rms(dev,a,w,b,rows,hidden,1e-6,4)==0
    for r in range(rows):
        scale=1/math.sqrt(sum(a[r*hidden+j]**2 for j in range(hidden))/hidden+1e-6)
        for j in range(hidden):
            value=b[r*hidden+j];expected=a[r*hidden+j]*scale*w[j]
            assert math.isfinite(value) and abs(value-expected)<1e-4,(dev,repeat,r,j,value,expected)
    assert list(b)[rows*hidden:]==[12345.]*8
print(f'PASS XM/RMS alternating device={dev} repeats=64 large_payload=1 guards=unchanged',flush=True)
