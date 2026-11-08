#!/usr/bin/env python3
import ctypes as c
import math
import os
import sys

dev=int(sys.argv[1]);assert dev in range(4)
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
run=lib.hetgpu_sifive_submit_rmsnorm_on
run.argtypes=[c.c_int]+[c.c_void_p]*3+[c.c_uint64]*2+[c.c_float,c.c_int];run.restype=c.c_int
rows,hidden=8,2048
for repeat in range(64):
    x=(c.c_float*(rows*hidden))(*[((i*7+repeat*3+dev)%31-15)/8 for i in range(rows*hidden)])
    w=(c.c_float*hidden)(*[.5+(i%5)/8 for i in range(hidden)])
    out=(c.c_float*(rows*hidden+8))(*([math.nan]*(rows*hidden)+[12345.]*8))
    assert run(dev,x,w,out,rows,hidden,1e-6,4)==0
    for r in range(rows):
        scale=1/math.sqrt(sum(x[r*hidden+j]**2 for j in range(hidden))/hidden+1e-6)
        assert all(math.isfinite(out[r*hidden+j]) and abs(out[r*hidden+j]-x[r*hidden+j]*scale*w[j])<1e-4 for j in range(hidden)),(dev,repeat,r)
    assert list(out)[rows*hidden:]==[12345.]*8
print(f'PASS RMS batch device={dev} rows=8 hidden=2048 repeats=64 guards=unchanged',flush=True)
