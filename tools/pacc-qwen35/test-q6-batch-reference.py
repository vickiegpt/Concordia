#!/usr/bin/env python3
"""Distinct activation columns, GGML dequantization and independent FP64 oracle."""
import ctypes as c
import json
import math
import os
from pathlib import Path
import random
import struct
import sys

device=int(sys.argv[1]);output=Path(sys.argv[2]);assert not output.exists()
root=Path('/srv/lanxin/qwen35-a3b')
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
ggml=c.CDLL(str(root/'llama.cpp/build-pacc-cuda-gcc14/bin/libggml-base.so'))
dequant=ggml.dequantize_row_q6_K
dequant.argtypes=[c.c_void_p,c.c_void_p,c.c_int64]
submit=lib.hetgpu_sifive_submit_q6k_q8_1_tiled
submit.argtypes=[c.c_int,c.c_void_p,c.c_void_p,c.c_void_p]+[c.c_uint32]*6
submit.restype=c.c_int
rng=random.Random(9173+device);results=[]
for m,n,k in ((37,4,256),(65,4,2048),(17,8,256)):
    xb=k//256*210;yb=k//32*36;sx=xb+16;sy=yb+16;sd=m+5
    x=bytearray(m*sx);y=bytearray(n*sy);activations=[]
    for row in range(m):
        for block in range(k//256):
            at=row*sx+block*210
            x[at:at+192]=bytes(rng.randrange(256) for _ in range(192))
            x[at+192:at+208]=struct.pack('16b',*(rng.randrange(-15,16) for _ in range(16)))
            x[at+208:at+210]=struct.pack('e',.0078125*(1+row%3))
    for col in range(n):
        values=[]
        for block in range(k//32):
            at=col*sy+block*36;scale=.015625*(1+col)
            integers=[rng.randrange(-127,128) for _ in range(32)]
            y[at:at+2]=struct.pack('e',scale)
            y[at+4:at+36]=struct.pack('32b',*integers)
            values.extend(scale*q for q in integers)
        activations.append(values)
    xp=c.create_string_buffer(bytes(x));yp=c.create_string_buffer(bytes(y))
    out=(c.c_float*(n*sd))(*([math.nan]*(n*sd)))
    assert submit(device,xp,yp,out,m,n,k,sx,sy,sd)==0
    worst=0.0
    for row in range(m):
        decoded=(c.c_float*k)();dequant(c.byref(xp,row*sx),decoded,k)
        for col in range(n):
            expected=sum(float(a)*b for a,b in zip(decoded,activations[col]))
            actual=out[col*sd+row];error=abs(actual-expected)
            assert math.isfinite(actual) and error<=1e-4+1e-5*abs(expected),(device,m,n,k,row,col,actual,expected)
            worst=max(worst,error)
    assert all(math.isnan(out[col*sd+row]) for col in range(n) for row in range(m,sd))
    results.append(dict(m=m,n=n,k=k,max_abs_error=worst,guards_unchanged=True))
output.write_text(json.dumps(dict(device=device,all_passed=True,cases=results),indent=2)+'\n')
print(json.dumps(results),flush=True)
