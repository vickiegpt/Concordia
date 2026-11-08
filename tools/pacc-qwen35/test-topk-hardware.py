#!/usr/bin/env python3
import ctypes as c
import math
import os
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
fn=lib.hetgpu_sifive_submit_topk_row
fn.argtypes=[c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_uint,c.c_uint,c.c_uint,c.c_float,c.c_float]
fn.restype=c.c_int
for n,k,flags in [(256,8,4),(128,8,0),(576,16,17),(32,5,8),(64,8,12),(32,4,20)]:
    xs=[c.c_float(((i*31)%41-20)*.137).value for i in range(n)]
    bias=[((i*7)%13-6)*.1 for i in range(n)]
    b=(c.c_float*(n*(2 if flags&16 else 1)))(*(xs+bias if flags&16 else xs))
    w=list(xs)
    if not flags&8:
        if flags&1:w=[1/(1+math.exp(-v)) for v in w]
        else:
            w=[math.exp(v-max(xs)) for v in xs];s=sum(w);w=[v/s for v in w]
    rank=[w[i]+(b[n+i] if flags&16 else 0) for i in range(n)]
    ids=sorted(range(n),key=lambda i:(-rank[i],i))[:k]
    ref=[w[i] for i in ids]
    if flags&4:s=max(sum(ref),1e-6);ref=[v/s for v in ref]
    if flags&8:vmax=max(ref);ref=[math.exp(v-vmax) for v in ref];s=sum(ref);ref=[v/s for v in ref]
    ref=[v*1.25 for v in ref]
    out=(c.c_float*(2*k+8))(*([math.nan]*(2*k)+[12345.]*8))
    assert fn(0,b,out,n,k,flags,1e-6,1.25)==0
    assert list(out)[k:2*k]==ids,(n,flags,list(out)[k:2*k],ids)
    err=max(abs(out[i]-ref[i]) for i in range(k));assert err<2e-6,(n,flags,err)
    assert list(out)[2*k:]==[12345.]*8
    print(f'PASS Top-k n={n} k={k} flags={flags} error={err:.9g}',flush=True)
