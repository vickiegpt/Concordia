#!/usr/bin/env python3
"""Compare real PACC head results against a separate FP64 recurrence reference."""
import ctypes as c
import math
import os

lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
submit=lib.hetgpu_sifive_submit_gdn_head_f32
submit.argtypes=[c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float)]+[c.c_uint]*5+[c.c_float]
submit.restype=c.c_int

def reference(data,s,n,kda,keep,slots,scale):
    q=data[:n*s]; k=data[n*s:2*n*s]; v=data[2*n*s:3*n*s]
    end=3*n*s+n*(s if kda else 1)
    g=data[3*n*s:end]; beta=data[end:end+n]
    state=list(data[end+n:]); out=[0.]*(n*s); saved=[None]*slots
    for t in range(n):
        for col in range(s):
            if kda:
                for i in range(s): state[col*s+i]*=math.exp(g[t*s+i])
                kv=sum(state[col*s+i]*k[t*s+i] for i in range(s))
                gate=1.
            else:
                gate=math.exp(g[t])
                kv=gate*sum(state[col*s+i]*k[t*s+i] for i in range(s))
            delta=(v[t*s+col]-kv)*beta[t]
            for i in range(s): state[col*s+i]=gate*state[col*s+i]+k[t*s+i]*delta
            out[t*s+col]=sum(state[col*s+i]*q[t*s+i] for i in range(s))*scale
        slot=n-1-t
        if keep and slot<slots: saved[slot]=list(state)
    if not keep: saved[0]=state
    return out+[v for slot in saved for v in slot]

for s,n,kda,keep,slots in [(16,1,0,0,1),(128,3,0,0,1),(32,5,0,1,3),(32,5,1,1,3),(128,2,1,0,1)]:
    ins=3*n*s+n*(s if kda else 1)+n+s*s
    outs=n*s+slots*s*s
    values=[((i*13)%47-23)*.003 for i in range(ins)]
    scale=c.c_float(1/math.sqrt(s)).value
    buf=(c.c_float*ins)(*values)
    output=(c.c_float*(outs+8))(*([math.nan]*outs+[12345.]*8))
    expected=reference(list(buf),s,n,kda,keep,slots,scale)
    assert submit(0,buf,output,s,n,kda,keep,slots,scale)==0
    worst=0.
    for i,ref in enumerate(expected):
        error=abs(output[i]-ref)
        assert math.isfinite(output[i]) and error<=3e-5+3e-5*abs(ref),(s,n,kda,keep,i,output[i],ref)
        worst=max(worst,error)
    assert list(output)[outs:]==[12345.]*8
    print(f'PASS PACC GDN width={s} tokens={n} kda={kda} keep={keep} slots={slots} max_abs_error={worst:.9g}',flush=True)
buf=(c.c_float*1)(0)
output=(c.c_float*1)(12345)
assert submit(0,buf,output,17,1,0,0,1,1.)!=0 and output[0]==12345
assert submit(0,buf,output,16,1,0,1,2,1.)!=0 and output[0]==12345
for key in ('HETGPU_SIFIVE_SKIP_JOB_WAIT','HETGPU_SIFIVE_ASSUME_SUCCESS_ON_WAIT_ERROR'):
    old=os.environ.get(key)
    os.environ[key]='1'
    assert submit(0,buf,output,16,1,0,0,1,1.)!=0 and output[0]==12345
    if old is None: del os.environ[key]
    else: os.environ[key]=old
print('PASS GDN invalid layouts and unsafe completion policy rejected',flush=True)
