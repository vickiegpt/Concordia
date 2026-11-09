#!/usr/bin/env python3
import ctypes as c
import json
import math
import os
from pathlib import Path
import struct
import sys

dev=int(sys.argv[1]);output=Path(sys.argv[2]);assert not output.exists()
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
exact=lib.hetgpu_sifive_submit_rms_exact_on
exact.argtypes=[c.c_int,c.c_void_p,c.c_void_p,c.c_void_p,c.c_uint64,c.c_uint64,c.c_float];exact.restype=c.c_int
old=lib.hetgpu_sifive_submit_rmsnorm_on
old.argtypes=exact.argtypes+[c.c_int];old.restype=c.c_int
class Layout(c.Structure):
    _fields_=[('hidden',c.c_uint64),('dims',c.c_uint64*3),('xs',c.c_uint64*3),
              ('ws',c.c_uint64*4),('wt',c.c_uint64*3),('eps',c.c_float)]
assert c.sizeof(Layout)==120
strided=lib.hetgpu_sifive_submit_rmsnorm_strided_f32
strided.argtypes=[c.c_int,c.c_void_p,c.c_void_p,c.c_void_p,c.POINTER(Layout)];strided.restype=c.c_int
os.environ['HETGPU_SIFIVE_RMS_EXACT']='1'
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
rows=[]
for count,hidden,eps,constant,weighted in [(1,2048,1e-6,math.sqrt(3.5),False),(3,128,1e-6,None,True),
    (2,2048,1e-6,None,True),(2,512,1e-5,0.0,False),(8,256,0.0,2.0,True)]:
    xs=[f32(constant if constant is not None else ((i*17+dev*3)%71-35)/13) for i in range(count*hidden)]
    ws=[f32(.5+(i%7)/8) for i in range(hidden)]
    x=(c.c_float*len(xs))(*xs);w=(c.c_float*hidden)(*ws) if weighted else None
    y=(c.c_float*(len(xs)+8))(*([math.nan]*len(xs)+[12345.0]*8))
    expected=[]
    for r in range(count):
        total=sum(f32(v*v) for v in xs[r*hidden:(r+1)*hidden])
        scale=f32(1/f32(math.sqrt(f32(f32(total/hidden)+f32(eps)))))
        expected.extend(f32(f32(xs[r*hidden+j]*scale)*(ws[j] if weighted else 1)) for j in range(hidden))
    rc=exact(dev,x,w,y,count,hidden,eps)
    assert rc==0,(dev,count,hidden,rc)
    error=max(abs(y[i]-expected[i]) for i in range(len(xs)))
    assert error==0,(dev,count,hidden,error)
    assert list(y)[len(xs):]==[12345.0]*8
    record=dict(rows=count,hidden=hidden,eps=eps,weighted=weighted,max_abs_error=error)
    padded=(c.c_float*(count*(hidden+13)))(*([9876.0]*(count*(hidden+13))))
    for r in range(count):
        for j in range(hidden):padded[r*(hidden+13)+j]=xs[r*hidden+j]
    z=(c.c_float*(len(xs)+8))(*([math.nan]*len(xs)+[12345.0]*8))
    layout=Layout(hidden,(count,1,1),(hidden+13,0,0),(hidden,1,1,1),(0,0,0),eps)
    assert strided(dev,padded,w,z,c.byref(layout))==0
    assert list(z)[:len(xs)]==expected and list(z)[len(xs):]==[12345.0]*8
    assert all(padded[r*(hidden+13)+j]==9876.0 for r in range(count) for j in range(hidden,hidden+13))
    record['strided_exact']=True
    if constant==math.sqrt(3.5):
        original=(c.c_float*len(xs))()
        rc=old(dev,x,None,original,count,hidden,eps,4)
        record.update(old_returncode=rc,old_first=original[0],new_first=y[0],expected=expected[0])
    rows.append(record)
bad=(c.c_float*8)(*([12345.0]*8))
assert exact(dev,None,None,bad,1,1,1e-6)!=0
assert exact(dev,x,w,bad,4096,32768,1e-6)!=0
assert list(bad)==[12345.0]*8
output.write_text(json.dumps(dict(device=dev,all_passed=True,cases=rows),indent=2)+'\n')
print(json.dumps(rows),flush=True)
