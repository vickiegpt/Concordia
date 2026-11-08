#!/usr/bin/env python3
import ctypes as c
import math
import os
from pathlib import Path
import struct
from cuda_abi_test_utils import Cuda,U

class Params(c.Structure):
    _fields_=[('nheads',c.c_int64),('n_head_log2',c.c_uint32)]+[(s,c.c_int64) for s in
       ['ncols','nrows_x','nrows_y','ne00','ne01','ne02','ne03','nb11','nb12','nb13','ne12','ne13']]+[(s,c.c_float) for s in ['scale','max_bias','m0','m1']]
assert c.sizeof(Params)==128
cuda=Cuda();cols=256;heads=2
x=[(i%13-6)*.137 for i in range(cols*heads)]
mask=[0.]+[-math.inf]*(cols-1)
xp=cuda.alloc(struct.pack('<%df'%len(x),*x));mp=cuda.alloc(struct.pack('<%df'%cols,*mask))
out=cuda.alloc(struct.pack('<%df'%(len(x)+8),*([math.nan]*len(x)+[12345.]*8)))
p=Params(heads,2,cols,heads,1,cols,1,heads,1,cols*4,cols*4,cols*4,1,1,.5,0,1,1)
name='_ZL12soft_max_f32ILb1ELi256ELi256EfEvPKfPKT2_S1_Pf15soft_max_params'
ptx=Path('/srv/lanxin/qwen35-a3b/ptx-cache/libggml_cuda_so_0/softmax.cu.ptx').read_bytes() if os.environ.get('SOFTMAX_REAL_PTX')=='1' else None
cuda.launch(name,[xp,mp,U(0),out,p],(1,heads,1),(256,1,1),ptx)
got=struct.unpack('<%df'%(len(x)+8),cuda.read(out,(len(x)+8)*4))
expect=([1.]+[0.]*(cols-1))*heads+[12345.]*8
assert all(math.isfinite(a) and abs(a-b)<1e-6 for a,b in zip(got,expect)),('masked softmax',got[:8],got[256:264])
print('PASS softmax masked columns=256 heads=2 guards=unchanged',flush=True)
for half in (False,True):
    n,heads,tokens,samples=7,3,2,2;count=n*heads*tokens*samples
    x=[((i*7)%31-15)*.137 for i in range(count)]
    xb=struct.pack('<%df'%(count+8),*(x+[12345.]*8));x=struct.unpack('<%df'%(count+8),xb)[:count]
    mask=[-math.inf if (k+t+s)%4==0 else -.2*(k+t+s) for s in range(samples) for t in range(tokens) for k in range(n)]
    mb=struct.pack('<%d%s'%(len(mask),'e' if half else 'f'),*mask);mask=struct.unpack('<%d%s'%(len(mask),'e' if half else 'f'),mb)
    sinks=[-.25,.1,.8];sb=struct.pack('<3f',*sinks);sinks=struct.unpack('<3f',sb)
    xp=cuda.alloc(xb);mp=cuda.alloc(mb);sp=cuda.alloc(sb)
    p=Params(heads,2,n,heads*tokens*samples,tokens,n,tokens,heads,samples,n*(2 if half else 4),n*tokens*(2 if half else 4),n*tokens*(2 if half else 4),1,samples,.37,4,.7,.3)
    expected=[]
    for s in range(samples):
      for h in range(heads):
        for t in range(tokens):
            slope=p.m0**(h+1) if h<2 else p.m1**(2*(h-2)+1)
            row=(s*heads+h)*tokens+t
            vals=[x[row*n+k]*p.scale+mask[(s*tokens+t)*n+k]*slope for k in range(n)]
            maxv=max(vals+[sinks[h]]);ex=[math.exp(v-maxv) for v in vals];den=sum(ex)+math.exp(sinks[h]-maxv)
            expected.extend(v/den for v in ex)
    expected += [12345.]*8
    name=f'_ZL12soft_max_f32ILb1ELi0ELi256E{"6__half" if half else "f"}EvPKfPKT2_S1_Pf15soft_max_params'
    cuda.launch(name,[xp,mp,sp,xp,p],(tokens,heads,samples),(256,1,1))
    got=struct.unpack('<%df'%(count+8),cuda.read(xp,(count+8)*4))
    error=max(abs(a-b) for a,b in zip(got,expected))
    assert all(math.isfinite(a) and abs(a-b)<2e-6 for a,b in zip(got,expected)),(half,error,got[:7],expected[:7])
    print(f'PASS softmax mask_half={half} alibi=True sinks=True alias=True cols=7 heads=3 tokens=2 samples=2 max_abs_error={error:.9g}',flush=True)
cuda.close()
