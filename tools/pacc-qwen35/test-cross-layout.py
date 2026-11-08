#!/usr/bin/env python3
"""Device concat tiling and sparse KV cast; no whole-cache staging."""
import ctypes as c
import struct
from cuda_abi_test_utils import Cuda

for a,b,rows in [(5,3,40000),(90001,70001,2)]:
    cuda=Cuda()
    x=(bytes(range(256))*((a*rows*4+255)//256))[:a*rows*4]
    y=(bytes(reversed(range(256)))*((b*rows*4+255)//256))[:b*rows*4]
    expected=b''.join(x[r*a*4:(r+1)*a*4]+y[r*b*4:(r+1)*b*4] for r in range(rows))
    xp=cuda.alloc(x);yp=cuda.alloc(y);out=cuda.alloc(b'\x5a'*(len(expected)+32))
    params=[xp,yp,out]+[c.c_int64(v) for v in [a,rows,1,a+b,rows,1]]
    cuda.launch('_ZL11concat_contIjLi0EEvPKT_S2_PS0_llllll',params,(((a+b)*rows+255)//256,1,1),(256,1,1))
    assert cuda.read(out,len(expected)+32)==expected+b'\x5a'*32
    print(f'PASS concat split a={a} b={b} rows={rows} bytes={len(expected)} guards=unchanged',flush=True)
    cuda.close()

for kind in ('6__half','f'):
    cuda=Cuda();width=8;headstride=1600040
    values=[0.,-0.,2**-25,2**-24,1.00048828125,1.00146484375,65504.,-65504.]*4
    ids=[200001,3,100003,1];fmt='e' if kind=='6__half' else 'f';elem=struct.calcsize(fmt)
    xp=cuda.alloc(struct.pack('<32f',*values));ip=cuda.alloc(struct.pack('<4q',*ids))
    expected=bytearray(b'\x5a'*((2*headstride+16)*elem));out=cuda.alloc(bytes(expected))
    for head in range(2):
        for row in range(2):
            start=(head*headstride+ids[head*2+row]*width)*elem
            expected[start:start+width*elem]=struct.pack('<8'+fmt,*values[(head*2+row)*width:(head*2+row+1)*width])
    fd=lambda n:(c.c_uint*3)(0,0,n)
    params=[xp,ip,out]+[c.c_int64(v) for v in [32,2,2,1,1,width,width*2,width*4,1,2,4,width,headstride,2*headstride]]+[fd(v) for v in [width,2,2,2,1]]
    name=f'_ZL10k_set_rowsIfl{kind}EvPKT_PKT0_PT1_llllllllllllll5uint3S9_S9_S9_S9_'
    cuda.launch(name,params,(1,1,1),(256,1,1))
    assert cuda.read(out,len(expected))==expected
    print(f'PASS KV sparse cast kind={kind} far_row=200001 rounding=RNE unselected_bytes=unchanged',flush=True)
    cuda.close()
