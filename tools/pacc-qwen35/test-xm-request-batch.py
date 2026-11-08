#!/usr/bin/env python3
"""Real XM N-column ABI checks with independent activations and MoE routing."""
import ctypes as c
import math
import os
from pathlib import Path
import runpy
import struct
from cuda_abi_test_utils import Cuda, U

ref=runpy.run_path(str(Path(__file__).with_name('test-q4k-xm.py')))
qtype=ref['qtype']; dtype=8 if qtype==8 else qtype+8
class Fusion(c.Structure): _fields_=[('ptrs',U*5),('op',c.c_int)]
for n in (2,4,8):
    for moe in (False,True):
        m,nb,experts=35,3,4
        channels=2 if moe else 1
        blocks=nb*(8 if qtype==8 else 1)
        rowbytes=blocks*(34 if qtype==8 else (144 if qtype==4 else 176))
        # Padding is poisoned and never part of the valid matrices.
        stride=m+3; col_stride=stride*channels; ids_stride=channels+1
        x,y=ref['generate'](m*experts,n*channels,nb,0)
        order=[(col+ch)%experts for col in range(n) for ch in range(channels)]
        ids=[v for col in range(n) for v in order[col*channels:(col+1)*channels]+[-99]]
        expected=[12345.]*(col_stride*n+8)
        for col in range(n):
            for ch in range(channels):
                expert=order[col*channels+ch] if moe else 0
                xi=x[expert*m*rowbytes:(expert+1)*m*rowbytes]
                yi=y[(col*channels+ch)*nb*288:(col*channels+ch+1)*nb*288]
                vals=ref['reference'](xi,yi,m,1,nb,0)
                start=col*col_stride+ch*stride;expected[start:start+m]=vals
        cuda=Cuda()
        xp=cuda.alloc(x);yp=cuda.alloc(y);ip=cuda.alloc(struct.pack('<%di'%len(ids),*ids)) if moe else U(0)
        out=cuda.alloc(struct.pack('<%df'%len(expected),*([12345.]*len(expected))))
        if moe:
            name=f'_ZL17mul_mat_vec_q_moeIL9ggml_type{dtype}ELi2EEvPKvS2_PKiPf j5uint3jjjjjjjjj'.replace(' ','')
            params=[xp,yp,ip,out,c.c_uint(nb*256),(c.c_uint*3)(0,0,channels),c.c_uint(m),
                    c.c_uint(blocks),c.c_uint(nb*8*channels),c.c_uint(col_stride),
                    c.c_uint(m*blocks),c.c_uint(nb*8),c.c_uint(stride),c.c_uint(n),c.c_uint(ids_stride)]
            grid=((m+1)//2,channels,1)
        else:
            # Dense ABI defines its row count through column stride, no output padding there.
            # Use padded channels only for MoE; dense output is MxN plus tail guard.
            expected=[v for col in range(n) for v in expected[col*col_stride:col*col_stride+m]]+[12345.]*8
            out=cuda.alloc(struct.pack('<%df'%len(expected),*([12345.]*len(expected))))
            name=f'_ZL13mul_mat_vec_qIL9ggml_type{dtype}ELi{n}ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'
            one=(c.c_uint*3)(0,0,1);zero=(c.c_uint*3)(0,0,0)
            params=[xp,yp,ip,Fusion(),out,c.c_uint(nb*256),zero,c.c_uint(blocks),c.c_uint(nb*8),
                    c.c_uint(m),one,c.c_uint(m*blocks),c.c_uint(nb*8*n),c.c_uint(m*n),one,
                    c.c_uint(0),c.c_uint(0),c.c_uint(0),c.c_uint(0)]
            grid=((m+1)//2,1,1)
        cuda.launch(name,params,grid,(32,n,1))
        got=struct.unpack('<%df'%len(expected),cuda.read(out,len(expected)*4))
        assert all(math.isfinite(a) and abs(a-b)<5e-5+5e-5*abs(b) for a,b in zip(got,expected)),(qtype,n,moe,got,expected)
        print(f'PASS request_batch qtype={qtype} n={n} moe={moe} distinct_inputs=1 routing=rotated guards=unchanged',flush=True)
        cuda.close()

# Some graph views expose requests as channels with ncols_dst=1.
for channels in (2,8):
    m,nb=35,3;blocks=nb*(8 if qtype==8 else 1)
    x,y=ref['generate'](m,channels,nb,0)
    expected=[]
    for ch in range(channels):expected+=ref['reference'](x,y[ch*nb*288:(ch+1)*nb*288],m,1,nb,0)
    expected+=[12345.]*8
    cuda=Cuda();xp=cuda.alloc(x);yp=cuda.alloc(y);out=cuda.alloc(struct.pack('<%df'%len(expected),*([12345.]*len(expected))))
    fd=lambda n:(c.c_uint*3)(0,0,n)
    params=[xp,yp,U(0),Fusion(),out,c.c_uint(nb*256),fd(0),c.c_uint(blocks),c.c_uint(nb*8),c.c_uint(m),
            fd(channels),c.c_uint(m*blocks),c.c_uint(nb*8),c.c_uint(m),fd(1),c.c_uint(0),c.c_uint(0),c.c_uint(0),c.c_uint(0)]
    name=f'_ZL13mul_mat_vec_qIL9ggml_type{dtype}ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'
    cuda.launch(name,params,(m,channels,1),(32,4,1))
    got=struct.unpack('<%df'%len(expected),cuda.read(out,len(expected)*4))
    assert all(math.isfinite(a) and abs(a-b)<5e-5+5e-5*abs(b) for a,b in zip(got,expected))
    print(f'PASS request_batch qtype={qtype} channel_requests={channels} distinct_inputs=1 guards=unchanged',flush=True)
    cuda.close()
