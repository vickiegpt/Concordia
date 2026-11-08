#!/usr/bin/env python3
import ctypes as c
import struct
from cuda_abi_test_utils import Cuda

def f32(x):return c.c_float(x).value
def reference(x,threads,mean):
    partial=[]
    for t in range(threads):
        sums=[0.]*8
        for start in range(t,len(x),threads*8):
            for j in range(8):
                i=start+j*threads
                if i<len(x):sums[j]=f32(sums[j]+x[i])
        total=0.
        for v in sums:total=f32(total+v)
        partial.append(total)
    warp=[]
    def reduce(v):
        for gap in (16,8,4,2,1):
            for j in range(gap):v[j]=f32(v[j]+v[j+gap])
        return v[0]
    for base in range(0,threads,32):warp.append(reduce(partial[base:base+32]))
    value=reduce(warp+[0.]*(32-len(warp)))
    return f32(value/len(x)) if mean else value

cuda=Cuda()
for cols,rows in ((8,17),(513,3),(4097,29)):
    x=[[f32(((j*13+r)%31-15)*.013) for j in range(cols)] for r in range(rows)]
    xp=cuda.alloc(struct.pack('<%df'%(rows*cols),*[v for row in x for v in row]))
    for threads in (32,128,512):
        for mean in (False,True):
            out=cuda.alloc(struct.pack('<%df'%(rows+8),*([-999.]*(rows+8))))
            cuda.launch(f'_ZL15reduce_rows_f32ILb{int(mean)}EEvPKfPfi',
                        [xp,out,c.c_int(cols)],(rows,1,1),(threads,1,1))
            actual=list(struct.unpack('<%df'%(rows+8),cuda.read(out,(rows+8)*4)))
            expected=[reference(row,threads,mean) for row in x]+[-999.]*8
            assert actual==expected,(cols,rows,threads,mean,actual[:3],expected[:3])
            print(f'PASS device reduce cols={cols} rows={rows} threads={threads} mean={mean} exact=True guards=unchanged',flush=True)
cuda.close()
