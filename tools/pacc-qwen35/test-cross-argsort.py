#!/usr/bin/env python3
import ctypes as c
import struct
from cuda_abi_test_utils import Cuda

def reference(values,ascending):
    n=len(values);pad=1<<(n-1).bit_length();ids=list(range(pad));width=2
    while width<=pad:
        gap=width//2
        while gap:
            for col in range(pad):
                other=col^gap
                if other<=col:continue
                a,b=ids[col],ids[other]
                if not col&width:
                    swap=a>=n or (b<n and (values[a]>values[b] if ascending else values[a]<values[b]))
                else:
                    swap=b>=n or (a<n and (values[a]<values[b] if ascending else values[a]>values[b]))
                if swap:ids[col],ids[other]=b,a
            gap//=2
        width*=2
    return ids[:n]

for n in (5,16,256,576):
    for order in (0,1):
        cuda=Cuda();rows=3
        x=[[float((j*17+r)%13-6) for j in range(n)] for r in range(rows)]
        xp=cuda.alloc(struct.pack('<%df'%(rows*n),*[v for row in x for v in row]))
        out=cuda.alloc(struct.pack('<%di'%(rows*n+8),*([-99]*(rows*n+8))))
        pad=1<<(n-1).bit_length()
        params=[xp,out,c.c_int(n),c.c_int(pad)]
        cuda.launch(f'_ZL17k_argsort_f32_i32IL15ggml_sort_order{order}EEvPKfPiii',params,(rows,1,1),(pad,1,1))
        actual=list(struct.unpack('<%di'%(rows*n+8),cuda.read(out,(rows*n+8)*4)))
        expected=[i for row in x for i in reference(row,order==0)]+[-99]*8
        assert actual==expected,(n,order)
        print(f'PASS device argsort n={n} order={order} ties=exact padding=exact guards=unchanged',flush=True)
        cuda.close()
