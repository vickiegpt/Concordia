#!/usr/bin/env python3
"""Hardware MMVF row/column tiling with FP32 precision and padding checks."""
import ctypes as c
import math
import os

lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so')
fn=lib.hetgpu_sifive_submit_gemm_mmvf_small_n
I=c.c_int; P=c.c_void_p; L=c.c_int64
fn.argtypes=[I]*5+[P,P,I,I,L,P,I,I,L,P,P,I,I,L,I,I]
fn.restype=I
for m,n,k,transa in [(256,1,2048,1),(259,3,2048,1),(11,19,62,0)]:
    lda=(k if transa else m)+3
    ldb=k+5; ldc=m+7
    A=(c.c_float*(lda*(m if transa else k)))(*([777.]*(lda*(m if transa else k))))
    B=(c.c_float*(ldb*n))(*([777.]*(ldb*n)))
    C=(c.c_float*(ldc*n+8))(*([12345.]*(ldc*n+8)))
    for r in range(m):
        for p in range(k): A[r*lda+p if transa else p*lda+r]=((r*7+p*11)%71-35)*.003137
    for col in range(n):
        for p in range(k): B[col*ldb+p]=((col*13+p*17)%53-26)*.004319
        for r in range(m): C[col*ldc+r]=.125
    alpha=c.c_float(.75); beta=c.c_float(.25)
    assert fn(transa,0,m,n,k,c.byref(alpha),A,4,lda,0,B,4,ldb,0,c.byref(beta),C,4,ldc,0,1,0)==0
    worst=0.
    for col in range(n):
        for r in range(m):
            ref=.75*sum(A[r*lda+p if transa else p*lda+r]*B[col*ldb+p] for p in range(k))+.25*.125
            err=abs(C[col*ldc+r]-ref)
            assert math.isfinite(C[col*ldc+r]) and err<2e-5,(m,n,k,r,col,C[col*ldc+r],ref)
            worst=max(worst,err)
        assert list(C)[col*ldc+m:(col+1)*ldc]==[12345.]*7
    assert list(C)[ldc*n:]==[12345.]*8
    print(f'PASS MMVF m={m} n={n} k={k} transa={transa} max_abs_error={worst:.9g}',flush=True)
