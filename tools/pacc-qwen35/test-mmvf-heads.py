#!/usr/bin/env python3
"""Multi-head/sample CUDA MMVF with FP32 activations and output guards."""
import ctypes as c
import math
import os
import struct

lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so');P=c.c_void_p;U=c.c_uint64
def call(name,types,*args):
    fn=getattr(lib,name);fn.argtypes=types;fn.restype=c.c_int
    rc=fn(*args);assert rc==0,(name,rc)
def alloc(data):
    p=U();call('cuMemAlloc_v2',[c.POINTER(U),c.c_size_t],c.byref(p),len(data))
    call('cuMemcpyHtoD_v2',[U,c.c_char_p,c.c_size_t],p,data,len(data));return p
call('cuInit',[c.c_uint],0);ctx=P();call('cuCtxCreate_v2',[c.POINTER(P),c.c_uint,c.c_int],c.byref(ctx),0,0)
for half in (False,True):
  for m,n,k,heads,samples in [(7,2,30,4,2),(256,1,256,16,1)]:
    cr=8 if heads==16 else 2;sr=2 if samples==2 else 1
    sx=k*2 if heads==16 else k+3;sy=k+8;sd=m+4
    cx=m*sx+5;cy=n*sy+7;cd=n*sd+13
    px=(heads//cr)*cx+7;py=heads*cy+17;pd=heads*cd+19
    x=[((i*7)%53-26)*.003137 for i in range(px)]
    xb=struct.pack('<%d%s'%(len(x),'e' if half else 'f'),*x)
    x=struct.unpack('<%d%s'%(len(x),'e' if half else 'f'),xb)
    y=[((i*17)%71-35)*.004319 for i in range(samples*py)]
    yb=struct.pack('<%df'%len(y),*y);y=struct.unpack('<%df'%len(y),yb)
    expected=[12345.]*(samples*pd)
    for sample in range(samples):
      for head in range(heads):
        for col in range(n):
          for r in range(m):
            expected[sample*pd+head*cd+col*sd+r]=sum(x[(sample//sr)*px+(head//cr)*cx+r*sx+j]*y[sample*py+head*cy+col*sy+j] for j in range(k))
    xp,yp,dp=alloc(xb),alloc(yb),alloc(struct.pack('<%df'%len(expected),*([12345.]*len(expected))))
    class Fusion(c.Structure):_fields_=[('ptrs',U*5),('op',c.c_int)]
    ratio=lambda d:(c.c_uint32*3)(1,int(math.log2(d)),d)
    params=[xp,yp,U(0),Fusion(),dp,c.c_int(k//2),ratio(1),c.c_int(sx),c.c_int(sy//2),c.c_int(sd),ratio(cr),
      c.c_int(cx),c.c_int(cy),c.c_int(cd),ratio(sr),c.c_int(px),c.c_int(py),c.c_int(pd),c.c_int(0)]
    name=f'_ZL13mul_mat_vec_fI{"6__half" if half else "f"}fLi{n}ELi128ELb0ELb0EEvPKT_PKfPKi31ggml_cuda_mm_fusion_args_devicePfi5uint3iiiSA_iiiSA_iiii'
    decl=[f'.param .align 8 .b8 p{i}[{c.sizeof(v)}]' if isinstance(v,(c.Array,c.Structure)) else f'.param .b{c.sizeof(v)*8} p{i}' for i,v in enumerate(params)]
    ptx=('.version 7.0\n.target sm_70\n.address_size 64\n.visible .entry '+name+'('+','.join(decl)+'){ret;}').encode()
    mod=P();fun=P();call('cuModuleLoadData',[c.POINTER(P),c.c_char_p],c.byref(mod),ptx)
    call('cuModuleGetFunction',[c.POINTER(P),P,c.c_char_p],c.byref(fun),mod,name.encode())
    argv=(P*19)(*[c.addressof(v) for v in params])
    call('cuLaunchKernel',[P]+[c.c_uint]*7+[P,c.POINTER(P),P],fun,m,heads,samples,128,1,1,0,None,argv,None)
    out=c.create_string_buffer(len(expected)*4);call('cuMemcpyDtoH_v2',[P,U,c.c_size_t],out,dp,len(out))
    got=struct.unpack('<%df'%len(expected),out.raw)
    error=max(abs(a-b) for a,b in zip(got,expected))
    assert all(math.isfinite(a) and abs(a-b)<2e-5 for a,b in zip(got,expected)),(half,m,n,k,error)
    print(f'PASS MMVF CUDA half={half} heads={heads} samples={samples} m={m} n={n} k={k} max_abs_error={error:.9g} guards=unchanged',flush=True)
    call('cuModuleUnload',[P],mod)
    for p in (xp,yp,dp):call('cuMemFree_v2',[U],p)
call('cuCtxDestroy_v2',[P],ctx)
