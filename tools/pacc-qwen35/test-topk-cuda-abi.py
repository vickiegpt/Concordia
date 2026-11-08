#!/usr/bin/env python3
import ctypes as c
import math
import os
import struct
lib=c.CDLL(os.environ['CONCORDIA_RELEASE']+'/libnvcuda.so');P=c.c_void_p;U=c.c_uint64
def call(name,types,*args):
    f=getattr(lib,name);f.argtypes=types;f.restype=c.c_int;rc=f(*args);assert rc==0,(name,rc)
def alloc(data):
    p=U();call('cuMemAlloc_v2',[c.POINTER(U),c.c_size_t],c.byref(p),len(data));call('cuMemcpyHtoD_v2',[U,c.c_char_p,c.c_size_t],p,data,len(data));return p
def read(p,fmt):
    b=c.create_string_buffer(struct.calcsize(fmt));call('cuMemcpyDtoH_v2',[P,U,c.c_size_t],b,p,len(b));return struct.unpack(fmt,b.raw)
call('cuInit',[c.c_uint],0);ctx=P();call('cuCtxCreate_v2',[c.POINTER(P),c.c_uint,c.c_int],c.byref(ctx),0,0)
n,k,rows=256,8,3
logits=[c.c_float(((i*11)%37-18)*.13).value for i in range(n*rows)]
x=alloc(struct.pack('<%df'%len(logits),*logits));w=alloc(struct.pack('<%df'%(k*rows+8),*([12345.]*(k*rows+8))))
ids=alloc(struct.pack('<%di'%(n*rows+8),*([-12345]*(n*rows+8))))
params=[x,w,ids,U(0),c.c_int(rows),c.c_int(k),c.c_float(1e-6),c.c_float(1.25),c.c_uint32(1<<16)]
name='_Z13topk_moe_cudaILi256ELb0EEvPKfPfPiS2_iiff15topk_moe_config'
decl=[f'.param .b{c.sizeof(v)*8} p{i}' for i,v in enumerate(params)]
ptx=('.version 7.0\n.target sm_70\n.address_size 64\n.visible .entry '+name+'('+','.join(decl)+'){ret;}').encode()
mod=P();fn=P();call('cuModuleLoadData',[c.POINTER(P),c.c_char_p],c.byref(mod),ptx)
call('cuModuleGetFunction',[c.POINTER(P),P,c.c_char_p],c.byref(fn),mod,name.encode())
argv=(P*9)(*[c.addressof(v) for v in params])
call('cuLaunchKernel',[P]+[c.c_uint]*7+[P,c.POINTER(P),P],fn,1,1,1,32,4,1,0,None,argv,None)
gotw=read(w,'<%df'%(k*rows+8));goti=read(ids,'<%di'%(n*rows+8))
for row in range(rows):
    a=logits[row*n:(row+1)*n];chosen=sorted(range(n),key=lambda i:(-a[i],i))[:k]
    assert list(goti[row*n:row*n+k])==chosen
    ex=[math.exp(a[i]-max(a)) for i in chosen];s=sum(ex);ex=[v*1.25/s for v in ex]
    assert all(abs(gotw[row*k+j]-ex[j])<2e-6 for j in range(k))
    assert list(goti[row*n+k:(row+1)*n])==[-12345]*(n-k)
assert list(goti[n*rows:])==[-12345]*8 and list(gotw[k*rows:])==[12345.]*8
print('PASS Top-k CUDA ABI experts=256 rows=3 ids_stride=256 tie_break=lowest_index guards=unchanged')
call('cuModuleUnload',[P],mod)
for p in [x,w,ids]:call('cuMemFree_v2',[U],p)
call('cuCtxDestroy_v2',[P],ctx)
